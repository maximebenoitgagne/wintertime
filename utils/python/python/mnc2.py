#!/usr/bin/env python
import sys
import os
import glob
import re
import numpy as np
from scipy.io.netcdf import netcdf_file

def sliceindices(slices, dims):
    try:
        slices[0]
    except TypeError:
        slices = (slices,)

    if Ellipsis in slices:
        cut = slices.index(Ellipsis)
        slices = slices[:cut] + (len(dims)-len(slices)+1)*(slice(0,None,None),) + slices[cut+1:]

    slices = slices + (len(dims)-len(slices))*(slice(0,None,None),)
    return tuple( hasattr(s,'indices') and s.indices(dim) or s for s,dim in zip(slices,dims) )


def sliceshape(slices, dims):
    ssss = sliceindices(slices,dims)

    res = []
    for sss in ssss:
        try:
            start,stop,step = sss
        except TypeError:
            pass
        else:
            res += [(stop-start+step-1)//step]

    return tuple(res)


class MNC(object):
    def __init__(self, patt):
        files = glob.glob(patt)
        self.tiles = [ int(re.search(r'\.t([0-9]*)\.nc$',f).group(1)) for f in files ]
        self.ds = [ netcdf_file(f,'r') for f in files ]
        ds = self.ds[0]
        self.Nx = int(ds.Nx)
        self.Ny = int(ds.Ny)
        self.Nr = int(ds.Nr)
        self.Nt = np.size(ds.variables['T'])
        self.dims = {'T': self.Nt, 'Z': self.Nr, 'Y':self.Ny, 'X':self.Nx}
        self.shape = (self.Nr, self.Ny, self.Nx)
        self.sNx = int(ds.sNx)
        self.sNy = int(ds.sNy)
        self.variables = ds.variables.keys()
        self.variables.sort()
        xs = list(set(np.concatenate([ds.variables['X'][:] for ds in self.ds])))
        ys = list(set(np.concatenate([ds.variables['Y'][:] for ds in self.ds])))
        xs.sort()
        ys.sort()
        self.i0 = [ xs.index(ds.variables['X'][0]) for ds in self.ds ]
        self.j0 = [ ys.index(ds.variables['Y'][0]) for ds in self.ds ]
        self.ie = [ i + self.sNx for i in self.i0 ]
        self.je = [ j + self.sNy for j in self.j0 ]

    def __getitem__(self,name):
        res = MNCvariable()
        res.mnc = self
        res.name = name
        var0 = self.ds[0].variables[name]
        for key in ['dimensions', 'description', 'ncattrs', 'ndim', 'units', 'dtype']:
            if hasattr(var0, key):
                setattr(res, key, getattr(var0, key))

        res.shape = tuple( self.dims[dim] for dim in var0.dimensions )
        return res

    def get(self, name, s=Ellipsis):
        var0 = self.ds[0].variables[name]
        dims = tuple( int(self.dims[dim]) for dim in var0.dimensions )
        ndims = len(dims)
        resshape = sliceshape(s, dims)
        try:
            s2 = (len(resshape)-2)*np.s_[:,] + s[ndims-2:]
            s = s[:ndims-2]
            resshape = resshape[:-2] + dims[-2:]
        except TypeError:
            s2 = Ellipsis

        res = np.zeros(sliceshape(s, dims), var0.data.dtype)
        if 'X' in var0.dimensions and 'Y' in var0.dimensions:
            for i0,ie,j0,je,ds in zip(self.i0,self.ie,self.j0,self.je,self.ds):
                res[...,j0:je,i0:ie] = ds.variables[name][s]
        elif 'X' in var0.dimensions:
            for i0,ie,ds in zip(self.i0,self.ie,self.ds):
                res[...,i0:ie] = ds.variables[name][s]
        elif 'Y' in var0.dimensions:
            for j0,je,ds in zip(self.j0,self.je,self.ds):
                res[...,j0:je] = ds.variables[name][s]
        else:  # assume same in each tile
            res[...] = var0[s]

        return res[s2]


    def getvec(self,patt,s=Ellipsis):
        regexp = re.compile(re.sub(r'%[\.0-9 -+]*d', '([0-9]+)', patt) + '$')
        names = {}
        for name in self.variables:
            m = regexp.match(name)
            if m:
                i = int(m.group(1))
                names[i] = name

        keys = names.keys()
        keys.sort()
        var0 = self.get(names[keys[0]], s)
        res = np.zeros((len(keys),)+var0.shape)
        offset = keys[0]
        for i in keys:
            res[i-offset] = self.get(names[i], s)

        return res
            

    def close(self):
        for ds in self.ds:
            ds.close()


class MNCvariable(object):
    def __getitem__(self, s=Ellipsis):
        return self.mnc.get(self.name, s)


def rdmnc(patt, var, s=Ellipsis):
    mnc = MNC(patt)
    res = mnc.get(var,s)
    mnc.close()
    return res


