#!/usr/bin/env python
from __future__ import print_function
import sys
import os
from collections import OrderedDict
import numpy as np
import pandas as pd
from h5py import File
from h5fa import FaFile
from gud import iofmt, ionum

ncs = 510
nface = 6
nhorz = ncs*ncs*nface

offlinedir = '/net/eofe-data002/micklab002/jahn/ecco2/offline'
offlinefields = 'ETAN KPPdiffS KPPghatK SALTanom THETA UVELMASS VVELMASS'.split()

def reshape(a, shape):
    try:
        res = np.reshape(a, shape)
    except:
        sys.stderr.write('Array size ' + str(a.size) + ' shape ' + str(shape) + '\n')
        raise

    return res


def concentrations(d):
    o = OrderedDict()
    for k in d:
        if k[:4] == 'TRAC':
            desc = d[k].description
            name = desc.replace(' concentration', '')
            o[name] = d[k]
    return o


########################################################################
# from params.py

class RunInfo(object):
    def __init__(self, rundir=None):
        if rundir:
            self.read(rundir)

    def read(self, rundir='.'):
        import fortran
        import nml
        self.names = [s.strip("'") for s in fortran.readnmlparam(rundir + '/data.ptracers','ptracers_names')]
#        grpnames = fortran.readnmlparam(rundir + '/data.gud','grp_names')
#        if grpnames:
#            if ',' in grpnames:
#                grpnames = grpnames.strip(', ').split(',')
#            self.grpnames = [s.strip("' ") for s in grpnames]
#        else:
#            self.grpnames = None
        fh=open(rundir + '/data.gud','r')
        while True:
            line=fh.readline()
            if '&GUD_TRAIT_PARAMS' in line:
                fh.readline()
                line=fh.readline()
                grpnames=line.split()
                grpnames.pop(0)
                break
            if not line:
                break
        fh.close()
        self.grpnames = grpnames
        ndig = 1
        while 'c{0:0{d}d}'.format(1, d=ndig) not in self.names and ndig < 10:
            ndig += 1
        self.format = ('{0}{1:0%dd}'%ndig).format
        self.ic = self.names.index('c{0:0{d}d}'.format(1, d=ndig))
        Chlname = 'Chl{0:0{d}d}'.format(1, d=ndig)
        if Chlname in self.names:
            self.iChl = self.names.index(Chlname)
        for i in range(len(self.names)-self.ic):
            if self.names[self.ic+i][0] != 'c':
                break
        self.nplk = i

        startdir = os.path.join(rundir, 'start')
        if not os.path.exists(os.path.join(startdir, 'gud_traits.txt')):
            startdir = rundir
        self.tr = nml.NmlFile(os.path.join(startdir, 'gud_traits.txt')).merge()
        try:
            self.nphy = self.tr['isphoto'].sum()
        except KeyError:
            for i in range(len(self.names)-self.iChl):
                if self.names[self.iChl+i][:3] != 'Chl':
                    self.nphy = i
                    break
            else:
                self.nphy = len(self.names) - self.iChl
            print('nphy', self.nphy)
        self.nzoo = self.nplk - self.nphy
        d = dict((k, v) for k, v in self.tr.items() if v.shape == (self.nplk,))
        self.f = pd.DataFrame(d)
        self.params = nml.NmlFile(os.path.join(startdir, 'gud_params.txt')).merge()
#         if 'grp_names' in self.params:
#             self.grpnames = [str.strip(_) for _ in self.params['grp_names']]
#         else:
#             self.grpnames = None
        if os.path.exists(os.path.join(startdir, 'gud_radtrans.txt')):
            self.rtparams = nml.NmlFile(os.path.join(startdir, 'gud_radtrans.txt')).merge()
            self.nlam = len(self.rtparams['wb_center'])
            d = dict((k, v.reshape(self.nlam, self.nplk)) for k, v in self.tr.items()
                                                if v.shape == (self.nplk*self.nlam,))
            self.rttraits = d
        self.grp = self.tr['group'] - 1
        self.vol = self.tr['biovol']
        # vol = (4*pi/3)*(dm/2)**3 = (pi/6)*dm**3
        # dm = (6*vol/np.pi)**(1./3)
        self.esd = (6*self.vol/np.pi)**(1./3)
        if len(self.vol) > 1:
            self.dlv = np.median(np.diff(sorted(set(np.log(self.vol)))))
            self.ivol = np.round(np.log(self.vol/self.vol.min())/self.dlv).astype(int)
        else:
            self.ivol = len(self.vol)*[0]
        self.ngrp = np.max(self.grp) + 1
        self.nvol = np.max(self.ivol) + 1
        self.f['grp'] = self.grp
        self.f['vol'] = self.vol
        self.f['esd'] = self.esd
        self.f['ivol'] = self.ivol
        if self.grpnames:
            self.f['group'] = np.r_[list(self.grpnames)][self.grp]
        self.esds = np.zeros((self.nvol,))
        self.esds.put(self.ivol, self.esd)
        self.group = list(self.f['group'])
        di = {}
        di['P'] = 'Prochlorococcus'
        di['S'] = 'Synechococcus'
        di['s'] = 'Small Eukaryotes'
        di['C'] = 'Other Eukaryotes'
        di['z'] = 'Diazotrophs'
        di['T'] = 'Trichodesmium'
        di['D'] = 'Diatoms'
        di['l'] = 'Mixo Dino'
        di['Z'] = 'Zooplankton'
        longgroup = [''] * len(self.group)
        for i, g in enumerate(self.group):
            longg = g
            if g in di:
                longg = di[g]
            longgroup[i] = longg
        self.longgroup = longgroup
        self.f['longgroup'] = self.longgroup
        # ref: https://plotnine.readthedocs.io/en/stable/tutorials/miscellaneous-order-plot-series.html#
        from pandas.api.types import CategoricalDtype
        # Create a categorical type
        longgroup_list=[]
        previous_grp=None
        for i,grp in enumerate(self.grp):
            if grp!=previous_grp:
                longgroup=self.f.loc[i,'longgroup']
                longgroup_list.append(longgroup)
            previous_grp=grp
        longgroup_cat = CategoricalDtype(categories=longgroup_list, ordered=True)
        # Cast the existing categories into the new category. Due to a bug in pandas
        # we need to do this via a string.
        self.f = self.f.assign(longgroup=self.f['longgroup'].astype(str).astype(longgroup_cat))

    def xr(self):
        import xarray as xr
        ds = xr.Dataset()
        if hasattr(self, 'rtparams'):
            d = dict(self.rtparams)
            ds['waveband'] = d.pop('wb_center')
            params = dict(self.params)
            ds['waveband_edges'] = params.pop('gud_waveband_edges')
            nopt = len(d['asize'])
            for k in d:
                v = d[k]
                shape = v.shape
                if v.shape == (self.nlam,):
                    dims = ('waveband',)
                elif v.shape == (nopt,):
                    dims = ('opttype',)
                elif v.shape == (1,):
                    dims = ()
                    shape = ()
                elif v.shape == (nopt*self.nlam,):
                    dims = ('opttype', 'waveband')
                    shape = (nopt, self.nlam)
                else:
                    dims = None
                if dims is None:
                    ds[k] = xr.DataArray(v)
                else:
                    ds[k] = (dims, v.reshape(shape))
            dimsize = {
                'group': self.ngrp,
                'plankton': self.nplk,
                'waveband': self.nlam,
                }
            dimmap = {
                1: (),
                self.ngrp: ('group',),
                self.nplk: ('plankton',),
                self.ngrp*self.ngrp: ('group', 'group'),
                self.ngrp*self.nplk: ('group', 'plankton'),
                self.nplk*self.nplk: ('plankton', 'plankton'),
                self.nlam*self.nplk: ('plankton', 'waveband'),
                self.nlam: ('waveband',),
                }
            for k in sorted(params):
                v = params[k]
                dims = dimmap[v.size]
                sh = tuple(dimsize[d] for d in dims)
                v = v.reshape(sh)
                print(k, dims, sh)
#                if v.dtype.kind == 'S' and v.size == 1:
#                    n = max(256, v.dtype.itemsize)
#                    v = v.astype( '|S'+str(n))
#                    ds.attrs[k] = v.tostring()
#                    continue
                if v.dtype.kind == 'S':
                    n = max(128, v.dtype.itemsize)
                    v = v.astype( '|S'+str(n))

                ds[k] = (dims, v.reshape(sh))

            traits = dict(self.tr)
            for k in sorted(traits):
                if k not in ds:
                    v = traits[k]
                    dims = dimmap[v.size]
                    sh = tuple(dimsize[d] for d in dims)
                    print(k, dims, sh)
                    ds[k] = (dims, v.reshape(sh))

        return ds

    def groups(self, p):
        n = len(p)
        assert n in [self.nplk, self.nphy]
        grp = np.zeros((self.ngrp,) + p[0].shape, p[0].dtype)
        count = np.zeros((self.ngrp,), int)
        for g in range(self.ngrp):
            for i in range(n):
                if self.grp[i] == g:
                    grp[g] += p[i]
                    count[g] += 1

            sz = np.sum(self.grp == g)
            if count[g] != sz:
                if count[g] == 0:
                    sys.stderr.write('Warning: group {} {} not present.\n'.format(g, self.grpnames[g]))
                else:
                    raise ValueError(
                    'Group {} has {} instead of {} members, have {} plankton need {} or {}\n'.format(
                               g, count[g], sz, n, self.nplk, self.nphy))

        return grp


    def sizes(self, p):
        assert p.shape[0] in [self.nplk, self.nphy]
        a = np.zeros((self.nvol,) + p.shape[1:], p.dtype)
        n = np.zeros((self.nvol,), int)
        for iv in range(self.nvol):
            for i in range(p.shape[0]):
                if self.ivol[i] == iv:
                    a[iv] += p[i]
                    n[iv] += 1

            sz = np.sum(self.ivol[:p.shape[0]] == iv)
            if n[iv] != sz:
                if n[iv] == 0:
                    sys.stderr.write('Size {} not present.\n'.format(iv))
                else:
                    sys.stderr.write(str(self.ivol) + '\n')
                    raise ValueError(
                    'Size {} has {} instead of {} members, have {} plankton need {} or {}\n'.format(
                              iv, n[iv], sz, p.shape[0], self.nplk, self.nphy))

        return a

    sizeclassnames=['pico', 'nano', 'micro', 'meso', 'macro']

    def sizeclasses(self, p, boundaries=[2.5, 20., 300., 1500.]):
        assert p.shape[0] in [self.nplk, self.nphy]
        o = np.zeros((len(boundaries)+1,) + p.shape[1:])
        for i in range(len(p)):
            c = np.searchsorted(boundaries, self.esd[i])
            o[c] += p[i]

        return o

    def mdsdirs(self, root):
        return GUDDirs(root, self)


import pickle
from mdsdir import MDSDirs
from slices import fixndim

class Group(object):
    def __init__(self, groups, name):
        self.g = groups
        self.name = name

    @property
    def shape(self):
        s = self.g.ptr.shape
        return s[:self.g.ptr.hastime] + s[self.g.ptr.hastime+1:]

    @property
    def ndim(self):
        return self.g.ptr.ndim - 1

    def __getitem__(self, idx):
        g = self.g[idx]
        return g[self.name]


class Groups(object):
    def __init__(self, ds, nplank, cache=False):
        self.ds = ds
        self.nplank = nplank
        self.cache = cache
        self.ptr = ds.d['ptr']
        self.ic = self.ds.info.ic
        self.names = self.ds.info.grpnames
        if 'Pro' in self.names and 'Syn' in self.names:
            self.names.append('Prok')
            if 'PicoEuk' in self.names:
                self.names.append('Picophy')
        self._cache = {}

    def __getitem__(self, idx):
        idx = fixndim(idx, self.ptr.ndim - 3)
        hsh = pickle.dumps(idx)
        if hsh in self._cache:
            g = self._cache[hsh]
        else:
            idx = idx[:self.ptr.hastime] + np.s_[self.ic:self.ic+self.nplank,] + idx[self.ptr.hastime:]
            p = self.ptr[idx]
            # read
            if self.ptr.hastime:
                idxs = np.arange(self.ptr.shape[0])[idx[0]]
                if np.ndim(idxs):
                    p = p.swapaxes(0, 1)
            g = dict(zip(self.ds.info.grpnames, self.ds.info.groups(p)))
            if 'Prok' in self.names:
                g['Prok'] = g['Pro'] + g['Syn']
            if 'Picophy' in self.names:
                g['Picophy'] = g['Prok'] + g['PicoEuk']

            if self.cache:
                self._cache[hsh] = g

        return g

    def variables(self):
        return {name: Group(self, name) for name in self.names}


class GUDDirs(MDSDirs):
    def __init__(self, root, info):
        self.populate(root)
        self.info = info
        self._cache = {}
        # make named ptracer variables, tendencies, etc.
        for spre, tpre in [('', 'TRAC'), ('gTrp', 'gTr')]:
            for i, name in enumerate(info.names):
                tgt = tpre + iofmt(i+1)
                if tgt in self.v:
                    self.v[spre+name] = self.v[tgt]

        self.groups = g = Groups(self, info.nplk)
        for k, v in g.variables().items():
            if k not in self.v:
                self.v[k] = v

if __name__ == '__main__':
    rundir, ncname = sys.argv[1:]
    info = RunInfo(rundir)
    ds = info.xr()
    ds.to_netcdf(ncname)
