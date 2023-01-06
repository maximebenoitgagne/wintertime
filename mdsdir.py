import sys
import os
from os.path import join as pjoin, split as psplit
import re
import glob
import MITgcmutils as mit
import numpy as np

metare = re.compile(r'^(.*)\.([0-9]{10})\.meta$')

class MDSVariable(object):
    def __init__(self, base, shape, hastime, hasrecords, fields=None):
        self.base = base
        self.shape = shape
        self.hastime = hastime
        self.hasrecords = hasrecords
        self.fields = fields
        self.iters = []

    @property
    def ndim(self):
        return len(self.shape)

    def __getitem__(self, idx):
        try:
            idx = list(idx)
        except:
            idx = [idx]
        if self.hastime:
            if len(idx) >= 1 and idx[0] is not Ellipsis:
                itrs = self.iters[idx.pop(0)]
            else:
                itrs = np.nan
        else:
            itrs = -1
        rec = None
        if self.hasrecords:
            # make a list of all requested records
            nrec = self.shape[self.hastime]
            rec = range(nrec)
            if len(idx) >= 1 and idx[0] is not Ellipsis:
                rec = rec[idx.pop(0)]
        levs = []
        for i,slc in enumerate(idx):
            assert i is not Ellipsis
            if not isinstance(slc, slice):
                slc = np.s_[slc:slc+1]
            n = self.shape[i+self.hastime+self.hasrecords]
            start,stop,step = slc.indices(n)
            if levs or (start,stop,step) != (0,n,1):
                levs.append(range(start, stop, step))
        levs = tuple(levs)
#        print self.base, itrs, levs
        return mit.rdmds(self.base, itrs, rec=rec, lev=levs)


class MDSDir(object):
    def __init__(self, dir):
        self.dir = dir
        self.findmds()

    def findmds(self):
        variables = {}
        for metafile in glob.glob(self.dir + '/*.meta'):
            _,fname = os.path.split(metafile)
            if fname[:6] != 'pickup':
                m = metare.match(fname)
                if m:
                    base = m.group(1)
                    iter = m.group(2)
                else:
                    base = fname[:-5]
                    iter = None
                hastime = iter is not None
                if base not in variables: 
                    gdims,i0s,ies,timestep,timeinterval,map2gl,meta = mit.mds.readmeta(metafile)
                    hasrecords = meta['nrecords'][0] > 1
                    if hasrecords:
                        shape = (meta['nrecords'][0],) + gdims
                    else:
                        shape = gdims
                    fields = meta.get('fldList', None)
                    variables[base] = MDSVariable(self.dir + '/' + base, shape,
                            hastime, hasrecords, fields)
                if hastime:
                    variables[base].iters.append(int(iter))
        for v in variables.values():
            if v.hastime:
                v.iters.sort()
                v.shape = (len(v.iters),) + v.shape
        self.variables = variables

class mds_reader(object):
    def __init__(self, dir):
        self.dir = dir

    @property
    def variables(self):
        return self

    def __getitem__(self, key):
        return mit.rdmds(self.dir + '/' + key)

    def __contains__(self, key):
        return bool(glob.glob(self.dir + '/' + key + '.*meta'))


class MDSRecord(object):
    def __init__(self, mdsvariable, idx=None):
        self.v = mdsvariable
        self.idx = idx

    @property
    def shape(self):
        return self.v.shape[:self.v.hastime] + self.v.shape[self.v.hastime+1:]

    @property
    def ndim(self):
        return len(self.shape)

    def __getitem__(self, idx):
        '''set record index if unset, else index underlying variable

        The record index is placed in the record dimension of the variable.
        '''
        if self.idx is None:
            return MDSRecord(self.v, idx)
        else:
            try:
                iter(idx)
            except:
                idx = (idx,)
            if self.v.hastime:
                if idx[0] is Ellipsis or idx[0] is slice(None, None, None):
                    idx = np.s_[:, self.idx] + idx
                else:
                    idx = idx[:1] + (self.idx,) + idx[1:]
            else:
                idx = (self.idx,) + idx
            return self.v[idx]


class MDSDirs(object):
    def __init__(self, root):
        self.populate(root)

    def populate(self, root):
        self.d = {}
        self.v = {}
        for f in os.listdir(root):
            dname = pjoin(root, f)
            d = MDSDir(dname)
            v = d.variables['_']
            self.d[f] = v
            if v.fields:
                for i, k in enumerate(v.fields):
                    self.v[k] = MDSRecord(self.d[f], i)


