from __future__ import print_function
import sys
import string
from collections import OrderedDict
#from functools32 import lru_cache
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(threadName)s: %(module)s: %(message)s')

formatter = string.Formatter()

def info(*args):
#    sys.stderr.write(' '.join(map(str, args)) + '\n')
    logging.info('%s' + (len(args) - 1)*' %s', *map(str, args))


def sed1(slc, n):
    try:
        s, e, d = slc.indices(n)
    except AttributeError:
        s = slc
        e = slc + 1
        d = 1

    return s, e, d


def size1(slc, n):
    try:
        s,e,d = slc.indices(n)
    except AttributeError:
        sh = ()
    else:
        sh = ((e + d - 1 - s)//d,)

    return sh


def sliceshape(slcs, dims):
    if Ellipsis in slcs:
        i = slcs.index(Ellipsis)
        slcs = slcs[:i] + (len(dims) + 1 - len(slcs))*np.s_[:,] + slcs[i+1:]
    elif len(slcs) < len(dims):
        slcs += (len(dims) - len(slcs))*np.s_[:,]
    return sum((size1(s, n) for s, n in zip(slcs, dims)), ())


class FlatIndexer:
    def __init__(self, tilereader):
        self.reader = tilereader

    def __getitem__(self, idx):
        if type(idx) != type(()):
            idx = (idx,)

        j, f, i = np.unravel_index(idx[-1], (self.reader.ncs, 6, self.reader.ncs))

        return self.reader[idx[:-1] + (j, f, i)]


class FacetIndexer:
    def __init__(self, tilereader):
        self.reader = tilereader
        sh = self.reader.shape
        self.shape = sh[-2:-1] + sh[:-2] + sh[-1:]

    def __getitem__(self, idx):
        if type(idx) != type(()):
            idx = (idx,)

        # make length self.ndim
        if Ellipsis in idx:
            i = idx.index(Ellipsis)
            idx = idx[:i] + (self.ndim + 1 - len(idx))*np.s_[:,] + idx[i+1:]
        elif len(idx) < self.ndim:
            idx += (self.ndim - len(idx))*np.s_[:,]

        odx = idx[1:-1] + idx[:1] + idx[-1:]
        #sys.stderr.write('i: ' + str(idx) +  '\no: ' + str(odx) +  '\n')

        o = self.reader.get(odx)
        n = self.ndim
        return o.transpose([n - 2] + range(n - 2) + [n - 1])

    @property
    def dtype(self):
        return self.reader.dtype

    @property
    def ndim(self):
        return len(self.shape)

class TileReader:
    '''
    Shape is (..., ncs, 6, ncs) with ncs = 510 for now.

    Two forms of indexing are supported:

    1. all slices or integers
    2. last 3 are arrays of integers of equal shape, others slices or integers

    This can be fed to dask.array.from_array with compatible chunk size,
    which will use the first form of indexing.
    '''
    def __init__(self, tmpl, dtype='>f4', shape=(), astype=None, 
                 fileindices={}, recdims=0, cache=0, cachemem=0, verbose=True):
        '''
        tmpl   :: file name template, e.g.,
                  'dir/{0}/res_{p:04d}/_.{1:010d}.{t:03d}.001.data'
                  p will be itile, t will be itile + 1, numbered fields
                  0, ... will form initial slots
                  numbers must start with 0, e.g.,
                  'res_{p:04d}/_.{0:010d}.{t:03d}.001.data'
        dtype  :: dtype of files
        shape  :: shape excluding horizontal
        astype :: dtype of returned values
        '''
        self.tmpl = tmpl
        self.itype = np.dtype(dtype)
        self.dtype = astype or self.itype.newbyteorder('=')
        self.pshape = shape
        self.ncs = 510
        self.tnx = 102
        self.tny = 51
        self.ntx = 5
        self.nty = 10
        self.verbose = verbose
        self.recdims = recdims
        self.recshape = self.pshape[self.recdims:] + (self.tny, self.tnx)
        self.recitems = int(np.prod(self.recshape))
        self.recsize = self.recitems*self.itype.itemsize
        if recdims:
            self.recindices = np.ravel_multi_index(np.indices(shape[:recdims]), shape[:recdims])
        else:
            self.recindices = np.array(0)
        if cachemem != 0:
            nmax = cachemem//self.recsize
            if cache:
                cache = min(cache, nmax)
            else:
                cache = nmax
        self.ncache = cache
        self.cache = OrderedDict()
        # numerical field names map to indices; count them
        d = {}
        n = 0
        l = formatter.parse(tmpl)
        for s, name, fmt, conv in l:
            if name is not None and name[:1] in string.digits:
                i = int(name)
                if i in fileindices:
                    d[i] = fileindices[i]
                else:
                    raise UserWarning('tileio: need to provide fileindices')

                n = max(n, i + 1)

        self.fileidx = [d[i] for i in range(n)]
        self.nfileidx = n

        self.shape = tuple(len(i) for i in self.fileidx) + shape + (510, 6, 510)

        self.ndim = len(self.shape)

#    @property
#    def ndim(self):
#        return len(self.shape)

    @property
    def chunks(self):
        return (self.nfileidx + self.recdims)*(1,) + self.pshape[self.recdims:] + (self.tny, 1, self.tnx)

    @property
    def maxchunkmem(self):
        return self.ncache*self.recsize

    def __getitem__(self, idx):
        # NB: no bool indexing so far!
        # make sure tuple
        if type(idx) != type(()):
            idx = (idx,)

        # make length self.ndim
        for i, ix in enumerate(idx):
            if ix is Ellipsis:
                idx = idx[:i] + (self.ndim + 1 - len(idx))*np.s_[:,] + idx[i+1:]
                break
        else:
            if len(idx) < self.ndim:
                idx += (self.ndim - len(idx))*np.s_[:,]

        return self.get(idx)

    def get(self, idx):
        # separate file sel., in-file and tile indices:
        #   1, ..., nfileidx, ..., infile, ..., j, f, i
        fshape = sliceshape(idx[:self.nfileidx], self.shape[:self.nfileidx])
        idxfile = [self.fileidx[i][idx[i]] for i in range(self.nfileidx)]
        jfi = idx[-3:]
        idx = idx[self.nfileidx:-3]

        allslice = True
        for x in jfi:
            if not isinstance(x, slice) and not np.isscalar(x):
                allslice = False

        if allslice:
            oshape = sliceshape(jfi, self.shape[-3:])
            slcs = []
            for d, slc in enumerate(jfi):
                if np.isscalar(slc):
                    l = [slc]
                else:
                    l = np.arange(self.shape[-3+d])[slc]

                slcs.append(l)

            j, f, i = slcs
            it, ti = divmod(np.array(i), self.tnx)
            jt, tj = divmod(np.array(j), self.tny)
            jt, ft, it = np.meshgrid(jt, f, it, indexing='ij', sparse=True)
            itile = (ft*self.nty + jt)*self.ntx + it

            recs = self.recindices[idx[:self.recdims]]
            sh = sliceshape(idx, self.pshape)
#            if self.verbose:
#                info('o shape', sh + itile.shape)
            fsh = tuple(map(len, idxfile))
            o = np.empty(fsh + sh + itile.shape, self.dtype)
            for fodx in np.ndindex(*fsh):
                fidx = [idxf[odxf] for idxf, odxf in zip(idxfile, fodx)]
                for t in set(itile.flat):
                    J, F, I = np.where(itile == t)
                    fname = self.tmpl.format(*fidx, p=t, t=t+1)
                    fh = None
                    try:
                        for odx, rec in np.ndenumerate(recs):
    #                        info('odx', odx, rec)
                            key = (fname, rec)
                            try:
                                a = self.cache[key]
                            except KeyError:
                                if self.verbose:
                                    if self.recdims:
                                        recidx = np.unravel_index(rec, self.pshape[:self.recdims])
                                    else:
                                        recidx = rec
                                    info('reading', fname, recidx)
                                if fh is None:
                                    fh = open(fname, 'r')
                                fh.seek(rec*self.recsize)
                                a = np.fromfile(fh, self.itype, self.recitems).reshape(self.recshape)
                                if self.ncache:
                                    count = 0
                                    while count < 10 and len(self.cache) >= self.ncache:
                                        try:
                                            self.cache.popitem(last=False)
                                        except KeyError:
                                            pass

                                        count += 1

                                    if self.verbose > 1:
                                        info('adding', *key)
                                    self.cache[key] = a

                            rhs = a[idx[self.recdims:]][..., tj[J], ti[I]]
                            #info('SET:', o.shape, odx, J.shape, F.shape, I.shape, rhs.shape)
    #                        sys.stderr.write(str(('SET:', o.shape, odx, J.shape, F.shape, I.shape, rhs.shape))+'\n')
                            # list index is always first, so need to transpose rhs
                            if fodx or odx:
                                rhs = np.rollaxis(rhs, -1, 0)
                            o[fodx + odx + (Ellipsis, J, F, I)] = rhs
                    finally:
                        if fh is not None:
                            fh.close()
        else:
            # tile indices are lists of integers
            oshapes = [np.array(i).shape for i in jfi]
            assert oshapes[0] == oshapes[1] and oshapes[1] == oshapes[2]
            oshape = oshapes[0]

            j, f, i = jfi
            it, ti = divmod(i, self.tnx)
            jt, tj = divmod(j, self.tny)
            itile = (f*self.nty + jt)*self.ntx + it

            sh = sliceshape(idx, self.pshape)
            o = np.empty(sh + itile.shape, self.dtype)
            for t in set(itile.flat):
                I = np.where(itile == t)
                fname = self.tmpl.format(*idxfile, p=t, t=t+1)
                if self.verbose:
                    info('reading ', fname)
                a = np.fromfile(fname, self.itype).reshape(self.pshape + (self.tny, self.tnx))[idx]
                o[np.s_[...,] + I] = a[..., tj[I], ti[I]]

        return o.reshape(fshape + sh + oshape)

    @property
    def csflat(self):
        return FlatIndexer(self)

    def todask(self, facets=False):
        import dask.array as da
        a = da.from_array(self, self.chunks)
        if facets:
            # move facet (second-last) axis to front
            axes = [self.ndim - 2] + range(self.ndim - 2) + [self.ndim - 1]
            a = a.transpose(axes)

        return a


if __name__ == '__main__':
    tmpl = 'n/rt/res_{p:04d}/_.0000070560.{t:03d}.001.data'
    tp = '>f4'
    sh = (5, 13, 50)
    recdims = 2  # read all levels at once, but only one field

    io = TileReader(tmpl, tp, sh, verbose=2, recdims=recdims, cache=100)
#    a = da.from_array(io, io.chunks)
    a = io.todask()
    aa = a[0, 0, 0, 30:60, 3:5, 360:430].sum().compute()
    print(aa)
#    f = a.transpose([4, 0, 1, 2, 3, 5])
    f = io.todask(facets=True)
    ff = f[3:5, 0, 0, 0, 30:60, 360:430].sum().compute()
    print(ff)

