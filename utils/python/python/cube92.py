from __future__ import print_function
import sys
import os
import numpy as np
from h5py import File
from h5fa import FaFile

ncs = 510
nface = 6
nhorz = ncs*ncs*nface

offlinedir = '/nfs/micklab002/jahn/ecco2/offline'
offlinefields = 'ETAN KPPdiffS KPPghatK SALTanom THETA UVELMASS VVELMASS'.split()

ptracer_names = [
    'DIC', 'NH4', 'NO2', 'NO3', 'PO4', 'SiO2', 'FeT', 'DOC', 'DON', 'DOP',
    'DOFe', 'POC', 'PON', 'POP', 'POSi', 'POFe', 'PIC', 'ALK', 'O2', 'CDOM',
    'c01', 'c02', 'c03', 'c04', 'c05', 'c06', 'c07', 'c08', 'c09', 'c10',
    'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'c19', 'c20',
    'c21', 'c22', 'c23', 'c24', 'c25', 'c26', 'c27', 'c28', 'c29', 'c30',
    'c31', 'c32', 'c33', 'c34', 'c35', 'c36', 'c37', 'c38', 'c39', 'c40',
    'c41', 'c42', 'c43', 'c44', 'c45', 'c46', 'c47', 'c48', 'c49', 'c50',
    'c51', 'Chl01', 'Chl02', 'Chl03', 'Chl04', 'Chl05', 'Chl06', 'Chl07', 'Chl08', 'Chl09',
    'Chl10', 'Chl11', 'Chl12', 'Chl13', 'Chl14', 'Chl15', 'Chl16', 'Chl17', 'Chl18', 'Chl19',
    'Chl20', 'Chl21', 'Chl22', 'Chl23', 'Chl24', 'Chl25', 'Chl26', 'Chl27', 'Chl28', 'Chl29',
    'Chl30', 'Chl31', 'Chl32', 'Chl33', 'Chl34', 'Chl35',
    ]

mon_days = np.r_[5*[
        31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 1992
        31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 1993
        31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 1994
        31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 1995
        ]]

def reshape(a, shape):
    try:
        res = np.reshape(a, shape)
    except:
        sys.stderr.write('Array size ' + str(a.size) + ' shape ' + str(shape) + '\n')
        raise

    return res

########################################################################
# from params.py

class RunInfo(object):
    def __init__(self, rundir='.'):
        import fortran
        import nml
        self.dir = rundir
        self.names = [s.strip("'") for s in fortran.readnmlparam(rundir + '/data.ptracers','ptracers_names')]
        grpnames = fortran.readnmlparam(rundir + '/data.gud','grp_names')
        if ',' in grpnames:
            grpnames = grpnames.strip(', ').split(',')
        self.grpnames = [s.strip("' ") for s in grpnames]
        self.ic = self.names.index('c01')
        for i in range(len(self.names)-self.ic):
            if self.names[self.ic+i][0] != 'c':
                break
        self.nplk = i
        self.nphy = int(self.names[-1][3:])
        self.nzoo = self.nplk - self.nphy

        startdir = os.path.join(rundir, 'start')
        if not os.path.exists(os.path.join(startdir, 'gud_traits.nml')):
            startdir = rundir
        self.tr = nml.NmlFile(os.path.join(startdir, 'gud_traits.nml')).merge()
        self.params = nml.NmlFile(os.path.join(startdir, 'gud_params.nml')).merge()
        if os.path.exists(os.path.join(startdir, 'gud_radtrans.nml')):
            self.rtparams = nml.NmlFile(os.path.join(startdir, 'gud_radtrans.nml')).merge()
        self.grp = self.tr['group'] - 1
        self.vol = self.tr['biovol']
        # vol = (4*pi/3)*(dm/2)**3 = (pi/6)*dm**3
        # dm = (6*vol/np.pi)**(1./3)
        self.esd = (6*self.vol/np.pi)**(1./3)
        self.dlv = np.median(np.diff(sorted(set(np.log(self.vol)))))
        self.ivol = np.round(np.log(self.vol/self.vol.min())/self.dlv).astype(int)
        self.ngrp = np.max(self.grp) + 1
        self.nvol = np.max(self.ivol) + 1


def groups(p, info):
    assert p.shape[0] in [info.nplk, info.nphy]
    grp = np.zeros((info.ngrp,) + p.shape[1:], p.dtype)
    n = np.zeros((info.ngrp,), int)
    for g in range(info.ngrp):
        for i in range(p.shape[0]):
            if info.grp[i] == g:
                grp[g] += p[i]
                n[g] += 1

        sz = np.sum(info.grp == g)
        if n[g] != sz:
            if n[g] == 0:
                sys.stderr.write('Warning: group {} {} not present.\n'.format(g, info.grpnames[g]))
            else:
                raise ValueError(
                'Group {} has {} instead of {} members, have {} plankton need {} or {}\n'.format(
                           g, n[g], sz, p.shape[0], info.nplk, info.nphy))

    return grp


def sizeclasses(p, info):
    assert p.shape[0] in [info.nplk, info.nphy]
    a = np.zeros((info.nvol,) + p.shape[1:], p.dtype)
    n = np.zeros((info.nvol,), int)
    for iv in range(info.nvol):
        for i in range(p.shape[0]):
            if info.ivol[i] == iv:
                a[iv] += p[i]
                n[iv] += 1

        sz = np.sum(info.ivol == iv)
        if n[iv] != sz:
            if n[iv] == 0:
                sys.stderr.write('Size {} not present.\n'.format(iv))
            else:
                raise ValueError(
                'Size {} has {} instead of {} members, have {} plankton need {} or {}\n'.format(
                          iv, n[iv], sz, p.shape[0], info.nplk, info.nphy))

    return a

########################################################################


class Files:
    def __init__(self):
        self.files = {}
        self.filenames = {
            'grid': '/nfs/micklab002/jahn/h5/cube84/grid.h5',
            'fagrid': '/nfs/micklab002/jahn/h5/cube84/fagrid.h5',
            'all':  '/nfs/micklab002/jahn/h5/cube84/ro/all.h5',
            'phy':  '/nfs/micklab001/jahn/h5/cube84/ro/L30phy.h5',
        }

    def open(self, name, mode='r', reader=File):
        # if it hasn't been opened or it has been closed, we have to open it
        if name not in self.files or not self.files[name].id:
            self.files[name] = reader(self.filenames[name], mode)
        return self.files[name]

#    @property
#    def grid(self):
#        # if it hasn't been opened or it has been closed, we have to open it
#        if 'grid' not in self.files or not self.files['grid'].id:
#            self.files['grid'] = File('/nfs/micklab002/jahn/h5/cube84/grid.h5')
#        return self.files['grid']

    @property
    def fagrid(self):
        return self.open('fagrid', reader=FaFile)

    @property
    def grid(self):
        return self.open('grid')

    @property
    def all(self):
        return self.open('all')

    @property
    def phy(self):
        return self.open('phy')

    def __del__(self):
        for f in self.files.values():
            f.close()

files = Files()

def index(l, s):
    try:
        return l.index(s)
    except ValueError:
        raise ValueError("'" + s + "' is not in list:\n\n" + ' '.join(l))


def readmeta(base, verbose=False):
    _,name = os.path.split(base)
    _,d1 = os.path.split(_)
    if d1 == 'res_0000':
        tmpl = os.path.join(_, 'res_{proc:04d}', name + '.{tile:03d}.001.data')
    else:
        tmpl = base + '.{tile:03d}.001.data'

    import mds
    meta = mds.Metadata.read(tmpl.format(proc=0, tile=1)[:-4] + 'meta')

    return meta


def readtiled(base, shape=None, dtype='>f4', mapIO=-1, recs=None, ks=None,
              flds=None, nf=6, ncs=510, tnx=102, tny=51, astype=None, verbose=False):
    '''
    readtiled('res_0000/ptr.0000000000', (50,))
    '''
    _,name = os.path.split(base)
    _,d1 = os.path.split(_)
    if d1 == 'res_0000':
        tmpl = os.path.join(_, 'res_{proc:04d}', name + '.{tile:03d}.001.data')
    else:
        tmpl = base + '.{tile:03d}.001.data'

    ntx = ncs//tnx
    nty = ncs//tny
    ntile = nf*ntx*nty

    if flds is not None:
        assert recs is None
        import mds
        meta = mds.Metadata.read(tmpl.format(proc=0, tile=1)[:-4] + 'meta')
        fldlst = [s.strip() for s in meta.metadata['fldList']]
        assert len(fldlst) == meta.nrecords
        if '-' in flds:
            flds = flds.split('-')
            recs = [index(fldlst, fld) if fld else None for fld in flds[:2]]
            recs[2:] = map(int, flds[2:])
            if len(recs) > 1 and recs[1] is not None:
                recs[1] += 1
        else:
            flds = flds.split(':')
            recs = [index(fldlst, fld) if fld else None for fld in flds[:2]]
            recs[2:] = map(int, flds[2:])
        if len(recs) == 1:
            recs = recs[0]
        else:
            recs = slice(*recs)
#        if np.isscalar(flds):
#            recs = index(fldlst, flds)
#        else:
#            recs = [index(fldlst, fld) for fld in flds]

        sys.stderr.write('Setting recs to ' + str(recs) + '\n')

        if shape is None:
            shape = tuple(meta.dims[:-2])
#            if meta.nrecords > 1:
            shape = (meta.nrecords,) + shape
            sys.stderr.write('Setting shape to ' + str(shape) + '\n')

    a = np.fromfile(tmpl.format(proc=0, tile=1), dtype)
    a = a.reshape(-1, tny, tnx)
    if shape is None:
        shape = a.shape[:1]
    elif shape and shape[0] < 0:
        n = a.shape[0]//np.prod(shape[1:])
        shape = (n,) + shape[1:]

    sys.stderr.write('Setting shape to ' + str(shape) + '\n')

    totcount = np.prod(shape, dtype=int)*tnx*tny

    dtype = np.dtype(dtype)
    otype = astype or dtype.newbyteorder('=')

    recshape = shape[1:]

    if recs is not None:
        try:
            start,stop,step = recs.indices(shape[0])
        except:
            start = recs
            stop = recs + 1
            step = 1
            recsshape = ()
        else:
            recsshape = (stop - start,)
        assert step == 1
        nrec = stop - start
        recsize = np.prod(recshape, dtype=int)*tnx*tny
        seek = start*recsize*dtype.itemsize
        count = nrec*recsize
        try:
            a = a.reshape(shape + (tny, tnx))
        except:
            raise ValueError('Cannot reshape ' + str(a.shape) + ' to ' + str(shape+(tny, tnx)))
        a = a[start:stop]
        shape = recsshape + recshape
    else:
        seek = 0
        count = totcount
        nrec = shape and shape[0] or 1

    if ks is not None:
        if len(recshape) == 0:
            raise ValueError("Have to specify shape of length 2 when using ks without flds")
        nk = shape[-1]
        seekinc = nk*tny*tnx*dtype.itemsize
        try:
            start,stop,step = ks.indices(nk)
        except:
            start = ks
            stop = ks + 1
            step = 1
            kshape = ()
        else:
            kshape = (stop - start,)
        assert step == 1
        nko = stop - start
        ksize = np.prod(recshape[1:], dtype=int)*tnx*tny
        seek += start*ksize*dtype.itemsize
        count = nko*ksize
        a = a.reshape(shape + (tny, tnx))[..., start:stop, :, :]
        shape = shape[:-1] + kshape
    else:
        seekinc = 0

    if mapIO == -1:
        tmpshape = shape + (nty, tny, nf, ntx, tnx)
        oshape = shape + (ncs, nf*ncs)
    elif mapIO == 'facets':
        tmpshape = (nf,) + shape + (nty, tny, ntx, tnx)
        oshape = (nf,) + shape + (ncs, ncs)
    else:
        tmpshape = shape + (nf, nty, tny, ntx, tnx)
        oshape = shape + (nf, ncs, ncs)

    print(tmpshape)
    res = np.zeros(tmpshape, otype)

    for itile in range(ntile):
        proc = itile
        _,itx = divmod(itile, ntx)
        f,ity = divmod(_, nty)
        if verbose:
            sys.stderr.write('Reading tile ' + str(itile) + '\n')
        fname = tmpl.format(proc=proc, tile=itile+1)
        if itile > 0:
            if seek or seekinc:
                with open(fname) as fid:
                    if seekinc:
                        a = np.empty((nrec,) + recshape[:-1] + (nko,) + (tny, tnx))
                        pos = seek
                        for irec in range(nrec):
                            fid.seek(pos)
                            a[irec] = np.fromfile(fid, dtype, count=count).reshape(nko, tny, tnx)
                            pos += seekinc
                    else:
                        fid.seek(seek)
                        a = np.fromfile(fid, dtype, count=count)
            else:
                a = np.fromfile(tmpl.format(proc=proc, tile=itile+1), dtype, count=count)
        a = reshape(a, shape + (tny, tnx))
        if mapIO == -1:
            res[..., ity, :, f, itx, :] = a
        elif mapIO == 'facets':
            res[f, ..., ity, :, itx, :] = a
        else:
            res[..., f, ity, :, itx, :] = a

    return res.reshape(oshape)


class Mapper(object):
    _mapd = {  #1: '/nobackup1/jahn/grid/ecco2/cs510_ll1deg/rmp_cs510_to_ll1deg_conserv.nc',
            1: '/nfs/micklab002/jahn/grid/remap/cs510_to_ll1deg_conserv_fixed_stitch90_stitch0_nonneg',
            5: '/nfs/micklab002/jahn/grid/remap/cs510_to_llfifthdeg_conserv_fixed_stitch0_nonneg',
            6: '/nfs/micklab002/jahn/grid/remap/cs510_to_llsixthdeg_conserv_fixed_stitch0_nonneg',
#            6: '/nfs/micklab002/jahn/grid/remap/e2_to_sixthexact/rmp_cs510_to_llsixthdeg_conserv.nc',
            }

    _smd = {
            1: '/nfs/micklab002/jahn/grid/remap/cs510_to_ll1deg_conserv_fixed_stitch90_stitch0_nonneg_smC',
            5: '/nfs/micklab002/jahn/grid/remap/cs510_to_llfifthdeg_conserv_fixed_stitch0_nonneg_smC',
            6: '/nfs/micklab002/jahn/grid/remap/cs510_to_llsixthdeg_conserv_fixed_stitch0_nonneg_smC',
            }

    def __init__(self, res=5, kslice=()):
        import oj.sparse
        import oj.num

        hshape = (res*180, res*360)
        kl = np.arange(50)[kslice]
        nk = kl.size
        kshape = kl.shape

        mapfile = self._mapd[res]
        if mapfile[-3:] == '.nc':
            M = oj.sparse.fromscrip(mapfile, hshape)
        else:
            M = oj.sparse.frommap(mapfile)
            assert M.shape == hshape

        with File('/nfs/micklab002/jahn/h5/cube84/grid.h5') as g:
            lm = g['hFacC'][kslice] == 0

        if res in self._smd:
            v = oj.num.loadbin(self._smd[res])[kslice]
        else:
            v = M(1.-lm)

        self.nk = nk
        self.kslice = kslice
        self.hshape = hshape
        self.M = M
        self.v = v
        self._v = v.reshape((nk,) + hshape)
        self._lm = lm.reshape((nk, 510, 6, 510))

    def map(self, a, astype=None):
        d = a.ndim - 1
        while np.prod(a.shape[d:]) != nhorz:
            d -= 1

        vshape = a.shape[:d]
        a = a.reshape((-1, self.nk, 510, 6, 510))
        n = a.shape[0]

        acp = np.zeros((510, 6, 510), astype)
        res = np.zeros((n, self.nk) + self.hshape, astype)
        for i in range(n):
            for ik in range(self.nk):
                acp[:] = a[i, ik]
                acp[self._lm[ik]] = 0.
                res[i, ik] = self.M(acp)
                res[i, ik] /= self._v[ik]

        return res.reshape(vshape + self.hshape)

    def __call__(self, a):
        return self.map(a)

