import os
import numpy as np
from haloarray import FacetArray, Exchange
from oj.num import myfromfile

if os.path.exists('/nobackup1b/jahn'):
    scratch = '/nobackup1b/jahn/'
elif os.path.exists('/scratch/jahn'):
    scratch = '/scratch/jahn/'
elif os.path.exists('/data/jahn'):
    scratch = '/data/jahn/'
else:
    raise IOError('No data directory found')

class GridFacet(object):
    def __init__(self, dir, nr, dims, exch):
        self.dir = dir
        self.nr = nr
        self.dims = dims
        self.exch = exch
        self._rf = None
        self._drf = None
        self._depth = None
        self._Ac = None
        self._Aw = None
        self._As = None
        self._Az = None
        self._dxg = None
        self._dyg = None
        self._dxc = None
        self._dyc = None
        self._hfc = None
        self._hfw = None
        self._hfs = None

    @property
    def drf(self):
        if self._drf is None:
            self._drf = np.fromfile(self.dir + '/DRF.data', '>f4').reshape((self.nr,1,1))
        return self._drf

    @property
    def rf(self):
        if self._rf is None:
            self._rf = np.fromfile(self.dir + '/RF.data', '>f4').reshape((self.nr+1,1,1))
        return self._rf

    @property
    def depth(self):
        if self._depth is None:
            self._depth = FacetArray.fromfile(self.dir + '/Depth.data', '>f4', (), self.dims)
            self.exch(self._depth)
        return self._depth

    @depth.deleter
    def depth(self):
        del self._depth
        self._depth = None

    @property
    def Ac(self):
        if self._Ac is None:
            self._Ac = FacetArray.fromfile(self.dir + '/RAC.data', '>f4', (), self.dims)
            self.exch(self._Ac)
        return self._Ac

    @property
    def Aw(self):
        if self._Aw is None:
            self._Aw = FacetArray.fromfile(self.dir + '/RAW.data', '>f4', (), self.dims)
            self.exch(self._Aw)
        return self._Aw

    @property
    def As(self):
        if self._As is None:
            self._As = FacetArray.fromfile(self.dir + '/RAS.data', '>f4', (), self.dims)
            self.exch(self._As)
        return self._As

    @property
    def Az(self):
        if self._Az is None:
            self._Az = FacetArray.fromfile(self.dir + '/RAZ.data', '>f4', (), self.dims)
            self.exch(self._Az)
        return self._Az

    @Ac.deleter
    def Ac(self):
        del self._Ac
        self._Ac = None

    @Aw.deleter
    def Aw(self):
        del self._Aw
        self._Aw = None

    @As.deleter
    def As(self):
        del self._As
        self._As = None

    @Az.deleter
    def Az(self):
        del self._Az
        self._Az = None

    @property
    def dxg(self):
        if self._dxg is None:
            self._dxg = FacetArray.fromfile(self.dir + '/DXG.data', '>f4', (), self.dims)
            self._dyg = FacetArray.fromfile(self.dir + '/DYG.data', '>f4', (), self.dims)
            self.exch.ws(self._dyg, self._dxg)
        return self._dxg

    @property
    def dyg(self):
        if self._dxg is None:
            self._dxg = FacetArray.fromfile(self.dir + '/DXG.data', '>f4', (), self.dims)
            self._dyg = FacetArray.fromfile(self.dir + '/DYG.data', '>f4', (), self.dims)
            self.exch.ws(self._dyg, self._dxg)
        return self._dyg

    @property
    def dxc(self):
        if self._dxc is None:
            self._dxc = FacetArray.fromfile(self.dir + '/DXC.data', '>f4', (), self.dims)
            self._dyc = FacetArray.fromfile(self.dir + '/DYC.data', '>f4', (), self.dims)
            self.exch.ws(self._dxc, self._dyc)
        return self._dxc

    @property
    def dyc(self):
        if self._dxc is None:
            self._dxc = FacetArray.fromfile(self.dir + '/DXC.data', '>f4', (), self.dims)
            self._dyc = FacetArray.fromfile(self.dir + '/DYC.data', '>f4', (), self.dims)
            self.exch.ws(self._dxc, self._dyc)
        return self._dyc

    @property
    def hfc(self):
        if self._hfc is None:
            self._hfc = FacetArray.fromfile(self.dir + '/hFacC.data', '>f4', (self.nr,), self.dims)
            self.exch(self._hfc)
        return self._hfc

    @hfc.deleter
    def hfc(self):
        del self._hfc
        self._hfc = None

    @property
    def hfw(self):
        if self._hfw is None:
            self._hfw = FacetArray.fromfile(self.dir + '/hFacW.data', '>f4', (self.nr,), self.dims)
            self._hfs = FacetArray.fromfile(self.dir + '/hFacS.data', '>f4', (self.nr,), self.dims)
            self.exch.ws(self._hfw, self._hfs)
        return self._hfw

    @hfw.deleter
    def hfw(self):
        del self._hfw
        del self._hfs
        self._hfw = None
        self._hfs = None

    @property
    def hfs(self):
        if self._hfw is None:
            self._hfw = FacetArray.fromfile(self.dir + '/hFacW.data', '>f4', (self.nr,), self.dims)
            self._hfs = FacetArray.fromfile(self.dir + '/hFacS.data', '>f4', (self.nr,), self.dims)
            self.exch.ws(self._hfw, self._hfs)
        return self._hfs

    @hfs.deleter
    def hfs(self):
        del self._hfw
        del self._hfs
        self._hfw = None
        self._hfs = None


class MITGridFacet(object):
    _fldnames = ['xc', 'yc', 'dxf', 'dyf', 'Ac', 'xg', 'yg', 'dxv', 'dyu', 'Az', 'dxc', 'dyc', 'Aw', 'As', 'dxg', 'dyg']
    _smate    = {'dxc':'dyc', 'dyg':'dxg', 'Aw':'As'}
    _wmate    = {'dyc':'dxc', 'dxg':'dyg', 'As':'Aw'}
    _zfields  = ['xg','yg','dxv','dyu','Az']
    _cfields  = ['xc','yc','dxf','dyf','Ac']

    def __init__(self, files, dims, exch, dtype='>f8'):
        self.files = files
        self.file_dtype = dtype
        self.dims = dims
        self.nx = dims[0::2]
        self.ny = dims[1::2]
        self.nfaces = len(self.nx)
        self.exch = exch
        self._fields = dict()

        if len(self.files) != self.nfaces:
            self.files = [ files.format(i+1) for i in range(self.nfaces) ]


    def _readfld(self, name):
        skip = self._fldnames.index(name)
        res = FacetArray.zeros((), self.dims, halo=1)
        for i in range(self.nfaces):
            if self.nx > 0 and self.ny > 0:
                res.face(i)[1:,1:] = myfromfile(self.files[i], self.file_dtype, (self.ny[i]+1,self.nx[i]+1), skip=skip)

        return res

    def __getattr__(self, name):
        if name in self._fldnames:
            if name not in self._fields:
                self._fields[name] = fld = self._readfld(name)
                if name in self._smate:
                    mate = self._smate[name]
                    self._fields[mate] = mfld = self._readfld(mate)
                    self.exch.ws(fld, mfld)
                elif name in self._wmate:
                    mate = self._wmate[name]
                    self._fields[mate] = mfld = self._readfld(mate)
                    self.exch.ws(mfld, fld)
                elif name in self._zfields:
                    self.exch.z(fld)
                else:
                    self.exch(fld)

            return self._fields[name]
        else:
            raise AttributeError

    def __delattr__(self, name):
        if name in self._fldnames:
            if name in self._fields:
                del self._fields[name]
                if name in self._smate:
                    mate = self._smate[name]
                elif name in self._wmate:
                    mate = self._wmate[name]
                else:
                    mate = None
                if mate is not None:
                    del self._fields[mate]
        else:
            raise AttributeError


# facets:
#
#              4   5
#
#          2   3
#
#      0   1
#
#  4   5
#                      N      S      E      W
csexch = Exchange([[ (2,1), (5,0), (1,0), (4,1)],
                   [ (2,0), (5,1), (3,1), (0,0)],
                   [ (4,1), (1,0), (3,0), (0,1)],
                   [ (4,0), (1,1), (5,1), (2,0)],
                   [ (0,1), (3,0), (5,0), (2,1)],
                   [ (0,0), (3,1), (1,1), (4,0)]])

mitgrid84 = MITGridFacet(scratch + 'grid/cube84/tile{0:03d}.mitgrid',12*[510],csexch)
grid84 = GridFacet(scratch + 'grid/cube84',50,12*[510],csexch)

