import numpy as np
import fa
import exchange

class CubedSphere(object):
    def __init__(self, n, gridtilefiles=None, griddir=None):
        """
        n = dimension of cubed-sphere face (6 faces of n x n grid cells)

        provide either gridtilefiles or griddir:

        gridtilefiles = 'tile{0:03d}.mitgrid'
        gridtilefiles = 'grid.face{0:03d}.bin'

        griddir must contains the grid files XC.data, ...
        """
        self.n = n
        self.exch = exchange.cs()
        if gridtilefiles is not None:
            self._grid = fa.MITGrid(gridtilefiles, 12*[n])
        elif griddir is not None:
            raise NotImplementedError("OLD_GRID_IO files are not supported.")
            #self._grid = MITGridDir(griddir, 12*[n])
        else:
            self._grid = None

    def loadc(self, fname, dtype, shape=(), halo=0, map=-1):
        a = fa.fromfile(fname, dtype, shape, 12*[self.n], halo=halo, map=map)
        self.exch(a, halo=halo)
        return a

    def loaduv(self, uname, vname, dtype, shape=(), halo=0, extra=1, map=-1, sign=-1):
        u = fa.fromfile(uname, dtype, shape, 12*[self.n], halo=halo, extrau=extra, map=map)
        v = fa.fromfile(vname, dtype, shape, 12*[self.n], halo=halo, extrav=extra, map=map)
        self.exch.uv(u, v, halo=halo, extra=extra)
        return u, v

    def loadws(self, uname, vname, dtype, shape=(), halo=0, extra=1, map=-1):
        return self.loaduv(uname, vname, dtype, shape, halo, extra, map, 1)
        
    def loadg(self, fname, dtype, shape=(), halo=0, extra=1, map=-1):
        a = fa.fromfile(fname, dtype, shape, dims=12*[self.n], halo=halo, extrau=extra, extrav=extra, map=map)
        self.exch.g(a, halo=halo, extra=extra)
        return a

    loadtracer = loadc
    loadvort = loadg

    def get_grid_c(self, name, halo=0, dtype=np.float64):
        a = fa.empty(dtype=dtype, dims=12*[self.n], halo=halo)
        self._grid.set(a, name, halo)
        if halo > 0:
            self.exch(a, halo=halo)
        return a

    def get_grid_ws(self, wname, sname, halo=0, extra=1, dtype=np.float64):
        w = fa.empty(dtype=dtype, dims=12*[self.n], halo=halo, extrau=extra)
        s = fa.empty(dtype=dtype, dims=12*[self.n], halo=halo, extrav=extra)
        self._grid.set(w, wname, halo)
        self._grid.set(s, sname, halo)
        if halo > 0:
            self.exch.ws(w, s, halo=halo, extra=extra)
        return w, s

    def get_grid_g(self, name, halo=0, extra=1, dtype=np.float64):
        a = fa.empty(dtype=dtype, dims=12*[self.n], halo=halo, extrau=extra, extrav=extra)
        print a.shape
        self._grid.set(a, name, halo)
        if halo > 0:
            self.exch.g(a, halo=halo, extra=extra)
        return a

    def xc(self, halo=0, dtype=np.float64):
        return self.get_grid_c('xc', halo, dtype)

    def yc(self, halo=0, dtype=np.float64):
        return self.get_grid_c('yc', halo, dtype)

    def dxf(self, halo=0, dtype=np.float64):
        return self.get_grid_c('dxf', halo, dtype)

    def dyf(self, halo=0, dtype=np.float64):
        return self.get_grid_c('dyf', halo, dtype)

    def ac(self, halo=0, dtype=np.float64):
        return self.get_grid_c('ac', halo, dtype)

    def aws(self, halo=0, extra=1, dtype=np.float64):
        return self.get_grid_ws('aw', 'as', halo, extra, dtype)

    def dxyc(self, halo=0, extra=1, dtype=np.float64):
        return self.get_grid_ws('dxc', 'dyc', halo, extra, dtype)

    def dyxg(self, halo=0, extra=1, dtype=np.float64):
        return self.get_grid_ws('dyg', 'dxg', halo, extra, dtype)

    def xg(self, halo=0, extra=1, dtype=np.float64):
        return self.get_grid_g('xg', halo, extra, dtype)

    def yg(self, halo=0, extra=1, dtype=np.float64):
        return self.get_grid_g('yg', halo, extra, dtype)

    def dxv(self, halo=0, extra=1, dtype=np.float64):
        return self.get_grid_g('dxv', halo, extra, dtype)

    def dyu(self, halo=0, extra=1, dtype=np.float64):
        return self.get_grid_g('dyu', halo, extra, dtype)

    def az(self, halo=0, extra=1, dtype=np.float64):
        return self.get_grid_g('az', halo, extra, dtype)

    def anglecs(self, halo=0, dtype=np.float64):
        return self.get_grid_c('anglecs', halo, dtype)

    def anglesn(self, halo=0, dtype=np.float64):
        return self.get_grid_c('anglesn', halo, dtype)

