import numpy as np
from scipy.optimize import bisect
from scipy.interpolate import interp1d
from colorspacious import cspace_converter
import oj.num

rgb2cam = cspace_converter('sRGB1', 'CAM02-UCS')
cam2rgb = cspace_converter('CAM02-UCS', 'sRGB1')

_scaledR = np.r_[1, 110, 160, 210, 240, 255]/255.
_unscaledR = np.r_[0, 30, 60, 120, 190, 255]/255.
_stretch = interp1d(_unscaledR, _scaledR)

def enhance(rgb):
    return _stretch(rgb)


def _bisect_fun(x, c):
    return cam2rgb(c*x).max() - 1.


def project_cam_to_rgb(Jab):
    rgb = cam2rgb(Jab)
    if rgb.max() > 1:
        f = bisect(_bisect_fun, 0., 1., (Jab,))
        rgb = cam2rgb(f*Jab)
    return rgb


def make_lut_2d(rgb1, rgb2, N=256):
    c1 = rgb2cam(rgb1)
    c2 = rgb2cam(rgb2)

    j, i = np.mgrid[:1.:N*1j,:1.:N*1j]

    rgb = np.zeros((N, N, 3))
    for J in range(N):
        for I in range(N):
            Jab = i[J,I]*c1 + j[J,I]*c2
            rgb[J,I] = project_cam_to_rgb(Jab)

    return rgb.clip(0., 1.)


def make_lut_nd(*rgbs, **kw):
    '''
    make_lut_nd(*rgb, N=256)
    '''
    N = kw.pop('N', 256)
    assert not kw
    n = len(rgbs)
    Nn = N**n
    oshape = n*(N,) + (3,)
    flatshape = (Nn, 3)

    cams = map(rgb2cam, rgbs)

    # grid the n-D unit hypercube
    t = np.linspace(0., 1., N)
    xi = n*[t]
    g = np.meshgrid(*xi, indexing='ij')

    Jab = np.zeros(oshape)
    for i in range(n):
        Jab += g[i]*cams[i]

    Jab.shape = flatshape
    rgb = np.zeros(flatshape)
    for i in range(Nn):
        rgb[i] = project_cam_to_rgb(Jab[i])

    return rgb.reshape(oshape).clip(0., 1.)


class Colormap2D(object):
    def __init__(self, rgb1=None, rgb2=None, N=256, lut=None, p=1.):
        if lut is not None:
            self._lut = lut
            self.N = len(lut)
        else:
            self._lut = make_lut_2d(rgb1, rgb2, N)
            self.N = N
        self.p = p

    @classmethod
    def fromfile(cls, fname, p=1.):
        lut = oj.num.loadbin(fname)
        return cls(lut=lut, p=p)

    def save(self, fname):
        oj.num.savebin(fname, self._lut)

    def __call__(self, x, y):
        x = np.clip(x, 0., None)
        y = np.clip(y, 0., None)
        s = x + y
        if self.p == 0.:
            f = enhance(s.clip(0., 1.))
        elif self.p < 0.:
            f = enhance(s.clip(0., 1.)*abs(self.p))/enhance(abs(self.p))
        else:
            f = s.clip(0., 1.)**self.p
        np.divide.at(f, s > 0., s[s>0.])
#        f = 1./(x + y).clip(1.)
        i = np.clip(f*x*self.N, 0., self.N - 1).astype(int)
        j = np.clip(f*y*self.N, 0., self.N - 1).astype(int)
        return self._lut[j, i]


class Colormap3D(object):
    def __init__(self, rgb1=None, rgb2=None, rgb3=None, N=256, lut=None, p=1.):
        if lut is not None:
            self._lut = lut
            self.N = len(lut)
        else:
            self._lut = make_lut_nd(rgb1, rgb2, rgb3, N=N)
            self.N = N
        self.p = p

    @classmethod
    def fromfile(cls, fname, p=1.):
        lut = oj.num.loadbin(fname)
        return cls(lut=lut, p=p)

    def save(self, fname):
        oj.num.savebin(fname, self._lut)

    def __call__(self, x, y, z):
        x = np.clip(x, 0., None)
        y = np.clip(y, 0., None)
        z = np.clip(z, 0., None)
        s = x + y + z
        if self.p == 0.:
            f = enhance(s.clip(0., 1.))
        elif self.p < 0.:
            f = enhance(s.clip(0., 1.)*abs(self.p))/enhance(abs(self.p))
        else:
            f = s.clip(0., 1.)**self.p
        np.divide.at(f, s > 0., s[s>0.])
#        f = 1./(x + y).clip(1.)
        i = np.clip(f*x*self.N, 0., self.N - 1).astype(int)
        j = np.clip(f*y*self.N, 0., self.N - 1).astype(int)
        k = np.clip(f*z*self.N, 0., self.N - 1).astype(int)
        return self._lut[k, j, i]


if __name__ == '__main__':
    rgb1 = np.r_[1., 0., 0.]
    rgb2 = np.r_[1., 1., 0.]
    cmap = Colormap2D(rgb1, rgb2)
    y, x = np.mgrid[-.1:1.5:.001, -.1:1.5:.001]
    rgb = cmap(x, y)

    from plt import myimshow
    myimshow(rgb)

