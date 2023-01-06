import numpy as np
from scipy.optimize import bisect
from scipy.interpolate import interp1d
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
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
    return rgb.clip(0., 1.)


class ColormapND(object):
    def __init__(self, *rgbs, **kw):
        '''
        Keyword arguments:
          p      [default: 1.]
        '''
        self.p = kw.pop('p', 1.)
        assert not kw

        self.n = n = len(rgbs)

        cams = map(rgb2cam, rgbs)
        Jmean = np.mean(cams, axis=0)[0]
        print Jmean
        cams = [rgb2cam((0., 0., 0.))] + cams + [np.r_[Jmean, 0., 0.]]

        p = np.zeros((n+2, n))
        for i in range(n):
            p[i+1, i] = 1.
        p[-1, :] = 1./n
        self.delaunay = Delaunay(p)

        self.I = LinearNDInterpolator(self.delaunay, cams)

    def __call__(self, x):
        '''
        x has shape (..., nd)
        '''
        # project to simplex
        x = np.clip(x, 0., None)
        s = np.sum(x, axis=-1, keepdims=True)
        if self.p == 0.:
            f = enhance(s.clip(0., 1.))
        elif self.p < 0.:
            f = enhance(s.clip(0., 1.)*abs(self.p))/enhance(abs(self.p))
        else:
            f = s.clip(0., 1.)**self.p
        np.divide.at(f, s > 0., s[s>0.])
        x *= f

        # interpolate in CAM02 space
        Jab = self.I(x.reshape(f.size, self.n))

        # project back to RGB
        rgb = np.zeros((f.size, 3))
        for i in range(f.size):
            rgb[i] = project_cam_to_rgb(Jab[i])

        return rgb.reshape(x.shape[:-1] + (3,))

class ListedColormapND(object):
    def __init__(self, lut):
        '''
        Keyword arguments:
          p      [default: 1.]
        '''
        self._lut = lut
        self.n = np.ndim(lut) - 1
        self.N = len(lut)

    def __call__(self, x):
        '''
        x has shape (..., nd)
        '''
        # project to simplex
        x = np.clip(x, 0., None)
        s = np.sum(x, axis=-1, keepdims=True)
#        f = s.clip(0., 1.)
#        np.divide.at(f, s > 0., s[s>0.])
#        x *= f
        x /= s.clip(1.)

        i = np.clip(x*self.N, 0., self.N - 1).astype(int)
        idx = tuple(np.rollaxis(i, -1))
        rgb = self._lut[idx]
        return rgb.clip(0., 1.)

