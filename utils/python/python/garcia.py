import numpy as np
from numpy import arange, pi, cos
from scipy.fftpack.realtransforms import dct, idct
from scipy.ndimage.morphology import distance_transform_edt

def inpaint_perx(y, n=100, y0=None, s0=3, order=2, relax=2.):
    nx = y.shape[-1]
    a = np.ma.zeros(y.shape[:-1] + (nx*3,))
    a[..., nx:-nx] = y
    a[..., :nx] = y
    a[..., -nx:] = y
    if not hasattr(y, 'mask'):
        a.mask = ~np.isfinite(a.data)

    inpaint = Inpaint(a, n, y0, s0, order, relax)
    while inpaint.iter():
        a[..., :nx] = a[..., nx:-nx]
        a[..., -nx:] = a[..., nx:-nx]

    b = inpaint.get()
    return b[..., nx:-nx]


class Inpaint:
    def __init__(self, y, n=100, y0=None, s0=3, order=2, relax=2., axis=None):
        '''
        Replaces the missing data in y by extra/interpolating the non-missing
        elements.  If y is a masked array, masked elements are considered
        missing, otherwise all non-finite values (NaN or +-Inf) are.

        n     :: number of iterations, increase if suspecting non-convergence
        y0    :: initial guess (default is nearest-neighbor interpolation)
                 useful for continuing iterations
        s0    :: initial log10 of smoothing parameter (will be reduced down to -6
                 during iteration)
        order :: power applied to smoothing kernel
        relax :: over/under-relaxation factor
        axis  :: axis to inpaint along (default all)

        Adapted from matlab code by Damien Garcia.

        References (please refer to the two following references)
        ----------
        1) Garcia D, Robust smoothing of gridded data in one and higher
        dimensions with missing values. Computational Statistics & Data
        Analysis, 2010;54:1167-1178.
        <a
        href="matlab:web('http://www.biomecardio.com/pageshtm/publi/csda10.pdf')">PDF download</a>
        2) Wang G, Garcia D et al. A three-dimensional gap filling method for
        large geophysical datasets: Application to global satellite soil
        moisture observations. Environ Modell Softw, 2012;30:139-142.
        <a
        href="matlab:web('http://www.biomecardio.com/pageshtm/publi/envirmodellsoftw12.pdf')">PDF download</a>
        '''
        try:
            mask = y.mask
        except AttributeError:
            finite = np.isfinite(y)
            mask = ~finite
        else:
            finite = ~mask
            y = y.data

        self.finite = finite
        self.relax = relax
        self.W = np.array(finite, dtype=np.float64)
        self.y = y = np.array(y, np.float64)
        if not np.any(mask) or y.size < 2:
            y[mask] = np.nan
            self.s = iter([])
            self.z = y0 or (np.zeros_like(y) + np.nan)
            return

        y[mask] = 0.
        sizy = y.shape

        if axis is None:
            axis = range(y.ndim)
        else:
            axis = np.ravel(axis)

        ## Creation of the Lambda tensor
        #---
        # Lambda contains the eingenvalues of the difference matrix used in this
        # penalized least squares process.
        # this is an "outer sum" of 1-d arrays
        siz1 = np.ones((y.ndim,), int)
        Lambda = np.zeros(sizy)
        for i in axis:
            siz1[:] = 1
            siz1[i] = sizy[i]
            Lambda1 = cos(pi*arange(sizy[i])/sizy[i])
            Lambda = Lambda + Lambda1.reshape(siz1)

        self.Lambda = (2.*abs(y.ndim - Lambda))**order

        if y0 is None:
            y0 = inpaint_nearest(y, mask)

        self.z = y0
        self.s = iter(np.logspace(s0, -6, n))

    def __next__(self):
        try:
            s = next(self.s)
        except StopIteration:
            return False

        Gamma = 1./(1. + s*self.Lambda)
        DCTy = apply_along_axes(dct, self.W*(self.y - self.z) + self.z)
        self.z[:] = self.relax*apply_along_axes(idct, Gamma*DCTy) + (1. - self.relax)*self.z
# should do this here?
#        self.z[self.finite] = self.y[self.finite]
        return True

    iter = __next__

    def __iter__(self):
        return self

    def get(self):
        self.z[self.finite] = self.y[self.finite]
        return self.z


def inpaintn(y, n=100, y0=None, s0=3, order=2, relax=2., axis=None):
    '''
    z = inpaintn(y) replaces the missing data in y by extra/interpolating
    the non-missing elements.  If y is a masked array, masked elements are
    considered missing, otherwise all non-finite values (NaN or +-Inf) are.

    n     :: number of iterations, increase if suspecting non-convergence
    y0    :: initial guess (default is nearest-neighbor interpolation)
             useful for continuing iterations
    s0    :: initial log10 of smoothing parameter (will be reduced down to -6
             during iteration)
    order :: power applied to smoothing kernel
    relax :: over/under-relaxation factor
    axis  :: axis to inpaint along (default all)

    Adapted from matlab code by Damien Garcia.

    References (please refer to the two following references)
    ----------
    1) Garcia D, Robust smoothing of gridded data in one and higher
    dimensions with missing values. Computational Statistics & Data
    Analysis, 2010;54:1167-1178.
    <a
    href="matlab:web('http://www.biomecardio.com/pageshtm/publi/csda10.pdf')">PDF download</a>
    2) Wang G, Garcia D et al. A three-dimensional gap filling method for
    large geophysical datasets: Application to global satellite soil
    moisture observations. Environ Modell Softw, 2012;30:139-142.
    <a
    href="matlab:web('http://www.biomecardio.com/pageshtm/publi/envirmodellsoftw12.pdf')">PDF download</a>
    '''
    try:
        mask = y.mask
    except AttributeError:
        finite = np.isfinite(y)
        mask = ~finite
    else:
        finite = ~mask
        y = y.data

    y = np.array(y, np.float64)
    if not np.any(mask) or y.size < 2:
        y[mask] = np.nan
        return y

    y[mask] = 0.
    sizy = y.shape

    W = np.array(finite, dtype=np.float64)

    if axis is None:
        axis = range(y.ndim)
    else:
        axis = np.ravel(axis)

    ## Creation of the Lambda tensor
    #---
    # Lambda contains the eingenvalues of the difference matrix used in this
    # penalized least squares process.
    # this is an "outer sum" of 1-d arrays
    siz1 = np.ones((y.ndim,), int)
    Lambda = np.zeros(sizy)
    for i in axis:
        siz1[:] = 1
        siz1[i] = sizy[i]
        Lambda1 = cos(pi*arange(sizy[i])/sizy[i])
        Lambda = Lambda + Lambda1.reshape(siz1)

    Lambda = (2.*abs(y.ndim - Lambda))**order

    if y0 is None:
        y0 = inpaint_nearest(y, mask)

    s = np.logspace(s0, -6, n)

    z = y0
    for i in range(n):
        Gamma = 1./(1. + s[i]*Lambda)
        DCTy = apply_along_axes(dct, W*(y - z) + z)
        z = relax*apply_along_axes(idct, Gamma*DCTy) + (1. - relax)*z

    z[finite] = y[finite]
    return z


def inpaint_nearest(y, mask):
    # nearest neighbor interpolation (in case of missing values)
    I = distance_transform_edt(mask, return_distances=False, return_indices=True)
    z = np.array(y)
    z[mask] = y[tuple(i[mask] for i in I)]
    return z


def apply_along_axes(f, data):
    '''apply f along all axes of data'''
    o = data
    for i in range(data.ndim):
        o = f(o, norm='ortho', type=2, axis=i)
    return o

