#!/usr/bin/env python
import numpy as np
from numpy import sqrt, exp
from scipy.linalg import solve_banded

def coeff(a, bt, bb, rmud, rmus=1/0.83, rmuu=1/0.4, rd=1.5, ru=3.0):
    '''return recarray of kd,k1,k2,r1,r2,x,y
    
    compute "intrinsic" coefficients
    these are independent of irradiances and layer thickness
    '''
    b = np.broadcast(a,bt,bb,rmud)
    c = np.recarray(b.shape, formats=7*['d'],
                    names=['kd','k1','k2','r1','r2','x','y'])

    c.kd = (a+bt)*rmud
    au = a*rmuu
    Bu = ru*bb*rmuu
    Cu = au+Bu
    As = a*rmus
    Bs = rd*bb*rmus
    Cs = As+Bs
    Bd = bb*rmud
    Fd = (bt-bb)*rmud
    bquad = Cs + Cu
    D = 0.5*(bquad + sqrt(bquad*bquad - 4.0*Bs*Bu))
    c.k1 = D - Cs
    c.k2 = Cs - Bs*Bu/D
    c.r1 = Bu/D
    c.r2 = Bs/D
    denom = (c.kd-Cs)*(c.kd+Cu) + Bs*Bu
    c.x = -((c.kd+Cu)*Fd+Bu*Bd)/denom
    c.y = (-Bs*Fd+(c.kd-Cs)*Bd)/denom
    return c

def irradiances(r, rf, Edf, amp1, amp2, c, kbot=None, axis=0):
    '''return Ed,Es,Eu

    kbot :: number of wet layers computed, last layer is assumed infinitely deep

    compute irradiances at arbitrary depths for solution
    '''
    assert np.all(np.diff(rf) < 0)
    k = len(rf) - 1 - np.searchsorted(rf[::-1], r)
    if kbot is not None:
        k = np.minimum(k, kbot - 1)
    drtop = rf[k] - r
    drbot = r - rf[k+1]
    if c.ndim > 1:
        drtop = np.rollaxis(drtop.reshape((c.ndim-1)*(1,) + (-1,)), -1, axis)
        drbot = np.rollaxis(drbot.reshape((c.ndim-1)*(1,) + (-1,)), -1, axis)

    f1 = exp(-c.k1.take(k, axis)*drbot)*amp1.take(k, axis)
    f2 = exp(-c.k2.take(k, axis)*drtop)*amp2.take(k, axis)
    E = np.zeros((3,)+f1.shape)
    print E.shape,c.shape,Edf.shape
    E[0] = exp(-c.kd.take(k, axis)*drtop)*Edf.take(k, axis)
    E[1] = f2 + c.r1.take(k, axis)*f1 + c.x.take(k, axis)*E[0]
    E[2] = c.r2.take(k, axis)*f2 + f1 + c.y.take(k, axis)*E[0]

    return E

def solve(Edsf, Essf, drf, c, kbot=None):
    '''return amp1,amp2

    kbot :: number of wet layers computed, last layer is assumed infinitely deep

    compute amplitudes of eigenmodes for given surface irradiances and layer thickness
    '''
    r1 = c.r1[:kbot]
    r2 = c.r2[:kbot]
    x = c.x[:kbot]
    y = c.y[:kbot]
    dx = np.diff(x, 1, 0)
    dy = np.diff(y, 1, 0)
    e1 = exp(-c.k1[:kbot]*drf[:kbot])
    e2 = exp(-c.k2[:kbot]*drf[:kbot])
    ed = exp(-c.kd[:kbot]*drf[:kbot])

    if kbot is None: kbot = len(r1)
    Edf = np.zeros((kbot+1,)+Edsf.shape)
    Edf[0] = Edsf
    Edf[1:] = np.cumprod(ed, axis=0)*Edsf

    abv = np.zeros(Edsf.shape + (3, 2*kbot))
    bv  = np.zeros(Edsf.shape + (2*kbot,))

    ab = abv.transpose([-2,-1] + range(Edsf.ndim))
    b  = bv.transpose([-1] + range(Edsf.ndim))

    ab[1, 0] = 1.
    ab[0, 1] = r1[0]*e1[0]
    b[0] = Essf - x[0]*Edsf

    ab[2, :-2:2]  = (1. - r2[:-1]*r1[1:])*e2[:-1]
    ab[1, 1:-1:2] = r1[:-1] - r1[1:]
    ab[0, 2::2]   = -1. + r2[1:]*r1[1:]
    b[1:-1:2] = (dx - r1[1:]*dy)*Edf[1:-1]

    ab[2, 1:-1:2] = 1. - r1[:-1]*r2[:-1]
    ab[1, 2::2]   = r2[:-1] - r2[1:]
    ab[0, 3::2]   = (-1. + r1[1:]*r2[:-1])*e1[1:]
    b[2::2] = (dy - r2[:-1]*dx)*Edf[1:-1]

    # bottom b.c.
    nk = (c.kd[:kbot]!=0).sum(0)        # nb wet layers
#    inds = tuple(np.indices(nk.shape))  # index grid
#    ab[(2, 2*nk-2)+inds] = 0.
#    ab[(1, 2*nk-1)+inds] = 1.
#    b[(2*nk-1,)+inds] = 0.

#    about = abv.copy()

    amp = np.zeros((2,kbot,)+Edsf.shape)
    for ind in np.ndindex(Edsf.shape):
        if nk[ind] > 0:
            ab = abv[ind][:,:2*nk[ind]]
            # bottom b.c.
            b = bv[ind][:2*nk[ind]]
            ab[1, -1] = 1.
            ab[2, -2] = 0.
            b[-1] = 0
            cs = solve_banded((1,1), ab, b, True, True)  # overwrite all
            amp[np.s_[1, :nk[ind],] + ind] = cs[0::2]
            amp[np.s_[0, :nk[ind],] + ind] = cs[1::2]
    return amp  #, nk, about

