#!/usr/bin/env python
import numpy as np
from numpy import sqrt, exp

def f2c(rf):
    return .5*(rf[:-1] + rf[1:])

def _c2f(rc):
    r = 0.
    yield r
    for c in rc:
        r += 2.*(c - r)
        yield r

def c2f(rc):
    '''compute rf from rc'''
    rf = np.r_[list(_c2f(rc))]
    return rf

rd   = 1.5  # these are taken from Ackleson, et al. 1994 (JGR)
ru   = 3.0

def coeff(a, bt, bb, rmud, rmus, rmuu):
    '''return cd,a1,a2,R1,R2,x,y
    
    compute "intrinsic" coefficients
    these are independent of irradiances and layer thickness
    '''
    cd = (a+bt)*rmud
    au = a*rmuu
    Bu = ru*bb*rmuu
    Cu = au+Bu
    As = a*rmus
    Bs = rd*bb*rmus
    Cs = As+Bs
    Bd = bb*rmud
    Fd = (bt-bb)*rmud
    bquad = Cu - Cs
    cquad = Bs*Bu - Cs*Cu
    sqarg = bquad*bquad - 4.0*cquad
    a1 = 0.5*(bquad + sqrt(sqarg))
    #a2 = 0.5*(bquad - sqrt(sqarg))  # K of Aas
    # more stable:
    a2 = cquad/a1
    R1 = (a1+Cs)/Bu
    R2 = (a2+Cs)/Bu
    denom = (cd-Cs)*(cd+Cu) + Bs*Bu
    x = -((cd+Cu)*Fd+Bu*Bd)/denom
    y = (-Bs*Fd+(cd-Cs)*Bd)/denom
    return cd,a1,a2,R1,R2,x,y

def amplitudes(Edf, Esf, Eubot, drf, cd, a1, a2, R1, R2, x, y, kbot):
    '''return c1,c2

    kbot :: number of wet layers computed, last layer is assumed infinitely deep

    compute amplitudes of eigenmodes for given irradiances and layer thickness
    '''
    expAddr = exp(-cd*drf)
    expmAudr = exp(-a1*drf)
    expAsdr = exp(a2*drf)
    idenom = 1./(R1-R2*expAsdr*expmAudr)
    c1  = np.zeros(drf.shape)
    c2  = np.zeros(drf.shape)
    for k in range(kbot-1):
        tmp = Esf[k] - x[k]*Edf[k]
        c1[k] = (Eubot[k]-R2[k]*expAsdr[k]*tmp-y[k]*Edf[k+1])*idenom[k]
        c2[k] = (R1[k]*tmp + y[k]*expmAudr[k]*Edf[k+1] - expmAudr[k]*Eubot[k])*idenom[k]
    # Aas b.c. in bottom layer
    k = kbot - 1
    c1[k] = 0
    c2[k] = Esf[k] - x[k]*Edf[k]
    return c1, c2

def calc(r, rf, Edf, c1, c2, cd, a1, a2, R1, R2, x, y, kbot):
    '''return Ed,Es,Eu

    kbot :: number of wet layers computed, last layer is assumed infinitely deep

    compute irradiances at arbitrary depths for solution
    '''
    k = kbot - 1 - np.searchsorted(rf[kbot-1::-1], r)
    drtop = rf[k] - r
    drbot = r - rf[k+1]

    Ed = exp(-cd[k]*drtop)*Edf[k]
    Es = exp(a2[k]*drtop)*c2[k] + exp(-a1[k]*drbot)*c1[k] + x[k]*Ed
    Eu = R2[k]*exp(a2[k]*drtop)*c2[k] + R1[k]*exp(-a1[k]*drbot)*c1[k] + y[k]*Ed

    return Ed, Es, Eu

def interp(r, rf, Edf, Esf, Eubot, a, bt, bb, rmud, rmus, rmuu, kbot):
    '''return Ed,Es,Eu

    kbot :: number of wet layers computed, last layer is assumed infinitely deep

    compute irradiances at arbitrary depths from model output
    '''
    drf = -np.diff(rf)
    cd,a1,a2,R1,R2,x,y = coeff(a, bt, bb, rmud, rmus, rmuu)
    c1,c2 = amplitudes(Edf, Esf, Eubot, drf, cd, a1, a2, R1, R2, x, y, kbot)
    Ed,Es,Eu = calc(r, rf, Edf, c1, c2, cd, a1, a2, R1, R2, x, y, kbot)
    return Ed, Es, Eu

def iterate_coeff(Edsf, Essf, drf, cd, a1, a2, R1, R2, x, y, kbot, niter):
    '''return Edf, Esf, Eubot

    kbot :: number of wet layers computed, last layer is assumed infinitely deep

    Esf[kbot] is computed but never used
    '''
    Nr = len(drf)
    Edf = np.zeros((Nr+1,))
    Esf = np.zeros((Nr+1,))
    Eubot = np.zeros((Nr,))

    expAddr = exp(-cd*drf)
    expmAudr = exp(-a1*drf)
    expAsdr = exp(a2*drf)
    idenom = 1./(R1-R2*expAsdr*expmAudr)

    # integrate Ed equation first
    Edf[0] = Edsf
    Edf[1:] = Edsf*np.cumprod(expAddr)

    # start with Aas solution (no increasing mode)
    Esf[0] = Essf
    for k in range(Nr):
        c2[k] = Esf[k] - x[k]*Edf[k]
        Esf[k+1] = np.maximum(0., c2[k]*expAsdr[k] + x[k]*Edf[k+1])

    for k in range(Nr):
        Eubot[k] = R2[k]*exp(a2[k]*drf[k])*c2[k] + y[k]*Edf[k+1]

    for it in range(niter):
        # Aas boundary condition in last layer
        k = kbot - 1
        c2[k] = Esf[k] - x[k]*Edf[k]
        Eubot[k] = R2[k]*exp(a2[k]*drf[k])*c2[k] + y[k]*Edf[k+1]

        # compute Eubot[k-1] from Esf[k] and Eubot[k]
        # need Eubot[kbot-1]
        for k in range(kbot-1, 1, -1):
            tmp = Esf[k] - x[k]*Edf[k]
            c1[k] = (Eubot[k]-R2[k]*expAsdr[k]*tmp-y[k]*Edf[k+1])*idenom[k]
            c2[k] = (R1[k]*tmp + y[k]*expmAudr[k]*Edf[k+1] - expmAudr[k]*Eubot[k])*idenom[k]
            Eunew = R2[k]*c2[k] + R1[k]*expmAudr[k]*c1[k] + y[k]*Edf[k]
            Eubot[k-1] = np.maximum(0., Eunew)

        # compute Esf[k+1] from Esf[k] and Eubot[k]
        # need Esf[0]
        for k in range(kbot):
            tmp = Esf[k] - x[k]*Edf[k]
            c1[k] = (Eubot[k]-R2[k]*expAsdr[k]*tmp-y[k]*Edf[k+1])*idenom[k]
            c2[k] = (R1[k]*tmp + y[k]*expmAudr[k]*Edf[k+1] - expmAudr[k]*Eubot[k])*idenom[k]
            Esnew = expAsdr[k]*c2[k] + c1[k] + x[k]*Edf[k+1]
            Esf[k+1] = np.maximum(0., Esnew)

    # Es is now continuous, but Eu is not
    # in particular, the bottom boundary condition is violated

    return Edf, Esf, Eubot
    
def iterate(Edsf, Essf, rf, a, bt, bb, rmud, rmus, rmuu, kbot, niter):
    '''return Edf, Esf, Eubot

    kbot :: number of wet layers computed, last layer is assumed infinitely deep

    Esf[kbot] is computed but never used
    '''
    drf = -np.diff(rf)
    cd,a1,a2,R1,R2,x,y = coeff(a, bt, bb, rmud, rmus, rmuu)
    Edf,Esf,Eubot = iterate_coeff(Edsf, Essf, drf, cd, a1, a2, R1, R2, x, y, kbot, niter)
    return Edf, Esf, Eubot


if __name__ == '__main__':
    import sys
    import oj.plt
    import matplotlib.pyplot as plt
    import MITgcmutils as mit

    # this should be the number of wet layers
    # last layer (kbot-1 in python) is assumed infinitely deep (no downward increasing mode)
    kbot = 22

    it = 3

    diagdir = sys.argv.pop(1)
    l = int(sys.argv.pop(1))

    lam = np.r_[400:701:25.]

    cols = np.loadtxt('rmud.txt').T
    rmud = cols[0,it]
    Edsf = cols[1:14,it][l]
    Essf = cols[14:27,it][l]
    a00 = cols[27,it]

    rmus = 1.0/0.83
    rmuu = 1.0/0.4

    nc = mit.rdmnc(diagdir + '/dar_tave.0000000000.t001.nc')

    rc = nc['Z'][:-1]
    rf = c2f(rc)
    drf = -np.diff(rf)

    a = np.array([ nc['dar_a{:02d}'.format(nl)][it,:,0,0] for nl in range(1,14) ])[l][:-1]
    bt = np.array([ nc['dar_bt{:02d}'.format(nl)][it,:,0,0] for nl in range(1,14) ])[l][:-1]
    bb = np.array([ nc['dar_bb{:02d}'.format(nl)][it,:,0,0] for nl in range(1,14) ])[l][:-1]

    Edf = np.array([ nc['dar_Ed{:02d}'.format(nl)][it,:,0,0] for nl in range(1,14) ])[l]
    Esf = np.array([ nc['dar_Es{:02d}'.format(nl)][it,:,0,0] for nl in range(1,14) ])[l]
    Euf = np.array([ nc['dar_Eu{:02d}'.format(nl)][it,:,0,0] for nl in range(1,14) ])[l]

    Eubot = Euf[1:]

    r = -np.r_[.00:500:.01]

    cd,a1,a2,R1,R2,x,y = coeff(a, bt, bb, rmud, rmus, rmuu)
    c1,c2 = amplitudes(Edf, Esf, Eubot, drf, cd, a1, a2, R1, R2, x, y, kbot)
    Ed,Es,Eu = calc(r, rf, Edf, c1, c2, cd, a1, a2, R1, R2, x, y, kbot)

#    Ed,Es,Eu = interp(r, rf, Edf, Esf, Eubot, a, bt, bb, rmud, rmus, rmuu, kbot)

    Eutop = np.array([ nc['dar_Eutop{:02d}'.format(nl)][it,:,0,0] for nl in range(1,14) ])[l][:-1]
    c1o = np.array([ nc['dar_c1_{:02d}'.format(nl)][it,:,0,0] for nl in range(1,14) ])[l]
    c2o = np.array([ nc['dar_c2_{:02d}'.format(nl)][it,:,0,0] for nl in range(1,14) ])[l]
    Edo,Eso,Euo = calc(r, rf, Edf, c1o, c2o, cd, a1, a2, R1, R2, x, y, kbot)

    plt.figure(1)
    plt.clf()
    plt.plot(-rf, Edf, '*')
    plt.plot(-rf, Esf, '*')
    plt.plot(-rf[:-1], Eutop, 'D')
    plt.plot(-rf[1:], Eubot, '*')
    plt.xlim(0, 100)
    k = np.where(rf>=-100)[0][-1]
    plt.ylim(min(Edf[:k].min(), Esf[:k].min(), Eubot[:k].min()), None)
    plt.yscale('log')

    plt.plot(-r, Ed, 'b')
    plt.plot(-r, Es, 'g')
    plt.plot(-r, Eu, 'r')

    plt.plot(-r, Edo, 'b--', lw=2)
    plt.plot(-r, Eso, 'g--', lw=2)
    plt.plot(-r, Euo, 'r--', lw=2)

    plt.draw()

    plt.figure(2)
    plt.clf()
    oj.plot.mystep(-rf, c1, 'b')
    oj.plot.mystep(-rf, c2, 'r')
    plt.plot(-rc, c1o[:-1], 'b*')
    plt.plot(-rc, c2o[:-1], 'r*')
    if c1.min()<0:
        oj.plot.mystep(-rf, -c1, 'b:', dashes=[2,2])
        plt.plot(-rc, -c1o[:-1], 'b+')
    plt.yscale('log')
    plt.xlim(0, 100)
    k = np.where(rf>=-100)[0][-1]
    plt.ylim(abs(np.r_[c2[:k],c2o[:k],c1[:k],c1o[:k]]).min(), None)

    plt.draw()

