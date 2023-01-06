import numpy as np
import matplotlib.pyplot as plt
#from pylab import *
#from hsv2rgb import *
from oj.misc import myimshow

def z2rgb(z,d=.04,dunit=0):
    dims = z.shape
    r = np.abs(z)
    hue = np.mod(np.angle(z)/2./np.pi,1.)
    # introduce jumps at 0 and .5 (real axis)
    #hue = (hue<.5)*(d+(1-2*d)*hue) + (hue>=.5)*(.5+d+(1-2*d)*(hue-.5))
    #hue = (d+(1-2*d)*hue)
    hue = ((hue>=d)&(hue<=1-d))*hue + (hue<d)*d + (hue>1-d)*(1-d)
    val = r/(1.+r)
    val[np.isinf(r)] = 1.
    sat = 4*(1.-val)*val
    #sat = np.minimum(r,1./r)
    val = (1-dunit)*val + (val>.5)*dunit
    ind = np.isnan(z)
    sat[ind] = 0.
    val[ind] = .5
    hue[ind] = 0.
    rgb = hsv2rgb(hue, sat, val)
    return rgb

def hsv2rgb(h,s,v):
    eps = 2.2204e-16
    h = 6*h
    k = np.fix(h-6*eps)
    f = h-k
    t = 1-s
    n = 1-s*f
    p = 1-(s*(1-f))
    e = np.ones(h.shape)
    r = (k==0)*e + (k==1)*n + (k==2)*t + (k==3)*t + (k==4)*p + (k==5)*e
    g = (k==0)*p + (k==1)*e + (k==2)*e + (k==3)*n + (k==4)*t + (k==5)*t
    b = (k==0)*t + (k==1)*t + (k==2)*p + (k==3)*1 + (k==4)*1 + (k==5)*n
    f = v/max(np.max(r), np.max(g), np.max(b))
    return np.dstack((f*r, f*g, f*b))


def zshow(z):
    myimshow(z2rgb(z), origin='lower')


def fshow(xv,yv,f):
    [x,y] = np.meshgrid(xv,yv)
    myimshow(z2rgb(f(x+y*1j)), extent=(xv[0], xv[-1], yv[0], yv[-1]), origin='lower')


def zplot(z,color=None,c=None):
    if c is not None and color is None: color = c
    if color is None: color = ''
    z = z.astype(complex)
    res = plt.plot(np.real(z), np.imag(z), color)
    plt.gca().set_aspect(1)
    if plt.isinteractive():
        plt.draw()
    return res


def zplotgrid(z,lim=None,thin=1,hold=False):
    if not hold:
        plt.clf()
    zplot(z[:,0::thin].astype(complex),'r')
    zplot(z[0::thin,:].T.astype(complex),'g')
    if lim is not None:
        plt.axis(lim)


