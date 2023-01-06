import numpy as np
import matplotlib as mpl
from oj.mycb import *

def addcolorssqrt(fields, cols, vmax, fun, landmask=None, landcol=None):
    """
        r = [1,0,0]
        o = [1,0.5,0]
        y = [1,1,0]
        g = [0,1,0]
        m = [1,0,1]
        b = [0,0,1]
        c = [0,1,1]

        cols = vstack((g,b,c,y,r))
        fields = [phygrp[i,:,:] for i in (0,2,3,4)]

        addcolorssqrt(fields, cols, 10., landmask, [.3,.3,.3])
    """
    dims = fields[0].shape
    ndim = len(dims)

    rgb = np.zeros(dims+(3,)) 
    for i in range(len(fields)):
        rgb += cols[i,:].reshape(ndim*(1,)+(3,)) * fields[i].reshape(dims+(1,))

    rgb /= vmax
    rgb[rgb<0] = 0
    rgb[rgb>1] = 1  # this make high values go white!

    # apply sqrt to max. component while retaining hue
    maxcomp = rgb.max(axis=ndim)
    f = fun(maxcomp)
    nz = maxcomp!=0
    f[nz] /= maxcomp[nz]
#    f[np.isnan(f)] = 0 
    rgb *= f.reshape(dims+(1,))
     
    if landmask is not None:
        for ic in range(3):
            rgb[:,:,ic][landmask] = landcol[ic]

    return rgb


def rgbcolorbar(im, cols, vmax, vmin=0, ticks=None, **kwargs):
    carr = np.zeros((0,3))
    for c in range(cols.shape[0]):
        carr = r_[carr, np.kron(np.sqrt(np.linspace(0,1,65).reshape(65,1)),cols[c:c+1,:])]
    im.cmap = mpl.colors.ListedColormap(carr)
    im.norm = mpl.colors.Normalize(vmin,vmax)
    if ticks is None:
        ticks = [vmin,vmax]
    return colorbar(im,ticks=ticks,**kwargs)


def rgbcb(cax, im, cols, vmax, vmin=0, **kwargs):
    carr = np.zeros((0,3))
    for c in [0,2,3,4]:
        carr = r_[carr, np.kron(np.sqrt(np.linspace(0,1,65).reshape(65,1)),cols[c:c+1,:])]
    im.cmap = mpl.colors.ListedColormap(carr)
    im.norm = mpl.colors.Normalize(vmin,vmax)
    return mycb(cax,im,ticks=[vmin,vmax],**kwargs)


_clabels = ['"Prochlorococcus"',
            '"Synechococcus"',
            'large',
            '"Diatoms"']

def colorbars(cols, pos=None, pad=0.01,size=0.01, gap=None, labels=_clabels):
    ax = plt.gca()

    try:
        x0,y0,w,h = pos
    except TypeError:
        x0,y0,w,h = list(ax.get_position().bounds)
        x0 = x0 + w + pad
        w = size

    if h > w:
        if gap is None:
            gap = .03*h

        ch = (h+gap)/4.-gap
        cw = w

    cbcols = cols[[0,2,3,4]]

    caxs = []
    for i in range(4):
        cax = plt.axes([x0,y0+i*(ch+gap),cw,ch])
        cbim = mpl.cm.ScalarMappable()
        cbim.set_array(np.arange(.1))
        cb = rgbcolorbar(cbim,cbcols[i:i+1],.1,orientation='vertical', cax=cax)
        cax.set_ylabel(labels[i])
        caxs += [cax]

    plt.axes(ax)

    return caxs

