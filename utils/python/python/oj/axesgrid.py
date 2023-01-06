#!/usr/bin/env python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from .colors import colorbar

_debug = False

_pcolorargs = { 'cmap':None, 'norm':None, 'alpha':1.0, 'vmin':None, 'vmax':None, 'origin':None, 'extent':None, 'shape':None, 'filternorm':1, 'filterrad':4.0, 'imlim':None, 'resample':None, 'url':None, 'hold':None, 'shading':'flat' }

_imshowargs = dict(_pcolorargs)
_imshowargs.update({'aspect':None, 'interpolation':None})

class PCGrid(object):
    def __init__(self, nrows_ncols, vmin=None, vmax=None, norm=None, titles=None, max_title=False, clf=True, figsize=None, dpi=80, **kwargs):
        if figsize is not None:
    #        fig = mpl.figure.Figure([s/float(dpi) for s in figsize], dpi)
            fs = [s/float(dpi) for s in figsize]
            fig = plt.figure(1,fs,dpi)
            fig.set_size_inches(fs)
            fig.set_dpi(dpi)
            kwargs['fig'] = fig

        nrows, ncols = nrows_ncols
        n = nrows*ncols

        cbar_mode = kwargs.get('cbar_mode', 'single')

        if cbar_mode != 'each':
            if norm is None:
                if vmin is None:
                    vmin = min(np.amin(p) for p in a)
                if vmax is None:
                    vmax = max(np.amax(p) for p in a)
                norm = plt.Normalize(vmin,vmax)

        try:
            iter(vmin)
        except TypeError:
            vmin = n*[vmin]

        try:
            iter(vmax)
        except TypeError:
            vmax = n*[vmax]

        try:
            iter(norm)
        except TypeError:
            norm = n*[norm]

        imshowargs = {}
        imshowkeys = [ key for key in kwargs if key in _pcolorargs ]
        for key in imshowkeys:
            imshowargs[key] = kwargs.pop(key)

        if nrows_ncols is not None:
            kwargs['nrows_ncols'] = nrows_ncols

        if 'nrows_ncols' not in kwargs:
            kwargs['nrows_ncols'] = (n,1)

        fig = kwargs.get('fig', plt.gcf())
        kwargs['fig'] = fig

        if 'rect' not in kwargs:
            kwargs['rect'] = [.05,.05,.9,.9]

        if 'cbar_mode' not in kwargs:
            kwargs['cbar_mode'] = 'single'

        if 'cbar_size' not in kwargs:
            kwargs['cbar_size'] = 12./72

        titleargs = {}
        titlekeys = [ key for key in kwargs if key.startswith('title') ]
        for key in titlekeys:
            titleargs[key[5:]] = kwargs.pop(key)

    #    if titles is not None:
    #        deftitlesize = mpl.rcParams['axes.titlesize']
    #        deffontsize = mpl.font_manager.font_scalings[deftitlesize]*mpl.rcParams['font.size']
    #        fontsize = titleargs.get('size', deffontsize)
    #        if 'axes_pad' not in kwargs:
    #            kwargs['axes_pad'] = 1.6*fontsize/72.

        if clf:
            fig.clf()

        agargs = {'share_all':True}
        padx = 0.02
        pady = 0.02
        padc = 0.02
        deftitlesize = mpl.rcParams['axes.titlesize']
        deffontsize = mpl.font_manager.font_scalings[deftitlesize]*mpl.rcParams['font.size']
        fontsize = titleargs.get('size', deffontsize)
        if cbar_mode == 'each':
            padx = 5*fontsize/72.

        if titles is not None:
            pady = 1.6*fontsize/72.
            if cbar_mode == 'each':
                padc = .5*fontsize/72.
            else:
                padc = pady

        if cbar_mode == 'each' or titles is not None:
            pady = max(.5*fontsize/72.,pady)
            padc = max(.5*fontsize/72.,padc)

        if padx < pady: padx = pady

        if padx+pady > 0:
            agargs['axes_pad'] = (padx,pady)
            agargs['cbar_pad'] = padc

        agargs.update(kwargs)

        self.ag = AxesGrid(**agargs)
        self.imshowargs = imshowargs
        self.norm = norm
        self.vmin = vmin
        self.vmax = vmax

#    ims = [ ax.pcolormesh(x, y, a2, vmin=mn, vmax=mx, norm=nm, **imshowargs) for ax,a2,nm,mn,mx in zip(ag,a,norm,vmin,vmax) ]

    def cbars(self, mpbls):
        if cbar_mode == 'each':
            cb = []
            for ax,cax,im in zip(ag,ag.cbar_axes,ims):
                cb.append(colorbar(im, cax))
                if max_title:
                    mn,mx = im.get_clim()
                    cax.set_title('{0:.3g}'.format(mx))
                else:
                    cax.yaxis.set_visible(1)
        else:
            cax = ag.cbar_axes[0]
            cb = colorbar(ims[0], cax)
            cax.yaxis.set_visible(1)

    def titles(self, titles):
        for ax, title in zip(ag, titles):
            ax.set_title(title, **titleargs)

    def add(self, ims):
        if clf:
            fig.clf()
        for ax, im in zip(ag,ims):
            fig.add_axes(ax)
            fig.add_axes(ax.cax)

    def draw(self):
        plt.draw_if_interactive()


def pcolorgrid(x, y, a, nrows_ncols=None, vmin=None, vmax=None, norm=None, titles=None, max_title=False, clf=True, figsize=None, dpi=80, **kwargs):
    if figsize is not None:
#        fig = mpl.figure.Figure([s/float(dpi) for s in figsize], dpi)
        fs = [s/float(dpi) for s in figsize]
        fig = plt.figure(1,fs,dpi)
        fig.set_size_inches(fs)
        fig.set_dpi(dpi)
        kwargs['fig'] = fig

    if isinstance(a,tuple):
        a = np.zeros(a)

    n = len(a)
    cbar_mode = kwargs.get('cbar_mode', 'single')

    if cbar_mode != 'each':
        if norm is None:
            if vmin is None:
                vmin = min(np.amin(p) for p in a)
            if vmax is None:
                vmax = max(np.amax(p) for p in a)
            norm = plt.Normalize(vmin,vmax)

    try:
        iter(vmin)
    except TypeError:
        vmin = n*[vmin]

    try:
        iter(vmax)
    except TypeError:
        vmax = n*[vmax]

    try:
        iter(norm)
    except TypeError:
        norm = n*[norm]

    imshowargs = {}
    imshowkeys = [ key for key in kwargs if key in _pcolorargs ]
    for key in imshowkeys:
        imshowargs[key] = kwargs.pop(key)

    if nrows_ncols is not None:
        kwargs['nrows_ncols'] = nrows_ncols

    if 'nrows_ncols' not in kwargs:
        kwargs['nrows_ncols'] = (n,1)

    fig = kwargs.get('fig', plt.gcf())
    kwargs['fig'] = fig

    if 'rect' not in kwargs:
        kwargs['rect'] = [.05,.05,.9,.9]

    if 'cbar_mode' not in kwargs:
        kwargs['cbar_mode'] = 'single'

    if 'cbar_size' not in kwargs:
        kwargs['cbar_size'] = 12./72

    titleargs = {}
    titlekeys = [ key for key in kwargs if key.startswith('title') ]
    for key in titlekeys:
        titleargs[key[5:]] = kwargs.pop(key)

#    if titles is not None:
#        deftitlesize = mpl.rcParams['axes.titlesize']
#        deffontsize = mpl.font_manager.font_scalings[deftitlesize]*mpl.rcParams['font.size']
#        fontsize = titleargs.get('size', deffontsize)
#        if 'axes_pad' not in kwargs:
#            kwargs['axes_pad'] = 1.6*fontsize/72.

    if clf:
        fig.clf()

    agargs = {'share_all':True}
    padx = 0.02
    pady = 0.02
    padc = 0.02
    deftitlesize = mpl.rcParams['axes.titlesize']
    deffontsize = mpl.font_manager.font_scalings[deftitlesize]*mpl.rcParams['font.size']
    fontsize = titleargs.get('size', deffontsize)
    if cbar_mode == 'each':
        padx = 5*fontsize/72.

    if titles is not None:
        pady = 1.6*fontsize/72.
        if cbar_mode == 'each':
            padc = .5*fontsize/72.
        else:
            padc = pady

    if cbar_mode == 'each' or titles is not None:
        pady = max(.5*fontsize/72.,pady)
        padc = max(.5*fontsize/72.,padc)

    if padx < pady: padx = pady

    if padx+pady > 0:
        agargs['axes_pad'] = (padx,pady)
        agargs['cbar_pad'] = padc

    if len(x) != len(a) or x[0].ndim not in [1, 2]:
        x = len(a)*[x]

    if len(y) != len(a) or y[0].ndim not in [1, 2]:
        y = len(a)*[y]

    agargs.update(kwargs)

    ag = AxesGrid(**agargs)
    ims = [ ax.pcolormesh(x2, y2, a2, vmin=mn, vmax=mx, norm=nm, **imshowargs) for ax,x2,y2,a2,nm,mn,mx in zip(ag,x,y,a,norm,vmin,vmax) ]

    if cbar_mode == 'each':
        cb = []
        for ax,cax,im in zip(ag,ag.cbar_axes,ims):
            cb.append(colorbar(im, cax))
            if max_title:
                mn,mx = im.get_clim()
                cax.set_title('{0:.3g}'.format(mx))
            else:
                cax.yaxis.set_visible(1)
    else:
        cax = ag.cbar_axes[0]
        cb = colorbar(ims[0], cax)
        cax.yaxis.set_visible(1)

    if titles is not None:
        for ax,title in zip(ag,titles):
            ax.set_title(title,**titleargs)

    if not kwargs.get('add_all',True):
        if clf:
            fig.clf()
        for ax,im in zip(ag,ims):
            fig.add_axes(ax)
            fig.add_axes(ax.cax)

    plt.draw_if_interactive()

    return ag,ims,cb


def imgridargs(a, nrows_ncols=None, vmin=None, vmax=None, norm=None, titles=None,
               max_title=False, mask=None, clf=True, figsize=None, dpi=80,
               colorbar=plt.colorbar, **kwargs):
    if figsize is not None:
#        fig = mpl.figure.Figure([s/float(dpi) for s in figsize], dpi)
        fs = [s/float(dpi) for s in figsize]
        fig = plt.figure(1,fs,dpi)
        fig.set_size_inches(fs)
        fig.set_dpi(dpi)
        kwargs['fig'] = fig

    if isinstance(a,tuple):
        a = np.zeros(a)

    if mask is not None:
        a = [ np.ma.masked_array(p, mask) for p in a ]

    n = len(a)
    cbar_mode = kwargs.get('cbar_mode', 'single')

    if cbar_mode == 'single':
        if norm is None:
            if vmin is None:
                vmin = min(np.nanmin(p) for p in a)
            if vmax is None:
                vmax = max(np.nanmax(p) for p in a)
            norm = plt.Normalize(vmin,vmax)

    try:
        iter(vmin)
    except TypeError:
        vmin = n*[vmin]

    try:
        iter(vmax)
    except TypeError:
        vmax = n*[vmax]

    try:
        iter(norm)
    except TypeError:
        norm = n*[norm]

    imshowargs = {'interpolation':'nearest','origin':'lower'}
    imshowkeys = [ key for key in kwargs if key in _imshowargs ]
    for key in imshowkeys:
        imshowargs[key] = kwargs.pop(key)

    if nrows_ncols is not None:
        kwargs['nrows_ncols'] = nrows_ncols

    if 'nrows_ncols' not in kwargs:
        kwargs['nrows_ncols'] = (n,1)

    fig = kwargs.get('fig', plt.gcf())
    kwargs['fig'] = fig

    if 'rect' not in kwargs:
        kwargs['rect'] = [.05,.05,.9,.9]

    if 'cbar_mode' not in kwargs:
        kwargs['cbar_mode'] = 'single'


    if 'cbar_size' not in kwargs:
        kwargs['cbar_size'] = 12./72

    titleargs = {}
    titlekeys = [ key for key in kwargs if key.startswith('title') ]
    for key in titlekeys:
        titleargs[key[5:]] = kwargs.pop(key)

#    if titles is not None:
#        deftitlesize = mpl.rcParams['axes.titlesize']
#        deffontsize = mpl.font_manager.font_scalings[deftitlesize]*mpl.rcParams['font.size']
#        fontsize = titleargs.get('size', deffontsize)
#        if 'axes_pad' not in kwargs:
#            kwargs['axes_pad'] = 1.6*fontsize/72.

    if clf:
        fig.clf()

    agargs = {'share_all':True}
    padx = 0.02
    pady = 0.02
    padc = 0.02
    deftitlesize = mpl.rcParams['axes.titlesize']
    deffontsize = mpl.font_manager.font_scalings[deftitlesize]*mpl.rcParams['font.size']
    fontsize = titleargs.get('size', deffontsize)
    if cbar_mode == 'each':
        padx = 5*fontsize/72.

    if titles is not None:
        pady = 1.6*fontsize/72.
        if cbar_mode == 'each':
            padc = .5*fontsize/72.
        else:
            padc = pady

    if cbar_mode == 'each' or titles is not None:
        pady = max(.5*fontsize/72.,pady)
        padc = max(.5*fontsize/72.,padc)

    if padx < pady: padx = pady

    if padx+pady > 0:
        agargs['axes_pad'] = (padx,pady)
        agargs['cbar_pad'] = padc

    agargs.update(kwargs)

    return fig,agargs,imshowargs,titleargs,cbar_mode,norm,vmin,vmax


def imgrid(a, nrows_ncols=None, vmin=None, vmax=None, norm=None, titles=None,
           max_title=False, mask=None, clf=True, figsize=None, dpi=80,
           colorbar=plt.colorbar, **kwargs):

    fig,agargs,imshowargs,titleargs,cbmode,norm,vmin,vmax = imgridargs(
            a, nrows_ncols, vmin, vmax, norm, titles, max_title, mask,
            clf, figsize, dpi, colorbar, **kwargs)

    ag = AxesGrid(**agargs)
    ims = [ ax.imshow(a2, vmin=mn, vmax=mx, norm=nm, **imshowargs) for ax,a2,nm,mn,mx in zip(ag,a,norm,vmin,vmax) ]

    if cbmode != 'single':
        cb = []
        for ax,cax,im in zip(ag,ag.cbar_axes,ims):
            cb.append(colorbar(im, cax))
            if max_title:
                mn,mx = im.get_clim()
                cax.set_title('{0:.3g}'.format(mx))
            else:
                cax.yaxis.set_visible(1)
    else:
        cax = ag.cbar_axes[0]
        cb = colorbar(ims[0], cax)
        cax.yaxis.set_visible(1)

    if titles is not None:
        for ax,title in zip(ag,titles):
            ax.set_title(title,**titleargs)

    if not kwargs.get('add_all',True):
        if clf:
            fig.clf()
        for ax,cax,im in zip(ag,ag.cbar_axes,ims):
            fig.add_axes(ax)
            fig.add_axes(cax)

    plt.draw_if_interactive()

    return ag,ims,cb
    

_plotargs = { 'origin':None, 'hold':None, 'drawstyle':None }

def plotgrid(a, nrows_ncols=None, ymin=None, ymax=None, titles=None, clf=True, plotaspect='auto', **kwargs):
    n = len(a)
    share_all = kwargs.get('share_all', False)

    if share_all:
        if ymin is None:
            ymin = np.amin(a)
        if ymax is None:
            ymax = np.amax(a)

    try:
        iter(ymin)
    except TypeError:
        ymin = n*[ymin]

    try:
        iter(ymax)
    except TypeError:
        ymax = n*[ymax]

    plotargs = {}
    plotkeys = [ key for key in kwargs if key in _plotargs ]
    for key in plotkeys:
        plotargs[key] = kwargs.pop(key)

    if nrows_ncols is not None:
        kwargs['nrows_ncols'] = nrows_ncols

    if 'nrows_ncols' not in kwargs:
        kwargs['nrows_ncols'] = (n,1)

    if 'fig' not in kwargs:
        kwargs['fig'] = plt.gcf()

    if 'rect' not in kwargs:
        kwargs['rect'] = [.05,.05,.9,.9]

    titleargs = {}
    titlekeys = [ key for key in kwargs if key.startswith('title') ]
    for key in titlekeys:
        titleargs[key[5:]] = kwargs.pop(key)

    if titles is not None:
        deftitlesize = mpl.rcParams['axes.titlesize']
        deffontsize = mpl.font_manager.font_scalings[deftitlesize]*mpl.rcParams['font.size']
        fontsize = titleargs.get('size', deffontsize)
        if 'axes_pad' not in kwargs:
            kwargs['axes_pad'] = 1.6*fontsize/72.

    fig = plt.gcf()
    if clf:
        fig.clf()

    ag = AxesGrid(**kwargs)
    liness = [ ax.plot(a2, **plotargs) for ax,a2 in zip(ag,a) ]

    for ax,mn,mx in zip(ag,ymin,ymax):
        ax.set_ylim(ymin=mn,ymax=mx)
        ax.set_aspect(plotaspect,adjustable='box-forced')

    if titles is not None:
        for ax,title in zip(ag,titles):
            ax.set_title(title,**titleargs)

    if not kwargs.get('add_all',True):
        if clf:
            fig.clf()
        for ax,_ in zip(ag,liness):
            fig.add_axes(ax)

    plt.draw_if_interactive()

    return ag,liness

