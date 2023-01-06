#!/usr/bin/env python
# coding=UTF-8
import sys
from namespace import Namespace
import numpy as np
from numpy.ma import MaskedArray
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import prod, mod, pi, log, tan, cos
from pylab import gca, pcolormesh, imshow, draw, axis
from matplotlib.colors import LogNorm
from .colors import colorbar

_debug = False

def rel2abs(relpos, axpos=None):
    if not axpos:
        axpos = gca().get_position().bounds
    abspos=[0,0,0,0]
    abspos[0] = axpos[0]+axpos[2]*relpos[0]
    abspos[1] = axpos[1]+axpos[3]*relpos[1]
    abspos[2] =          axpos[2]*relpos[2]
    abspos[3] =          axpos[3]*relpos[3]
    return abspos                                            
    
    
def xzeroaxis(ax=None, ls=':k'):
    if ax is None:
        ax = gca()
    trans = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.plot([0,1], [0,0], ls, transform=trans)


def yzeroaxis(ax=None, ls=':k'):
    if ax is None:
        ax = gca()
    trans = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot([0,0], [0,1], ls, transform=trans)


def myfig(num=None, size=(1920,1150), dpi=80, facecolor=None, edgecolor=None, frameon=True, clear=False, forward=False, toolbarsize=40., **kwargs):
    dpi = float(dpi)
    figsize = [ float(pix)/dpi for pix in size ]
    fig = plt.figure(num, figsize, dpi, facecolor, edgecolor, frameon, **kwargs)
    fig.set_dpi(dpi)
    if forward:
        fig.set_size_inches((size[0]/dpi, (size[1]+toolbarsize)/dpi), forward=forward)
    fig.set_size_inches(figsize)
    if clear:
        fig.clear()
    try:
        canvas = fig.canvas.get_tk_widget()
    except:
        pass
    else:
        canvas.config(bd=0)
    return fig

def spadj(fig=None, left=None, bottom=None, right=None, top=None,
          wspace=None, hspace=None):
    if fig is None:
        fig = plt.gcf()
    w,h = fig.get_size_inches()
    left /= w
    right = 1. - right/w
    top = 1. - top/h
    bottom /= h
    if hspace is not None:
        ax = fig.axes[0]
        n = ax.numRows
        hspace = n*hspace/((top-bottom)*h - (n-1)*hspace)
    fig.subplots_adjust(left, bottom, right, top, wspace, hspace)


def inch(a, tot):
    if np.iscomplex(a):
        a = a.real + a.imag/tot
    return a

def set_size_inches(fig, *args, **kwargs):
    if not isinstance(fig, mpl.figure.Figure):
        fig = plt.figure(fig)

    if len(args)==1:
        w,h = args[0]
    else:
        w,h = args

    fig.set_size_inches(*args, **kwargs)
    if kwargs.get('forward', False):
        fig.canvas.draw()
        w1,h1 = fig.get_size_inches()
#        print h,h1,fig.dpi, h1 != h
        if h1 != h:
#            print h+h-h1
            fig.set_size_inches(w, h+h-h1, **kwargs)
            w1,h1 = fig.get_size_inches()
            sys.stderr.write('fig size {} {}/{}\n'.format(w1, h1, h))

def myspadjust(nrows_ncols, fig=None, **kwargs):
    nr,nc = nrows_ncols
    if fig is None:
        fig = plt.gcf()
    elif type(fig) == type(1):
        fig = plt.figure(fig)

    fw,fh = fig.get_size_inches()

    left   = inch(kwargs.get('left',   4j/6), fw)
    right  = inch(kwargs.get('right',1-4j/6), fw)
    wgap   = inch(kwargs.get('wgap',   4j/6), fw)
    top    = inch(kwargs.get('top',  1-2j/6), fh)
    bottom = inch(kwargs.get('bottom', 2j/6), fh)
    hgap   = inch(kwargs.get('hgap',   2j/6), fh)
    wspace = wgap*nc/(right-left-wgap*(nc-1))
    hspace = hgap*nr/(top-bottom-hgap*(nr-1))
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom,
                        wspace=wspace, hspace=hspace)
    pars = Namespace(left=left, right=right, top=top, bottom=bottom,
                     wgap=wgap, hgap=hgap, nrows=nr, ncols=nc)
    return pars

def myfigtext(x, y, s, fig=None, **kwargs):
    if fig is None:
        fig = plt.gcf()
    elif type(fig) == type(1):
        fig = plt.figure(fig)
    fw,fh = fig.get_size_inches()
    x = inch(x, fw)
    y = inch(y, fh)
    fig.text(x, y, s, **kwargs)

def mysavefig(fname, size=(1920,1150), dpi=None, width=None, height=None, restore=True, **kwargs):
    fig = kwargs.get('fig', plt.gcf())
    w,h = fig.get_size_inches()
    fs = None
    if width is not None or height is not None:
        # compute dpi from width or height, discard size
        if height is None:
            dpi = width/w
        elif width is None:
            dpi = height/h
        else:
            sys.stderr.write('mysavefig: cannot adjust width and height at the same time.\n')
            raise ValueError
    else:
        # compute fig_size_inches from size and dpi
        if dpi is None:
            dpi = fig.dpi
        fs = [ sz/float(dpi) for sz in size ]
        fig.set_size_inches(fs)

    res = plt.savefig(fname, dpi=dpi, **kwargs)

    if restore and fs is not None:
        fig.set_size_inches([w,h])

    return res


def secshow(X, cmap=None, norm=None, aspect='auto', interpolation=None,
            alpha=1.0, vmin=None, vmax=None, origin='upper', extent=None,
            symmetric=False, fixaxis=True, mask=None, maskval=None,
            cb=False, ij=False, fig=None, e=0, hold='replace',
            **kwargs):
    return myimshow(X, cmap, norm, aspect, interpolation,
             alpha, vmin, vmax, origin, extent,
             symmetric, fixaxis, mask, maskval,
             cb, ij, fig, e, hold, **kwargs)


def symshow(X, cmap=None, norm=None, aspect=None, interpolation=None,
            alpha=1.0, vmin=None, vmax=None, origin=None, extent=None,
            symmetric=False, fixaxis=True, mask=None, maskval=None,
            cb=False, ij=False,
            **kwargs):
    from .cm import lBuRd
    from .colors import SymNorm
    if cmap is None: cmap = 'balance'  #lBuRd
    if norm is None: norm = SymNorm()
    return myimshow(X, cmap, norm, aspect, interpolation,
             alpha, vmin, vmax, origin, extent,
             symmetric, fixaxis, mask, maskval,
             cb, ij, **kwargs)


def myimshow(X, cmap=None, norm=None, aspect=None, interpolation=None,
             alpha=1.0, vmin=None, vmax=None, origin=None, extent=None,
             symmetric=False, fixaxis=True, mask=None, maskval=None,
             cb=False, ij=False, fig=None, e=0, hold='replace',
             cbargs={},
             **kwargs):
    if fig is not None:
        plt.figure(fig)
    if ij and origin is None:
        origin = 'upper'
    if origin is None:
        origin = 'lower'
    if interpolation is None:
        interpolation = 'nearest'
    # delete colorbars of old images
    if 'axes' in kwargs:
        ax = kwargs['axes']
    else:
        ax = gca()
#    if 'hold' not in kwargs:
#        kwargs['hold'] = ax.ishold()
    if hold in [False, 'replace']:
        for im in ax.images:
            if im.colorbar is not None:
                try:
                    cax = im.colorbar.ax
                except AttributeError:
                    cax = im.colorbar[1]
                if cax in im.figure.axes:
                    if hold == 'replace':
                        if extent is None:
                            ny,nx = X.shape[:2]
                            if e:
                                myext = (e-.5, nx+e-.5, e-.5, ny+e-.5)
                            else:
                                myext = (-.5, nx-.5, -.5, ny-.5)
                        else:
                            myext = extent

                        if myext != im.get_extent():
                            hold = False

                    im.figure.delaxes(cax)
                    im.colorbar = None
                    ax.images.remove(im)
    kwargs['hold'] = True if hold == 'replace' else hold
    if kwargs['hold']:
        del kwargs['hold']
    if mask is None and maskval is not None:
        mask = X == maskval
    if mask is not None:
        X = MaskedArray(X,mask)
    if symmetric is True:
        if vmax is not None:
            if vmin is not None:
                vmax = max(vmax,-vmin)
            vmin = -vmax
        elif vmin is not None:
            vmax = -vmin
        elif norm is not None:
            norm.vmax = max(norm.vmax,-norm.vmin)
            norm.vmin = -norm.vmax
        else:
            vmax = max(np.max(X),-np.min(X))
            vmin = -vmax
    if e and not extent:
        ny,nx = X.shape[:2]
        extent = (e-.5, nx+e-.5, e-.5, ny+e-.5)
    im = imshow(X, cmap, norm, aspect, interpolation,
                alpha, vmin, vmax, origin, extent, **kwargs)
    if fixaxis and interpolation.lower() != 'none':
        ex = im.get_extent()
        if axis() == ex:
            x0,x1,y0,y1 = ex
            #x1 += 1.*(x1-x0)/w
            #y1 += 1.5*(y1-y0)/w
            ax = im.axes
            x1f,y1f = ax.transData.transform((x1,y1))
            x1p,y1p = ax.transData.inverted().transform((x1f+1.,y1f+1.99))
            axis((x0,x1p,y0,y1p))

    if cb:
        cb = colorbar(im,ax=gca(),**cbargs)
        return im,cb
    else:
        return im


def axes_size_inches(ax=None):
    if ax is None: ax = gca()
    pos = ax.get_position()
    fig = ax.get_figure()
    w,h = fig.get_size_inches()
    return np.array([w*pos.width, h*pos.height])


def im_dpi(im=None):
    if im is None: im = gci()
    h,w = im.get_size()
    win,hin = axes_size_inches(im.axes)
    return w/win, h/hin
#    if im is None: im = gci()
#    ax = im.axes
#    sz = im.get_size()
#    w,h = ax.transData.transform(sz)-ax.transData.transform((0,0))


def im_size_pixels(im=None, extent=None):
    if im is None: im = gci()
    ax = im.axes
#    fig = ax.get_figure()
#    sz = im.get_size()
    if extent is None:
        extent = im.get_extent()
    ext = np.array(extent).reshape(2,2).T
    extpix = ax.transData.transform(ext)
    return extpix[1,:] - extpix[0,:]


def fix_extent(im, extent=None):
    ''' fix_extent(im, extent=None)

    Do not call im.set_extent(...) before calling this.  Call fix_extent with
    the 'extent' argument instead.
    '''
    if extent is None:
        try:
            extent = im._extent_orig
        except AttributeError:
            extent = im.get_extent()
    im._extent_orig = tuple(extent)
    nx,ny = im_size_pixels(im, extent)
    # width->width*(nx-1)/nx
    dx = (extent[1]-extent[0])/nx
    dy = (extent[3]-extent[2])/ny
    extent = (extent[0], extent[1]-dx, extent[2], extent[3]-dy)
    im.set_extent(extent)


def im_size_axes(im=None):
    if im is None: im = gci()
    ax = im.axes
#    fig = ax.get_figure()
#    sz = im.get_size()
    ext = np.array( im.get_extent()).reshape(2,2).T
    extpix = ax.transAxes.inverted().transform(ax.transData.transform(ext))
    return extpix[1,:] - extpix[0,:]


def im_scale_factor(im=None):
    if im is None: im = gci()
    ax = im.axes
    fig = ax.get_figure()
    h0,w0 = im.get_size()
    w,h = im_size_pixels(im)
    return np.sqrt((w0/w)*(h0/h))
    

def im_fig_size(im=None):
    if im is None: im = plt.gci()
    ax = im.axes
    fig = ax.get_figure()
    sz = fig.get_size_inches()
    h0,w0 = im.get_size()
    w,h = im_size_pixels(im)
    return np.array([sz[0]*(w0/w), sz[1]*(h0/h)])


def im_fig_dpi(im=None):
    if im is None: im = plt.gci()
    ax = im.axes
    fig = ax.get_figure()
    sz = fig.get_size_inches()
    h0,w0 = im.get_size()
    w,h = im_size_pixels(im)
    return fig.dpi*(w0/w)


def im_axes_size(im=None):
    if im is None: im = plt.gci()
    ax = im.axes
    h0,w0 = im.get_size()
    w,h = im_size_pixels(im)
    aw,ah = ax.get_position().size
    return np.array([aw*(w0/w),ah*(h0/h)])


def calc_fig_size(wh, imex, axex, axpos, dpi):
    w,h = wh
    imexw,imexh = imex
    axexw,axexh = axex
    axposw,axposh = axpos
    fw = float(w*axexw)/float(imexw*axposw*dpi)
    fh = float(h*axexh)/float(imexh*axposh*dpi)
    return np.array([fw,fh])


def subaxes(pos, ax=None, trans=None):
    if ax is None:
        ax = gca()
    if trans is None:
        trans = ax.transAxes
    fig = ax.get_figure()
    # map LL and UR corners to normalized figure coords
#    print [pos[:2],[pos[0]+pos[2],pos[1]+pos[3]]]
    figpoints = fig.transFigure.inverted().transform(trans.transform([pos[:2],[pos[0]+pos[2],pos[1]+pos[3]]]))
#    print figpoints
    # turn into pos
    figpos = np.r_[figpoints[0,:], figpoints[1,:]-figpoints[0,:]]
#    print figpos
    return fig.add_axes(figpos)
    
    
def pos_borders_inches(l,b=None,r=None,t=None,fig=None):
    if t is None:
        l,b,r,t = l
    if fig is None:
        fig = plt.gcf()
    w,h = fig.get_size_inches()
    return [l/w, b/h, (w-l-r)/w, (h-b-t)/h]


def pos_borders_fs(l,b=None,r=None,t=None,fs=None,fig=None):
    if t is None:
        l,b,r,t = l
    if fig is None:
        fig = plt.gcf()
    if fs is None:
        fs = mpl.rcParams['font.size']
    fs /= 72.
    w,h = fig.get_size_inches()
    return [l*fs/w, b*fs/h, (w-(l+r)*fs)/w, (h-(b+t)*fs)/h]


def axes_borders_inches(borders, ax=None):
    l,b,r,t = borders
    if ax is None:
        ax = plt.gca()
    fig = ax.figure
    w,h = fig.get_size_inches()
    ax.set_position([l/w, b/h, (w-l-r)/w, (h-b-t)/h])


def axes_borders_fs(borders, ax=None):
    l,b,r,t = borders
    if ax is None:
        ax = plt.gca()
    fsx = ax.yaxis.offsetText.get_fontsize()/72.
    fsy = ax.xaxis.offsetText.get_fontsize()/72.
    axes_borders_inches([l*fsx, b*fsy, r*fsx, t*fsy], ax)


def fixtext(th):
    th.figure.canvas.draw()
    th.set_position(th.get_window_extent().bounds[:2])
    th.set_ha('left')
    th.set_va('baseline')
    th.set_transform(mpl.transforms.IdentityTransform())


def pause(*args):
    draw()
    raw_input(*args)


def _latformatfunc(val, pos=None, sep='$^o$'):
    if val > 0:
        return '%G' % val + sep + 'N'
    elif val == 0:
        return '0' + sep
    else:
        return '%G' % -val + sep + 'S'

LatitudeFormatter = mpl.ticker.FuncFormatter(_latformatfunc)

def _lonformatfunc(val, pos=None, sep='$^o$'):
    val = np.mod(val,360)
    if val == 0 or val >= 359.9999999:
        return '0' + sep
    elif val < 179.9999999:
        return '%G' % val + sep + 'E'
    elif val <= 180.0000001:
        return '180' + sep
    else:
        return '%G' % (360-val) + sep + 'W'

def LatitudeFormatter(sep=u'°'):
    return mpl.ticker.FuncFormatter(lambda val,pos=None:_latformatfunc(val,pos,sep))

def LongitudeFormatter(sep=u'°'):
    return mpl.ticker.FuncFormatter(lambda val,pos=None:_lonformatfunc(val,pos,sep))


def cycle_ls(lhs=None, n=0, lscycle=['-','--',':','-.']):
    if lhs is None:
        lhs = plt.gca().get_lines()
    if n == 0:
        n = len(mpl.rcParams['axes.color_cycle'])
    for i,lh in enumerate(lhs):
        lh.set_ls(lscycle[i//n%len(lscycle)])
    plt.draw_if_interactive()


def mystep(x, y=None, *args, **kwargs):
    if y is None or mpl.cbook.is_string_like(y):
        if y is not None:
            args = (y,) + args
        y = x
        x = np.arange(len(x) + 1)
    y = np.asfarray(y)
    ax = kwargs.pop('ax', None)
    if ax is not None:
        step = ax.step
    else:
        step = plt.step
    y = np.insert(y, 0, np.nan, axis=0)
    return step(x, y, *args, **kwargs)

def mysteph(y, x=None, *args, **kwargs):
    if x is None or mpl.cbook.is_string_like(x):
        if x is not None:
            args = (x,) + args
        x = y
        y = np.arange(len(y) + 1)
    ax = kwargs.pop('ax', None)
    if ax is not None:
        plot = ax.plot
    else:
        plot = plt.plot
    x = x[..., None].repeat(2, -1).reshape(x.shape[:-1] + (-1,))
    y = y[..., None].repeat(2, -1).reshape(y.shape[:-1] + (-1,))[1:-1]
#    print x.shape, y.shape
    return plot(x, y, *args, **kwargs)

def sobolcycle(n=32):
    import sobol
    sob = sobol.sobol(3,0)
    scols = [sob.next() for _ in range(n)]
    cols = [plt.cm.colors.colorConverter.to_rgb(k) for k in ['b','g','r','c','m','y','k','.5']]
    cols = np.r_[cols, scols[3:]]
    try:
        from cycler import cycler
        plt.rc('axes', prop_cycle=cycler('color', cols.tolist()))
    except ImportError:
        plt.rc('axes', color_cycle=cols.tolist())
    return cols

def tight(pad=1.08, h_pad=None, w_pad=None, rect=None, axis='both'):
    plt.autoscale(True, axis=axis, tight=True)
    plt.tight_layout(pad, h_pad, w_pad, rect)

