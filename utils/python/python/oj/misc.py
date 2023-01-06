#!/usr/bin/env python
import os
import sys, time, gzip
import re
from glob import glob
import exceptions
import numpy as np
from numpy import frombuffer, fromfile
from numpy.ma import MaskedArray
import matplotlib.pyplot as plt
from pylab import gca, prod, mod, pcolormesh, pi, log, tan, cos, imshow, draw, axis
from matplotlib.colors import LogNorm
import matplotlib as mpl
from colors import colorbar

_debug = False

def fromgzipfile(file, dtype=float, count=-1, offset=0):
    fid = gzip.open(file, 'rb')
    res = frombuffer(fid.read(), dtype, count, offset)
    fid.close()
    return res

#def myfromfile(file, dtype=float, count=-1):
#    if file.endswith('.gz'):
#        return fromgzipfile(file, dtype, count)
#    elif os.path.exists(file):
#        return fromfile(file, dtype, count)
#    elif os.path.exists(file + '.gz'):
#        print 'myfromfile: reading', file + '.gz'
#        return fromgzipfile(file + '.gz', dtype, count)
#    else:
#        # this will most likely raise an IOError
#        return fromfile(file, dtype, count)

def myfromfid(fid, dtype=float, shape=None, count=-1, skip=-1):
    size = np.dtype(dtype).itemsize
    if shape is not None:
        size *= np.prod(shape)
        if count >= 0:
            shape = (count,) + shape
        if count >= 0 or skip >= 0:
            count = np.prod(shape)
 
    if skip > 0:
        fid.seek(skip*size)

#    print count, shape
    a = fromfile(fid, dtype, count)

    if shape is not None:
        a = a.reshape(shape)

    return a


def myfromfile(file, dtype=float, shape=None, count=-1, skip=-1, skipbytes=0):
    zipped = False
    if file.endswith('.gz'):
        zipped = True
    elif os.path.exists(file):
        zipped = False
    elif os.path.exists(file + '.gz'):
        if _debug: print 'myfromfile: reading', file + '.gz'
        zipped = True
        file = file + '.gz'
    else:
        # this will most likely raise an IOError
        pass

    if zipped:
        openf = gzip.open
    else:
        openf = open

    countbytes = -1

    size = np.dtype(dtype).itemsize
    if shape is not None:
        size *= np.prod(shape)
        if count >= 0:
            shape = (count,) + shape
        if count >= 0 or skip >= 0 or skipbytes > 0:
            count = np.prod(shape)

    if skip > 0:
        skipbytes += skip*size
 
# gzip doesn't support the 'with', so we do it ourselves       
#    with openf(file, 'rb') as fid:
#        if skip > 0:
#            size = np.dtype(dtype).itemsize
#            if shape is not None:
#                size *= np.prod(shape)
#            fid.seek(skip*size)
#        a = fromfile(fid, dtype, count)

    fid = openf(file, 'rb')
    exc = True
    try:
        try:
            if skipbytes > 0:
                fid.seek(skipbytes)
#            print dtype,count
            if zipped:
                a = frombuffer(fid.read(), dtype, count)
            else:
                a = fromfile(fid, dtype, count)
        except:
            exc = False
            fid.close()
            raise
    finally:
        if exc:
            fid.close()

    if shape is not None:
        try:
            a = a.reshape(shape)
        except ValueError:
            if count >=0:
                raise IOError('Could only read {0} items, expecting {1}'.format(len(a),count))
            else:
                raise IOError('Wrong file size: read {0} items, expecting {1}'.format(len(a),np.prod(shape)))

    return a


_typemap = {'>f4':'R4', '>f8':'R8', '>c8':'C8', '>c16':'C16'}
_invtypemap = dict((v,k) for k, v in _typemap.iteritems())

def str2type(type):
    try:
        type = _invtypemap[type]
    except KeyError:
        m = re.match(r'([A-Z])', type)
        if m:
            l = m.group(1)
            type = re.sub(r'^' + l, '>' + l.lower(), type)
    return type


def type2str(dt):
    dtypes = str(dt)
    if '>' in dtypes:
        try:
            dtypes = _typemap[dtypes]
        except KeyError:
            m = re.match(r'>([a-z])', dtypes)
            if m:
                l = m.group(1)
                dtypes = re.sub(r'>' + l, l.upper(), dtypes)

    return dtypes


def toraw(a,f,dtype=None):
    if dtype is not None:
        try:
            dtype = np.dtype(dtype)
        except TypeError:
            dtype = str2type(dtype)
        a = a.astype(dtype)

    dtypes = type2str(a.dtype)
#    dtypes = str(a.dtype)
#    if '>' in dtypes:
#        try:
#            dtypes = _typemap[dtypes]
#        except KeyError:
#            m = re.match(r'>([a-z])', dtypes)
#            if m:
#                l = m.group(1)
#                dtypes = re.sub(r'>' + l, l.upper(), dtypes)

    fname = f + '.' + 'x'.join([str(i) for i in a.shape]) + '_' + dtypes + '.raw'
    return a.tofile(fname)


def rawname(shape,f,dtype):
    if dtype is not None:
        if not isinstance(dtype, np.dtype):
            dtype = str2type(dtype)

    dtypes = type2str(dtype)

    fname = f + '.' + 'x'.join([str(i) for i in shape]) + '_' + dtypes + '.raw'
    return fname


def rawparams(f):
    # does f have grid info in it already?
    m = re.search(r'[\._]([0-9x]*)(_([<>a-zA-Z0-9]*))?\.raw$', f)
    if m:
        file = f
    else:
        if not '*' in f and not '?' in f:
            f = f + '.[0-9]*.raw'

        files = glob(f)
        for file in files:
            m = re.search(r'[\._]([0-9x]*)(_([<>a-zA-Z0-9]*))?\.raw$', file)
            if m: break
        else:
            sys.stderr.write('file not found: ' +  f + '\n')
            raise IOError

    if not m:
        raise exceptions.IOError

    dims = tuple( int(s) for s in m.group(1).split('x') )
    type = str2type(m.group(3))

    return dims, type


def fromraw(f, mask_val=None, astype=None, rec=None):
    # does f have grid info in it already?
    m = re.search(r'[\._]([-0-9x]*)(_([<>a-zA-Z0-9]*))?\.raw$', f)
    if m:
        file = f
    else:
        if not '*' in f and not '?' in f:
            f = f + '.[-0-9]*.raw'

        files = glob(f)
        for file in files:
            m = re.search(r'[\._]([-0-9x]*)(_([<>a-zA-Z0-9]*))?\.raw$', file)
            if m: break
        else:
            raise IOError('file not found: ' +  f)

    if not m:
        raise IOError('file not found: ' +  f)

    dims = [ int(s) for s in m.group(1).split('x') ]
    type = str2type(m.group(3))
#    try:
#        type = _invtypemap[type]
#    except KeyError:
#        m = re.match(r'([A-Z])', type)
#        if m:
#            l = m.group(1)
#            type = re.sub(r'^' + l, '>' + l.lower(), type)

    if rec is not None:
        a = myfromfile(file, dtype=np.dtype(type), shape=tuple(dims[1:]), skip=rec)
    else:
        a = np.fromfile(file, dtype=np.dtype(type)).reshape(dims)

    if mask_val is not None:
        a = np.ma.MaskedArray(a, a==mask_val)

    if astype is not None:
        a = a.astype(astype)

    return a


def globits(patt):
    files = glob(patt)
    pre,suf = patt.split('*')
    its = []
    for file in files:
        s = re.sub(r'^' + re.escape(pre), '', file)
        s = re.sub(re.escape(suf) + r'$', '', s)
        s = re.sub(r'^0*', '', s)
        its.append(int(s))

    its.sort()
    return its


def fromunformatted(file,dtype='float32', shape=None, skip=-1, count=-1):
    if skip >= 0:
        endcount = 1
    else:
        endcount = -1

    try:
        file.seek(0,1)
    except AttributeError:
        file = open(file)

    if skip > 0 or count >= 0:
        for i in range(skip):
            n1, = np.fromfile(file,'int32',count=1)
            file.seek(n1+4,1)

    if count > 0:
        res = np.empty((count,)+shape,dtype)
        for c in range(count):
            res[c,...] = fromunformatted(file,dtype,shape,skip=0)

        return res

    try:
        # skip header
        n1, = np.fromfile(file,'int32',count=1)
    except TypeError:
        raise
    else:
        n1 /= np.dtype(dtype).itemsize
        data = np.fromfile(file, dtype, count=n1)
        n2, = np.fromfile(file,'int32',count=endcount)

        if shape is not None:
            data = data.reshape(shape)

        return data


def grid2cell(x):
    return .5*(x[1:]+x[:-1])


def block1d(a,n=2,f=np.mean,axis=-1):
    dims = a.shape
    nx = dims[axis]
    dimsl = dims[:axis]
    if axis == -1:
        dimsr = ()
    else:
        dimsr = dims[axis+1:]

    if axis >= 0:
        axis += 1

    tmp = f(a.reshape(dimsl + (nx/n,n) + dimsr), axis=axis)
    return tmp


def block2d(a,n=2,f=np.mean):
    dims = a.shape
    ny,nx = dims[-2:]
    tmp = f(f(a.reshape(dims[:-2] + (ny/n,n,nx/n,n)), axis=-1), axis=-2)
    return tmp


def unblock2d(a,n=2):
    dims = a.shape
    ny,nx = dims[-2:]
    tmp = a.reshape(dims[:-2] + (ny,1,nx,1)
                   ) * np.ones(len(dims[:-2])*(1,) + (1,n,1,n))
    return tmp.reshape(dims[:-2] + (ny*n,nx*n))


def unblock1d(a,n=2,axis=-1):
    dims = a.shape
    nx = dims[axis]
    dimsl = dims[:axis]
    if axis == -1:
        dimsr = ()
    else:
        dimsr = dims[axis+1:]

    tmp = a.reshape(dimsl + (nx,1) + dimsr) * \
          np.ones(len(dimsl)*(1,) + (1,n) + len(dimsr)*(1,)) 
    return tmp.reshape(dimsl + (nx*n,) + dimsr)


def it2ymdhms(it, dt=1200, start=694224000):
    """
        step and start are in seconds since the epoch
        default for start is the equivalent of 19920101 UTC
    """
    date = time.gmtime(start+dt*it)
    return date[0:6]


def it2date(it, dt=1200, start=694224000):
    """
        step and start are in seconds since the epoch
        default for start is the equivalent of 19920101 UTC
    """
    return time.strftime('%Y%m%d %H%M%S', time.gmtime(start+dt*it))


def it2day(it, dt=1200, start=694224000, sep=''):
    """
        dt and start are in seconds since the epoch
        default for start is the equivalent of 19920101 UTC
    """
    return time.strftime('%Y' + sep + '%m' + sep + '%d', time.gmtime(start+dt*it-86400))


def it2dayl(it, dt=1200, start=694224000):
    """
        dt and start are in seconds since the epoch
        default for start is the equivalent of 19920101 UTC
    """
    return time.strftime('%Y-%m-%d', time.gmtime(start+dt*it-86400))
    

def it2mon(it, dt=1200, start=694224000):
    """
        dt and start are in seconds since the epoch
        default for start is the equivalent of 19920101 UTC
    """
    (y,m,d,H,M,S,xx,yy,zz) = time.gmtime(start+dt*it)
    m = m - 1
    if m == 0:
        y = y - 1
        m = 12
    return time.strftime('%Y-%m', (y,m,d,H,M,S,xx,yy,zz))
    

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


def myfig(num=None, size=(1920,1150), dpi=80, facecolor=None, edgecolor=None, frameon=True, **kwargs):
    figsize = [ float(pix)/dpi for pix in size ]
    fig = plt.figure(num, figsize, dpi, facecolor, edgecolor, frameon, **kwargs)
    fig.set_dpi(dpi)
    fig.set_size_inches(figsize)
    try:
        canvas = fig.canvas.get_tk_widget()
    except:
        pass
    else:
        canvas.config(bd=0)
    return fig


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
            print 'mysavefig: cannot adjust width and height at the same time.'
            raise exceptions.ValueError
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


def myimshow(X, cmap=None, norm=None, aspect=None, interpolation=None,
             alpha=1.0, vmin=None, vmax=None, origin=None, extent=None,
             symmetric=False, fixaxis=True, mask=None, maskval=None,
             cb=False, ij=False,
             **kwargs):
    if ij and origin is None:
        origin = 'upper'
    if origin is None:
        origin = 'bottom'
    if interpolation is None:
        interpolation = 'nearest'
    if 'hold' not in kwargs:
        kwargs['hold'] = False
    if kwargs['hold'] is False:
        # delete colorbars of old images
        if 'axes' in kwargs:
            ax = kwargs['axes']
        else:
            ax = gca()
        for im in ax.images:
            if im.colorbar is not None and im.colorbar[1] in im.figure.axes:
                im.figure.delaxes(im.colorbar[1])
                im.colorbar = None
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
    im = imshow(X, cmap, norm, aspect, interpolation,
                alpha, vmin, vmax, origin, extent, **kwargs)
    if fixaxis:
        ex = im.get_extent()
        if axis() == ex:
            x0,x1,y0,y1 = ex
            #x1 += 1.*(x1-x0)/w
            #y1 += 1.5*(y1-y0)/w
            ax = im.get_axes()
            x1f,y1f = ax.transData.transform((x1,y1))
            x1p,y1p = ax.transData.inverted().transform((x1f+1.,y1f+1.99))
            axis((x0,x1p,y0,y1p))

    if cb:
        cb = colorbar(im,ax=gca())
        return im,cb
    else:
        return im


def mercatory(lat):
    """ y coordinate of Mercator projection (in degrees) """
    #return 180./pi*log(tan(lat*pi/180.) + 1./cos(lat*pi/180))
    return 180./np.pi*np.log(np.tan(np.pi/4.+lat*np.pi/360.))


def axes_size_inches(ax=None):
    if ax is None: ax = gca()
    pos = ax.get_position()
    fig = ax.get_figure()
    w,h = fig.get_size_inches()
    return np.array([w*pos.width, h*pos.height])


def im_dpi(im=None):
    if im is None: im = gci()
    h,w = im.get_size()
    win,hin = axes_size_inches(im.get_axes())
    return w/win, h/hin
#    if im is None: im = gci()
#    ax = im.get_axes()
#    sz = im.get_size()
#    w,h = ax.transData.transform(sz)-ax.transData.transform((0,0))


def im_size_pixels(im=None):
    if im is None: im = gci()
    ax = im.get_axes()
#    fig = ax.get_figure()
#    sz = im.get_size()
    ext = np.array( im.get_extent()).reshape(2,2).T
    extpix = ax.transData.transform(ext)
    return extpix[1,:] - extpix[0,:]


def im_size_axes(im=None):
    if im is None: im = gci()
    ax = im.get_axes()
#    fig = ax.get_figure()
#    sz = im.get_size()
    ext = np.array( im.get_extent()).reshape(2,2).T
    extpix = ax.transAxes.inverted().transform(ax.transData.transform(ext))
    return extpix[1,:] - extpix[0,:]


def im_scale_factor(im=None):
    if im is None: im = gci()
    ax = im.get_axes()
    fig = ax.get_figure()
    h0,w0 = im.get_size()
    w,h = im_size_pixels(im)
    return np.sqrt((w0/w)*(h0/h))
    

def im_fig_size(im=None):
    if im is None: im = plt.gci()
    ax = im.get_axes()
    fig = ax.get_figure()
    sz = fig.get_size_inches()
    h0,w0 = im.get_size()
    w,h = im_size_pixels(im)
    return np.array([sz[0]*(w0/w), sz[1]*(h0/h)])
    

def im_axes_size(im=None):
    if im is None: im = plt.gci()
    ax = im.get_axes()
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
    print [pos[:2],[pos[0]+pos[2],pos[1]+pos[3]]]
    figpoints = fig.transFigure.inverted().transform(trans.transform([pos[:2],[pos[0]+pos[2],pos[1]+pos[3]]]))
    print figpoints
    # turn into pos
    figpos = np.r_[figpoints[0,:], figpoints[1,:]-figpoints[0,:]]
    print figpos
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


def smax(x,axis=None):
    " signed maximum: max if max>|min|, min else "
    mx=np.max(x,axis)
    mn=np.min(x,axis)
    neg=abs(mn)>abs(mx)
    return (1-neg)*mx+neg*mn


def maxabs(x,axis=None):
    " maximum modulus "
    return np.max(abs(x),axis)


def indmin(a,axis=None):
    flatindex = np.argmin(a,axis)
    return np.unravel_index(flatindex, a.shape)


def indmax(a,axis=None):
    flatindex = np.argmax(a,axis)
    return np.unravel_index(flatindex, a.shape)


def nanindmin(a,axis=None):
    flatindex = np.nanargmin(a,axis)
    return np.unravel_index(flatindex, a.shape)


def nanindmax(a,axis=None):
    flatindex = np.nanargmax(a,axis)
    return np.unravel_index(flatindex, a.shape)


def indmaxabs(a,axis=None):
    flatindex = np.argmax(np.abs(a),axis)
    return np.unravel_index(flatindex, a.shape)


def max2(a):
    return max(max(a,axis=-1),axis=-1)


def maxabs2(a):
    return np.max(np.max(abs(a),axis=-1),axis=-1)


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

def LatitudeFormatter(sep='$^o$'):
    return mpl.ticker.FuncFormatter(lambda val,pos=None:_latformatfunc(val,pos,sep))

def LongitudeFormatter(sep='$^o$'):
    return mpl.ticker.FuncFormatter(lambda val,pos=None:_lonformatfunc(val,pos,sep))

# for netCDF3
def untile(a,nty,ntx):
    sh = a.shape[1:-2]
    ny,nx = a.shape[-2:]
    n = len(sh)
    if not hasattr(a,'reshape'):
        a = a[:]

    return a.reshape((nty,ntx)+sh+(ny,nx)).transpose(range(2,2+n)+[0,2+n,1,3+n]).reshape(sh+(nty*ny,ntx*nx))

