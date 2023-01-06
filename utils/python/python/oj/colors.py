#!/usr/bin/env python
import sys
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.widgets import  RectangleSelector
from matplotlib import cbook
from numpy import ma

def scale_segmentdata(sd,x0,x1):
    sc = x1 - x0
    res = {}
    for c,lst in sd.items():
        res[c] = tuple((x0+sc*t[0],t[1],t[2]) for t in lst)
    return res


def cat_cmaps(cmap1, cmap2, x, name=None):
    if type(cmap1) == type(''):
        cmap1 = mpl.cm.get_cmap(cmap1)
    if type(cmap2) == type(''):
        cmap2 = mpl.cm.get_cmap(cmap2)
    try:
        sd1 = cmap1._segmentdata
    except:
        sd1 = cmap1
    try:
        sd2 = cmap2._segmentdata
    except:
        sd2 = cmap2
    if name is None:
        name = cmap1.name + '+' + cmap2.name
    sd1 = scale_segmentdata(sd1, 0, x)
    sd2 = scale_segmentdata(sd2, x, 1)

    for c in sd1.keys():
        c2 = sd2.get(c, ((0., 1., 1.), (1., 1., 1.)))
        sd1[c] = sd1[c][:-1] + ((x,sd1[c][-1][1],c2[0][2]),) + c2[1:]

    return LinearSegmentedColormap(name, sd1)

def hsv2rgb(hsv, axis=0):
    h,s,v = np.rollaxis(np.asanyarray(hsv), axis)
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

def hsl2rgb(hsl, axis=0):
    h,s,l = np.rollaxis(np.asanyarray(hsl), axis)
    c = (1. - abs(2*l-1.))*s
    k = np.mod((6*h)//1., 6)
    k[c==0] = 0
    x = c*(1. - abs(np.mod(6*h, 2.) - 1.))
    x[c==0] = 0
    r = (k==0)*c + (k==1)*x + (k==2)*0 + (k==3)*0 + (k==4)*x + (k==5)*c
    g = (k==0)*x + (k==1)*c + (k==2)*c + (k==3)*x + (k==4)*0 + (k==5)*0
    b = (k==0)*0 + (k==1)*0 + (k==2)*x + (k==3)*c + (k==4)*c + (k==5)*x
    m = l - .5*c
    return np.dstack((r+m, g+m, b+m))

def rgb2hsl(rgb, axis=0):
    M = rgb.max(axis=axis)
    m = rgb.min(axis=axis)
    C = M - m
    i = rgb.argmax(axis=axis)
    r,g,b = np.rollaxis(np.asanyarray(rgb), axis)
    H = np.choose(i, [np.mod((g-b)/C, 6.), (b-r)/C + 2., (r-g)/C + 4.])/6.
    L = .5*(M + m)
    S = C/(1 - abs(2*L-1))
    return np.dstack((H, S, L))

def make_cmap_seawifs():
    mydir,_ = os.path.split(os.path.realpath(__file__))
    data = np.fromfile(os.path.join(mydir,'cmseawifs_512x3.raw'))
    return ListedColormap(data.reshape(512,3))


def make_seawifs_blue(red0=0., **kwargs):
    cmap = mpl.colors.LinearSegmentedColormap('SeaWiFSblue',{
    #  'red':  (( 0./16.,0.6,0.6),
      'red':  (( 0./16.,0.0,red0),
               ( 3./16.,0.0,0.0),
               ( 8./16.,0.0,0.0),
               (10./16.,1.0,1.0),
               (14./16.,1.0,1.0),
               ( 1.0   ,0.4,0.4)),
      'green':(( 0./16.,0.0,0.0),
               ( 3./16.,0.0,0.0),
               ( 6./16.,1.0,1.0),
               (10./16.,1.0,1.0),
               (14./16.,0.0,0.0),
               ( 1.0   ,0.0,0.0)),
      'blue': (( 0./16.,0.4,0.4),
               ( 3./16.,1.0,1.0),
               ( 6./16.,1.0,1.0),
               ( 8./16.,0.0,0.0),
               ( 1.0   ,0.0,0.0)),
    }, **kwargs)
    return cmap


def colorbar_position(ax,pad,size,**kw):
    pos = list(ax.get_position().bounds)
    pos[0] = pos[0]+pos[2]+pad
    pos[2] = size
    return pos


def eval_linear_segmented(x, data):
    xa,y0,y1 = np.array(data).T
    ind = np.searchsorted(xa, x)
    v = ((x-xa[ind-1])/(xa[ind]-xa[ind-1]))*(y0[ind]-y1[ind-1]) + y1[ind-1]
    return np.clip(v, 0., 1.)

def sym_segment_data(cmap, vmin, vmax):
    sd = cmap._segmentdata
    m = max(vmax, -vmin)
    if m == 0:
        return sd
#    # color at lower end
#    rgba = cmap((vmin+m)/(2*m) + 1e-12)
#    d0 = dict(zip(['red', 'green', 'blue', 'alpha'], rgba))
#    # color at upper end
#    rgba = cmap((vmax+m)/(2*m) - 1e-12)
#    d1 = dict(zip(['red', 'green', 'blue', 'alpha'], rgba))

    d = {}
    for c,v in sd.items():
        _x = None
        _y0 = None
        l = []
        for i,(x,y0,y1) in enumerate(v):
            x = (x*2*m-vmin-m)/(vmax-vmin)
#            if x > 1:
#                if i > 0:
#                    X,Y0,Y1 = v[i-1]
#                    X = (X-vmin-m)*2*m/(vmax-vmin)
#                    w = (x-1.)/(x-X)
#                    y0 = Y1*w + y0*(1-w)
#                x = 1.
#            if x < 0:
#                if i+1 < len(v):
#                    X,Y0,Y1 = v[i+1]
#                    X = (X-vmin-m)*2*m/(vmax-vmin)
#                    w = (0-x)/(X-x)
#                    y1 = Y0*w + y1*(1-w)
#                x = 0.
#            if x == _x:
#                l[-1] = (x, _y0, y1)
#            else:
#                l.append((x, y0, y1))
#                _x = x
#                _y0 = y0
            if 0. <= x <= 1.:
                l.append((x, y0, y1))

        if l[0][0] != 0.0:
            y = eval_linear_segmented((vmin+m)/(2*m), v)
            l.insert(0, (0.0, y, y))

        if l[-1][0] != 1.0:
            y = eval_linear_segmented((vmax+m)/(2*m), v)
            l.append((1.0, y, y))

#        if l[0][0] != 0:
#            sys.stderr.write('sym_segment_data({}, {}, {}) does not start with 0:\n{}\n'.format(
#                vmin,vmax,m,l))
#        if l[-1][0] != 1:
#            if l[-1][0] > .999:
#                l[-1] = (1.,) + l[-1][1:]
#            else:
#                sys.stderr.write('sym_segment_data({}, {}, {}) does not end with 1:\n{}\n'.format(
#                    vmin,vmax,m,l))
##            l.append((1., l[-1][2], l[-1][2]))

        d[c] = l

    return d


def colorbar(mappable=None, cax=None, ax=None, pad=12/72., size=12/72.,
             center=None, centerextend=None, **kw):
#    oldax = plt.gca()
    if mappable is None:
        try:
            norm = kw.pop('norm')
            cmap = kw.pop('cmap')
        except KeyError:
            mappable = plt.gci()
        else:
            mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    if ax is None:
        try:
            ax = mappable.axes
        except AttributeError:
            # for ContourSet
            ax = mappable.ax

    fig = ax.figure

    orientation = kw.get('orientation','vertical')

    if np.iterable(cax):
        cax = plt.axes(cax)
    elif cax is None:
        try:
            cb,cax = mappable.colorbar
        except TypeError:
            try:
                cax = ax.cax
            except AttributeError:
                divider = make_axes_locatable(ax)
                if orientation == 'horizontal':
                    cax = divider.new_vertical(size,pad)
                else:
                    if kw.get('extend', 'neither') != 'neither':
                        divider.get_vsize_hsize()[0]._list[0]._aspect = 1.1
                    cax = divider.new_horizontal(size,pad)

                fig.add_axes(cax)
        else:
            pass

    cb = fig.colorbar(mappable,cax,ax,**kw)

    cb.zoom = zoomify(cb)

    if center is not None:
        extend = centerextend or kw.get('extend', 'neither')
        def tmp(im=mappable, cmap0=mappable.cmap, callback=cb.on_mappable_changed,
                extend=extend):
            sd = sym_segment_data(cmap0, *im.get_clim())
            cmap = LinearSegmentedColormap('_sym_cb_map', sd)
            if cmap0._rgba_under is not None:
                cmap.set_under(cmap0._rgba_under)
            elif extend in ['both', 'min']:
                cmap.set_under([sd[c][0][1] for c in ['red', 'green', 'blue']])
            if cmap0._rgba_over is not None:
                cmap.set_over(cmap0._rgba_over)
            elif extend in ['both', 'max']:
                cmap.set_over([sd[c][-1][2] for c in ['red', 'green', 'blue']])
            im.cmap = cmap
            callback(im)
            im.figure.canvas.draw()
        cb.center_callback = mappable.callbacksSM.connect('changed', tmp)

    fig.sca(ax)

    fig.canvas.draw_idle()

    if center:
        tmp()

    return cb


def log_ticks(cb, subs=[1,2,3,4,5,6,7,8,9]):
    locator = mpl.ticker.LogLocator(10,subs=subs)
    locator.create_dummy_axis()
    locator.set_view_interval(*cb.get_clim())
    locator.set_data_interval(*cb.get_clim())
    locator.numticks = 1000
    b = np.array(locator())
    ticks = cb._locate(b)
    return b,ticks


def minor_log_ticks(cb, subs=[1,2,3,4,5,6,7,8,9]):
    b,ticks = log_ticks(cb,subs)
    if cb.orientation == 'vertical':
        ax = cb.ax.yaxis
    else:
        ax = cb.ax.xaxis

    ax.set_minor_locator(mpl.ticker.FixedLocator(ticks))

    if plt.isinteractive():
        plt.draw()

    return b,ticks


def major_log_ticks(cb, subs=[1,2,3,4,5,6,7,8,9]):
    b,ticks = log_ticks(cb,subs)
    if cb.orientation == 'vertical':
        ax = cb.ax.yaxis
    else:
        ax = cb.ax.xaxis

    ax.set_major_locator(mpl.ticker.FixedLocator(ticks))
    formatter = cb.formatter
    formatter.create_dummy_axis()
    formatter.set_view_interval(*cb.get_clim())
    formatter.set_data_interval(*cb.get_clim())
    formatter.set_locs(b)
    formatter.labelOnlyBase = False
    ticklabels = [formatter(t, i) for i, t in enumerate(b)]
    offset_string = formatter.get_offset()
    ax.set_major_formatter(mpl.ticker.FixedFormatter(ticklabels))
    if plt.isinteractive():
        plt.draw()

    return ticks, ticklabels, offset_string


def oldcolorbar(mappable=None, cax=None, ax=None, pad=0.01, size=0.01, **kw):
#    oldax = plt.gca()
    if mappable is None:
        mappable = plt.gci()

    if ax is None:
        try:
            ax = mappable.axes
        except AttributeError:
            # for ContourSet
            ax = mappable.ax

    fig = ax.figure

    if np.iterable(cax):
        cax = plt.axes(cax)

    if cax is None:
        pos = colorbar_position(ax,pad,size,**kw)
        cax = plt.axes(pos)
        kw['orientation'] = kw.get('orientation', 'vertical')

    cax.hold(True)
    cb = mpl.colorbar.Colorbar(cax, mappable, **kw)
    cb.pad = pad
    cb.size = size
    cb.kw   = kw

    def on_changed(m):
#        print 'calling on changed', m.get_cmap().name
        cb.set_cmap(m.get_cmap())
        cb.set_clim(m.get_clim())
        cb.update_bruteforce(m)

    def on_resize(event):
#        print 'calling on draw',event,cb.mappable.get_cmap().name
        m = cb.mappable
        try:
            ax = mappable.axes
        except AttributeError:
            # for ContourSet
            ax = mappable.ax

        pos = colorbar_position(ax,cb.pad,cb.size,**cb.kw)
        cax.set_position(pos)

    def on_draw(event):
        m = cb.mappable
        cb.update_bruteforce(m)

    fig.cbid = mappable.callbacksSM.connect('changed', on_changed)
#    fig.canvas.mpl_connect('resize_event', on_resize)
    fig.canvas.mpl_connect('draw_event', on_resize)
    mappable.set_colorbar(cb, cax)
    fig.sca(ax)

    return cb


class SymNorm(mpl.colors.Normalize):
    def __init__(self, vmax=None, clip=False):
        self._vmin = None
        self.vmax = vmax
        self.clip = clip

    def absmax(self):
        if self._vmax is None and self._vmin is None: return None
        return max(self._vmax or 0, -(self._vmin or 0))

    @property
    def vmin(self):
        mx = self.absmax()
        # keep None
        return mx and -mx
    @vmin.setter
    def vmin(self, v):
        # keep None
        self._vmin = v and min(0, v)

    @property
    def vmax(self):
        mx = self.absmax()
        return mx
    @vmax.setter
    def vmax(self, v):
        # keep None
        self._vmax = v and max(0, v)

    def autoscale_None(self, A):
        if self._vmin is None and self._vmax is None and np.size(A) > 0:
            mx = np.ma.max(A)
            mn = np.ma.min(A)
            mx = max(mx, -mn)
            self.vmax = mx
            self.vmin = -mx


def symLogNorm(mn,mx):
    def tmp(x,mn=mn,mx=mx):
        absx = np.minimum(mx,np.maximum(mn,np.abs(x)))
        return np.sign(x)*(np.log(absx)-np.log(mn))*.5/np.log(mx/mn) + .5

    return tmp


class TwoSegmentNorm(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcut=None, ocut=None, clip=False):
        self.vmin = vmin
        self.vmax = vmax
        self.vcut = vcut
        self.ocut = ocut
        self.clip = clip

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

#        result = ma.masked_less_equal(result, 0, copy=False)

        self.autoscale_None(result)
        vmin, vmax = self.vmin, self.vmax
        vcut, ocut = self.vcut, self.ocut
        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin==vmax:
            result.fill(0)
        elif vcut < vmin or vcut > vmax:
            raise ValueError("cut value must fall within range")
        else:
            if clip:
                mask = ma.getmask(result)
                val = ma.array(np.clip(result.filled(vmax), -vmax, vmax),
                                mask=mask)
            #result = (ma.log(result)-np.log(vmin))/(np.log(vmax)-np.log(vmin))
            # in-place equivalent of above can be much faster
            resdat = np.asarray(result.data)
            res1 = resdat - vmin
            res1 /= (vcut - vmin)
            res1 *= ocut
            res2 = resdat - vcut
            res2 /= (vmax - vcut)
            res2 *= 1. - ocut
            res2 += ocut
            resdat = np.where(resdat <= vcut, res1, res2)
            result = np.ma.array(resdat, mask=result.mask, copy=False)
        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax = self.vmin, self.vmax
        vcut, ocut = self.vcut, self.ocut

        if cbook.iterable(value):
            value = ma.asarray(value)

        res1 = value/ocut
        res1 *= vcut - vmin
        res1 += vmin
        res2 = value - ocut
        res2 /= 1. - ocut
        res2 *= vmax - vcut
        res2 += vcut
        res = np.where(value <= ocut, res1, res2)
        return res

    def autoscale(self, A):
        '''
        Set *vmin*, *vmax* to min, max of *A*.
        '''
#        A = ma.masked_less_equal(A, 0, copy=False)
        self.vmin = ma.min(A)
        self.vmax = ma.max(A)

    def autoscale_None(self, A):
        ' autoscale only None-valued vmin or vmax'
        if self.vmin is not None and self.vmax is not None:
            return
#        A = ma.masked_less_equal(A, 0, copy=False)
        if self.vmin is None:
            self.vmin = ma.min(A)
        if self.vmax is None:
            self.vmax = ma.max(A)


#class SymLogNorm(mpl.colors.Normalize):
#    def __call__(self, value, clip=None):
#        if clip is None:
#            clip = self.clip
#
#        result, is_scalar = self.process_value(value)
#
##        result = ma.masked_less_equal(result, 0, copy=False)
#
#        self.autoscale_None(result)
#        vmin, vmax = self.vmin, self.vmax
#        if vmin > vmax:
#            raise ValueError("minvalue must be less than or equal to maxvalue")
#        elif vmin<=0:
#            raise ValueError("values must all be positive")
#        elif vmin==vmax:
#            result.fill(0)
#        else:
#            if clip:
#                mask = ma.getmask(result)
#                val = ma.array(np.clip(result.filled(vmax), -vmax, vmax),
#                                mask=mask)
#            #result = (ma.log(result)-np.log(vmin))/(np.log(vmax)-np.log(vmin))
#            # in-place equivalent of above can be much faster
#            resdat = result.data
#            mask = result.mask
##            if mask is np.ma.nomask:
##                mask = (resdat <= 0)
##            else:
##                mask |= resdat <= 0
##            np.putmask(resdat, mask, 1)
##            np.log(resdat, resdat)
##            resdat -= np.log(vmin)
##            resdat /= (np.log(vmax) - np.log(vmin))
#            signx = np.sign(resdat)
#            np.abs(resdat, resdat)
#            np.maximum(vmin,resdat,resdat)
##            np.minimum(vmax,resdat,resdat)
#            np.log(resdat, resdat)
#            resdat -= np.log(vmin)
#            resdat *= .5*signx
#            resdat /= (np.log(vmax) - np.log(vmin))
#            resdat += .5
#            result = np.ma.array(resdat, mask=mask, copy=False)
#        if is_scalar:
#            result = result[0]
#        return result
#
#    def inverse(self, value):
#        if not self.scaled():
#            raise ValueError("Not invertible until scaled")
#        vmin, vmax = self.vmin, self.vmax
#
#        if cbook.iterable(value):
#            val = ma.asarray(value)
#            val -= .5
#            signx = np.sign(val)
#            val /= .5
#            np.abs(val, val)
##            val *= (np.log(vmax) - np.log(vmin))
#            return signx * vmin * ma.power((vmax/vmin), val)
#        else:
#            value -= .5
#            signx = np.sign(value)
#            value /= .5
##            value *= (np.log(vmax) - np.log(vmin))
#            return signx * vmin * pow((vmax/vmin), abs(value))
#
#    def autoscale(self, A):
#        '''
#        Set *vmin*, *vmax* to min, max of *A*.
#        '''
##        A = ma.masked_less_equal(A, 0, copy=False)
#        self.vmin = ma.min(A)
#        self.vmax = ma.max(A)
#
#    def autoscale_None(self, A):
#        ' autoscale only None-valued vmin or vmax'
#        if self.vmin is not None and self.vmax is not None:
#            return
##        A = ma.masked_less_equal(A, 0, copy=False)
#        if self.vmin is None:
#            self.vmin = ma.min(A)
#        if self.vmax is None:
#            self.vmax = ma.max(A)


from matplotlib.widgets import SpanSelector

class ButtonSpanSelector(SpanSelector):
    '''SpanSelector for any mouse button
    '''
    def __init__(self, ax, onselect, direction, minspan=None, useblit=False,
                 rectprops=None, onmove_callback=None, button=[1]):
        SpanSelector.__init__(self, ax, onselect, direction, minspan, useblit,
                 rectprops, onmove_callback)
        self.button = button
        self.haveeventpress = hasattr(self, 'eventpress')

    @property
    def myeventpress(self):
        if self.haveeventpress:
            return self.eventpress
        else:
            return self.buttonDown

    def releaseeventpress(self):
        if self.haveeventpress:
            self.eventpress = None
        else:
            self.buttonDown = False

    def ignore(self, event):
        'return *True* if *event* should be ignored'
        widget_off = not self.visible # or not self.active
        non_event = event.inaxes!=self.ax or event.button not in self.button

        # If a button was pressed, check if the release-button is the
        # same. If event is out of axis, limit the data coordinates to axes
        # boundaries.
        if self.myeventpress and event.inaxes != self.ax:
            (xdata, ydata) = self.ax.transData.inverted().transform_point((event.x, event.y))
            x0, x1 = self.ax.get_xbound()
            y0, y1 = self.ax.get_ybound()
            xdata = max(x0, xdata)
            xdata = min(x1, xdata)
            ydata = max(y0, ydata)
            ydata = min(y1, ydata)
            event.xdata = xdata
            event.ydata = ydata
            return False

        return widget_off or non_event

    def release(self, event):
        'on button release event'
        if self.ignore(event) and not self.myeventpress:
            return
        if self.pressv is None:
            return
        self.releaseeventpress()

        self.rect.set_visible(False)
        self.canvas.draw()
        vmin = self.pressv
        if self.direction == 'horizontal':
            vmax = event.xdata or self.prev[0]
        else:
            vmax = event.ydata or self.prev[1]

        if vmin>vmax: vmin, vmax = vmax, vmin
        span = vmax - vmin
        if self.minspan is not None and span<self.minspan: return
        self.onselect(vmin, vmax, event.button)
        self.pressv = None
        return False


def zoomify(cb, minspan=.05, useblit=True, rectprops={}):
    def line_select_callback(y1, y2, button):
        'eclick and erelease are the press and release events'
        im = line_select_callback.mappables[0]
        mn,mx = im.get_clim()
        if abs((y2-y1)) < .05: return
        if abs((1.-y2)) < .02: y2 = 1.
        if abs((y2-0.)) < .02: y2 = 0.
        y1,y2 = min(y1,y2), max(y1,y2)
        if button == 1:
            for im in line_select_callback.mappables:
                im.set_clim(mn+y1*(mx-mn),mn+y2*(mx-mn))
        else:
            f = (mx-mn)/(y2-y1)
            for im in line_select_callback.mappables:
                im.set_clim(mn-f*y1, mn-f*y1+f)
        im.figure.canvas.draw_idle()

    try:
        ims = cb.mappables
    except AttributeError:
        ims = [cb.mappable]
    line_select_callback.mappables = ims

    rp = dict(facecolor='0.75', alpha=0.5)
    rp.update(rectprops)

    # drawtype is 'box' or 'line' or 'none'
    rs = ButtonSpanSelector(cb.ax, line_select_callback, cb.orientation,
                            button=[1,3], minspan=minspan, useblit=useblit,
                            rectprops=rp)
    cb.zoom_selector = rs
    return rs

class ColorZoom:
    def __init__(self, cb, mappables=None, minspan=.05, useblit=True, rectprops={}):
        self.colorbar = cb
        self.mappables = mappables
        if self.mappables is None:
            try:
                self.mappables = cb.mappables
            except AttributeError:
                self.mappables = [cb.mappable]

        rp = dict(facecolor='0.75', alpha=0.5)
        rp.update(rectprops)

        # drawtype is 'box' or 'line' or 'none'
        self.selector = ButtonSpanSelector(cb.ax, self.callback, cb.orientation,
                                button=[1,3], minspan=minspan, useblit=useblit,
                                rectprops=rp)

    def callback(self, y1, y2, button):
        'eclick and erelease are the press and release events'
        im = self.colorbar.mappable
        mn,mx = im.get_clim()
#        if abs((y2-y1)) < .05: return
        if abs((1.-y2)) < .02: y2 = 1.
        if abs((y2-0.)) < .02: y2 = 0.
        y1,y2 = min(y1,y2), max(y1,y2)
        if button == 1:
            for im in self.mappables:
                im.set_clim(mn+y1*(mx-mn),mn+y2*(mx-mn))
        else:
            f = (mx-mn)/(y2-y1)
            for im in self.mappables:
                im.set_clim(mn-f*y1, mn-f*y1+f)

        if plt.isinteractive():
            for fig in set(im.figure for im in self.mappables):
                fig.canvas.draw_idle()

    def add(self, *mappables):
        self.mappables.extend(mappables)


def cmcycle(cmap, N, ax=None):
    from cycler import cycler
    from matplotlib import cm
    cmap = cm.get_cmap(cmap)
    cmap._init()
    colors = cmap._lut[:-3,:3]
    n = len(colors)
    colors = colors[::n//N]
    cyc = cycler(color=colors)
    if ax is not None:
        ax.set_prop_cycle(cyc)
    else:
        mpl.rc('axes', prop_cycle=cyc)
    return cyc

