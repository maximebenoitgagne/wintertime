#!/usr/bin/env python
import os
import re
import pylab
import matplotlib as mpl
from pylab import *
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

def scale_segmentdata(sd,x0,x1):
    sc = x1 - x0
    res = {}
    for c,lst in sd.items():
        res[c] = tuple((x0+sc*t[0],t[1],t[2]) for t in lst)
    return res

        
def cat_cmaps(cmap1, cmap2, x):
    sd1 = scale_segmentdata(cmap1._segmentdata, 0, x)
    sd2 = scale_segmentdata(cmap2._segmentdata, x, 1)

    for c in sd1.keys():
        sd1[c] = sd1[c][:-1] + ((x,sd1[c][-1][1],sd2[c][0][2]),) + sd2[c][1:]

    return LinearSegmentedColormap(cmap1.name + '+' + cmap2.name, sd1)


def purplejet(x):
    return cat_cmaps(LinearSegmentedColormap.from_list('purplejet', [(1,0,1),(0,0,.5)]), mpl.cm.jet, x)

        
def make_cmap_seawifs():
    _mydir,_ = os.path.split(os.path.realpath(__file__))
    data = np.fromfile(os.path.join(_mydir,'cmseawifs_512x3.raw'))
    return ListedColormap(data.reshape(512,3))


cmap_ldeo = LinearSegmentedColormap.from_list('iri/ldeo',
    [(0,0,1),(0,1,1),(0,1,0),(1,1,0),(1,0,0),(1,0,1)])

_cdom_index_colors = np.array([
       [ 76,   0, 255],
       [ 32,   1, 255],
       [  0,  11, 250],
       [  1,   9, 255],
       [  0,  55, 254],
       [  0,  99, 255],
       [  0, 142, 254],
       [  0, 187, 255],
       [  0, 231, 255],
       [  2, 230, 255],
       [ 36, 179,   0],
       [ 52, 185,   0],
       [ 68, 191,   0],
       [ 83, 195,   0],
       [101, 198,   3],
       [ 98, 199,   1],
       [115, 202,   1],
       [132, 206,   0],
       [152, 212,   2],
       [174, 215,   0],
       [182, 219,   0],
       [186, 221,   3],
       [255, 255,   0],
       [255, 248,   0],
       [255, 238,   0],
       [255, 229,   0],
       [254, 222,   0],
       [255, 221,   0],
       [255, 211,   0],
       [255, 204,   0],
       [253, 193,   0],
       [254, 185,   2],
       [253, 187,   5],
       [254, 177,   1],
       [255, 166,   0],
       [255, 158,   1],
       [255, 149,   1],
       [252, 141,   0],
       [255, 139,   4],
       [255, 133,   0],
       [255, 123,   0],
       [255, 113,   1],
       [255, 105,   0],
       [255,  95,   1],
       [255,  96,   0],
       [255,  87,   0],
       [255,  80,   1],
       [255,  69,   0],
       [255,  62,   0],
       [254,  53,   1],
       [255,  51,   0],
       [255,  44,   0],
       [255,  35,   0],
       [255,  26,   0],
       [255,  15,   0],
       [253,   9,   0],
       [254,  10,   2],
       [254,   1,   0],
       [  0,   0,   0]])

cmap_cdom_index = ListedColormap(_cdom_index_colors[:-1], 'cdom_index')
cmap_cdom_index.set_over(_cdom_index_colors[-1])

def calc_linear_segmented(x,data,right=False):
    adata = np.array(data)
    xs = adata[:,0]
    y0 = adata[:,1]
    y1 = adata[:,2]
    n = len(xs)
    if right:
        ind = n - np.searchsorted(-xs[::-1], -x)
    else:
        ind = np.searchsorted(xs, x)
    if ind < 1:
        return y0[0]
    elif ind >= n:
        return y1[-1]
    else:
        return ( ((x - xs[ind-1]) / (xs[ind] - xs[ind-1]))
                 * (y0[ind] - y1[ind-1]) + y1[ind-1]
               )


def cmap2xrgb(cmap):
    xs = set()
    for comp in cmap._segmentdata.values():
        for x,c0,c1 in comp:
            xs.add(x)
    xs = list(xs)
    xs.sort()
    rgb = [ tuple( calc_linear_segmented(x,cmap._segmentdata[k]) for k in ['red','green','blue'] ) for x in xs ]
    rgb_r = [ tuple( calc_linear_segmented(x,cmap._segmentdata[k],True) for k in ['red','green','blue'] ) for x in xs ]
    return xs,rgb,rgb_r


def makecmap(name,xs,rgbs,rgbs_r=None,N=256):
    if rgbs_r is None: rgbs_r = rgbs
    rgbl = [[],[],[]]
    for x,rgb,rgb_r in zip(xs,rgbs,rgbs_r):
        for c in range(3):
            rgbl[c].append((x,rgb[c],rgb_r[c]))

    return LinearSegmentedColormap(name,{'red':rgbl[0], 'green':rgbl[1], 'blue':rgbl[2]},N)


class MyLogFormatter(LogFormatter):
    def __init__(self, mn, mx):
        self.mn = mn
        self.mx = mx
    def __call__(self, val, pos=None):
        return '%G' % (10.0**(self.mn+(self.mx-self.mn)*val))

        
class MyMathLogFormatter(LogFormatter):
    def __init__(self, mn, mx, fmt=r'$10^{%g}$'):
        self.mn = mn
        self.mx = mx
        self.fmt = fmt
    def __call__(self, val, pos=None):
        expstr = '%f' % (self.mn+(self.mx-self.mn)*val)
        expstr = re.sub(r'\.?0*$', '', expstr)
        return r'$10^{' + expstr + r'}$'

        
def mycb(cax=None, im=None,
         log=False, ticks=None, nticks=None, ticklabels=None, color=None, ticklines=True,
         title=None, titlepos=None, titlesize=None, label=None, endticks=None,
         width=.01,
         **kwargs):
    oldax = gca()
    if im is None:
        im = gci()
    try:
        ax = im.axes
    except AttributeError:
        # for ContourSet
        ax = im.ax
    if np.iterable(cax):
        cax = plt.axes(cax)
    if cax is None:
        pos = list(ax.get_position().bounds)
        pad = .02
        if 'pad' in kwargs:
            pad = kwargs['pad']
            del kwargs['pad']
        pos[0] = pos[0]+pos[2]+pad
        pos[2] = width
        cax = axes(pos)
        # make vertical unless given
        kwargs['orientation'] = kwargs.get('orientation', 'vertical')
    axes(cax)
    cla()
    if im.colorbar is not None and im.colorbar[1] in ax.figure.axes:
        ax.figure.delaxes(im.colorbar[1])
        im.colorbar = None
    mn, mx = get(im, 'clim')
    if ticks is None and nticks is not None:
        if log:
            dtick = float(mx-mn)/nticks
            print 'dtick:', dtick
            if dtick > log10(2.):
                dtick = ceil(dtick)
                print 'every', dtick, 'power of 10'
                ticks = arange(ceil(mn), mx+dtick, dtick);
            elif dtick > log10(10./9.):
                print '1,2,5'
                ticks = concatenate((arange(ceil(mn), mx+1., 1.),
                                     arange(floor(mn)+log10(2.), mx+1., 1.),
                                     arange(floor(mn)+log10(5.), mx+1., 1.)))
                ticks.sort()
            elif dtick > log10(10./9.)/2.:
                print 'all digits'
                print ceil(mn), mx+1
                ticks = concatenate((arange(ceil(mn), mx+1., 1.),
                                     arange(floor(mn)+log10(2.), mx+1., 1.),
                                     arange(floor(mn)+log10(3.), mx+1., 1.),
                                     arange(floor(mn)+log10(4.), mx+1., 1.),
                                     arange(floor(mn)+log10(5.), mx+1., 1.),
                                     arange(floor(mn)+log10(6.), mx+1., 1.),
                                     arange(floor(mn)+log10(7.), mx+1., 1.),
                                     arange(floor(mn)+log10(8.), mx+1., 1.),
                                     arange(floor(mn)+log10(9.), mx+1., 1.)))
                ticks.sort()
            else:
                print 'log10(linearly spaced ticks)'
                print 10.**float(mx), 10.**float(mn), nticks
                print (10.**float(mx)-10.**float(mn))/nticks
                decade, frac = divmod(log10((10.**mx-10.**mn)/nticks), 1.)
                dig = 10.0**frac
                if dig > 5.000001:
                    dig = 10.
                elif dig > 2.000001:
                    dig = 5.0
                elif dig > 1.000001:
                    dig = 2.0
                else:
                    dig = 1.0
                dtick = dig*10.0**decade
                print decade, frac, dig, dtick
                ticks = log10(arange(ceil(10.**mn/dtick)*dtick, 10.**mx+dtick, dtick))
        else:
            decade, frac = divmod(log10(float(mx-mn)/nticks), 1.)
            dig = 10.0**frac
            print 'dig:', dig
            if dig > 5.000001:
                dig = 10.
            elif dig > 2.000001:
                dig = 5.0
            elif dig > 1.000001:
                dig = 2.0
            else:
                dig = 1.0
            dtick = dig*10.0**decade
            print 'decade, frac, dig, dtick =', decade, frac, dig, dtick
            ticks = arange(mn, mx+dtick, dtick);
        if endticks:
            if endticks in ['max', 'both']:
                print (mx-ticks[-1])/(mx-mn), .5/nticks
                if (mx-ticks[-1])/(mx-mn) > .5/nticks:
                    ticks = concatenate((ticks, [mx]))
            if endticks in ['min', 'both']:
                print (ticks[-1]-mn)/(mx-mn), .5/nticks
                if (ticks[-1]-mn)/(mx-mn) > .5/nticks:
                    ticks = concatenate(([mn], ticks))
        print ticks
    if 'orientation' not in kwargs: kwargs['orientation'] = 'horizontal'
    cb = colorbar(im, cax=cax, ticks=ticks, **kwargs)
    if ticklabels is not None:
        if kwargs['orientation'].startswith('h'):
            cb.ax.set_xticklabels(ticklabels)
        else:
            cb.ax.set_yticklabels(ticklabels)
    if log:
        cax.xaxis.set_major_formatter(MyLogFormatter(mn, mx))
    if color:
        for child in getp(cax, 'children'):
            if isinstance(child, mpl.lines.Line2D):
                setp(child, color=color)
        setp(cax.patch, edgecolor=color)
        if kwargs['orientation'].startswith('h'):
            labels = cax.get_xticklabels() + cax.get_xticklines()
        else:
            labels = cax.get_yticklabels() + cax.get_yticklines()
        for tl in labels:
            tl.set_color(color)
    if not ticklines:
        for tl in cax.get_xticklines():
            tl.set_visible(False)
    th = None
    if title:
        args={}
        if titlesize:
            args['size']=titlesize
        th = pylab.title(title, **args)
        if color:
            th.set_color(color)
        if titlepos:
            th.set_position((0.5, titlepos))
    lh = None
    if label:
        if kwargs['orientation'].startswith('h'):
            lh = pylab.xlabel(label)
        else:
            lh = pylab.ylabel(label)
        if color:
            lh.set_color(color)
    if isinteractive():
        draw()
    axes(oldax)
    return cb, cax, th, lh

