from __future__ import division
import sys
from copy import copy, deepcopy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def inch_or_pixel(a, dpi):
    if np.iscomplex(a):
        a = a.real + a.imag/float(dpi)
    return a


class GridSpecInches(object):
    def __init__(self, widths, heights, wgap=1./6, hgap=1./6, subgrids=None, dpi=80.):
        self.widths  = [ inch_or_pixel(w, dpi) for w in widths ]
        self.heights = [ inch_or_pixel(h, dpi) for h in heights ]
        self.wgap    = inch_or_pixel(wgap, dpi)
        self.hgap    = inch_or_pixel(hgap, dpi)
        self.dpi = dpi
        self.subgrids = subgrids
        self.parent  = None
        self.slice   = None
        self._figure  = None
        self._gridspec = None
        if subgrids is not None:
            for i,subgrid in enumerate(subgrids):
                subgrid.parent = self
                subgrid.slice = i

    @property
    def nrows(self):
        return len(self.heights)

    @property
    def ncols(self):
        return len(self.widths)

    @property
    def wspace(self):
        return self.wgap*len(self.widths)/sum(self.widths)

    @property
    def hspace(self):
        return self.hgap*len(self.heights)/sum(self.heights)

    @property
    def width(self):
        """ total width """
        return sum(self.widths) + (len(self.widths)-1)*self.wgap

    @property
    def height(self):
        """ total height """
        return sum(self.heights) + (len(self.heights)-1)*self.hgap

    def clone(self, nrows, ncols, wgap=1./6, hgap=1./6):
        wgap = inch_or_pixel(wgap, self.dpi)
        hgap = inch_or_pixel(hgap, self.dpi)
#        wid = ncols*self.width + (ncols-1)*wgap
#        hei = nrows*self.height + (nrows-1)*hgap
        parent = GridSpecInches(ncols*[self.width], nrows*[self.height], wgap, hgap,
                                [copy(self) for _ in range(nrows*ncols)])
        self.parent = parent
        return parent

    def makefigure(self, fignum, left=4./6,right=4./6,bottom=4./6,top=4./6,
            dpi=None,maxwidth=9999999999,maxheight=9999999999,
#            forward=True,toolbarsize=34.):
            forward=True,toolbarsize=None):
        if dpi is None:
            dpi = self.dpi
        left   = inch_or_pixel(left  , self.dpi)
        right  = inch_or_pixel(right , self.dpi)
        bottom = inch_or_pixel(bottom, self.dpi)
        top    = inch_or_pixel(top   , self.dpi)

        wid = left + right + self.width
        hei = bottom + top + self.height

        fac = 1
        if hei > maxheight:
            fac = maxheight/hei
        if wid*fac > maxwidth:
            fac = maxwidth/wid
        if fac < 1.:
            sys.stderr.write('Scaling figure by {}\n'.format(fac))
        wid *= fac
        hei *= fac
        widpix = int(round(wid*dpi))
        heipix = int(round(hei*dpi))
        wid = 2.*np.ceil(widpix/2.)/dpi
        hei = 2.*np.ceil(heipix/2.)/dpi

        fig = plt.figure(fignum,(wid,hei),dpi=dpi)
        if forward:
            if toolbarsize is None:
                fig.set_size_inches([wid,hei], forward=True)
                plt.draw()
                w,h = fig.get_size_inches()
                if h != hei:
                    fig.set_size_inches([wid,2*hei-h], forward=True)
            else:
                fig.set_size_inches((wid, hei+toolbarsize/dpi), forward=forward)
        fig.set_size_inches(wid,hei)
        fig.set_dpi(dpi)
        fig.clear()
        wid,hei = fig.get_size_inches()

        self._figure = fig
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top

        return fig

    @property
    def figure(self):
        if self._figure is None:
            if self.parent is not None:
                self._figure = self.parent.figure
            else:
                raise ValueError('Need to call makefigure method first')

        return self._figure

    @property
    def gridspec(self):
        if self._gridspec is None:
            if self.parent is not None:
                gs = self.parent.gridspec[self.slice]
                self._gridspec = mpl.gridspec.GridSpecFromSubplotSpec(self.nrows, self.ncols, gs,
                                           wspace=self.wspace, hspace=self.hspace,
                                           width_ratios=self.widths, height_ratios=self.heights)
            else:
                wid,hei = self.figure.get_size_inches()
                self._gridspec = mpl.gridspec.GridSpec(self.nrows, self.ncols,
                                           left=self.left/wid, right=1.-self.right/wid,
                                           bottom=self.bottom/hei, top=1.-self.top/hei,
                                           wspace=self.wspace, hspace=self.hspace,
                                           width_ratios=self.widths, height_ratios=self.heights)

        return self._gridspec

    def add_subplot(self, idx, **kwargs):
        return self.figure.add_subplot(self.gridspec[idx], **kwargs)

    def get(self, idx, **kwargs):
        if self.subgrids is not None:
            return self.subgrids[idx]
        else:
            return self.figure.add_subplot(self.gridspec[idx], **kwargs)

    def __getitem__(self,idx):
        if self.subgrids is not None:
            return self.subgrids[idx]
        else:
            return self.figure.add_subplot(self.gridspec[idx])

    def myaxes(self, slices=None, off=False, **kwargs):
        if off:
            # turn anything visible off unless specified by user
            kwargs['frameon'] = kwargs.get('frameon', False)
            kwargs['xticks'] = kwargs.get('xticks', [])
            kwargs['yticks'] = kwargs.get('yticks', [])

        if slices is None:
            ag = np.empty((self.nrows, self.ncols), object)
            for j in range(self.nrows):
                for i in range(self.ncols):
                    ag[j,i] = self.add_subplot((j,i), **kwargs)
        else:
            ag = [ self.add_subplot(s, **kwargs) for s in slices ]

        return ag


    def axes(self, slices=None, **kwargs):
        if self.subgrids is not None:
            ag = np.empty((self.nrows, self.ncols), object)
            for i in range(ag.size):
                ag.flat[i] = self[i].axes(slices, **kwargs)
        else:
            if slices is None:
                slices = [(j,i) for j in range(self.nrows) for i in range(self.ncols)]

            ag = [ self[s] for s in slices ]

        return ag

    def add_cax(self, width=1./6, gap=None):
        if gap is None:
            gap = width
        return hbox([self, width], wgap=gap)


def hbox(grids, wgap=1./6):
    # convert numbers to single-cell GridSpecInches of that width
    grids = [ np.isscalar(g) and GridSpecInches([g], [1e-32]) or g for g in grids ]
    widths = [ g.width for g in grids ]
    height = max( g.height for g in grids )
    parent = GridSpecInches(widths, [height], wgap, 0., grids)
    return parent


def vbox(grids, hgap=1./6):
    # convert numbers to single-cell GridSpecInches of that width
    grids = [ np.isscalar(g) and GridSpecInches([1e-32], [g]) or g for g in grids ]
    width = max( g.width for g in grids )
    heights = [ g.height for g in grids ]
    parent = GridSpecInches([width], heights, 0., wgap, grids)
    return parent


