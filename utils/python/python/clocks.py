from __future__ import division
import sys
import numpy as np
from numpy import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
import oj.cal

class Dayclock(object):
    def __init__(self, pos, xyear=0, yyear=0, fig=None, transform=None,
                      bgcol='0.0', facecol='0.3', labcol='0.7', handcol='y', yearcol=None,
                      monfrac=0.8, moncol=None, monsize=None,
                      lw=None,
                      r=0.6, rhand=0.6,
                      ha='left', va='baseline', fontsize=14,
                      dt=1200., t0=oj.cal.dateargs2num(1992, 1, 1), offset=-86400.):
        if fig is None:
            fig = plt.gcf()

        if len(fig.axes):
            currax = fig.gca()
        else:
            currax = None

        if yearcol is None:
            yearcol = labcol

        if monsize is None:
            monsize = fontsize

        if transform is not None:
            x0,y0,w,h = pos
            trf = fig.transFigure.inverted().transform
            tra = transform.transform
            x1,y1 = trf(tra((x0+w, y0+h)))
            x0,y0 = trf(tra((x0, y0)))
            pos = (x0, y0, x1 - x0, y1 - y0)
            #sys.stderr.write('Clock position in figure: {}\n'.format(pos))

        self.ax = dax = fig.add_axes(pos,polar=True)
        dax.set_rmin(0)
        dax.set_rmax(1)
        dax.set_xticks([])
        dax.set_yticks([])
        dax.set_axis_bgcolor(facecol)
        ticks = np.linspace(0,2*pi,12,endpoint=0)
        self.lhs = dax.plot(np.array([ticks,ticks]),np.array(12*[[r,1.]]).T,bgcol, lw=lw)
        self.p1 = mpl.patches.Polygon(np.array([np.linspace(0,2*pi,121),121*[r]]).T,edgecolor=bgcol,facecolor=bgcol)
        dax.add_patch(self.p1)
        self.dxy = np.array([[pi/2,pi/2,pi/2+pi/180,pi/2+pi/180],[rhand,1.,1.,rhand]]).T
        self.patch = mpl.patches.Polygon(self.dxy,facecolor=handcol,edgecolor=handcol,zorder=10)
        dax.add_patch(self.patch)
        dax.set_rmin(0)
        dax.set_rmax(1)
#        self.monh = dax.text(3*pi/2,.10*fontsize/14,'Jan',ha='center',va='baseline',color=labcol,fontsize=fontsize)
        _,fh = fig.get_size_inches()
        h = fh*dax.get_position().height

        if moncol:
            tlab = 'JFMAMJJASOND'
            dax.set_thetagrids(75-np.arange(0, 360, 30), tlab, monfrac, fontsize=monsize, color=moncol)
            dax.grid(False)
            self.monh = None
            self.yearh = dax.text(.5, .5-.3*fontsize/72/h, '1999',
                                  ha='center',va='baseline',color=labcol,fontsize=fontsize,
                                  transform=dax.transAxes)
        else:
            self.monh = dax.text(.5, .5-.3*fontsize/72/h, 'Jan',
                                 ha='center',va='baseline',color=labcol,fontsize=fontsize,
                                 transform=dax.transAxes)
            self.yearh = fig.text(xyear, yyear, '1999', ha=ha, va=va,
                                  color=yearcol, fontsize=fontsize)

        self.dt = dt
        self.t0 = t0
        self.offset = offset

        if currax:
            plt.axes(currax)

    def update(self,it):
        date = oj.cal.it2date(it,dt=self.dt,t0=self.t0,offset=self.offset)
        year,mon,day,_,_,_,_,_,_ = date.timetuple()
        yday = oj.cal.dateargs2num(year,mon,day) - oj.cal.dateargs2num(year,1,1)
        self.dxy[:2,0] = 2*pi*(.25-(yday-0)/366.)
        self.dxy[2:,0] = 2*pi*(.25-(yday+2)/366.)
        self.patch.set_xy(self.dxy)
        if self.monh:
            self.monh.set_text(oj.cal.it2date(it,dt=self.dt,t0=self.t0,offset=self.offset).strftime('%b'))
        self.yearh.set_text(str(year))

    def yday(self, yday, year, yearlen=360.):
        self.dxy[:2,0] = 2*pi*(.25-(yday-0)/yearlen)
        self.dxy[2:,0] = 2*pi*(.25-(yday+2)/yearlen)
        self.patch.set_xy(self.dxy)
        if self.monh:
            self.monh.set_text('')
        self.yearh.set_text(str(year))

