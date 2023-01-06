import numpy as np
from numpy import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
import cal

class Dayclock(object):
    def __init__(self,pos,x,y,fig=None,
                      bgcol='0.0',facecol='0.3',labcol='0.7',handcol='y',
                      r=0.6,rhand=0.6,
                      monr=0.1, monoff=None,
                      ha='left',va='baseline',fontsize=14,yearcol=None,
                      dt=1200.,t0=cal.dateargs2num(1992,1,1),offset=-86400.):
        '''
        monoff :: 0.3
        '''
        if yearcol is None:
            yearcol = labcol

        if fig is None:
            fig = plt.gcf()

        if len(fig.axes):
            currax = fig.gca()
        else:
            currax = None

        self.ax = dax = fig.add_axes(pos,polar=True)
        dax.set_xticks([])
        dax.set_yticks([])
        dax.set_axis_bgcolor(facecol)
        ticks = np.linspace(0,2*pi,12,endpoint=0)
        dax.plot(np.array([ticks,ticks]),np.array(12*[[r,1.]]).T,color=bgcol)
        dax.add_patch(mpl.patches.Polygon(np.array([np.linspace(0,2*pi,121),121*[r]]).T,edgecolor=bgcol,facecolor=bgcol))
        self.dxy = np.array([[pi/2,pi/2,pi/2+pi/180,pi/2+pi/180],[rhand,1.,1.,rhand]]).T
        self.patch = mpl.patches.Polygon(self.dxy,facecolor=handcol,edgecolor=handcol,zorder=10)
        dax.add_patch(self.patch)
        dax.set_rmin(0)
        dax.set_rmax(1)
        if monoff is not None:
            _,fh = fig.get_size_inches()
            h = fh*dax.get_position().height
            self.monh = dax.text(.5, .5-monoff*fontsize/72/h, 'Jan',
                                 ha='center',va='baseline',color=labcol,fontsize=fontsize,
                                 transform=dax.transAxes)
        else:
            self.monh = dax.text(3*pi/2,monr,'Jan',ha='center',va='baseline',color=labcol,fontsize=fontsize)
        self.yearh = fig.text(x,y,'1999',ha=ha,va=va,color=yearcol,fontsize=fontsize)
        self.dt = dt
        self.t0 = t0
        self.offset = offset

        if currax:
            plt.axes(currax)

    def update(self,it):
        date = cal.it2date(it,dt=self.dt,t0=self.t0,offset=self.offset)
        year,mon,day,_,_,_,_,_,_ = date.timetuple()
        yday = cal.dateargs2num(year,mon,day) - cal.dateargs2num(year,1,1)
        self.dxy[:2,0] = 2*pi*(.25-(yday-0)/366.)
        self.dxy[2:,0] = 2*pi*(.25-(yday+2)/366.)
        self.patch.set_xy(self.dxy)
        self.monh.set_text(cal.it2date(it,dt=self.dt,t0=self.t0,offset=self.offset).strftime('%b'))
        self.yearh.set_text(str(year))


