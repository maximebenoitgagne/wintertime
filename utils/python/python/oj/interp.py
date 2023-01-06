from __future__ import division, print_function
import sys
import numpy as np
import matplotlib.mlab as mlab

def slopes(x,y,ypin=None):
    x = np.asarray(x, np.float_)
    y = np.asarray(y, np.float_)
    if ypin is None:
        ypin = np.ma.masked_all(y.shape)
    else:
        ypin = np.ma.masked_invalid(ypin)

#    sys.stderr.write('Given slopes: {}\n'.format(str(ypin)))

    #yp = np.zeros(y.shape, np.float_)
    yp = np.zeros((2*y.shape[0]-1,) + y.shape[1:], np.float_)

    # Cast key variables as float.
    x = np.r_[x[:1], x, x[-1:]]
    y = np.r_[y[:1], y, y[-1:]]

    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    with np.errstate(divide='ignore'):
        dydx = dy/dx
        yp[::2] = np.where(ypin.mask, 
              (dydx[:-1] * dx[1:] + dydx[1:] * dx[:-1])/(dx[1:] + dx[:-1]),
              ypin.data)

        yp[1::2] = dy[1:-1]/dx[1:-1]

#    sys.stderr.write('Linear slopes: {}\n'.format(str(yp)))

    # extrapolate from right
    ind = (dx[:-2]==0)&(dx[1:-1]!=0)&(dx[2:]!=0)
    yp[:-1:2][ind] = 2.*yp[1::2][ind] - yp[2::2][ind]

    ind = (dx[:-2]==0)&(dx[1:-1]!=0)&(dx[2:]==0)
    yp[:-1:2][ind] = yp[1::2][ind]

    # extrapolate from left
    ind = (dx[:-2]!=0)&(dx[1:-1]!=0)&(dx[2:]==0)
    yp[2::2][ind] = 2.*yp[1::2][ind] - yp[:-1:2][ind]

    ind = (dx[:-2]==0)&(dx[1:-1]!=0)&(dx[2:]==0)
    yp[2::2][ind] = yp[1::2][ind]

#    sys.stderr.write('Computed slopes: {}\n'.format(str(yp)))

    yp = np.where(ypin.mask, yp[::2], ypin)

#    sys.stderr.write('Returned slopes: {}\n'.format(str(yp)))
    return yp

def stineman(xi, x, y, yp=None, logslopes=False):
    ''' like mlab.stineman_interp, but compute slopes for masked (or nan)
    values in yp while keeping valid ones.
    If 2 consecutive x coincide, extrapolate slope from both sides (resulting
    in linear interpolation if 2 duplicates follow each other).
    '''
    yp = slopes(x, y, yp)
    if logslopes:
        sys.stderr.write('Slopes: {}\n'.format(str(yp)))
    try:
        res = mlab.stineman_interp(xi, x, y, yp)
    except ValueError:
        sys.stderr.write('Slopes: {}\n'.format(str(yp)))
        raise
    return res

class StinemanInteractor:
    """
    An editor for stineman interpolants

    Key-bindings

      'i' insert vertices/slopes
      'd' delete vertices/slopes
      '0' set slope to zero or align with closest neighbor
      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them
      'P' print current parameters and toggle printing
      'q' delete callbacks
      'Q' delete callbacks and sys.exit


    """

    showverts = True
    printonchange = False
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, ax, x, y, yp=None, n=50):

        self.ax = ax
        canvas = self.ax.figure.canvas

        x = np.asanyarray(x)
        y = np.asanyarray(y)
        if yp is None:
            yp = len(x)*[np.nan]
        yp = np.asanyarray(yp)
        self.n = n
        xx = np.zeros((3*len(x)-2,))
        yy = np.zeros((3*len(x)-2,))
        dx = np.diff(x)/3.
        dx = np.r_[dx[0], dx, dx[-1]]
        xp = 1. - np.isnan(yp)
        dy = np.where(xp, yp*dx[1:], 0.)
        xx = np.r_[[x],[x+xp*dx[1:]]]
        yy = np.r_[[y],[y+xp*dy]]
        self.lines = ax.plot(xx,yy,marker='o', markerfacecolor='r', color='k', linewidth=1, animated=True)

        xf = np.r_[(x[:-1,None] + np.arange(0., 1., 1./n)*3*dx[1:-1,None]).reshape(-1), x[-1]]
        yf = stineman(xf, x, y, yp)

        self.fline, = ax.plot(xf, yf, 'm', linewidth=3, animated=True)

        self._ind = None # the active vert

        self.cids = []
        self.cids.append(canvas.mpl_connect('draw_event', self.draw_callback))
        self.cids.append(canvas.mpl_connect('button_press_event', self.button_press_callback))
        self.cids.append(canvas.mpl_connect('key_press_event', self.key_press_callback))
        self.cids.append(canvas.mpl_connect('button_release_event', self.button_release_callback))
        self.cids.append(canvas.mpl_connect('motion_notify_event', self.motion_notify_callback))
        self.canvas = canvas


    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
#        self.ax.draw_artist(self.pathpatch)
        self.ax.draw_artist(self.fline)
        for line in self.lines:
            self.ax.draw_artist(line)
        self.canvas.blit(self.ax.bbox)

    def pathpatch_changed(self, pathpatch):
        'this method is called whenever the pathpatchgon object is called'
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
#        Artist.update_from(self.line, pathpatch)
        self.line.set_visible(vis)  # don't use the pathpatch visibility state


    def get_ind_under_point(self, event):
        'get the index of the vertex under point if within epsilon tolerance'

        # display coords
        xy = np.concatenate([line.get_data() for line in self.lines], -1).T
        xyt = self.lines[0].get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.sqrt((xt-event.x)**2 + (yt-event.y)**2)
        ind = d.argmin()

        if d[ind]>=self.epsilon:
            ind = None

        return ind

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        if not self.showverts: return
        if event.inaxes != self.ax: return
        if event.button != 1: return
        self._ind = self.get_ind_under_point(event)

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        if not self.showverts: return
        if event.button != 1: return
        self._ind = None

    def key_press_callback(self, event):
        'whenever a key is pressed'
        if event.inaxes != self.ax: return
        if event.key=='q':
            for cid in self.cids:
                self.canvas.mpl_disconnect(cid)
        elif event.key=='Q':
            for cid in self.cids:
                self.canvas.mpl_disconnect(cid)
            sys.exit()
        elif event.key=='t':
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts: self._ind = None
        elif event.key=='d':
            ind = self.get_ind_under_point(event)
            if ind is not None:
#                nline = len(self.lines)
                n = len(self.lines[0].get_data()[0])
                iline,i = divmod(ind, n)
                if i == 1:
                    x,y = self.lines[iline].get_data()
                    x[1] = x[0]
                    y[1] = y[0]
                    self.lines[iline].set_data(x,y)
                else:
                    self.lines[iline].remove()
                    del self.lines[iline]
                self.redraw()
        elif event.key=='i':
            xy = np.concatenate([line.get_data() for line in self.lines], -1).T
            xyt = self.lines[0].get_transform().transform(xy)
            xt, yt = xyt[:, 0], xyt[:, 1]
            d = np.sqrt((xt-event.x)**2).min()
            if d > self.epsilon:
                xx,yy = np.concatenate([line.get_data() for line in self.lines], -1)
                nline = len(self.lines)
                xx = xx.reshape(nline, -1)
                yy = yy.reshape(nline, -1)
                i = np.searchsorted(xx[:,0], event.xdata)
#                dx = xx[i,0] - event.xdata
#                dy = yy[i,0] - event.ydata
                x = [event.xdata, event.xdata]#+dx/3.]
                y = [event.ydata, event.ydata]#+dy/3.]
                self.lines[i:i] = self.ax.plot(x,y,marker='o', markerfacecolor='r', color='k', animated=True)
                self.redraw()
            else:
                ind = self.get_ind_under_point(event)
                if ind is not None:
    #                nline = len(self.lines)
                    n = len(self.lines[0].get_data()[0])
                    iline,i = divmod(ind, n)
                    x,y = self.lines[iline].get_data()
                    if x[1] == x[0] and y[1] == y[0]:
                        iline2 = iline + 1
                        if iline2 >= len(self.lines):
                            iline2 = iline - 1
                        x2,y2 = self.lines[iline2].get_data()
                        dx = x2[0] - x[0]
                        dy = y2[0] - y[0]
                        x[1] = x[0] + dx/3.
                        y[1] = y[0] + dy/3.
                        self.lines[iline].set_data(x,y)
                        self.redraw()
                
        elif event.key=='0':
            ind = self.get_ind_under_point(event)
            if ind is not None:
                n = len(self.lines[0].get_data()[0])
                iline,i = divmod(ind, n)
                x,y = self.lines[iline].get_data()
                if i == 1:
                    y[1] = y[0]
                    self.lines[iline].set_data(x,y)
                else:
                    other = [j for j in [iline-1, iline+1] if j>=0 and j<len(self.lines)]
                    yother = np.array([self.lines[j].get_data()[1][0] for j in other])
                    j = np.argmin(abs(y[1]-yother))
                    y[1] = yother[j] + y[1] - y[0]
                    y[0] = yother[j]
                    self.lines[iline].set_data(x,y)
                self.redraw()
        elif event.key=='P':
            xx,yy = np.concatenate([line.get_data() for line in self.lines], -1)
            nline = len(self.lines)
            xx = xx.reshape(nline, -1)
            yy = yy.reshape(nline, -1)
            yp = (yy[:,1]-yy[:,0])/(xx[:,1]-xx[:,0])
            print()
            print('x  =', repr(xx[:,0])[6:-1])
            print('y  =', repr(yy[:,0])[6:-1])
            print('yp =', repr(yp)[6:-1])
            self.printonchange = not self.printonchange

        self.canvas.draw()

    def motion_notify_callback(self, event):
        'on mouse movement'
        if not self.showverts: return
        if self._ind is None: return
        if event.inaxes != self.ax: return
        if event.button != 1: return
        x,y = event.xdata, event.ydata

        xx,yy = np.concatenate([line.get_data() for line in self.lines], -1)

        if self._ind%2 == 0:
            if event.key == 'shift':
                # do not change
                y = yy[self._ind]
            elif event.key == 'control':
                # do not change
                x = xx[self._ind]
            xx[self._ind+1] += x - xx[self._ind]
            yy[self._ind+1] += y - yy[self._ind]
        else:
            if event.key == 'shift':
                # do not change slope
                x0 = xx[self._ind-1]
                x1 = xx[self._ind]
                y0 = yy[self._ind-1]
                y1 = yy[self._ind]
                y = y0 + (x-x0)*(y1-y0)/(x1-x0)
            
        xx[self._ind] = x
        yy[self._ind] = y
        nline = len(self.lines)
        xx = xx.reshape(nline, -1)
        yy = yy.reshape(nline, -1)
        for i in range(nline):
            self.lines[i].set_data((xx[i], yy[i]))
        dx = np.diff(xx[:,0])
        xf = np.r_[(xx[:-1,0,None] + np.arange(0., 1., 1./self.n)*dx[:,None]).reshape(-1), xx[-1,0]]
        yp = (yy[:,1]-yy[:,0])/(xx[:,1]-xx[:,0])
        yf = stineman(xf, xx[:,0], yy[:,0], yp)
        self.fline.set_data((xf, yf))

        self.canvas.restore_region(self.background)
#        self.ax.draw_artist(self.pathpatch)
        self.ax.draw_artist(self.fline)
        for line in self.lines:
            self.ax.draw_artist(line)
        self.canvas.blit(self.ax.bbox)
#        self.redraw()

        if self.printonchange:
            print()
            print('x  =', repr(xx[:,0])[6:-1])
            print('y  =', repr(yy[:,0])[6:-1])
            print('yp =', repr(yp)[6:-1])

    def redraw(self):
        xx,yy = np.concatenate([line.get_data() for line in self.lines], -1)
        nline = len(self.lines)
        xx = xx.reshape(nline, -1)
        yy = yy.reshape(nline, -1)
        dx = np.diff(xx[:,0])
        xf = np.r_[(xx[:-1,0,None] + np.arange(0., 1., 1./self.n)*dx[:,None]).reshape(-1), xx[-1,0]]
        yp = (yy[:,1]-yy[:,0])/(xx[:,1]-xx[:,0])
        yf = stineman(xf, xx[:,0], yy[:,0], yp)
        self.fline.set_data((xf, yf))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.fline)
        for line in self.lines:
            self.ax.draw_artist(line)
        self.canvas.blit(self.ax.bbox)


def editor(ax, x, y, yp=None, n=50):
    return StinemanInteractor(ax, x, y, yp, n)

