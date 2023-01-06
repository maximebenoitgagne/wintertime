#!/usr/bin/env python
import sys,os
import numpy as np
from numpy.ma import MaskedArray
import matplotlib.pyplot as plt
#from pylab import *
from matplotlib.ticker import FixedLocator
from matplotlib.widgets import Slider, Button, RadioButtons
from oj.num import *
from oj.plt import *
#import gobject

def layerplot(a, norm='auto', mask=None, ticklabels=None, axislabels=[], start=None, **kwargs):
    """ cb1,cb2,axi = layerplot(a,norm,mask,ticklabels,axislabels,start,...)
        
        a           array/TiledFiles/...  :: data to plot
        norm        matplotlib.colors.Normalize instance / 'auto' / 'fixed'
        mask        array                 :: mask, at least 2d
        ticklabels  [[f1,f2,...], [it1,it2,...], ...]
        axislabels  ['fld', 'it']
        start       [i0,i1,...]           :: indices of layer to show at start

        cb1,cb2     figure.canvas callbacks
        axi[]       list of slider axes
    """
    global layerind, layernorm, guicbax, layerplay

    dim = a.shape[:-2]
    ndim = len(dim)

    if norm == 'fixed':
        norm = None
    if norm == 'auto':
        normarg = None
    else:
        normarg = norm

    if ticklabels is None:
        if hasattr(a,'labels'):
            ticklabels = a.labels
        elif hasattr(a,'fields'):
            ticklabels = [a.fields]
            if hasattr(a,'its'):
                ticklabels.append(a.its)
        else:
            ticklabels = []

    labelsarenumbers = [ np.all( [ type(l) in (int,long) for l in tl ] ) for tl in ticklabels ]
    ticklabels = [[ str(l) for l in tl ] for tl in ticklabels ]

    layerind = ndim*[0]

    if start is not None:
        for i,ind in enumerate(start):
            if type(ind) not in (int,long) and len(ticklabels) > i:
                try:
                    ind = ticklabels[i].index(ind)
                except ValueError:
                    pass
            layerind[i] = ind
#        layerind[:len(start)] = list(start)

    plt.clf()
    off = max(.25,.06 + ndim*.08)
    print off
    guicbax = cax = plt.axes([.93,off,.01,.95-off])
    off = .06 + ndim*.08
    ax = plt.axes([.05,off,.85,.95-off])
    #plt.subplots_adjust(bottom=0.25)

    indcur = tuple(layerind)+np.s_[:,:]
    mymask = mask
    ndimmask = 0
    if mask is not None:
        ndimmask = len(mask.shape)
        maskarg = mymask[indcur[-ndimmask:]]
    else:
        maskarg = mymask

    im = myimshow(a[tuple(layerind)+np.s_[:,:]], norm=normarg, mask=maskarg, animated=True, **kwargs)

    mycb(cax=cax,orientation='vertical');

    layernorm = norm

    axi = ndim*[None]
    sli = ndim*[None]
    upax = ndim*[None]
    dnax = ndim*[None]
    plax = ndim*[None]
    layerplay = ndim*[False]
    for i in range(ndim):
        itlo = 0
        ithi = dim[i]-1
        off = .06+i*.08
        axi[i] = plt.axes([0.1, off, 0.6, 0.03])
        label = 'i%d'%i
        if len(axislabels) > i:
            label = axislabels[i]
        sli[i] = Slider(axi[i], label, itlo-.5, ithi+.5, valinit=layerind[i], valfmt='%.0f') #,closedmin=True,closedmax=True)
        tickstep = 1
        ntickmax = 20.
        if len(ticklabels) > i:
            ntickmax = max(3, 60./len(ticklabels[i][-1]))
        if (ithi-itlo)/tickstep > ntickmax and len(ticklabels) <= i or labelsarenumbers[i]:
            tickstep = max(1,int(np.floor((ithi-itlo)/ntickmax)))
        axi[i].set_xticks(np.arange(itlo,ithi+1,tickstep))
        if len(ticklabels) > i:
            axi[i].set_xticklabels(ticklabels[i][0::tickstep])
        if tickstep > 1:
            axi[i].xaxis.set_minor_locator(FixedLocator(range(dim[i])))

        def update(val,il=i):
            global layerind, layernorm
            newind = int(round(val))
            if newind != layerind[il]:
                layerind[il] = newind
                indcur = tuple(layerind)+np.s_[:,:]
                if mymask is not None:
                    im.set_data(MaskedArray(a[indcur],mymask[indcur[-ndimmask:]]))
                else:
                    im.set_data(a[indcur])
                if layernorm == 'auto':
                    im.autoscale()
                sli[il].set_val(layerind[il])
                plt.draw()

        sli[i].on_changed(update)

        upax[i] = plt.axes([0.83, off, 0.03, 0.04])
        button = Button(upax[i], '+')
        def tmp(event,i=i):
            if layerind[i]+1 < dim[i]:
                sli[i].set_val(layerind[i]+1)
        button.on_clicked(tmp)

        dnax[i] = plt.axes([0.79, off, 0.03, 0.04])
        button = Button(dnax[i], '-')
        def tmp(event,i=i):
            if layerind[i] > 0:
                sli[i].set_val(layerind[i]-1)
        button.on_clicked(tmp)

        plax[i] = plt.axes([0.87, off, 0.02, 0.04])
        button = Button(plax[i], '>')
        def tmp(event,i=i,button=button):
            global layerplay
            layerplay[i] = not layerplay[i]
            if layerplay[i]:
                plt.gcf().canvas.manager.window.after(10,callback,i)
                #gobject.idle_add(callback,i)
                button.label.set_text('||')
            else:
                button.label.set_text('>')
        button.on_clicked(tmp)

    def callback(i,*args):
        global layerind, layerplay
        newind = (layerind[i]+1)%dim[i]
        sli[i].set_val(newind)
        if layerplay[i]:
            plt.gcf().canvas.manager.window.after(10,callback,i)
            #gobject.idle_add(callback,i)


    normax = plt.axes([0.93, .06, 0.05, 0.16], axisbg='.75')
    radio = RadioButtons(normax, ('fix', 'auto'), active=norm=='auto')
    def normfunc(label):
        global layernorm
        layernorm = label
        if layernorm == 'fix':
            layernorm = plt.Normalize(*im.get_clim())
            im.set_norm(layernorm)
        else:
            im.autoscale()
        plt.draw()
    radio.on_clicked(normfunc)

    def onpress(event):
        global guiypress, guicbax
#        print 'button=%d, x=%d, y=%d'%(
#            event.button, event.x, event.y)
#        print event.inaxes, guicbax
        if event.inaxes == guicbax and event.ydata is not None:
#            print 'xdata=%f, ydata=%f'%(
#                event.xdata, event.ydata)
            clim = plt.gci().get_clim()
            guiypress = event.ydata
        else:
            guiypress = None

    def onrelease(event):
        global guiypress, guicbax, layernorm
#        print 'button=%d, x=%d, y=%d'%(
#            event.button, event.x, event.y)
        if event.inaxes == guicbax and event.ydata is not None and guiypress is not None:
            layernorm = None  # fixed
#            print 'xdata=%f, ydata=%f'%(
#                event.xdata, event.ydata)
            clim = plt.gci().get_clim()
            yhi = max(guiypress, event.ydata)
            ylo = min(guiypress, event.ydata)
#            print clim, ylo,yhi
            if event.button == 1:
                # zoom in
                chi = clim[0] + (clim[1]-clim[0])*yhi
                clo = clim[0] + (clim[1]-clim[0])*ylo
            elif event.button == 3:
                # zoom out
                clo = (clim[0]*yhi-clim[1]*ylo)/(yhi-ylo)
                chi = (clim[0]*(1.-yhi)-clim[1]*(1.-ylo))/(ylo-yhi)
            else:
                # pan
                clo = clim[0]-(event.ydata-guiypress)*(clim[1]-clim[0])
                chi = clim[1]-(event.ydata-guiypress)*(clim[1]-clim[0])
#            print clo,chi
            plt.clim(clo,chi)
            plt.draw()

    cid1 = plt.gcf().canvas.mpl_connect('button_press_event', onpress)
    cid2 = plt.gcf().canvas.mpl_connect('button_release_event', onrelease)

#    plt.gcf().canvas.manager.window.after(100,callback)

    return cid1,cid2,axi

#    off = .10+(ndim-1)*.08
#    fixax = plt.axes([0.93, off, 0.05, 0.04])
#    button = Button(fixax, 'fix')
#    def fixnorm(event):
#        norm = plt.Normalize(*im.get_clim())
#        im.set_norm(norm)
#    button.on_clicked(fixnorm)

#    off = .10+(ndim-2)*.08
#    autoax = plt.axes([0.93, off, 0.05, 0.04])
#    button = Button(autoax, 'auto')
#    def autonorm(event):
#        norm = 'auto'
#        im.autoscale()
#    button.on_clicked(autonorm)

    plt.show()

#args = sys.argv[1:]
#fpatt = args[0]
#itlo,ithi,itstep,w,h = map(int, args[1:6])
#mn,mx = map(float, args[6:8])

def scalelabels(axi):
    for i,ax in enumerate(axi):
        tls = ax.get_xticklabels()
        fs = tls[0].get_fontsize()
        while fs > 6 and min([r.get_window_extent().xmin-l.get_window_extent().xmax for l,r in zip(tls[:-1],tls[1:])]) < 0:
            fs -= 1
            plt.setp(tls, fontsize=fs)

    plt.draw()


def disconnect(*cids):
    fig = plt.gcf()
    for cid in cids:
        fig.canvas.mpl_disconnect(cid)

