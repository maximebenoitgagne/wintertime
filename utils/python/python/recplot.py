import sys
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

class UnitRecArray(np.recarray):
#    def __new__(subtype, shape, dtype=None, buf=None, offset=0, strides=None,
#                formats=None, names=None, titles=None,
#                byteorder=None, aligned=False, units=None):
#        obj = np.recarray.__new__(subtype, shape, dtype, buf, offset, strides,
#                                  formats, names, titles, byteorder, aligned)
#        if units is not None and not hasattr(units,'items'):
#            units = { k:u for k,u in zip(obj.dtype.names, units) }
#        obj.units = units
#        return obj

    def __new__(cls, input_array, units=None):
        obj = np.asarray(input_array).view(cls)
        if units is not None and not hasattr(units,'items'):
            units = { k:u for k,u in zip(obj.dtype.names, units) }
        obj.units = units
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.units = getattr(obj, 'units', None)

#    def __getitem__(self,i):
#        res = super(UnitRecArray,self).__getitem__(i)
#        res.units = self.units
#        return res

def recfromdict(d,names=None):
    if names is None:
        names = d.keys()
    return np.rec.fromrecords([tuple(d[k] for k in names)],names=names)


def recfromlist(dl,names=None,units=None):
    if names is None:
        names = dl[0].keys()
    if units is not None:
        try:
            units = { k:units[k] for k in names }
        except TypeError:
            units = { k:t for k,t in zip(names,units) }
    res = np.rec.fromrecords([tuple(d[k] for k in names) for d in dl],names=names)
    return UnitRecArray(res, units)


def recfromlists(dls,names=None):
    if names is None:
        names = [dl[0].keys() for dl in dls]
    rows = []
    for i in xrange(len(dls[0])):
        row = sum([ tuple(dl[i][k] for k in nam) for dl,nam in zip(dls,names) ], ())
        rows.append(row)
    print rows
    return np.rec.fromrecords(rows, names=sum(names,[]))


def dictfromrec(ra):
    return [ {k:r[k] for k in ra.dtype.names} for r in ra]


def plotrec(x,a=None,vmaxd={},*args,**kwargs):
    if isinstance(x,np.recarray):
        a = x
        x = np.arange(len(a))
    names = a.dtype.names
    n = len(names)
    res = []
    for i,name in enumerate(names):
        ax = plt.subplot(n,1,i+1)
        kw = kwargs.copy()
        res.extend(plt.plot(x,a[name],*args,**kw))
        if name in vmaxd:
            ax.set_ylim(top=vmaxd[name])
        ax.set_ylabel(name)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    return res


def plotrecs(x,a=None,vmaxd={},subs=None,fig=0,clf=False,xmax=None,names=None,units=None,
             sharex=None,pos=None,col=None,hgapfac=None,gapfrac=0.2,title=None,llw=1.5, 
             hlines=[],hlinelabels=[],
             xlabel=None,
             *args,**kwargs):
    if fig == -1:
        fig = plt.figure()
    elif fig > 0:
        fig = plt.figure(fig)
    else:
        fig = plt.gcf()

    if clf:
        plt.clf()

    if a is None:
        a = x
        x = np.arange(len(a))

    try:
        a.dtype
    except AttributeError:
        a = recfromlist(a,units=units)

    if xmax is not None:
        imax = np.where(np.array(x)<=xmax)[0][-1] + 1
        x = x[:imax]
        a = a[:imax]

    minx = np.min(x)
    maxx = np.max(x)

    if names is None:
        names = a.dtype.names

    try:
        units = [ a.units[k.lower()] for k in names ]
    except AttributeError:
        units = [ '' for k in names ]

    legends = subs is not None

    if col is not None:
        col,ncol = col
        if pos is None:
            pos = (.05,.05,.9,.9)
#        w = pos[2]/(ncol+gapfrac*(ncol-1))
        if hgapfac is None: hgapfac=0.9
        hgap = hgapfac*(1-pos[2])
        w = (pos[2]-(ncol-1)*hgap)/float(ncol)
        pos = (pos[0]+col*(w+hgap), pos[1], w, pos[3])

    n = len(names)
    if subs is None:
        subs = [[s] for s in names]
    nsub = len(subs)
    try:
        units = [ a.units[s[0].lower()] for s in subs ]
    except AttributeError:
        units = [ '' for s in subs ]

    axs = []
    axd = {}
    lines = []
    lined = {}
    for i,sub in enumerate(subs):
        if pos is not None:
            h = pos[3]/(nsub+gapfrac*(nsub-1))
            subpos = (pos[0], pos[1]+(nsub-1-i)*h*(1+gapfrac), pos[2], h)
            ax = plt.axes(subpos,sharex=sharex)
        else:
            ax = plt.subplot(nsub,1,i+1,sharex=sharex)

        if i == 0:
#            ax = ax0 = plt.subplot(nsub,1,i+1)
            if sharex is None:
                sharex = ax

        for name in sub:
            kw = kwargs.copy()
            ls = plt.plot(x,a[name.lower()],label=name,*args,**kw)
            lines.extend(ls)
            lined[name], = ls
            axd[name] = ax

        if name in vmaxd:
            ax.set_ylim(top=vmaxd[name])

        if not legends:
            lab = name
            if units[i] != '':
                lab = lab + ' [' + re.sub(r'\*\*','',units[i]) + ']'
            ax.set_ylabel(lab)
        else:
            mpl.rcParams['legend.fontsize'] = 'small'
            leg = plt.legend(loc=6,bbox_to_anchor=(1,0.5))
            plt.setp(leg.get_lines(), lw=llw)
            if units is not None:
                ax.set_ylabel(re.sub(r'\*\*','',units[i]))

        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        if i == nsub - 1:
            if xlabel is not None:
                ax.set_xlabel(xlabel)
        else:
            plt.setp(ax.xaxis.get_ticklabels(),visible=False)

        if len(hlines) > i and hlines[i] is not None:
            ax.hlines(hlines[i],minx,maxx, linestyles='dashed', colors=['b','g','r','c','m','y','k'])
            if len(hlinelabels) > i and hlinelabels[i] is not None:
                labels = hlinelabels[i]
                for j in range(len(labels)-1):
                    if hlines[i][j] == hlines[i][j+1]:
                        labels[j+1] = labels[j] + ',' + labels[j+1]
                        labels[j] = ''
                ax2 = ax.twinx()
                ax2.set_yticks(hlines[i])
                ax2.set_yticklabels(labels)
                ax2.set_ylim(ax.get_ylim())

        axs.append(ax)

    if title is not None:
        axs[0].set_title(title)

    return axs,lined


