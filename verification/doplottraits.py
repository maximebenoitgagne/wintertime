#!/usr/bin/env python
import sys
from collections import OrderedDict
import numpy as np
from nml import NmlFile
from plt import *

UNDEF = 123456.7
pzk = ['palat', 'asseff', 'exportfracpreypred']

args = sys.argv[1:]
fname, = args

tr = NmlFile(fname).merge()

ntot = tr.physize.size
nphy = (tr.pcmax>0).sum()
nzoo = ntot - nphy

discrete = [k for k,v in tr.items() if v.dtype == np.dtype('i8')]

typed = {
    3: 'P',
    15: 'S',
    16: 'T',
    47: 'D',
    79: 'C',
    255: 'Z',
}
colors = OrderedDict()
colors['P'] = (0.,1.,0.)
colors['S'] = (0.,.8,.8)
colors['D'] = (.8,0.,.2)
colors['C'] = (.2,.4,1.)
colors['T'] = (.85,.8,0.)
colors['Z'] = (0.,0.,0.)

denomd = dict(mort=86400, mort2=86400, acclimtimescl=86400, pcmax=86400,
    r_fec=120, r_pc=120, r_nc=120, r_sic=120)

f = (tr.usenh4) + (tr.useno2<<1) + (tr.useno3<<2) + (tr.combno<<3) + (tr.diazotroph<<4) + (tr.diacoc<<5)
f[f<=0] = 255
print '   ', ' CDAn324', 'g', ' PCmax(/day)'
for i in range(nphy):
    print '{:3d} {:8b} {:1s} {:9.6f}'.format(i+1, f[i], typed.get(f[i], str(f[i])), tr.pcmax[i]*86400)

pk = sorted(k for k,v in tr.items() if v[:nphy].std() > 1e-12*abs(v[:nphy].mean()) and k not in discrete)
if 'alpha_mean' in pk and 'alphachl' in pk:
    pk.remove('alpha_mean')

print
print 'phyto:'
for k in sorted(tr):
    if k not in pk and k not in discrete:
        a = tr[k][:nphy]
        if a.max() > 0:
            s = a.std()
            if s == 0 or s < abs(a.mean())/1e12: s = ''
            m = a.mean()
            if m == 0: m = 0
            d = denomd.get(k, 1.)
            if d != 1:
                m = '{}/{}'.format(m*d, d)
            print '{:18s}'.format(k), m, s

import gapgrid

nc = 2
nr = -(-len(pk)//nc)

#fig = figure('phy', (13.5, 13.9))
sz = (20., 13.3)
fig = figure('phy', sz)
clf()
fig.set_size_inches(sz)
gg = gapgrid.gapgrid(nc*[1.], nr*[1.], 1.2j, .3j, 1.j,
        add_all=False, share_x='all', share_y=False, rect=[0,0,.67,1])

for i,k in enumerate(pk):
    ax = gg.add_axes(*divmod(i,nc))
    tp = [typed[ff] for ff in f[:nphy]]
    cols = [colors[t] for t in tp]
    y = tr[k][:nphy]
    if k in denomd:
        d = denomd[k]
        y *= d
        ax.set_ylabel('/{}'.format(d))
    ax.scatter(np.r_[1:nphy+1], y, 32, c=cols, edgecolor='None')
#    ax.scatter(np.r_[nphy+.6], tr[k][nphy], 8, c='k', edgecolor='None')
    ax.scatter(np.r_[:nzoo]*0./nzoo+nphy+.5, tr[k][nphy:], 8, c='k', edgecolor='None')
    ax.set_title(k)
    ax.set_xlim((.1, nphy+.9))
    ax.set_ylim((0, tr[k].max()*1.05))
#    ax.tick_params('y', labelleft=True, label=True)
    ax.axis["left"].toggle(ticklabels=True, label=True)

    ax.set_xticks(range(2, nphy+1, 1 if nphy<20 else 2))

ax = gg.axes_all[0]
#axy = ax.twiny()
#axy.xaxis.set_ticks(range(nphy))
#axy.xaxis.set_ticklabels([typed.get(ff) for ff in f])
#ax.tick_params('x', labelbottom=False)
#axy.tick_params('x', labeltop=True)

ax = gg.axes_row[0][-1]
_,y = ax.get_ylim()
y = .1*y
#for i in range(nphy):
#    ax.text(i+1, y, typed[f[i]], va='bottom', ha='center')
for k,c in colors.items():
    sz = k == 'Z' and 3 or 6
    ax.plot([], [], 'o', color=c, label=k, markeredgecolor='None', markersize=sz)
ax.legend(loc='upper left', bbox_to_anchor=(1.02,1), borderaxespad=0,
          numpoints=1, labelspacing=.05, handlelength=1., handletextpad=0.25)#, fontsize=12)

draw()

#fig.set_size_inches((13.5, 13.9))
#savefig('traits_phy.png', dpi=fig.dpi)

zz,pp = np.mgrid[nphy:ntot,:nphy] + 1

kw = dict()#edgecolor='None')

#fig = figure('zoo-phy')
#clf()
gg = gapgrid.gapgrid([1.], len(pzk)*[1.], 1.2j, .3j, 1.j,
        share_x='all', share_y=False, rect=[.67,.5,.33,.5])
for i,k in enumerate(pzk):
    ax = gg.axes_all[i]
    a = tr[k]
    if i != 0:
        a[tr[pzk[0]]<=0] = 0
    a = a.reshape(ntot,ntot)[nphy:,:nphy]
    ax.scatter(pp, zz, a*50, c=cols, **kw)
    if i == 0:
        msk = a < 0
        ax.plot(pp[msk], zz[msk], 'k_')
    ax.set_title(k)
    ax.set_xticks(range(2, nphy+1, 1 if nphy<20 else 2))
    #ax.set_yticks(range(nphy+1, ntot+1))
    ax.set_xlim((0, nphy+1))
    ax.set_ylim((nphy+.1, ntot+.9))
    ax.grid(True)

draw()

#fig.set_size_inches((8., 6.))
#savefig('traits_zoo-phy.png', dpi=fig.dpi)


zk = sorted(k for k,v in tr.items() if v[nphy:].std() > 1e-12*abs(v[nphy:].mean()) and k not in discrete and v.size<ntot*ntot)
if 'exportfracgraz' in tr:
    zk.append('exportfracgraz')
zk = sorted(set(zk))

#fig = figure('zoo')
#clf()
gg = gapgrid.gapgrid(1*[1.], len(zk)*[1.], 1.2j, .3j, 1.j, share_x='all', share_y=False, rect=[.67,0,.33,.5])

print
print 'zoo:'
for k in sorted(tr):
    if k not in zk and k not in discrete and k not in pzk:
        a = tr[k][nphy:]
        if a.size and a.max() > 0 and np.any(a != UNDEF):
            s = a.std()
            if s == 0 or s < abs(a.mean())/1e12: s = ''
            m = a.mean()
            d = denomd.get(k, 1.)
            if d != 1:
                m = '{}/{}'.format(m*d, d)
            print '{:18s}'.format(k), m, s

for i,k in enumerate(zk):
    ax = gg.axes_all[i]
    y = tr[k][nphy:]
    ax.scatter(np.r_[nphy+1:ntot+1], y, 32, c='k', edgecolor='None')
    ax.set_title(k)
#    ax.legend()
    ax.set_xticks(range(nphy+1, ntot+1))
    ax.set_xlim((nphy+.1, ntot+.9))
    ax.set_ylim((0, tr[k].max()*1.05))
    ax.tick_params('y', labelleft=True)

draw()

#fig.set_size_inches((8., 2.5*len(zk)))
#savefig('traits_zoo.png', dpi=fig.dpi)
savefig('traits.png', dpi=fig.dpi)

