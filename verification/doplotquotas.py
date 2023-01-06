#!/usr/bin/env python
'''Usage: doplotts.py [options] RUNDIR [REFDIR]

Options:
    -c    plot biomass too
'''
from docopt import docopt
from collections import OrderedDict
from plt import *
from h5py import File
import gapgrid
import nml
import util

args = docopt(__doc__)
dir = args['RUNDIR']
refdir = args['REFDIR']
dobiomass = args['-c']

mpl.rc_file_defaults()

nk = 6
kw = dict(lw=1.)
legendkw = dict(fontsize=10, labelspacing=0.2, 
         frameon=False,
#         borderpad=0, borderaxespad=.5,
         )

traits = nml.NmlFile(dir + '/gud_traits.nml').merge()

w,depth = util.getweight(nk)
p,t,names = util.getglobal(dir, nk)
nt = t.size
dt = diff(t)[0]

u = 86400.
xt = None
xl = 'days'
if t.max() - t.min() > 360*u:
    u *= 30.
    xt = r_[t.min()-dt:t.max()+dt:6*u]/u
    xl = 'months'
    if t.max() - t.min() > 36*u:
        u *= 12.
        xt = None
        xl = 'years'
t /= u
t0 = t.min() - dt/u
t1 = t.max()
t -= .5*dt/u

qelems = ['n', 'p', 'si', 'fe','Chl']
iDOC = names.index('DOC')
iPOC = names.index('POC')
iALK = names.index('ALK')
ic = names.index('O2') + 1
elemd = OrderedDict((k,names.index(k+'1') if k+'1' in names else names.index(k+'01')) for k in qelems if k+'1' in names or k+'01' in names)
iChl = names.index('Chl1')
iPP  = names.index('PP')
nN = iDOC
nD = iPOC - iDOC
nP = iALK - iPOC
nA = ic - iALK
nc = iChl - ic
if elemd:
    nc = elemd.values()[0] - ic
nChl = iPP - iChl
npp = len(names) - iPP
nq = len(elemd)

if refdir:
    p0 = util.getref(refdir, names, nk, nt)

cols = sobolcycle(100)

fig = figure(1)
clf()
_,cwd = os.path.split(os.getcwd())
suptitle('top {depth:.0f}m   {}/{} vs {}'.format(cwd, dir, refdir, depth=depth), y=.995)
nr = len(qelems)+dobiomass
gg = gapgrid.gapgrid(nc*[1.], nr*[1.], .6j, .25j, .6j, .25j, direction='column', add_all=False)
y0 = 0
if dobiomass:
    for i in range(nc):
        gg.add_axes(y0, i)
        ip = ic + i
        name = names[ip]
        plot(t, p[ip], color=cols[i], label=name, **kw)
        if refdir and name in p0:
            plot(t, p0[name], color=cols[i], linestyle='--')
        ylim(0, None)
    y0 += 1

for y,elem in enumerate(qelems):
    if elem in elemd:
        for i in range(nc):
            if elem != 'Chl' or i < nChl:
                gg.add_axes(y0+y, i)
                ip = elemd[elem] + i
                name = names[ip]
                plot(t, p[ip]/p[ic+i], color=cols[i], label='Q'+name, **kw)
                if refdir and name in p0:
                    plot(t, p0[name]/p0[names[ic+i]], color=cols[i], linestyle='--')
                if elem == 'Chl':
                    mn = traits['chl2cmin'][i]
                    mx = traits['chl2cmax'][i]
                else:
                    mn = traits['q'+elem+'min'][i]
                    mx = traits['q'+elem+'max'][i]
                hlines([mn, mx], t0, t1, colors=cols[i], linestyle='dotted', **kw)
                a,b = mn,mx
                ylim(max(0., a*1.05-b*.05), b*1.05-a*.05)

if xt is not None:
    gg.set(xticks=xt)
gg.set(xlabel=xl)
gg.set(xlim=(t0, t1))

for ax in gg.axes_all:
    if ax in fig.axes:
        ax.legend(**legendkw)
        if ax.legend_ is not None:
            lhs = ax.legend_.get_lines()
            lw = 2 if len(lhs) > 1 else 0
            setp(lhs, lw=lw)

show()

fw,fh = fig.get_size_inches()
fig.set_size_inches((40.,9.7))
if refdir:
    comps = refdir.split(os.sep)
    ref = '_'.join([''] + comps[-3:])
else:
    ref = ''
dir,subdir = os.path.split(dir)
savefig(dir + '/quotas_' + subdir +ref+'.png', dpi=fig.dpi)

fig.set_size_inches((fw,fh))
draw()

