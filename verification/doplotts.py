#!/usr/bin/env python
'''Usage: doplotts.py [options] RUNDIR [REFDIR] [RANGEDIRS...]

Options:
    -t SLICE  time records to plot
'''
from docopt import docopt
from collections import OrderedDict
from plt import *
from h5py import File
import gapgrid
import nml

args = docopt(__doc__)
dir = args['RUNDIR']
refdir = args['REFDIR']
refdirs = [d for d in args['RANGEDIRS'] if d not in [dir, refdir]]
tslc = args['-t'] or ':'
havet = tslc != ':'
tslc = eval('s_['+tslc+']')

mpl.rc_file_defaults()

nk = 6
kw = dict(lw=1.)
legendkw = dict(fontsize=10, labelspacing=0.2, 
         frameon=False,
#         borderpad=0, borderaxespad=.5,
         )

traits = nml.NmlFile(dir + '/gud_traits.nml').merge()

with File('../../input/eccov3/grid.h5', 'r') as g:
    dr = g['drF'][:nk]
    a = g['rA']
    h = g['HFacC'][:nk]
    v = h*a*dr[:,None,None]
    w = v/v.sum()

fname = dir+'/L{}.h5'.format(nk)
haveL = os.path.exists(fname)
if not haveL:
    fname = dir+'/3d.h5'
print fname
with File(fname, 'r') as h:
    t = h['T']
    nttot = t.size
    t = t[tslc]
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

    names = h.attrs['ptracer_names'].tolist() + ['PP', 'pCO2', 'pH']
    if haveL:
        p = array([h[k][tslc] for k in names])
    else:
        p = array([(h[k][tslc,:nk]*w).reshape(nt,v.size).sum(-1) for k in names])

# definite limits
inds = tslc.indices(nttot)
tslc = slice(*inds)
if inds[-1] == 1:
    inds = inds[:-1]
tname = '_'+'-'.join(map(str, inds)) if havet else ''

iDOC = names.index('DOC')
iPOC = names.index('POC')
iALK = names.index('ALK')
ic = names.index('O2') + 1
elemd = OrderedDict((k,names.index(k+'1') if k+'1' in names else names.index(k+'01')) for k in ['n', 'p', 'si', 'fe'] if k+'1' in names or k+'01' in names)
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

p0 = None
mins = {}
maxs = {}
if refdir:
    for mydir in [refdir] + refdirs:
        fname = mydir+'/L{}.h5'.format(nk)
        haveL = os.path.exists(fname)
        if not haveL:
            fname = mydir+'/3d.h5'
        if os.path.exists(fname):
            print fname
            with File(fname, 'r') as h0:
                p0names = dict((k,k) for k in names)
                for i in range(nc):
                    name = names[ic+i]
                    name1 = 'c{}'.format(i+1)
                    name2 = 'c{:02d}'.format(i+1)
                    if name not in h0:
                        if name1 in h0:
                            p0names[name] = name1
                        elif name2 in h0:
                            p0names[name] = name2
                        else:
                            del p0names[name]
                        print name, p0names.get(name, '-')

                if haveL:
                    pref = dict((k, h0[kk][tslc]) for k in p0names for kk in [p0names[k]] if kk in h0)
                else:
                    pref = dict((k, (h0[kk][tslc,:nk]*w).reshape(nt,v.size).sum(-1)) for k in p0names
                                                                                    for kk in [p0names[k]]
                                                                                    if kk in h0)
                if p0 is None:
                    p0 = pref
                    tslc = s_[:]
                else:
                    for k in pref:
                        mins[k] = minimum(mins.get(k, 1e38), pref[k].min())
                        maxs[k] = maximum(maxs.get(k, -1e38), pref[k].max())
                    del pref
        else:
            print '-'+mydir

cols = sobolcycle(100)

fig = figure(1)
clf()
_,cwd = os.path.split(os.getcwd())
suptitle('top {depth:.0f}m   {}/{} vs {}'.format(cwd, dir, refdir, depth=dr.sum()), y=.995)
gg = gapgrid.gapgrid(5*[1.], 7*[1.], 1j, .25j, 1j, .25j, direction='column', add_all=False)
for i in range(nN):
    ip = r_[0,4,6,5,1,2,3][i]
    gg.add_axes(i, 0)
    plot(t, p[ip], label=names[ip], **kw)
    if refdir: plot(t, p0[names[ip]], ':')
    plot(t[:1], [mins.get(names[ip], nan)], '-')
    plot(t[:1], [maxs.get(names[ip], nan)], '-')

for i in range(nD):
    ip = iDOC + r_[0,2,3,1][i]
    gg.add_axes(i, 1)
    plot(t, p[ip], label=names[ip], **kw)
    if refdir: plot(t, p0[names[ip]], ':')
    plot(t[:1], [mins.get(names[ip], nan)], '-')
    plot(t[:1], [maxs.get(names[ip], nan)], '-')

for i in range(nP):
    ip = iPOC + r_[0,2,4,3,1,5][i]
    gg.add_axes(i, 2)
    plot(t, p[ip], label=names[ip], **kw)
    if refdir: plot(t, p0[names[ip]], ':')
    plot(t[:1], [mins.get(names[ip], nan)], '-')
    plot(t[:1], [maxs.get(names[ip], nan)], '-')

for i in range(nA):
    ip = iALK + i
    gg.add_axes(4+i, 1)
    plot(t, p[ip], label=names[ip], **kw)
    if refdir: plot(t, p0[names[ip]], ':')
    plot(t[:1], [mins.get(names[ip], nan)], '-')
    plot(t[:1], [maxs.get(names[ip], nan)], '-')

for i in range(1, npp):
    ip = iPP + i
    gg.add_axes(-1, npp-i)
    plot(t, p[ip], label=names[ip], **kw)
    if refdir: plot(t, p0[names[ip]], ':')
    plot(t[:1], [mins.get(names[ip], nan)], '-')
    plot(t[:1], [maxs.get(names[ip], nan)], '-')

vol = 34180464067389348.0
Pg = 1E-3*12E-15*365.25*86400*vol
ip = iPP
gg.add_axes(-1, -1)
plot(t, p[ip]*Pg, label=names[ip], **kw)
if refdir: plot(t, p0[names[ip]]*Pg, ':')
plot(t[:1], [mins.get(names[ip], nan)*Pg], '-')
plot(t[:1], [maxs.get(names[ip], nan)*Pg], '-')

for x,(i0,ie) in enumerate([(0,nChl), (nChl,nc)]):
    y = 0
    gg.add_axes(y, 3+x)
    for i in range(i0, ie):
        ip = ic + i
        name = names[ip]
        plot(t, p[ip], color=cols[i], label=name, **kw)
        if refdir and name in p0:
            plot(t, p0[name], color=cols[i], linestyle='--')
        plot(t[:1], [mins.get(name, nan)], '-')
        plot(t[:1], [maxs.get(name, nan)], '-')
    ylim(0., None)

    for elem in ['p', 'fe', 'si', 'n']:
        y += 1
        if elem in elemd:
            gg.add_axes(y, 3+x)
            for i in range(i0,ie):
                ip = elemd[elem] + i
                name = names[ip]
                plot(t, p[ip], color=cols[i], label=name, **kw)
                if refdir and name in p0:
                    plot(t, p0[names[ip]], color=cols[i], linestyle='--')
                plot(t[:1], [mins.get(name, nan)], '-')
                plot(t[:1], [maxs.get(name, nan)], '-')
            ylim(0., None)

    # log plot
    if x < 1:
        y += 1
        if y < 6:
            gg.add_axes(y, 3+x)
            for i in range(i0,ie):
                ip = ic + i
                name = names[ip]
                semilogy(t, p[ip], color=cols[i], label=name, **kw)
                if refdir and name in p0:
                    plot(t, p0[name], color=cols[i], linestyle='--')
                plot(t[:1], [mins.get(name, nan)], '-')
                plot(t[:1], [maxs.get(name, nan)], '-')
                ylim(1e-5, 1e1)

    for elem in ['p', 'fe', 'si', 'n']:
        if elem in elemd:
            y += 1
            if y < 6:
                gg.add_axes(y, 3+x)
                kw2 = dict(kw)
                for i in range(i0,ie):
                    ip = elemd[elem] + i
                    name = names[ip]
    #                kw2['lw'] = 2 if i >= nChl else kw['lw']
                    plot(t, p[ip]/p[ic+i], color=cols[i], label='Q'+name, **kw2)
                    if refdir and name in p0:
                        plot(t, p0[name]/p0[names[ic+i]], color=cols[i], linestyle='--')
                    plot(t[:1], [mins.get(name, nan)], '-')
                    plot(t[:1], [maxs.get(name, nan)], '-')
                mn = traits['q'+elem+'min']
                mx = traits['q'+elem+'max']
                s = s_[i0:ie]
                hlines(mn[s], t0, t1, colors=cols[s], linestyle='dotted', **kw2)
                hlines(mx[s], t0, t1, colors=cols[s], linestyle='dotted', **kw2)
                a,b = mn[s].min(),mx[s].max()
                ylim(max(0., a*1.03-b*.03), b*1.03-a*.03)

    gg.add_axes(6, 3)
    for i in range(i0,nChl):
        ip = iChl + i
        name = names[ip]
        plot(t, p[ip], label=name, **kw)
        if refdir and name in p0:
            plot(t, p0[name], color=cols[i], linestyle='--')
        plot(t[:1], [mins.get(name, nan)], '-')
        plot(t[:1], [maxs.get(name, nan)], '-')

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
fig.set_size_inches((23.75,13.))
if refdir:
    comps = refdir.split(os.sep)
    ref = '_'.join([''] + comps[-3:])
else:
    ref = ''
if refdirs:
    extra = '+'.join(''.join(d.split(os.sep)[-3:]) for d in refdirs)
    ref = ref + '+' + extra.replace('rrun10y', '').replace('quota_eccov3_6+4_','')
dir,subdir = os.path.split(dir)
savefig(dir + '/ts' + tname + '_' + subdir +ref+'.png', dpi=fig.dpi)

fig.set_size_inches((fw,fh))
draw()

