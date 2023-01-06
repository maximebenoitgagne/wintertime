#!/usr/bin/env python
from collections import OrderedDict
import re
from plt import *
from oj.plt import mycb
from matplotlib.ticker import MaxNLocator
from h5py import File
import gapgrid

numre = re.compile(r'([^0-9])0')

def reaspect(div, forward=False, axes=None, renderer=None):
    figW,figH = div._fig.get_size_inches()
    x,y,w,h = div.get_position_runtime(axes, renderer)
    hsizes = div.get_horizontal_sizes(renderer)
    vsizes = div.get_vertical_sizes(renderer)
    k_h = div._calc_k(hsizes, figW*w)
    k_v = div._calc_k(vsizes, figH*h)
    k = min(k_h, k_v)
    ox = div._calc_offsets(hsizes, k)
    oy = div._calc_offsets(vsizes, k)
    ww = (ox[-1] - ox[0])/figW
    hh = (oy[-1] - oy[0])/figH
    figW *= ww
    figH *= hh
    div._fig.set_size_inches(figW, figH)
    return figW, figH


rc('axes', facecolor='k')
nticks = 7
ticksteps = [1., 2., 2.5, 5., 10.]

RNCmax = .35 #40./120.
RPCmax = 1.25/120.
RFeCmax = 5e-3/120.
RSiCmax = 50./120.
RChlCmax = .5
clims = dict(
    DIC  = (1500, 2200),
    PO4  = (0.0, 2.),
    FeT  = (0.0, 0.001),
    SiO2 = (0.0, 114.),
    NH4  = (0.0, 1.),
    NO2  = (0.0, 1.),
    NO3  = (0.0, 30.),
    DOC  = (0.0, 60.),
    DOFe = (0.0, 0.0005),
    DON  = (0.0, 8.),
    DOP  = (0.0, 0.5),
    POC  = (0.0, 30.),
    POFe = (0.0, 0.0003),
    PON  = (0.0, 5.),
    POP  = (0.0, 0.3),
    POSi = (0.0, .5),
    PIC  = (0.0, 200.),
    c1   = (0.0, 1.5),
    c2   = (0.0, .2),
    c3   = (0.0, .8),
    c4   = (0.0, .8),
    c5   = (0.0, .2),
    c6   = (0.0, 1.),
    c7   = (0.0, 1.),
    c8   = (0.0, 1.),
    c9   = (0.0, 1.),
    c10  = (0.0, 1.),
    n1   = (0.0, RNCmax),
    n2   = (0.0, RNCmax),
    n3   = (0.0, RNCmax),
    n4   = (0.0, RNCmax),
    n5   = (0.0, RNCmax),
    n6   = (0.0, RNCmax),
    n7   = (0.0, RNCmax),
    n8   = (0.0, RNCmax),
    p1   = (0.0, RPCmax),
    p2   = (0.0, RPCmax),
    p3   = (0.0, RPCmax),
    p4   = (0.0, RPCmax),
    p5   = (0.0, RPCmax),
    p6   = (0.0, RPCmax),
    p7   = (0.0, RPCmax),
    p8   = (0.0, RPCmax),
    si1   = (0.0, RSiCmax),
    si2   = (0.0, RSiCmax),
    si3   = (0.0, RSiCmax),
    si4   = (0.0, RSiCmax),
    si5   = (0.0, RSiCmax),
    si6   = (0.0, RSiCmax),
    si7   = (0.0, RSiCmax),
    si8   = (0.0, RSiCmax),
    fe1   = (0.0, RFeCmax),
    fe2   = (0.0, RFeCmax),
    fe3   = (0.0, RFeCmax),
    fe4   = (0.0, RFeCmax),
    fe5   = (0.0, RFeCmax),
    fe6   = (0.0, RFeCmax),
    fe7   = (0.0, RFeCmax),
    fe8   = (0.0, RFeCmax),
    Chl1 = (0.0, RChlCmax),
    Chl2 = (0.0, RChlCmax),
    Chl3 = (0.0, RChlCmax),
    Chl4 = (0.0, RChlCmax),
    Chl5 = (0.0, RChlCmax),
    Chl6 = (0.0, RChlCmax),
    ALK  = (800, 2405.),
    O2   = (140, 380.),
    PP   = (0.0, 2./86400.),
    pH   = (7.6, 8.4),
    pCO2 = (0.0002, 0.002),
    )

lnames = dict(
    c1='Diatom 8',
    c2='Euk 8',
    c3='Syn 3',
    c4='Pro 1',
    c5='Tricho 8',
    c6='Cocco 6',
    c7='Zoo 1',
    c8='Zoo 3',
    c9='Zoo 6',
    c10='Zoo 8',
)

args = sys.argv[1:]
dir = args.pop(0)
tslc = args and args.pop(0) or ':'
tslc = eval('s_['+tslc+']')

nk = 6
kw = dict(origin='lower', interpolation='none', extent=(0,360,-80,80))

with File('../../input/eccov3/grid.h5','r') as g:
    dr = g['drF'][:nk]
    a = g['rA']
    h = g['HFacC'][:nk]
    v = h*a*dr[:,None,None]
with errstate(invalid='ignore'):
    w = v/v.sum(0)
nr,ny,nx = w.shape

with File(dir+'/3d.h5','r') as h:
    t = h['T'][:]
    t0,te,ts = tslc.indices(t.size)
    t = t[tslc]
    nt = t.size
    names = h.attrs['ptracer_names'].tolist() + ['PP', 'pCO2', 'pH']
    p = [(h[k][tslc,:nk]*w).reshape((nt*nk,ny,nx)).sum(0)/nt for k in names]

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

sobolcycle(100)

fig = figure(1)
clf()
_,cwd = os.path.split(os.getcwd())
suptitle('{}/{}   top {:.0f}m   time {}-{}'.format(cwd, dir, dr.sum(), t0, te), y=.995)
gg = gapgrid.gapgrid((3+nc)*[360.], 7*[160.], .8j, .3j, .1j, .8j, .6j, .2j,
                     direction='column', aspect=1,
                     label_mode='none', cbar_mode='each', cbar_pad=.05j, cbar_size=.12j,
                     add_all=False)
axd = {}
for i in range(nN):
    ip = r_[0,4,6,5,1,2,3][i]
    ax = gg.add_axes(i, 0)
    im = imshow(p[ip], **kw)
    clim(clims.get(names[ip], (None, None)))
    title(names[ip])
    fig.add_axes(ax.cax)
#    cb = colorbar(im, ax.cax)
#    cb.set_ticks(MaxNLocator(nticks, steps=ticksteps))
    axd[names[ip]] = ax

for i in range(nD):
    ip = iDOC + r_[0,2,3,1][i]
    ax = gg.add_axes(i, 1)
    im = imshow(p[ip], **kw)
    clim(clims.get(names[ip], (None, None)))
    title(names[ip])
    fig.add_axes(ax.cax)
#    cb = colorbar(im, ax.cax)
#    cb.set_ticks(MaxNLocator(nticks, steps=ticksteps))
    axd[names[ip]] = ax

for i in range(nP):
    ip = iPOC + r_[0,2,4,3,1,5][i]
    ax = gg.add_axes(i, 2)
    im = imshow(p[ip], **kw)
    clim(clims.get(names[ip], (None, None)))
    title(names[ip])
    fig.add_axes(ax.cax)
#    cb = colorbar(im, ax.cax)
#    cb.set_ticks(MaxNLocator(nticks, steps=ticksteps))
    axd[names[ip]] = ax

for i in range(nA):
    ip = iALK + i
    ax = gg.add_axes(-1, nA-i)
    im = imshow(p[ip], **kw)
    clim(clims.get(names[ip], (None, None)))
    title(names[ip])
    fig.add_axes(ax.cax)
#    cb = colorbar(im, ax.cax)
#    cb.set_ticks(MaxNLocator(nticks, steps=ticksteps))
    axd[names[ip]] = ax

y = 0
for i in range(nc):
    ip = ic + i
    ax = gg.add_axes(y, 3+i)
    im = imshow(p[ip], **kw)
#    mn = percentile(p[ip][isfinite(p[ip])], 10.)
    mx = percentile(p[ip][isfinite(p[ip])], 99.)
    mx = percentile(p[ip][p[ip]>=.05*mx], 99.)
    mn,mx = clims.get('c{}'.format(i+1))
    clim((0., mx))
    name = numre.sub(r'\1', names[ip])
    title(lnames.get(name, name))
    fig.add_axes(ax.cax)
#    cb = colorbar(im, ax.cax)
#    cb.set_ticks(MaxNLocator(nticks, steps=ticksteps))
    axd[names[ip]] = ax

elemorder = ['p', 'fe', 'si', 'n']
for elem in elemorder:
    if elem in elemd:
        y = 1 + elemorder.index(elem)
        for i in range(nc):
            ip = elemd[elem] + i
            ax = gg.add_axes(y, 3+i)
            im = imshow(p[ip]/p[ic+i], **kw)
            clim(clims.get(names[ip], (None, None)))
            title(elem+':c'+names[ip][1:])
            fig.add_axes(ax.cax)
#            cb = colorbar(im, ax.cax)
#            cb.set_ticks(MaxNLocator(nticks, steps=ticksteps))
            axd[names[ip]] = ax

y = 5
for i in range(nChl):
    ip = iChl + i
    ax = gg.add_axes(y, 3+i)
    im = imshow(p[ip]/p[ic+i], **kw)
    clim(clims.get(names[ip], (None, None)))
    title('Chl:c'+names[ip][3:])
    fig.add_axes(ax.cax)
#    cb = colorbar(im, ax.cax)
#    cb.set_ticks(MaxNLocator(nticks, steps=ticksteps))
    axd[names[ip]] = ax

for i in range(npp):
    ip = iPP + i
    ax = gg.add_axes(-1, 3+i)
    im = imshow(p[ip], **kw)
    clim(clims.get(names[ip], (None, None)))
    title(names[ip])
    fig.add_axes(ax.cax)
#    cb = colorbar(im, ax.cax)
#    cb.set_ticks(MaxNLocator(nticks, steps=ticksteps))
    axd[names[ip]] = ax

for ax in gg.axes_all:
    if ax.images:
        cb = colorbar(ax.images[0], ax.cax)
        cb.set_ticks(MaxNLocator(nticks, steps=ticksteps))

gg.set(xlim=(0,360))
gg.set(ylim=(-80,80))
show()

fw,fh = fig.get_size_inches()
print fw, fh
fig.set_size_inches((43.75, 11.85))
reaspect(gg.get_divider())
dir,subdir = os.path.split(dir)
savefig(dir + '/maps_' + subdir + '_{}-{}.png'.format(t0, te), dpi=fig.dpi)

fig.set_size_inches((fw,fh))
draw()

