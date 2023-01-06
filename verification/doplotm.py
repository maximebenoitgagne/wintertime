#!/usr/bin/env python
'''Usage: mknamemap.py DARWINOUT k y x [mtadcnpf] [ini]

  m map
  t tmap
  a all
  d differences
  c C
  n N
  p P
  f Fe
  ini ignore differences at t=0

Run from run/ directory
'''
import sys
import os
from scipy.io.netcdf import netcdf_file
import MITgcmutils as mit
import numpy as np
from darwin import iofmt

def int_or_None(s):
    if len(s): return int(s)
    else: return None

y,x = 20,0
k = np.s_[:1]

args = sys.argv[1:]
try:
    fname = args.pop(0)
    k = map(int_or_None, args.pop(0).split(':'))
    y = map(int_or_None, args.pop(0).split(':'))
    x = map(int_or_None, args.pop(0).split(':'))
except IndexError:
    sys.exit(__doc__)

refdir = '/home/jahn/gud/darwin2sub/verification'

if len(k) == 1: k = [k[0], k[0]+1]
if len(y) == 1: y = [y[0], y[0]+1]
if len(x) == 1: x = [x[0], x[0]+1]

k = slice(*k)
y = slice(*y)
x = slice(*x)

wd = os.getcwd()
d,_ = os.path.split(wd)
d,exp = os.path.split(d)

names0 = np.genfromtxt('{}/{}/names'.format(refdir, exp), object)
names = np.genfromtxt('../names',object)
namel0 = names0.tolist()
namel = names.tolist()

ndigphy = 'Phy1' in names0 and 1 or 2

ii = []
ff = []
nphy = 0
for name in namel:
    if name in namel0:
        ii.append(namel0.index(name))
        ff.append(1.)
        continue
    if name[-1] == 'C':
        name = name[:-1] + 'P'
        if name in namel0:
            ii.append(namel0.index(name))
            ff.append(120.)
            continue
    if name.startswith('c'):
        name0 = 'Phy{:0{}d}'.format(int(name[1:]), ndigphy)
        if name0 in namel0:
            ii.append(namel0.index(name0))
            ff.append(120.)
            nphy += 1
            continue
        name0 = 'ZOO{:d}P'.format(int(name[1:])-nphy)
        if name0 in namel0:
            ii.append(namel0.index(name0))
            ff.append(120.)
            print name, name0, namel0.index(name0)
            continue
    if name.startswith('zc'):
        name0 = 'ZOOC' + str(int(name[2:]))
        if name0 in namel0:
            ii.append(namel0.index(name0))
            ff.append(1.)
            continue
        name = 'ZOO' + str(int(name[2:])) + 'P'
        if name in namel0:
            ii.append(namel0.index(name))
            ff.append(120.)
            continue
    if name.startswith('Chl'):
        name0 = 'Chl{:0{}d}'.format(int(name[3:]), ndigphy)
        if name0 in namel0:
            ii.append(namel0.index(name0))
            ff.append(1.)
            print name, name0, namel0.index(name0)
            continue
    ii.append(0)
    ff.append(0.)
    print name

i0s = ii
f0s = ff
n = len(namel)

domap = 'm' in args
if domap:
    i = args.index('m')
    args.pop(i)
    imap = args.pop(i)

if 't' in args:
    i = args.index('t')
    args.pop(i)
    tmap = int(args.pop(i))
else:
    tmap = -1

dir,name = os.path.split(fname)
_,ext = os.path.splitext(name)
_,ext2 = os.path.splitext(_)
ext = ext2 + ext

gridf = netcdf_file('grid{}'.format(ext))
#gridf = mit.mnc_files('grid*.nc')
drf = gridf.variables['drF'][:]
hFacC = gridf.variables['HFacC'][:]
AC = gridf.variables['rA'][:]
gridf.close()

nc0 = netcdf_file('{}/{}/run/{}'.format(refdir, exp, fname))
if domap:
#    p0 = np.array([(nc0.variables['TRAC{}'.format(iofmt(i))][...,k,y,x]*hFacC[k,y,x]).sum(-3)/hFacC[k,y,x].sum(-3) for i in range(1,max(i0s)+2)])
    p0 = np.array([(nc0.variables[namel0[i-1]][...,k,y,x]*hFacC[k,y,x]).sum(-3)/hFacC[k,y,x].sum(-3) for i in range(1,max(i0s)+2)])
else:
#    p0 = np.array([(nc0.variables['TRAC{}'.format(iofmt(i))][...,y,x]*hFacC[:,y,x]).sum(-1).sum(-1)/hFacC[:,y,x].sum(-1).sum(-1) for i in range(1,max(i0s)+2)])
    p0 = np.array([(nc0.variables[namel0[i-1]][...,y,x]*hFacC[:,y,x]).sum(-1).sum(-1)/hFacC[:,y,x].sum(-1).sum(-1) for i in range(1,max(i0s)+2)])
p00 = np.array([f*p0[i] for i,f in zip(i0s, f0s)])

nc = netcdf_file(fname)
if domap:
#    p = np.array([(nc.variables['TRAC{}'.format(iofmt(i))][...,k,y,x]*hFacC[k,y,x]).sum(-3)/hFacC[k,y,x].sum(-3) for i in range(1,n+1)])
    p = np.array([(nc.variables[namel[i-1]][...,k,y,x]*hFacC[k,y,x]).sum(-3)/hFacC[k,y,x].sum(-3) for i in range(1,n+1)])
    drf = drf[k]
else:
#    p = np.array([(nc.variables['TRAC{}'.format(iofmt(i))][...,y,x]*hFacC[:,y,x]).sum(-1).sum(-1)/hFacC[:,y,x].sum(-1).sum(-1) for i in range(1,n+1)])
    p = np.array([(nc.variables[namel[i-1]][...,y,x]*hFacC[:,y,x]).sum(-1).sum(-1)/hFacC[:,y,x].sum(-1).sum(-1) for i in range(1,n+1)])
ics = [i for i,name in enumerate(namel) if name[:1]=='c']
iChls = [i for i,name in enumerate(namel) if name[:3]=='Chl']
if iChls:
    ics = ics[:len(iChls)]
if 'DIC' in names0:
    idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] + range(16, ics[0])
else:
    idx = [1,2,3,4,5,6,8,9,10,12,13,14,15] + range(16, ics[0])

if 'a' in args:
    idx += range(16, n)

RPC = 1./120.
RNC = 16./120.
RFeC = RPC/1e3
RNCphy = np.ones((len(ics),))*RNC
if len(ics) == 9:
    RNCphy = np.r_[5*[RNC]+2*[1./3]+2*[RNC]]

if len(ics):
    p = np.r_[p, p[ics].sum(0)[None], (p[ics]*RNCphy[np.s_[:,]+(p.ndim-1)*np.s_[None,]]).sum(0)[None]]
    p00 = np.r_[p00, p00[ics].sum(0)[None], (p00[ics]*RNCphy[np.s_[:,]+(p.ndim-1)*np.s_[None,]]).sum(0)[None]]
    namel += ['phy', 'phyN']
    idx += [n,n+1]
if len(iChls):
    p = np.r_[p, p[iChls].sum(0)[None]]
    p00 = np.r_[p00, p00[iChls].sum(0)[None]]
    namel += ['chl']
    idx += [n+2]

if 'ini' in args:
    p *= p00[:,0,None]/p[:,0,None]

nchl = len(iChls)
zc1 = 'c' + str(nchl+1)
zc2 = 'c' + str(nchl+2)
idx += [namel.index(zc1), namel.index(zc2)]

iiC = [0,7,11,16,n]+idx[-2:], [1.,1.,1.,1.,1.,1.,1.]
iiN = [1,2,3,8,12,n+1]+idx[-2:], [1.,1.,1.,1.,1.,1.,RNC,RNC]
iiP = [4,9,13,n]+idx[-2:], [1.,1.,1.,RPC,RPC,RPC]
iiFe = [6,10,15,n]+idx[-2:], [1.,1.,1.,RFeC,RFeC,RFeC]

ii,ff=iiN
p00int = (p00[ii,:,:]*drf).sum(-1)/drf.sum()
totN0 = (np.r_[ff].T*p00int.T).T.sum(0)
pint = (p[ii,:,:]*drf).sum(-1)/drf.sum()
totN = (np.r_[ff].T*pint.T).T.sum(0)

#idx = range(len(namel))

names = np.array(namel, object)

#from oj.plt import *
#figure(3);plot((p/p00)[idx,:,0].T,hold=0,lw=3);xlim(0,5);legend(names[idx],loc=0,ncol=(len(idx)+50)//50,fontsize=10)
#ylim(exp(r_[-1,1]*1e-6))

lw = 4

if True:
    from pylab import *
    from oj.plt import *
    sobolcycle()

    disp = os.environ.get('DISPLAY', ':0.0')
    host,dsp = disp.split(':')
    if float(dsp) > 9:
        client = os.environ.get('SSH_CLIENT', 'localhost')
        if client.startswith('174.'):
            rc('legend', labelspacing=0.1) #, fontsize='x-small', markerscale=1.5)
#            rc('lines', linewidth=4)
            lw = 5

    legparm = dict(loc=0, prop=dict(size=12))

    fmt = lambda x: x is not None and str(x) or ''
    slices = '[' + ', '.join(':'.join(map(fmt, [xx.start,xx.stop,xx.step])) for xx in [k,y,x]) + ']'

    if domap:
        import oj.plt
        try:
            imap = int(imap)
        except ValueError:
            imap = namel.index(imap)
        figure(2)
        if 'd' in args:
            a = (p-p00)[imap, tmap]
            ttl = '{0} - {0}_0'.format(names[imap])
        else:
            a = (p/p00)[imap, tmap]
            ttl = '{0}/{0}_0'.format(names[imap])
        oj.plt.myimshow(a, cb=1)
        plt.title('{}   {}'.format(ttl, slices))
    else:
        figure(3)
        xm = 1.2*p.shape[1]
        if 'c' in args:
            ii,ff = iiC
            a = r_[ff]*((p-p00)[ii,:,k]*drf[k]).sum(-1).T/drf[k].sum()
            ttl = 'C-C00'
        elif 'n' in args:
            ii,ff = iiN
            a = r_[ff]*((p-p00)[ii,:,k]*drf[k]).sum(-1).T/drf[k].sum()
            ttl = 'N-N00'
        elif 'p' in args:
            ii,ff = iiP
            a = r_[ff]*((p-p00)[ii,:,k]*drf[k]).sum(-1).T/drf[k].sum()
            ttl = 'P-P00'
        elif 'f' in args:
            ii,ff = iiFe
            a = r_[ff]*((p-p00)[ii,:,k]*drf[k]).sum(-1).T/drf[k].sum()
            ttl = 'Fe-Fe00'
        elif 'd' in args:
            ii = idx
            a = ((p-p00)[ii,:,k]*drf[k]).sum(-1).T/drf[k].sum()
            ttl = 'p-p00'
        else:
            ii = idx
            a = (p[ii,:,k]*drf[k]).sum(-1).T/(p00[ii,:,k]*drf[k]).sum(-1).T
            ttl = 'p/p00'
            if np.nanmax(a)/np.nanmin(a) < 1+1e-13:
                a -= 1
                ttl += '-1'
        plot(a, hold=0, lw=lw)
        xlim(0, xm)
        legend(names[ii], ncol=(len(idx)+50)//50, **legparm)
        title('{}   {}'.format(ttl, slices))
    #    ylim(exp(r_[-1,1]*1e-6))
        print 'nan:', ' '.join(names[ii[i]] for i in range(len(ii)) if np.any(np.isnan(a[:,i])))
        
    draw()
    show()

nc.close()
nc0.close()
