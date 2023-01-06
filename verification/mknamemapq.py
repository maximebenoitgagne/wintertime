#!/usr/bin/env python
import sys
import os
from scipy.io.netcdf import netcdf_file
import numpy as np
from darwin import iofmt

def int_or_None(s):
    if len(s): return int(s)
    else: return None

k = np.s_[:1]

args = sys.argv[1:]
try:
    fname = args.pop(0)
    k = map(int_or_None, args.pop(0).split(':'))
    y = map(int_or_None, args.pop(0).split(':'))
    x = map(int_or_None, args.pop(0).split(':'))
except IndexError:
    sys.exit(__doc__)

domap = 'm' in args

refdir = '/home/jahn/darwin3/Darwin2/darwin2/verification'

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

ii = []
ff = []
for name in namel:
    f = 1.
    if name in namel0:
        ii.append(namel0.index(name))
        ff.append(f)
        continue
#    if name[-1] == 'P':
#        name = name[:-1] + 'C'
#        f = 1./120.
    if name.startswith('c'):
        name = 'BIO_1_{:03d}'.format(int(name[1:]))
        if name in namel0:
            ii.append(namel0.index(name))
            ff.append(f)
            continue
    if name.startswith('n'):
        name = 'BIO_2_{:03d}'.format(int(name[1:]))
        if name in namel0:
            ii.append(namel0.index(name))
            ff.append(f)
            continue
    if name.startswith('fe'):
        name = 'BIO_3_{:03d}'.format(int(name[2:]))
        if name in namel0:
            ii.append(namel0.index(name))
            ff.append(f)
            continue
    if name.startswith('Chl'):
        name = 'BIO_4_{:03d}'.format(int(name[3:]))
        if name in namel0:
            ii.append(namel0.index(name))
            ff.append(f)
            continue
    if name.startswith('DO') and name[2] in 'CNF':
        name = 'OM_{:d}_1'.format('CNF'.index(name[2])+1)
        if name in namel0:
            ii.append(namel0.index(name))
            ff.append(f)
            continue
    if name.startswith('PO') and name[2] in 'CNF':
        name = 'OM_{:d}_2'.format('CNF'.index(name[2])+1)
        if name in namel0:
            ii.append(namel0.index(name))
            ff.append(f)
            continue
    ii.append(0)
    ff.append(0.)
    print name

i0s = ii
f0s = ff
n = len(namel)

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
    p0 = np.array([(nc0.variables['TRAC{}'.format(iofmt(i))][...,k,y,x]*hFacC[k,y,x]).sum(-3)/hFacC[k,y,x].sum(-3) for i in range(1,max(i0s)+2)])
else:
    p0 = np.array([(nc0.variables['TRAC{}'.format(iofmt(i))][...,y,x]*hFacC[:,y,x]).sum(-1).sum(-1)/hFacC[:,y,x].sum(-1).sum(-1) for i in range(1,max(i0s)+2)])
p00 = np.array([f*p0[i] for i,f in zip(i0s, f0s)])

nc = netcdf_file(fname)
if domap:
    p = np.array([(nc.variables['TRAC{}'.format(iofmt(i))][...,k,y,x]*hFacC[k,y,x]).sum(-3)/hFacC[k,y,x].sum(-3) for i in range(1,n+1)])
else:
    p = np.array([(nc.variables['TRAC{}'.format(iofmt(i))][...,y,x]*hFacC[:,y,x]).sum(-1).sum(-1)/hFacC[:,y,x].sum(-1).sum(-1) for i in range(1,n+1)])

#grid=netcdf_file('grid.t001.nc')
#drf = grid.variables['drF'][:]

##nc0 = netcdf_file('/home/jahn/darwin3/Darwin2/darwin2/verification/{}/run/darwin.0000000000.t001.nc'.format(exp))
#p0 = np.array([nc0.variables['TRAC{}'.format(iofmt(i))][...,0,0] for i in range(1,max(i0s)+2)])
#p00 = np.array([f*p0[i] for i,f in zip(i0s, f0s)])

#nc = netcdf_file(fname) #'darwin.0000000000.t001.nc')
#p = np.array([nc.variables['TRAC{}'.format(iofmt(i))][...,0,0] for i in range(1,n+1)])

ics = [i for i,name in enumerate(namel) if name[:1]=='c']
ins = [i for i,name in enumerate(namel) if name[:1]=='n']
ifs = [i for i,name in enumerate(namel) if name[:2]=='fe']
iChls = [i for i,name in enumerate(namel) if name[:3]=='Chl']
if 's' in args:
    idx = [0,1,2,3,6,7,8,10,11,12,15]
else:
    idx = range(len(p))

nptr = len(p)

nn = n
if len(ics):
    p = np.r_[p, p[ics[:-1]].sum(0)[None], p[ics[-1:]]]
    p00 = np.r_[p00, p00[ics[:-1]].sum(0)[None], p00[ics[-1:]]]
    namel += ['phyc', 'zooc']
    idx += [nn]; nn += 1
    idx += [nn]; nn += 1
if len(ins):
    p = np.r_[p, p[ins[:-1]].sum(0)[None], p[ins[-1:]]]
    p00 = np.r_[p00, p00[ins[:-1]].sum(0)[None], p00[ins[-1:]]]
    namel += ['phyn', 'zoon']
    idx += [nn]; nn += 1
    idx += [nn]; nn += 1
if len(ifs):
    p = np.r_[p, p[ifs[:-1]].sum(0)[None], p[ifs[-1:]]]
    p00 = np.r_[p00, p00[ifs[:-1]].sum(0)[None], p00[ifs[-1:]]]
    namel += ['phyfe', 'zoofe']
    idx += [nn]; nn += 1
    idx += [nn]; nn += 1
if len(iChls):
    p = np.r_[p, p[iChls].sum(0)[None]]
    p00 = np.r_[p00, p00[iChls].sum(0)[None]]
    namel += ['chl']
    idx += [nn]; nn += 1

#idx += [namel.index('zc1'), namel.index('zc2')]

iiC = ([0,7,11,16,nptr,nptr+1])#, 6*[1.])
iiN = ([1,2,3,8,12,nptr+2,nptr+3])#, 6*[1.])
iiP = ([4,9,13])#, 6*[1.])
iiF = ([6,10,15,nptr+4,nptr+5])#, 6*[1.])

if 's' not in args:
    iiC = iiC[:-2] + range(17,17+16) + iiC[-2:]
    iiN = iiN[:-2] + range(33,33+16) + iiN[-2:]
    iiF = iiF[:-2] + range(49,49+16) + iiF[-2:]

names = np.array(namel, object)

if 'P' not in args:
    from pylab import *
    from oj.plt import *
    sobolcycle()

    figure(3)
    xm = 1.2*p.shape[1]
    if 'c' in args:
        ii = iiC
        plot(((p-p00)[ii,:,k]*drf).sum(-1).T/drf.sum(),hold=0,lw=3);xlim(0,xm);legend(names[ii],loc=0,ncol=(len(ii)+50)//50,fontsize=12);title('C-C00')
    elif 'n' in args:
        ii = iiN
        plot(((p-p00)[ii,:,k]*drf).sum(-1).T/drf.sum(),hold=0,lw=3);xlim(0,xm);legend(names[ii],loc=0,ncol=(len(ii)+50)//50,fontsize=12);title('N-N00')
    elif 'p' in args:
        ii = iiP
        plot(((p-p00)[ii,:,k]*drf).sum(-1).T/drf.sum(),hold=0,lw=3);xlim(0,xm);legend(names[ii],loc=0,ncol=(len(ii)+50)//50,fontsize=12);title('P-P00')
    elif 'f' in args:
        ii = iiF
        plot(((p-p00)[ii,:,k]*drf).sum(-1).T/drf.sum(),hold=0,lw=3);xlim(0,xm);legend(names[ii],loc=0,ncol=(len(ii)+50)//50,fontsize=12);title('Fe-Fe00')
    elif 'd' in args:
        plot(((p-p00)[idx,:,k]*drf).sum(-1).T/drf.sum(),hold=0,lw=3);xlim(0,xm);legend(names[idx],loc=0,ncol=(len(idx)+50)//50,fontsize=12);title('p-p00')
    else:
        plot(((p/p00)[idx,:,k]*drf).sum(-1).T/drf.sum(),hold=0,lw=3);xlim(0,xm);legend(names[idx],loc=0,ncol=(len(idx)+50)//50,fontsize=12);title('p/p00')
#    ylim(exp(r_[-1,1]*1e-6))
    draw()
