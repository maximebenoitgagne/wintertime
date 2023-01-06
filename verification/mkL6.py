#!/usr/bin/env python
'''Usage: doplotts.py RUNDIR
'''
import sys
from numpy import *
from h5py import File
import gapgrid

try:
    dir, = sys.argv[1:]
except IndexError:
    sys.exit(__doc__)
    
nk = 6

with File('../../input/eccov3/grid.h5', 'r') as g:
    dr = g['drF'][:nk]
    a = g['rA']
    h = g['HFacC'][:nk]
    v = h*a*dr[:,None,None]
    w = v/v.sum()

with File(dir+'/3d.h5','r') as h:
    t = h['T'][:]
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

    pnames = h.attrs['ptracer_names'].tolist()
    names = pnames + ['PP', 'pCO2', 'pH']
    with File(dir+'/L{}.h5'.format(nk), 'w') as out:
        out.attrs['ptracer_names'] = pnames
        out['T'] = h['T'][:]
        out['iter'] = h['iter'][:]
        for k in names:
            out[k] = (h[k][:,:nk]*w).reshape(nt,v.size).sum(-1)

