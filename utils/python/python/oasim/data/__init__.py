__all__ = ['atmo', 'sl', 'water', 'make_WtouEins', 'WtouEins']

import os
import numpy as np

pkgdir,_ = os.path.split(__file__)

fname = os.path.join(pkgdir, 'atmo25b.dat')
type = [('lam','i')] + [(k,'d') for k in ['Fobar', 'thray', 'aoz', 'awv', 'ao', 'aco2']]
atmo = np.loadtxt(fname, type, skiprows=4).view(np.recarray)

fname = os.path.join(pkgdir, 'abw25b.dat')
type = [('lam','i')] + [(k,'d') for k in ['a', 'b']]
water = np.loadtxt(fname, type, skiprows=6).view(np.recarray)

slingofile = os.path.join(pkgdir, 'slingo.dat')
slingotype = [(k,'d') for k in ['rnl1','rnl2', 'a', 'b', 'e', 'f', 'c', 'd']]
sl = np.loadtxt(slingofile, slingotype, skiprows=3).view(np.recarray)

def make_WtouEins(lam):
    h = 6.6256E-34   # Plancks constant J sec
    c = 2.998E8      # speed of light m/sec
    hc = 1.0/(h*c)
    oavo = 1.0/6.023E23   #  1/Avogadros number
    hcoavo = hc*oavo
    rlamm = np.asfarray(lam)*1.0E-9   # lambda in m
    WtouEins = 1.0E6*rlamm*hcoavo     # Watts to quanta conversion
    return WtouEins

WtouEins = make_WtouEins(atmo.lam)

