import sys
import os
import numpy as np

esdbounds = np.r_[  2.5,     20.,    250.,   1500.,]
sizeclasses = ['pico', 'nano', 'micro', 'meso', 'macro']

class RunInfo(object):
    def __init__(self, rundir='.'):
        import fortran
        import nml
        self.dir = rundir
        self.names = [s.strip("'") for s in fortran.readnmlparam(rundir + '/data.ptracers','ptracers_names')]
        grpnames = fortran.readnmlparam(rundir + '/data.gud','grp_names')
        if ',' in grpnames:
            grpnames = grpnames.strip(', ').split(',')
        self.grpnames = [s.strip("' ") for s in grpnames]
        self.ic = self.names.index('c01')
        for i in range(len(self.names)-self.ic):
            if self.names[self.ic+i][0] != 'c':
                break
        self.nplk = i
        self.nphy = int(self.names[-1][3:])
        self.nzoo = self.nplk - self.nphy

        startdir = os.path.join(rundir, 'start')
        if not os.path.exists(os.path.join(startdir, 'gud_traits.nml')):
            startdir = rundir
        self.tr = nml.NmlFile(os.path.join(startdir, 'gud_traits.nml')).merge()
        self.params = nml.NmlFile(os.path.join(startdir, 'gud_params.nml')).merge()
        if os.path.exists(os.path.join(startdir, 'gud_radtrans.nml')):
            self.rtparams = nml.NmlFile(os.path.join(startdir, 'gud_radtrans.nml')).merge()
        self.grp = self.tr['group'] - 1
        self.vol = self.tr['biovol']
        # vol = (4*pi/3)*(dm/2)**3 = (pi/6)*dm**3
        # dm = (6*vol/np.pi)**(1./3)
        self.esd = (6*self.vol/np.pi)**(1./3)
        self.dlv = np.median(np.diff(sorted(set(np.log(self.vol)))))
        self.ivol = np.round(np.log(self.vol/self.vol.min())/self.dlv).astype(int)
        self.ngrp = np.max(self.grp) + 1
        self.nvol = np.max(self.ivol) + 1


def groups(p, info):
    assert p.shape[0] in [info.nplk, info.nphy]
    grp = np.zeros((info.ngrp,) + p.shape[1:], p.dtype)
    n = np.zeros((info.ngrp,), int)
    for g in range(info.ngrp):
        for i in range(p.shape[0]):
            if info.grp[i] == g:
                grp[g] += p[i]
                n[g] += 1

        sz = np.sum(info.grp == g)
        if n[g] != sz:
            if n[g] == 0:
                sys.stderr.write('Group {} {} not present.\n'.format(g, info.grpnames[g]))
            else:
                raise ValueError(
                'Group {} has {} instead of {} members, have {} plankton need {} or {}\n'.format(
                           g, n[g], sz, p.shape[0], info.nplk, info.nphy))

    return grp


def sizeclasses(p, info, maxesds=None):
    assert p.shape[0] in [info.nplk, info.nphy]
    if maxesds is None:
        maxesds = sorted(info.esd)[:-1]
        assert len(maxesd) == info.nvol - 1

    ncls = len(maxesds) + 1

    a = np.zeros((ncls,) + p.shape[1:], p.dtype)
    n = np.zeros((ncls,), int)
    for iv in range(ncls):
        for i in range(p.shape[0]):
            icls = np.searchsorted(maxesds, info.esd[i])
            if icls == iv:
                a[iv] += p[i]
                n[iv] += 1

    return a

import numpy as np
from numpy import sin, cos, pi, arcsin, arccos
import string

digits = '0123456789abcdefghijklmnopqrstuvxyzABCDEFGHIJKLMNOPQRSTUVXYZ'

def ionum(s):
    try:
        i = int(s)
    except ValueError:
        try:
            i0 = int(s[0])
        except ValueError:
            i0 = digits[10:].index(s[0])
            i1 = digits.index(s[1])
            i = 620 + i0*62 + i1
        else:
            i1 = digits[10:].index(s[1])
            i = 100 + i0*52 + i1
    return i


def iofmt(i):
    if i < 100:
        return '{0:02d}'.format(i)
    elif i < 620:
        a,b = divmod(i-100, 52)
        return '{0}{1}'.format(a, digits[10+b])
    else:
        a,b = divmod(i-620, 62)
        return '{0}{1}'.format(digits[10+a], digits[b])


class Diversities(object):
    '''
    Diver1 = # types > 1e-6 uM C
    Diver2 = # types > 1e-3 * total phyto biomass
    Diver3 = # types comprizing .999 of total phyto biomass
    Diver4 = # types > 1e-5 * biomass of most abundant type
    Shannon = Shannon index (of types >= 1e-10 uM C)
    Simpson = Simpson index (of types >= 1e-10 uM C)
    RichnessAbs = # types >= 1e-6 uM C
    RichnessRel = # types >= 1e-3 * total phyto biomass
    EvennessAbs = Shannon/ln(RichnessAbs)
    EvennessRel = Shannon/ln(RichnessRel)
    EvennessSimpsonAbs = Simpson/RichnessAbs

    only types >= 1e-10 uM C are taken into account for relative thresholds

    '''
    def __init__(self):
        self.thresh0 = 1e-10   # uM C  thresh for sum_i P
        self.thresh1 = 1e-6    # uM C  thresh for P
        self.thresh2 = 1e-3    #       thresh for P / sum_i P
        self.thresh3 = .999    #       biomass fraction
        self.thresh4 = 1e-5    #       thresh for P / max_i P
        self.threshsh = 1e-10  # uM C  just for numerics
        self.threshrich = 1e-6 # uM C  abs.threshold for P
        self.threshrichrel = 1e-3 #    thresh for P / sum_i P

    def calc(self, phy, rfac=1., rfacsh=1., rfacrich=None):
        ''' overwrites phy!

        absolute thresholds are multiplied by rfac
        (rfacsh for Shannon and rfacrich for Richness)
        '''
        if rfacrich is None: rfacrich = rfac
        nphy = phy.shape[0]
        shape = phy.shape[1:]

        phy[phy<0] = 0.
        total = phy.sum(0)
        mx = phy.max(0)

        diver1 = np.zeros(shape, int)
        diver2 = np.zeros(shape, int)
    #    diver3 = np.zeros(shape, int)
        diver4 = np.zeros(shape, int)
        shannon = np.zeros(shape)
        simpson = np.zeros(shape)
        richnessabs = np.zeros(shape, int)
        richnessrel = np.zeros(shape, int)
    #    present = np.zeros(shape, bool)

        valid = total >= self.thresh0*rfac
        for i in range(nphy):
    #        present[...] = 0
    #        for k in range(kmax):
    #            present |= p[i,k,:,:] > self.thresh1
    #        diver1[:,:] += present
            diver1[...] += phy[i] > self.thresh1*rfac
            diver2[...] += valid & (phy[i] > self.thresh2*total)
            diver4[...] += valid & (phy[i] > self.thresh4*mx)

        diver3 = self.calcdiver3(phy.copy())

        # for consistency recompute total with self.threshold
        phy[phy<self.threshsh*rfacsh] = 0
        total = phy.sum(0)
        for i in range(nphy):
            valid = phy[i] >= self.threshsh*rfacsh
            tmp = phy[i,valid]/total[valid]
            shannon[valid] += tmp*np.log(tmp)
            simpson[valid] += tmp*tmp
            richnessrel[valid] += tmp >= self.threshrichrel
            valid = phy[i] >= self.threshrich*rfacrich
            richnessabs[valid] += 1

        shannon *= -1
        valid = simpson != 0
        simpson[valid] = 1./simpson[valid]

    #    biotot = np.zeros(shape)
    #    for i2 in range(nphy):
    #        imx = np.argmax(phy,axis=0)
    #        mx = np.max(phy,axis=0)
    #        diver3[biotot < self.thresh3*total] += 1
    #        biotot += mx
    #        for ind in np.ndindex(shape):
    #            phy[(imx[ind],)+ind] = 0.

        d = dict(
            Diver1=diver1,
            Diver2=diver2,
            Diver3=diver3,
            Diver4=diver4,
            Shannon=shannon,
            Simpson=simpson,
            RichnessAbs=richnessabs,
            RichnessRel=richnessrel,
            )

        valid = richnessabs != 0
        tmp = simpson.copy(); tmp[valid] /= richnessabs[valid]; d['EvennessSimpsonAbs'] = tmp
        valid = richnessabs > 1
        tmp = shannon.copy(); tmp[valid] /= np.log(richnessabs[valid]); d['EvennessAbs'] = tmp
        valid = richnessrel > 1
        tmp = shannon.copy(); tmp[valid] /= np.log(richnessrel[valid]); d['EvennessRel'] = tmp

        return d

    names = set('Diver1 Diver2 Diver3 Diver4 Shannon Simpson RichnessAbs RichnessRel EvennessSimpsonAbs EvennessAbs EvennessRel'.split())

    def calcdiver3(self, phy):
        ''' overwrites phy! '''
        nphy = phy.shape[0]
        shape = phy.shape[1:]

        phy[phy<0] = 0.
        total = phy.sum(0)

        phy.sort(axis=0)
        physort = phy[::-1]
        diver3 = np.zeros(shape, int)
        biotot = np.zeros(shape)
        for i in range(nphy):
            diver3[biotot < self.thresh3*total] += 1
            biotot += physort[i]

        return diver3

    def calcshannons(self, phy, rfacsh=1., rfacrich=1.):
        ''' overwrites phy! '''
        nphy = phy.shape[0]
        shape = phy.shape[1:]

        shannon = np.zeros(shape)
        simpson = np.zeros(shape)
        richness = np.zeros(shape, int)

        # for consistency recompute total with self.threshold
        phy[phy<self.threshsh*rfacsh] = 0
        total = phy.sum(0)
        for i in range(nphy):
            valid = phy[i] >= self.threshsh*rfacsh
            tmp = phy[i,valid]/total[valid]
            shannon[valid] += tmp*np.log(tmp)
            simpson[valid] += tmp*tmp
            valid = phy[i] >= self.threshrich*rfacrich
            richness[valid] += 1

        shannon *= -1
        valid = simpson != 0
        simpson[valid] = 1./simpson[valid]

        return shannon, simpson, richness


def calcshannonabs(phy, thresh=1e-6):
    '''
    Threshold is for Richness.
    Default 1E-6 uM C is for local/average biomass in C units.
    For integrated biomass or deep averaging, modify threshold, e.g.,

      thresh=1E-4 for 100m integral
      thresh=1E-7 for 1000m average (assuming biomass is present in top ~100m)

    '''
    nphy = phy.shape[0]
    shape = phy.shape[1:]

    shannon = np.zeros(shape)
    richness = np.zeros(shape, int)
    evenness = np.zeros(shape)

    phy = np.where(phy >= thresh, phy, 0.)
    total = phy.sum(0)
    for i in range(nphy):
        valid = phy[i] > 0.
        tmp = phy[i, valid]/total[valid]
        shannon[valid] += tmp*np.log(tmp)
        richness += valid

    shannon *= -1

    valid = richness > 0.
    evenness[valid] = shannon[valid]/np.log(richness[valid])

    d = dict(Shannon=shannon, Richness=richness, Evenness=evenness)

    return d

