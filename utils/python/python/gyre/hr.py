import os
from os.path import join as pjoin
import numpy as np
import tiles

homedir = os.environ['HOME']
rundir = homedir + '/gyre/run/B9/monod/run.017'
outdir = homedir + '/gyre/output/B54V3onB9/monod/run.017'
nr,ny,nx = 30,180,270

planktontype = np.dtype(dict(formats=18*['f8']+2*[int], names=[
    'dico', 'diaz', 'sz', 'mu', 'mort', 'Rnp', 'Rfep', 'Rsip', 'wsink',
    'KsP', 'KsN', 'KsFe', 'KsSi', 'palat1', 'palat2', 'Kpar', 'Kinh', 'Topt', 'nsrc', 'np']))
plankton = np.loadtxt(pjoin(rundir, 'plankton_ini_char_nohead.dat'), planktontype).view(np.recarray)
fgroup = (plankton.nsrc + plankton.sz + plankton.dico).astype(int) - 1
fgroupnames = ['Pro', 'ProNH4', 'Syn', 'Dinoflag', 'Diatom']
useNO3 = [ 0, 0, 1, 1, 1 ]
useNO2 = [ 1, 0, 1, 1, 1 ]
useSi  = [ 0, 0, 0, 0, 1 ]

its = range(720, 172801, 720)
monvars = [
'ALK',
'ChlCloer',
'ChlDoney',
'ChlGeide',
'DIC',
'DOC',
'DOFe',
'DON',
'DOP',
'Diver1',
'Diver2',
'Diver3',
'Diver4',
'FeT',
'NH4',
'NO2',
'NO3',
'O2',
'PAR',
'PIC',
'PO4',
'POC',
'POFe',
'PON',
'POP',
'POSi',
'PP',
'Shannon',
'Si',
'Simpson',
]
monZooVars = ['ZooC', 'ZooFe', 'ZooN', 'ZooP', 'ZooSi', ]
monSurfVars = [ 'DICCFLX', 'DICOFLX', 'DICPCO2', 'DICPHAV', 'DICTFLX', ]

mon3d = tiles.mdata(rundir+'/diags/mon/{d0}/mon.{d1:010d}.data',
                    '>f4', (nr+1, ny+2, nx+2), d0=monvars, d1=its, fast=2, mmap=False)
monPhy = tiles.mdata(rundir+'/diags/mon/{d0}/mon.{d1:010d}.data',
                     '>f4', (100, nr+1, ny+2, nx+2), d0=['Phy'], d1=its, fast=2, mmap=False)
monZoo = tiles.mdata(rundir+'/diags/mon/{d0}/mon.{d1:010d}.data',
                     '>f4', (2, nr+1, ny+2, nx+2), d0=monZooVars, d1=its, fast=2, mmap=False)
monSurf = tiles.mdata(rundir+'/diags/mon/{d0}/mon.{d1:010d}.data',
                      '>f4', (ny+2, nx+2), d0=monSurfVars, d1=its, fast=2, mmap=False)

mon = mon3d.todict()
mon.update(monPhy.todict())
mon.update(monZoo.todict())
mon.update(monSurf.todict())


its2day = range(48, 172801, 48)
phys2day = [
'Phy4',
'Phy13',
'Phy14',
'Phy16',
'Phy17',
'Phy20',
'Phy21',
'Phy23',
'Phy27',
'Phy35',
'Phy36',
'Phy38',
'Phy44',
'Phy51',
'Phy57',
'Phy59',
'Phy61',
'Phy62',
'Phy63',
'Phy65',
'Phy68',
'Phy70',
'Phy74',
'Phy76',
'Phy80',
'Phy81',
'Phy83',
'Phy84',
'Phy90',
'Phy92',
'Phy94',
'Phy96',
]
iphys2day = [ int(s[3:])-1 for s in phys2day ]
vars2day = [
'ALK',
'DIC',
'DOC',
'DOFe',
'DON',
'DOP',
'FeT',
'NH4',
'NO2',
'NO3',
'O2',
'PIC',
'PO4',
'POC',
'POFe',
'PON',
'POP',
'POSi',
'PP',
'Si',
] + phys2day

av2day = tiles.mdata(rundir+'/diags/2day/{d0}/2day.{d1:010d}.data',
                     '>f4', (nr+1, ny+2, nx+2), d0=vars2day, d1=its2day, fast=2, mmap=False)
varsana2day = [
'Diver1',
'Diver2',
'Diver3',
'Diver4',
'Richness',
'Shannon',
'Simpson',
]
ana2day = tiles.mdata(outdir+'/2day/{d0}loc2day/2day.{d1:010d}.data',
                     '>f4', (nr+1, ny+2, nx+2), d0=varsana2day, d1=its2day, fast=2, mmap=False)

varsintr2day = [
'Diver1',
'Diver2',
'Diver3',
'Diver4',
'Richness',
'Shannon',
'Simpson',
]
intr2day = tiles.mdata(outdir+'/2day/{d0}intr2day/2day.{d1:010d}.data',
                     '>f4', (1, ny+2, nx+2), d0=varsintr2day, d1=its2day, fast=2, mmap=False)

