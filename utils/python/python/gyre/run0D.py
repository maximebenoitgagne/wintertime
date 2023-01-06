import os
from os.path import join as pjoin
import numpy as np
import tiles

rundir = '/data/jahn/gyre/run/B9/monod/run.020'
mondir = '/data/jahn/gyre/run/B9/monod/run.020/diags'
nr,ny,nx = 30,180,270

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
'Si',
'PP',
'Shannon',
'Simpson',
'KPPdiffS',
]
monZooVars = ['ZooC', 'ZooFe', 'ZooN', 'ZooP', 'ZooSi', ]
monSurfVars = [ 'DICCFLX', 'DICOFLX', 'DICPCO2', 'DICPHAV', 'DICTFLX', ]

mon3d = tiles.mdata('%s/{d0}/mon.{d1:010d}.data' % mondir,
                    '>f4', (nr+1, ny+2, nx+2), d0=monvars, d1=its, fast=2, mmap=False)
monPhy = tiles.mdata('%s/{d0}/mon.{d1:010d}.data' % mondir,
                     '>f4', (100, nr+1, ny+2, nx+2), d0=['Phy'], d1=its, fast=2, mmap=False)
monZoo = tiles.mdata('%s/{d0}/mon.{d1:010d}.data' % mondir,
                     '>f4', (2, nr+1, ny+2, nx+2), d0=monZooVars, d1=its, fast=2, mmap=False)
monSurf = tiles.mdata('%s/{d0}/mon.{d1:010d}.data' % mondir,
                      '>f4', (ny+2, nx+2), d0=monSurfVars, d1=its, fast=2, mmap=False)

mon = mon3d.todict()
mon.update(monPhy.todict())
mon.update(monZoo.todict())
mon.update(monSurf.todict())

