import os
import numpy as np
import tiles
import fa

tilesopts = dict(mmap=False)

if os.path.exists('/nobackup1b/jahn'):
    scratch = '/nobackup1b/jahn/'
elif os.path.exists('/scratch/jahn/grid'):
    scratch = '/scratch/jahn/'
elif os.path.exists('/mit/jahn/l'):
    scratch = '/mit/jahn/l/beagle/'
elif os.path.exists('/data/jahn'):
    scratch = '/data/jahn/'
else:
    raise IOError('No data directory found')
tiledir = scratch + 'tiles/ecco2/cube84/'
itsmon0 = [
 52704,
 54936,
 56952,
 59184,
 61344,
 63576,
 65736,
 67968,
 70200,
 72360,
 74592,
 76752,
 78984,
 81216,
 83232,
 85464,
 87624,
 89856,
 92016,
 94248,
 96480,
 98640,
 100872,
 103032,
 105264,
 107496,
 109584,
 111816,
 113976,
 116208,
 118368,
 120600,
 122832,
 124992,
 127224,
 129384,
 131616,
 133848,
 135864,
 138096,
 140256,
 142488,
 144648,
 146880,
 149112,
 151272,
 153504,
 155664,
 157896,
 160128,
 162144,
 164376,
 166536,
 168768,
 170928,
 173160,
 175392,
 177552,
 179784,
 181944,
 184176,
 186408,
 188424,
 190656,
 192816,
 195048,
 197208,
 199440,
 201672,
 203832,
 206064,
 208224,
 210456,
]
itsmon = [ it - 72 for it in itsmon0 ][1:]

itsyear = {
           1994: range( 52704,  78984, 72),
           1995: range( 78984, 105264, 72),
           1996: range(105264, 131616, 72),
           1997: range(131616, 157896, 72),
           1998: range(157896, 184176, 72),
           1999: range(184176, 210456, 72),
          }

its6hr = range(52650, 210384+1, 18)
its = range(52704,210384+1,72)
chl2cits = range(131616,157896,72)

plankton = np.genfromtxt(scratch + '/grid/cube84/plankton-ini-char.dat',
                         names=True).view(np.rec.recarray)
group = plankton['size'] + plankton['diat'] + plankton['nsrc'] - 1
groupnames = ['Pro', 'Pro2', 'Syn', 'large', 'Diatom']

chl2cvars = ['Chl2C',
    'Chl2CClo',
    'Chl2CDon',
    'PARday',
    ]
L30vars = [
    'biomass',
    'ChlCloer',
    'ChlDoney',
    'Chla',
    'Diver1',
    'Diver2',
    'Diver3',
    'Diver4',
    'Diver4small',
    'Shannonsmall',
    'PAR',
    'PP',
    ]
ptracervars = [
    'PO4', 'NO3', 'FeT', 'SiO2', 'DOP', 'DON', 'DOFe',
    'Zoo1P', 'Zoo1N', 'Zoo1Fe', 'Zoo1Si',
    'Zoo2P', 'Zoo2N', 'Zoo2Fe', 'Zoo2Si',
    'POP', 'PON', 'POFe', 'POSi', 'NH4', 'NO2',
    ] + [ 'Phy%02d'%i for i in range(1,79) ]
pinds = dict((k,i) for i,k in enumerate(ptracervars))

L50vars = ['KPPdiffS',
    'KPPghat',
    'S',
    'T',
    'UVEL',
    'VVEL',
    'WVEL',
    ]
L1vars = [
    'SIarea',
    'SIheff',
    'SIhsalt',
    'SIhsnow',
    'SIqsw',
    'SIuice',
    'SIvice',
   ]
L16hrvars = [
    'ETAN',
    'KPPhbl',
    'KPPmld',
    'oceQsw',
    'oceTAUX',
    'oceTAUY',
    'PHIBOT',
    'SSS',
    'SST',
    'surForcS',
    'surForcT',
    'UVEL_k1',
    'VVEL_k1',
   ]
L1assvars = [
    'ETAN',
    'PHIBOT',
    'KPPhbl',
    'KPPmld',
    'oceTAUX',
    'oceTAUY',
    'surForcT',
    'oceTAU',
    'rStarFacC',
    'rStarFacS',
    'rStarFacW',
   ]
k1vars = [
    'T',
    'ChlCloer',
    'NO3',
   ]
k1vars5 = [
    'PhyGrp',
   ]
k2vars = [
    'VEL',
   ]
intrvars = [
    'PP',
    'biomass',
   ] + ['PhyGrp{0}'.format(i) for i in range(1,6)
   ] + ['TRAC{0:02d}'.format(i) for i in range(1,100)
   ]
L50mondpvars = [
    'THETA',
    'SALTanom',
    'SALTSQan',
    'THETASQ',
    ]
L50monvars = [
    'DRHODR',
    'RHOAnoma',
    'RHOANOSQ',
    'URHOMASS',
    'USLTMASS',
    'UTHMASS',
    'UVELMASS',
    'UVELSQ',
    'UV_VEL_Z',
    'VRHOMASS',
    'VSLTMASS',
    'VTHMASS',
    'VVELMASS',
    'VVELSQ',
    'WRHOMASS',
    'WSLTMASS',
    'WTHMASS',
    'WU_VEL',
    'WVELMASS',
    'WVELSQ',
    'WV_VEL',
    ]
L50monassvars = [
    'UEMASS',
    'VNMASS',
    'UEMASSmskmean',
    'VNMASSmskmean',
    ]
L1monvars = [
    'TRELAX',
    'EXFhs',
    'EXFhl',
    'EXFlwnet',
    'oceFWflx',
    'oceSflux',
    'oceQnet',
    'SRELAX',
    'TFLUX',
    'SFLUX',
]
L50mondpvars = [
    'SALTanom',
    'THETA',
    'SALTSQan',
    'THETASQ',
]
L50monspvars = [
    'UVELMASS',
    'VVELMASS',
    'WVELMASS',
    'UVELSQ',
    'VVELSQ',
    'WVELSQ',
    'UV_VEL_Z',
    'WU_VEL',
    'WV_VEL',
    'UTHMASS',
    'VTHMASS',
    'WTHMASS',
    'USLTMASS',
    'VSLTMASS',
    'WSLTMASS',
    'RHOAnoma',
    'DRHODR',
    'RHOANOSQ',
    'URHOMASS',
    'VRHOMASS',
    'WRHOMASS',
]

def mkk1PhyGrp():
    tmpl = [ scratch+'ecco2/cube84/assembled/global.k1/k1{d0}/k1{d0}_day.{d1:010d}.data' ]
    return tiles.mdata(tmpl,'>f4',(5,510,3060),d0=k1vars5,d1=its,fast=2,**tilesopts)

def mkk1():
    tmpl = [ scratch+'ecco2/cube84/assembled/global.k1/k1{d0}/k1{d0}_day.{d1:010d}.data' ]
    return tiles.mdata(tmpl,'>f4',(510,3060),d0=k1vars,d1=its,fast=2,**tilesopts)

def mkk2():
    tmpl = [ scratch+'ecco2/cube84/assembled/global.k1/k2{d0}/k2{d0}_day.{d1:010d}.data' ]
    return tiles.mdata(tmpl,'>f4',(510,3060),d0=k2vars,d1=its,fast=2,**tilesopts)

def mkintr():
    tmpl = [ scratch+'ecco2/cube84/assembled/global.intr/intr{d0}/day.{d1:010d}.data' ]
    return tiles.mdata(tmpl,'>f4',(510,3060),d0=intrvars,d1=its,fast=2,**tilesopts)

def mkL1ass():
    tmpl = [ scratch+'ecco2/cube84/assembled/L1/{d0}/{d0}_day.{d1:010d}.data' ]
    return tiles.mdata(tmpl,'>f4',(510,3060),d0=L1assvars,d1=its,fast=2,**tilesopts)

def mkL50monass():
    tmpl = [ scratch+'ecco2/cube84/assembled/L50/mon/{d0}/mon.{d1:010d}.data' ]
    return tiles.mdata(tmpl,'>f4',(50,510,3060),d0=L50monassvars,d1=itsmon,fast=2,**tilesopts)

def mkL16hr():
    tmpl = [ scratch+'output/ecco2/cube84/tiles300.6hr/res_{r:04d}/{{d0}}/{{d0}}.{{d1:010d}}.{t:03d}.001.data'.format(r=r,t=r+1) for r in range(300) ]
    return tiles.mcs(tmpl,'>f4',(51,102),d0=L16hrvars,d1=its6hr,fast=2,**tilesopts)

def mkL1():
    tmpl = [ tiledir+'L1/res_{r:04d}/{{d0}}/{{d0}}.{{d1:010d}}.{t:03d}.001.data'.format(r=r,t=r+1) for r in range(300) ]
    return tiles.mcs(tmpl,'>f4',(51,102),d0=L1vars,d1=its,fast=2,**tilesopts)

def mkL50():
    tmpl = [ tiledir+'L50/res_{r:04d}/{{d0}}/{{d0}}_day.{{d1:010d}}.{t:03d}.001.data'.format(r=r,t=r+1) for r in range(300) ]
    return tiles.mcs(tmpl,'>f4',(50,51,102),d0=L50vars,d1=its,fast=2,**tilesopts)

def mkL1mon():
    tmpl = [ tiledir+'L50mon/res_{r:04d}/{{d0}}/{{d0}}.{{d1:010d}}.{t:03d}.001.data'.format(r=r,t=r+1) for r in range(300) ]
    return tiles.mcs(tmpl,'>f4',(51,102),d0=L1monvars,d1=itsmon,fast=2,**tilesopts)

def mkL50mon():
    tmpl = [ tiledir+'L50mon/res_{r:04d}/{{d0}}/{{d0}}.{{d1:010d}}.{t:03d}.001.data'.format(r=r,t=r+1) for r in range(300) ]
    return tiles.mcs(tmpl,'>f4',(50,51,102),d0=L50monvars,d1=itsmon,fast=2,**tilesopts)

def mkL50mondp():
    tmpl = [ tiledir+'L50mon/res_{r:04d}/{{d0}}/{{d0}}.{{d1:010d}}.{t:03d}.001.data'.format(r=r,t=r+1) for r in range(300) ]
    return tiles.mcs(tmpl,'>f8',(50,51,102),d0=L50mondpvars,d1=itsmon,fast=2,**tilesopts)

def mkL30():
    tmpl = [ tiledir+'L30/res_{r:04d}/{{d0}}/{{d0}}_day.{{d1:010d}}.{t:03d}.001.data'.format(r=r,t=r+1) for r in range(300) ]
    return tiles.mcs(tmpl,'>f4',(30,51,102),d0=L30vars,d1=its,fast=2,**tilesopts)

def mkChl2C():
    tmpl = [ tiledir+'L30Chl2C/res_{r:04d}/{{d0}}/{{d0}}_day.{{d1:010d}}.{t:03d}.001.data'.format(r=r,t=r+1) for r in range(300) ]
    return tiles.mcs(tmpl,'>f4',(30,51,102),d0=chl2cvars,d1=chl2cits,fast=2,**tilesopts)

def mkptracers():
    tmpl = [ tiledir+'L30/res_{r:04d}/ptracers/ptracers_day.{{d0:010d}}.{t:03d}.001.data'.format(r=r,t=r+1) for r in range(300) ]
    return tiles.mcs(tmpl,'>f4',(99,30,51,102),d0=its,fast=2,**tilesopts)

def mkptracersann():
    tmpl = [ tiledir+'L30ann/ptracers/1999/ptracers.{t:03d}.001.data'.format(r=r,t=r+1) for r in range(300) ]
    return tiles.mcs(tmpl,'>f4',(99,30,51,102),fast=2,**tilesopts)

def mkPhyGrp():
    tmpl = [ tiledir+'L30/res_{r:04d}/PhyGrp/PhyGrp_day.{{d0:010d}}.{t:03d}.001.data'.format(r=r,t=r+1) for r in range(300) ]
    return tiles.mcs(tmpl,'>f4',(5,30,51,102),d0=its,fast=2,**tilesopts)

gridvars3d = [
    'hFacC',
    'hFacS',
    'hFacW',
    ]
gridvars1C = [
    'DRC',
    'DRF',
    'PHrefC',
    'RC',
    ]
gridvars1F = [
    'PHrefF',
    'RF',
    ]
gridvars2d = [
    'AngleCS',
    'AngleSN',
    'Depth',
    'DXC',
    'DXG',
    'DYC',
    'DYG',
    'RAC',
    'RAS',
    'RAW',
    'RAZ',
    'XC',
    'XG',
    'YC',
    'YG',
    ]

gridtmpl = scratch+'grid/cube84/{d0}'
griddatatmpl = gridtmpl + '.data'

def mkgrid3d():
    return tiles.mmds(gridtmpl,d0=gridvars3d,fast=2,**tilesopts)

def mkgrid2d():
    return tiles.mmds(gridtmpl,d0=gridvars2d,fast=2,**tilesopts)

def grid1C(name):
    return np.fromfile(griddatatmpl.format(d0=name),'>f4').reshape(50)

def grid1F(name):
    return np.fromfile(griddatatmpl.format(d0=name),'>f4').reshape(51)

grid2d = mkgrid2d()
grid3d = mkgrid3d()
L1 = mkL1()
k1 = mkk1()
k2 = mkk2()
k1PhyGrp = mkk1PhyGrp()['PhyGrp']
intr = mkintr()
L1ass = mkL1ass()
L30 = mkL30()
L50 = mkL50()
L1mon = mkL1mon()
L50mon = mkL50mon()
L50mondp = mkL50mondp()
L50monass = mkL50monass()
Chl2C = mkChl2C()
PhyGrp = mkPhyGrp()
ptracers = mkptracers()
ptracerdict = dict((k,ptracers[:,i]) for k,i in pinds.items())

del mkL1, mkL30, mkL50, mkL50mon, mkChl2C, mkPhyGrp, mkptracers
del mkgrid2d, mkgrid3d
#del L1vars, L30vars, L50vars, L50monvars, chl2cvars
#del gridvars2d, gridvars3d

class UntiledVariable(object):
    def __init__(self, fname, dtype, shape):
        self.fname = fname
        self.dtype = dtype
        self.shape = shape

    def __repr__(self):
        return 'Variable' + str(self.shape)

    __str__ = __repr__

    def __call__(self):
        return np.fromfile(self.fname, self.dtype).reshape(self.shape)

    def __array__(self, *args):
        return self().__array__(*args)


def mkvariables():
    variables = {}
    for name in [
                'grid2d',
                'grid3d',
                'L1',
                'L1ass',
                'L30',
                'L50',
                'L1mon',
                'L50mon',
                'L50mondp',
                'Chl2C',
                'PhyGrp',
                'ptracers',
               ]:
        var = globals()[name]
        dv = var.dimvals
        if type(dv[0][0]) == type(''):
            for i,fld in enumerate(dv[0]):
                variables[fld] = var[i]
        else:
            variables[name] = var

    for nr,names in [(51, [ 'PHrefF', 'RF']),
                     (50, [ 'DRC', 'DRF', 'PHrefC', 'RC'])]:
        for name in names:
            variables[name] = UntiledVariable(griddatatmpl.format(d0=name), '>f4', (nr,))

    return variables

variables = mkvariables()

del mkvariables

from namespace import Namespace

ns = Namespace(**variables)

def lmC():
    return np.fromfile(griddatatmpl.format(d0='lmC'), bool).reshape(50,510,3060)

def lmCblacksea():
    return np.fromfile(griddatatmpl.format(d0='lmCblacksea'), bool).reshape(510,3060)

intrvars = ['biomass', 
            'PP',
            ] + [
            'PhyGrp{0}'.format(i) for i in range(1,6)] + [
            'TRAC{0:02d}'.format(i) for i in range(1,100)]

def mkglobalintr():
    tmpl = [ scratch+'ecco2/cube84/assembled/global.intr/intr{d0}/day.{d1:010d}.data' ]
    return tiles.mdata(tmpl,'>f4',(510,3060),d0=intrvars,d1=its,fast=2,**tilesopts)

globalintr = mkglobalintr()

mitgrid = fa.MITGrid(scratch+'grid/cube84/tile{0:03d}.mitgrid', 12*[510])

