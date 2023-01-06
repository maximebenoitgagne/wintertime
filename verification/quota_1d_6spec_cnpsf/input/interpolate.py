#!/usr/bin/env python
import os
from plt import *

rc0 = r_[ 5, 15, 27.5, 45, 65, 87.5, 117.5, 160, 222.5, 310, 435, 
          610, 847.5, 1160, 1542.5, 1975, 2450, 2950, 3450, 3950, 4450, 
          4950, 5450 ]
rc1 = r_[ 5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 
          135, 147.5, 162.5, 177.5, 192.5, 207.5, 225, 245, 265, 285, 
          320, 370, 445, 570, 720, 895, 1145, 1495, 1895, 2295, 2745, 
          3245 ]
i0 = 0
j0 = 35

for fname in [
        '/home/stephdut/Input/input_ecco/glodap_tco2_ann-3d.bin',
        '/home/stephdut/Input/GUD_ECCO_run17/nh4_y2000.data',
        '/home/stephdut/Input/GUD_ECCO_run17/no2_y2000.data',
        '/home/stephdut/Input/input_ecco/lev01_nitrate_ann-3d.bin',
        '/home/stephdut/Input/input_ecco/lev01_phosphate_ann-3d.bin',
        '/home/stephdut/Input/input_ecco/lev01_silicate_ann-3d.bin',
        '/home/stephdut/Input/GUD_ECCO_run17/fet_y2000.data',
        '/home/stephdut/Input/GUD_ECCO_run17/doc_y2000.data',
        '/home/stephdut/Input/GUD_ECCO_run17/don_y2000.data',
        '/home/stephdut/Input/GUD_ECCO_run17/dop_y2000.data',
        '/home/stephdut/Input/GUD_ECCO_run17/dofe_y2000.data',
        '/home/stephdut/Input/GUD_ECCO_run17/poc_y2000.data',
        '/home/stephdut/Input/GUD_ECCO_run17/pon_y2000.data',
        '/home/stephdut/Input/GUD_ECCO_run17/pop_y2000.data',
        '/home/stephdut/Input/GUD_ECCO_run17/posi_y2000.data',
        '/home/stephdut/Input/GUD_ECCO_run17/pofe_y2000.data',
        '/home/stephdut/Input/GUD_ECCO_run17/pic_y2000.data',
        '/home/stephdut/Input/input_ecco/glodap_alk_ann-3d.bin',
        '/home/stephdut/Input/input_ecco/lev01_oxygen_micromolar_ann-3d.bin',
        '/home/stephdut/Input/GUD_ECCO_run17/phy_6species_y2000.data',
        '/home/stephdut/Input/GUD_ECCO_run17/phy_6species_y2000.data',
        '/home/stephdut/Input/GUD_ECCO_run17/phy_6species_y2000.data',
        '/home/stephdut/Input/GUD_ECCO_run17/phy_6species_y2000.data',
        '/home/stephdut/Input/GUD_ECCO_run17/phy_6species_y2000.data',
        '/home/stephdut/Input/GUD_ECCO_run17/phy_6species_y2000.data',
        '/home/stephdut/Input/GUD_ECCO_run17/zoop_y2000.data',
        '/home/stephdut/Input/GUD_ECCO_run17/zoop_y2000.data',
    ]:
    _,base = os.path.split(fname)
    a = fromfile(fname, '>f4').reshape(len(rc0), 160, 360)
    b = interp(rc1, rc0, a[:, j0, i0])
    b.astype('>f4').tofile('input/loc1_' + base)

for fname in [
        '../../monod_eccov3_6spec/input/input/tren_speed_mth-2d.bin',
        '/home/stephdut/Input/input_ecco/mahowald2009_solubile_current_smooth_oce_mth-2d.bin',
        '/home/stephdut/Input/input_ecco/ecco_oasim_total_below_oneband_einm2d.bin',
    ]:
    _,base = os.path.split(fname)
    a = fromfile(fname, '>f4').reshape(-1, 160, 360)
    b = a[:, j0, i0]
    b.astype('>f4').tofile('input/loc1_' + base)
