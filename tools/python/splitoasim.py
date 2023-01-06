#!/usr/bin/env python
import numpy as np

inf = '../input/input/loc1_oasim_edp_below.bin'
out = '../input/input/loc1_oasim_edp{:02d}_below.bin'

a = np.fromfile(inf,'>f4').reshape((12,13))
for i in range(13):
    a[:,i].astype('>f4').tofile(out.format(i+1))

inf = '../input/input/loc1_oasim_esp_below.bin'
out = '../input/input/loc1_oasim_esp{:02d}_below.bin'

a = np.fromfile(inf,'>f4').reshape((12,13))
for i in range(13):
    a[:,i].astype('>f4').tofile(out.format(i+1))

