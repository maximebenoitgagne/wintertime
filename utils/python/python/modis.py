#!/usr/bin/env python
import numpy as np
from scipy.interpolate import interp1d

_scaledR = np.r_[1, 110, 160, 210, 240, 255]/255.
_unscaledR = np.r_[0, 30, 60, 120, 190, 255]/255.
_stretch = interp1d(_unscaledR, _scaledR)

def enhance(rgb):
    return _stretch(rgb)

