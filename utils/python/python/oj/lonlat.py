from pylab import *
from scipy.special import ellipeinc

def lonlatdist(lonv, latv):
    """ (spherical) distance along piece-wise linear paths in lat-lon space """
    d = 0.
    res = [d]
    for i in range(len(lonv)-1):
        lon0 = lonv[i]
        lon1 = lonv[i+1]
        lat0 = latv[i]
        lat1 = latv[i+1]
        if lat1-lat0 == 0.:
            dd = cos(lat0*pi/180.)*(lon1-lon0)
        else:
            slope = (lon1-lon0)/(lat1-lat0)
            slope2 = slope*slope
            k = slope2/(1.+slope2)
            dd = 180./pi * sqrt(1+slope2) * abs(ellipeinc(lat1*pi/180., k) - ellipeinc(lat0*pi/180., k))
        d += dd
        res.append(d)
    return res

