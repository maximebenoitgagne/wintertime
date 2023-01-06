import numpy as np

def _sq(x):
    return x*x

def arcdist(lon1,lat1,lon2,lat2):
    dlon = lon2-lon1
    c1 = np.cos(lat1)
    s1 = np.sin(lat1)
    c2 = np.cos(lat2)
    s2 = np.sin(lat2)
    sd = np.sin(dlon)
    cd = np.cos(dlon)
    return np.arctan2(np.sqrt(_sq(c2*sd)+_sq(c1*s2-s1*c2*cd)),
                      s1*s2+c1*c2*cd)

def arcdistdeg(lon1,lat1,lon2,lat2):
    return arcdist(lon1*np.pi/180,lat1*np.pi/180,
                   lon2*np.pi/180,lat2*np.pi/180)


