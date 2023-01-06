import numpy as np
import calendar

def m2it(im):
    iy, m = divmod(im,12)
    y = 1992+iy
    m = m + 1
    return (calendar.timegm((y,m,1,0,0,0))-694224000)/1200

def date2endit(y,m=None,d=None,H=None,M=None,S=None,dt=1200,beyond=True):
    off = 0
    if m is None:
        y += 1
        m = 1
        d = 1
        H = 0
        M = 0
        S = 0
    elif d is None:
        m += 1
        if m > 12:
            y += 1
            m -= 12
        d = 1
        H = 0
        M = 0
        S = 0
    elif H is None:
        off = 86400
        H = 0
        M = 0
        S = 0
    elif M is None:
        off = 3600
        M = 0
        S = 0
    elif S is None:
        off = 60
        S = 0

    if beyond:
        while m > 12:
            m -= 12
            y += 1

    it = (calendar.timegm((y,m,d,H,M,S))+off-694224000)/dt
    return int(np.rint(it))


def itrange(d1,d2,step=86400.,dt=1200):
    d1 = d1 + (6-len(d1))*(None,)
    d2 = d2 + (6-len(d2))*(None,)
    return range(date2endit(*d1,dt=dt), date2endit(*d2,dt=dt), step/dt)

