import numpy as np

mon_days = np.r_[[
        31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 1992
        31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 1993
        31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 1994
        31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,  # 1995
        ]*8]                                             # ...
dayf = np.cumsum(np.r_[0, mon_days])

def date2it_day(year, month, day):
    '''for offline fields'''
    mon = (year - 1992)*12 + month - 1
    daynum = dayf[mon] + day - 1
    it = (daynum + 1)*72
    return it

def date2it_3day(year, month, day):
    mon = (year - 1992)*12 + month - 1
    daynum = dayf[mon] + day - 1
    i3day = daynum//3
    it = (i3day + 1)*72
    return it

def it2date_3day(it):
    d = it//24 - 1
    im = dayf.searchsorted(d) - 1
    y, m = divmod(im, 12)
    y += 1992
    m += 1
    d = d - dayf[im]
    return (y, m, d)

def it2date_day(it):
    d = it//72 - 1
    im = dayf.searchsorted(d) - 1
    y, m = divmod(im, 12)
    y += 1992
    m += 1
    d = d - dayf[im]
    return (y, m, d)

def it2yyday_3day(it):
    d = it//24 - 1
    iy = dayf[::12].searchsorted(d) - 1
    y = 1992 + iy
    d = d - dayf[12*iy]
    return (y, d)


def monav_its_wgts(year, month):
    y = year - 1992
    m = month - 1
    mon = y*12 + m
    dl0, dle = dayf[mon:mon+2]
    i0 = (dl0//3).clip(1, None)
    ie = (dle+2)//3

    its = np.arange(i0, ie)*72 + 72

    wgts = np.ones((ie-i0,))
    wgts[0] = (((i0 + 1)*3 - dl0) / 3.).clip(0., 1.)
    wgts[-1] = (dle - (ie - 1)*3) / 3.
    wgts /= wgts.sum()

    return its, wgts

def loadbin_monav(tmpl, year, month, div=1, offset=0, verbose=False):
    from oj.num import loadbin
    its, wgts = monav_its_wgts(year, month)
    it = its[0]//div + offset
    if verbose: print year, month, it
    sm = loadbin(tmpl.format(it)).astype('f8')*wgts[0]
    for i in range(1, len(its)):
        it = its[i]//div + offset
        a = loadbin(tmpl.format(it)).astype('f8')
        sm += a*wgts[i]

    return sm

