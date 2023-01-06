#!/usr/bin/env python
from matplotlib import dates
from datetime import datetime as date
from datetime import timedelta as delta

modelstart = dates.datestr2num('20111118')
dt = 900.

def it2date(it,dt=dt,t0=modelstart,offset=0.):
    return dates.num2date(t0+(dt*it+offset)/86400.)


def dateargs2num(*args):
    return dates.date2num(dates.datetime.datetime(*args))


def date2it(d,dt=dt,t0=modelstart):
    return int((dates.date2num(d)-t0)*86400)//dt


def datestr2it(d,dt=dt,t0=modelstart):
    return int((dates.datestr2num(d)-t0)*86400)//dt


def it2ymd(it,dt=dt,t0=modelstart,offset=0.,sep=''):
    return it2date(it,dt,t0,offset).strftime('%Y'+sep+'%m'+sep+'%d')


def it2ymdhms(it,dt=dt,t0=modelstart,offset=0.):
    return it2date(it,dt,t0,offset).strftime('%Y%m%d%H%M%S')


def it2datestr(it,dt=dt,t0=modelstart):
    return it2date(it,dt,t0).strftime('%Y-%m-%d+%H:%M:%S')


def it2day(it,dt=dt,t0=modelstart,sep='-'):
    return it2ymd(it,dt,t0,-86400.,sep)


def it2yday(it,dt=dt,t0=modelstart,sep='-'):
    return it2date(it,dt,t0,-86400).strftime('%Y'+sep+'%j')

def it2ydaylen(it,dt=dt,t0=modelstart,sep='-'):
    d = it2date(it,dt,t0,-86400)
    jan1st = d.replace(month=1,day=1)
    nextyear = jan1st.replace(year=d.year+1)
    days = (nextyear-jan1st).days
    return d.year, int(d.strftime('%j')), days


def datetup2it(d,dt=dt,t0=modelstart):
    d = dates.datetime.datetime(*d)
    return date2it(d,dt,t0)


def itrange(d1,d2,step=86400.,dt=dt,t0=modelstart):
    dit = int(step//dt)
    d1 = dates.datetime.datetime(*d1)
    d2 = dates.datetime.datetime(*d2)
    return range(date2it(d1,dt,t0)+dit, date2it(d2,dt,t0)+dit, dit)

if __name__ == '__main__':
    import sys
    sys.stdout.write(eval(' '.join(sys.argv[1:])) + '\n')
