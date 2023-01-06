#!/usr/bin/env python
from logging import warning
import numpy as np
from matplotlib import dates
from datetime import datetime as date
from datetime import timedelta as delta

def date2secs(self, d):
    return (dates.date2num(d) - t0)*86400


class DateIt(object):
    def __init__(self, startdate, dt, freq, phase, labelsecs):
        self.t0 = float(dates.datestr2num(startdate))
        self.dt = dt
        self.freq = freq
        self.phase = phase
        self.laboff = labelsecs

    def date2secs(self, d):
        return (dates.date2num(d) - self.t0)*86400

    def date(self, d):
        s = self.date2secs(d)
        s = (s-self.phase)//self.freq*self.freq + self.phase + self.laboff
        it = int(np.round(s/self.dt))
        delta = s - it*self.dt
        if abs(delta) > 1e-6:
            warning('cal.date: delta %f', delta)
        return it

    def tup(self, *args):
        d = dates.datetime.datetime(*args)
        return self.date(d)


modelstart = dates.datestr2num('19920101')

def it2date(it,dt=1,t0=modelstart,offset=0.):
    return dates.num2date(t0+(dt*it+offset)/86400.)


def dateargs2num(*args):
    return dates.date2num(dates.datetime.datetime(*args))


def date2it(d,dt=1,t0=modelstart):
    return int((dates.date2num(d)-t0)*86400)//dt


def datestr2it(d,dt=1,t0=modelstart):
    return int((dates.datestr2num(d)-t0)*86400)//dt


def it2ymd(it,dt=1,t0=modelstart,offset=0.,sep=''):
    return it2date(it,dt,t0,offset).strftime('%Y'+sep+'%m'+sep+'%d')


def it2ymdhms(it,dt=1,t0=modelstart,offset=0.):
    return it2date(it,dt,t0,offset).strftime('%Y%m%d%H%M%S')


def it2datestr(it,dt=1,t0=modelstart):
    return it2date(it,dt,t0).strftime('%Y-%m-%d+%H:%M:%S')


def it2day(it,dt=1,t0=modelstart,sep='-'):
    return it2ymd(it,dt,t0,-86400.,sep)


def it2yday(it,dt=1,t0=modelstart,sep='-'):
    return it2date(it,dt,t0,-86400).strftime('%Y'+sep+'%j')

def it2ydaylen(it,dt=1,t0=modelstart,sep='-'):
    d = it2date(it,dt,t0,-86400)
    jan1st = d.replace(month=1,day=1)
    nextyear = jan1st.replace(year=d.year+1)
    days = (nextyear-jan1st).days
    return d.year, int(d.strftime('%j')), days


def datetup2it(d,dt=1,t0=modelstart):
    d = dates.datetime.datetime(*d)
    return date2it(d,dt,t0)


def itrange(d1,d2,step=86400.,dt=1200,t0=modelstart):
    dit = int(step//dt)
    d1 = dates.datetime.datetime(*d1)
    d2 = dates.datetime.datetime(*d2)
    return range(date2it(d1,dt,t0)+dit, date2it(d2,dt,t0)+dit, dit)

if __name__ == '__main__':
    import sys
#   args = sys.argv[1:]
#   opts = [ arg for arg in args if '=' in arg ]
#   for opt in opts:
#        args.remove(opt)
#
#    kw = dict(opt.split('=') for opt in opts)
#    for k,v in kw.items():
#        try:
#            v = float(v)
#        except ValueError:
#            pass
#        else:
#            kw[k] = v
    sys.stdout.write(eval(' '.join(sys.argv[1:])) + '\n')
