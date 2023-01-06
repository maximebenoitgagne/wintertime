#!/usr/bin/env python
import sys
import re
import fortran
import maptracers
from darwin import iofmt, ionum

srcdir = '../../darwin2fixedzoo/verification/'

args = sys.argv[1:]
if '-q' in args:
    args.remove('-q')
    maptrac = maptracers.mapquota
else:
    maptrac = maptracers.mapmonod
exp, = args

monre = re.compile(r'%MON ')
ptrre = re.compile(r'%MON trcstat_ptracer([0-9a-zA-Z][0-9a-zA-Z])_([a-z0-9]*)  *=([ 0-9.E+-]*) *$')
endre = re.compile(r'End MONITOR ptracers field statistics')
numre = re.compile(r'%MON time_tsnumber *=([ 0-9]*) *$')

names0 = [s.strip("'") for s in fortran.readnmlparam(srcdir + exp + '/input/data.ptracers', 'ptracers_names')]
names = [s.strip("'") for s in fortran.readnmlparam(exp + '/input/data.ptracers', 'ptracers_names')]
i0s,ff = maptrac(names0, names)
id = dict((i0+1,i+1) for i,i0 in enumerate(i0s))

o = {}

with open(exp + '/results/output.txt', 'w') as fo:
#with sys.stdout as fo:
    with open(srcdir + exp + '/results/output.txt') as fi:
        for l in fi:
            m = ptrre.search(l[:-1])
            if m:
                i,s,v = m.groups()
                i0 = ionum(i)
#                if i0 in id:
#                    i = id[i0]
#                v = ff[i-1]*float(v)
#                    ol = '(PID.TID 0000.0001) %MON trcstat_ptracer{:02d}_{:<11s}= {:21.13E}\n'.format(i, s, v)
                o.setdefault(i0-1, []).append((s,v))
            elif endre.search(l):
                for i in range(len(names)):
                    i0 = i0s[i]
                    print i, names[i], names0[i0], ff[i]
                    #fo.write(''.join(o[i]))
                    for s,v in o[i0]:
                        v = ff[i]*float(v)
                        fo.write('(PID.TID 0000.0001) %MON trcstat_ptracer{}_{:<11s}= {:24.16E}\n'.format(iofmt(i+1), s, v))

                o = {}
            elif monre.search(l):
                fo.write(l)
