#!/usr/bin/env python
#$ -N mpl
#$ -o sge.o$JOB_ID.$TASK_ID
#$ -e sge.e$JOB_ID.$TASK_ID
#$ -j n
#$ -cwd
#$ -V
#$ -q darwin
#$ -t 1-6

import sys,os

debug = False

sge = False
id = 1
n = 1
first = None
last = None
step = None

def roundrobin(tasks, nhost=None, ihost=None):
    if nhost is None: nhost = n
    if ihost is None: ihost = id-1
    hosttasks = [[] for i in range(nhost)]
    ih = 0
    for task in tasks:
        hosttasks[ih].append(task)
        ih = (ih+1) % nhost
    if ihost >= 0:
        return hosttasks[ihost]
    else:
        return hosttasks
    

def fillup(tasks, nhost=None, ihost=None):
    if nhost is None: nhost = n
    if ihost is None: ihost = id-1
    neach, nextra = divmod(len(tasks), nhost)
    itask = 0
    hosttasks = []
    # the first nextra hosts have one more task...
    for i in range(0, nextra):
        iend = itask + neach + 1
        hosttasks.append(tasks[itask:iend])
        itask = iend
    # ...the others have exactly neach tasks
    for i in range(nextra, nhost):
        iend = itask + neach
        hosttasks.append(tasks[itask:iend])
        itask = iend
    if ihost >= 0:
        return hosttasks[ihost]
    else:
        return hosttasks
    

def rr(tasks, nhost=None, args=None):
    if sge:
        if nhost is None: nhost = n
        mytasks = roundrobin(tasks, nhost)
    else:
        if args is None: args = sys.argv[1:]
        mytasks = [ tasks[int(s)] for s in args ]
    tasks[:] = mytasks


if 'SGE_TASK_ID' in os.environ:
    sge = True
    id = int(os.environ['SGE_TASK_ID'])
    first = int(os.environ.get('SGE_TASK_FIRST', id))
    last = int(os.environ.get('SGE_TASK_LAST', first))
    step = int(os.environ.get('SGE_TASK_STEPSIZE', 1))
    n = (last-first)//step + 1
    if debug: print 'TASK:', id, 'of', n, '(%d-%d:%d)' % (first, last, step)
elif 'SLURM_ARRAY_TASK_ID' in os.environ:
    sge = True
    id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    if debug: print 'TASK:', id


if __name__ == '__main__':
    args = sys.argv[1:]
#    print 'args:', args

    print id,'rr',roundrobin(args, n)
    print id,'fu',fillup(args, n)

