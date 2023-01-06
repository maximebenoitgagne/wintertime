import sys,os

debug = True
array = False
id = 0
offset = 0

def roundrobin(tasks, nhost, ihost=None):
    if ihost is None: ihost = id-offset
    hosttasks = [[] for i in range(nhost)]
    ih = 0
    for task in tasks:
        hosttasks[ih].append(task)
        ih = (ih+1) % nhost
    if ihost >= 0:
        return hosttasks[ihost]
    else:
        return hosttasks
    

def fillup(tasks, nhost, ihost=None):
    if ihost is None: ihost = id-offset
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
    

def rr(tasks, nhost=None, args=None, verbose=False):
    '''If an array job, select tasks appropriate for task id.
    If not, pick task indices for all numerical arguments and
    return remaining arguments.

    Example:
        tasks = [0, 1, 2, 3, 4, 5]
        rr(tasks, 3)
        job 0: tasks = [0, 3]
        job 1: tasks = [1, 4]
        job 2: tasks = [2, 5]

    equivalent:
        tasks = rr(6, 3)
    '''
    try:
        iter(tasks)
    except:
        if verbose:
            sys.stderr.write('Setting tasks = range(0, ' + str(tasks) + ')\n')
        tasks = range(tasks)
        returntasks = True
    else:
        returntasks = False
    if nhost is None: nhost = len(tasks)
    if args is None: args = sys.argv[1:]
    if array:
        mytasks = roundrobin(tasks, nhost)
        other = args
    elif args:
        mytasks = []
        other = []
        for arg in args:
            try:
                i = int(arg)
            except ValueError:
                other.append(arg)
            else:
                mytasks.append(tasks[i])
    else:
        mytasks = tasks
        other = []
    if returntasks:
        return mytasks
    else:
        tasks[:] = mytasks
        return other

def master():
    if array:
        return id == offset
    else:
        return True

if 'SLURM_ARRAY_TASK_ID' in os.environ:
    array = True
    id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    offset = 0
    if debug: sys.stderr.write('TASK: {} SLURM\n'.format(id))
elif 'SGE_TASK_ID' in os.environ:
    array = True
    id = int(os.environ['SGE_TASK_ID'])
    offset = 1
    if debug: sys.stderr.write('TASK: {} SGE\n'.format(id))
else:
    array = False

