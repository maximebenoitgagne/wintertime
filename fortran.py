from __future__ import print_function
import sys
import os
import re
from collections import OrderedDict

__all__ = ['readparameters','readnmlparam']

def readparameters(*files, **kwargs):
    sloppy = kwargs.pop('sloppy', False)
    returnconditions = kwargs.pop('conditions', False)

    dir = kwargs.pop('dir', '.')
    assert len(kwargs) == 0

    filestack = []
    for f in files:
        if not hasattr(f,'readlines'):
            f = open(f)
        filestack[:0] = [f]

    conds = {}

    code = ''
    cond = []
    while len(filestack):
        file = filestack[-1]
        for line in file:
            m = re.match(r'^ *# *include *"([^"]*)"', line)
            if m:
                fname = os.path.join(dir, m.group(1))
                try:
                    file = open(fname)
                except IOError:
                    if not sloppy:
                        raise
                else:
                    filestack.append(file)
                continue

            m = re.match(r'^ *# *ifdef  *(\S*)', line)
            if m:
                cond.append(line)
                continue

            m = re.match(r'^ *# *endif', line)
            if m:
                cond.pop()
                continue

            m = re.match(r'^ *parameter *\( *(.*) *\)', line, re.IGNORECASE)
            if m:
                codeline = m.group(1)
                codeline = re.sub(r',', ';', codeline)
                codeline = re.sub(r'\bin\b', 'IN', codeline)
                code += codeline + '\n'
                try:
                    name,val = m.group(1).split('=')
                except ValueError:
                    pass
                else:
                    conds[name.strip().lower()] = list(cond)
        filestack.remove(file)

    d = OrderedDict()
    exec(code, d)
    if 'IN' in d:
        dnew = OrderedDict()
        for k,v in d.items():
            if k == 'IN':
                k = 'in'
            dnew[k] = v
        d = dnew

    try:
        del d['__builtins__']
    except:
        pass

    if returnconditions:
        return d, conds
    else:
        return d

def readnmlparam(file,name):
    if not hasattr(file,'readlines'):
        file = open(file)

    d = {}
    for line in file.readlines():
        m = re.match(r' *(\w*) *(\([^)]*\))? *= *(.*?),?\s*$', line)
        if m:
            key,arg,val = m.groups()
            if key.lower() == name.lower():
                if arg is not None:
                    d[int(arg[1:-1])] = val
                else:
                    return val
    if d:
        n = max(d.keys())
        arr = [d.get(i, None) for i in range(1, n+1)]
        return arr

def parsenml(f):
    if not hasattr(f,'readlines'):
        f = open(f)

    nmls = {}
    for line in f:
        m = re.match(r' *&(\w*) *$', line)
        if m and m.group(1) != '':
            nmlname = m.group(1)
            nmls[nmlname] = {}
            nml = nmls[nmlname]

        m = re.match(r' *(\w*) *(\([^)]*)? *= *(.*?),?\s*$', line)
        if m:
            key,arg,val = m.groups()
            if key not in nml:
                nml[key] = {}
            nml[key][arg] = val
            print(nmlname,key,arg,val)

    res = {}
    for nmlname,nml in nmls.items():
        res[nmlname] = {}
        for var,d in nml.items():
            if d.keys() == [None]:
                res[nmlname][var] = d[None]
            else:
                print(var,d)
    #            inds = d.keys()[0]
    #            ndim = inds.count(',') + 1
                inds = [ i.split(',') for i in d.keys() ]
                shape = map(max,zip(*inds))
                try:
                    float(d.values()[0])
                except ValueError:
                    tp = 'S'
                else:
                    tp = 'f'

                res[nmlname][var] = np.zeros(shape,tp)
                for key,val in d.items():
                    inds = tuple( int(s)-1 for s in key.split(',') )
                    if tp == 'f':
                        val = float(val)
                    res[nmlname][var][inds] = val

    return res


def readnml(filename):
    import f90nml
    parser = f90nml.Parser()
    parser.comment_tokens += '#'
    nml = parser.read(filename)
    return nml


if __name__ == '__main__':
    d = readparameters(*sys.argv[1:])
    wid = max(len(k) for k in d)
    for k,v in d.items():
        print('{:{}s} = {}'.format(k, wid, v))

