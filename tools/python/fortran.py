import sys
import os
import re
from UserDict import UserDict

__all__ = ['OrderedDict', 'readparameters','readnmlparam']

class OrderedDict(UserDict):
    """A UserDict that preserves insert order whenever possible."""
    def __init__(self, dict=None, **kwargs):
        self._order = []
        self.data = {}
        if dict is not None:
            if hasattr(dict,'keys'):
                self.update(dict)
            else:
                for k,v in dict: # sequence
                    self[k] = v
        if len(kwargs):
            self.update(kwargs)
    def __repr__(self):
        return '{'+', '.join([('%r: %r' % item) for item in self.items()])+'}'

    def __setitem__(self, key, value):
        if not self.has_key(key):
            self._order.append(key)
        UserDict.__setitem__(self, key, value)

    def copy(self):
        return self.__class__(self)

    def __delitem__(self, key):
        UserDict.__delitem__(self, key)
        self._order.remove(key)

    def iteritems(self):
        for item in self._order:
            yield (item, self[item])

    def items(self):
        return list(self.iteritems())

    def itervalues(self):
        for item in self._order:
            yield self[item]

    def values(self):
        return list(self.itervalues())

    def iterkeys(self):
        return iter(self._order)

    __iter__ = iterkeys

    def keys(self):
        return list(self._order)

    def popitem(self):
        key = self._order[-1]
        value = self[key]
        del self[key]
        return (key, value)

    def setdefault(self, item, default):
        if self.has_key(item):
            return self[item]
        self[item] = default
        return default

    def update(self, d):
        for k, v in d.items():
            self[k] = v


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
    varlist = []
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
                    varlist.append(name.strip())
        filestack.remove(file)

#    d = OrderedDict()
    d = dict()
    exec code in d
    dnew = OrderedDict.fromkeys(varlist)
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

    for line in file.readlines():
        m = re.match(r' *(\w*) *(\([^)]*)? *= *(.*?),?\s*$', line)
        if m:
            key,arg,val = m.groups()
            if key.lower() == name.lower():
                return val

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
            print nmlname,key,arg,val

    res = {}
    for nmlname,nml in nmls.items():
        res[nmlname] = {}
        for var,d in nml.items():
            if d.keys() == [None]:
                res[nmlname][var] = d[None]
            else:
                print var,d
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


if __name__ == '__main__':
    d = readparameters(*sys.argv[1:])
    wid = max(len(k) for k in d)
    for k,v in d.items():
        print '{0:{1}s} = {2}'.format(k, wid, v)

