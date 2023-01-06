import numpy as np

def lcmp(a,b):
    return cmp(a[0].lower(),b[0].lower())

class UnitRecArray(np.recarray):
#    def __new__(subtype, shape, dtype=None, buf=None, offset=0, strides=None,
#                formats=None, names=None, titles=None,
#                byteorder=None, aligned=False, units=None):
#        obj = np.recarray.__new__(subtype, shape, dtype, buf, offset, strides,
#                                  formats, names, titles, byteorder, aligned)
#        if units is not None and not hasattr(units,'items'):
#            units = { k:u for k,u in zip(obj.dtype.names, units) }
#        obj.units = units
#        return obj

    def __new__(cls, input_array, units=None):
        obj = np.asarray(input_array).view(cls)
        if units is not None and not hasattr(units,'items'):
            units = dict( (k,u) for k,u in zip(obj.dtype.names, units) )
        if units:
            units = dict((k,v) for k,v in units.items() if k in obj.dtype.names)
        obj.units = units
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        units = getattr(obj, 'units', None)
        self.units = units

    def pretty(self):
        us = self.units
        vs = self[:1]
        w = max([len(k) for k in us])
        for k,u in sorted(us.items(),lcmp):
            v = float(getattr(vs,k))
            if u.endswith('/s'):
                v *= 86400.
                u = u[:-1] + 'd'
            if u.endswith('*s') or u == 's':
                v /= 86400.
                u = u[:-1] + 'd'
            v = str(v)
            if v == '0.0': v = '0.'
            if '.' in v[:5]:
                v = (5-v.index('.'))*' ' + v
            print '{0:{1}} : {2} {3}'.format(k,w,v,u)



def recfromdict(d,names=None):
    if names is None:
        names = d.keys()
    return np.rec.fromrecords([tuple(d[k] for k in names)],names=names)


def recfromlist(dl,names=None,units=None):
    if names is None:
        names = dl[0].keys()
    if units is not None:
        try:
            units = dict( (k,units[k]) for k in names )
        except TypeError:
            units = dict( (k,t) for k,t in zip(names,units) )
    res = np.rec.fromrecords([tuple(d[k] for k in names) for d in dl],names=names)
    return UnitRecArray(res, units)


def recfromlists(dls,names=None):
    if names is None:
        names = [dl[0].keys() for dl in dls]
    rows = []
    for i in xrange(len(dls[0])):
        row = sum([ tuple(dl[i][k] for k in nam) for dl,nam in zip(dls,names) ], ())
        rows.append(row)
    print rows
    return np.rec.fromrecords(rows, names=sum(names,[]))


def dictfromrec(ra):
    return [ dict( (k,r[k]) for k in ra.dtype.names ) for r in ra]


def rcat(ras):
    names = ras[0].dtype.names
    return np.rec.fromrecords([tuple(r) for ra in ras for r in ra],names=names)


class AttributeDict(dict):
    def __getitem__(self,key):
        return dict.__getitem__(self, key)
    def __setitem__(self,key,val):
        return dict.__setitem__(self, key, val)
    def __delitem__(self,key):
        return dict.__delitem__(self, key)
    def __getattr__(self,key):
        return dict.__getitem__(self, key)
    def __setattr__(self,key,val):
        return dict.__setitem__(self, key, val)
    def __delattr__(self,key):
        return dict.__delitem__(self, key)
    def __contains__(self,key):
        return dict.__contains__(self, key)
    def __dir__(self):
        return self.keys()
    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).iteritems():
            self[k] = v


class CaselessDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self)
        if len(args) == 1:
            arg = args[0]
            try:
                arg = arg.iteritems()
            except AttributeError:
                pass
            for k,v in arg:
                self[k.lower()] = v
        self.update(**kwargs)
    def __getitem__(self,key):
        return dict.__getitem__(self, key.lower())
    def __setitem__(self,key,val):
        return dict.__setitem__(self, key.lower(), val)
    def __delitem__(self,key):
        return dict.__delitem__(self, key.lower())
    def __getattr__(self,key):
        return dict.__getitem__(self, key.lower())
    def __setattr__(self,key,val):
        return dict.__setitem__(self, key.lower(), val)
    def __delattr__(self,key):
        return dict.__delitem__(self, key.lower())
    def __contains__(self,key):
        return dict.__contains__(self, key.lower())
    def __dir__(self):
        return self.keys()
    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).iteritems():
            self[k.lower()] = v


def zipdicts(l):
    names = l[0].keys()
    return dict((k, np.array([d[k] for d in l])) for k in names)

class DictArray():
    '''
    A dictionary of arrays, hopefully with some common dimensions.

    Key access is via the __call__ method, i.e., round parenthesis.
    Slicing acts on all arrays simultaneously.
    Item assignment is not supported.

    Constructors
    ------------
    DictArray(d) :: from a dictionary of arrays
    fromlist(l)  :: from a list of dictionaries (will add list dimensions to arrays)
    '''

    dict = None

    def __init__(self, d, caseless=False, dictclass=None):
        if dictclass is None:
            if caseless:
                dictclass = CaselessDict
            else:
                dictclass = AttributeDict
        self.__dict__['dict'] = dictclass(d)

    def __getitem__(self, ind):
        return DictArray(((k, v[ind]) for k,v in self.dict.items()),
                         dictclass=type(self.dict))

    @property
    def shape(self):
        return self.dict.itervalues().next().shape

    def __call__(self, k):
        return self.dict[k]

    def __getattr__(self, k):
        return self.dict.__getitem__(k)

    def __setattr__(self, k, v):
        self.dict.__setitem__(self, k, v)

    def __delattr__(self, k):
        return self.dict.__delitem__(k)

    def keys(self):
        return self.dict.keys()

    def __dir__(self):
        return self.keys()

    def __contains__(self,key):
        return self.dict.__contains__(key)

    def update(self, *args, **kwargs):
        return self.dict.update(*args, **kwargs)

    def __repr__(self):
        return repr(self.dict)

    def __str__(self):
        return str(self.dict)

    @classmethod
    def fromlist(cls, l, caseless=False):
        return cls(zipdicts(l), caseless)

