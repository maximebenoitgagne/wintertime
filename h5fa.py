import sys
import numpy as np
from h5py import File
from h5py._hl.group import Group
import fa

py3 = sys.version_info[0] == 3

# copied from older h5py._hl.base
class DictCompat(object):

    """
        Contains dictionary-style compatibility methods for groups and
        attributes.
    """

    def get(self, name, default=None):
        """ Retrieve the member, or return default if it doesn't exist """
        try:
            return self[name]
        except KeyError:
            return default

    if py3:
        def keys(self):
            """ Get a view object on member names """
            return KeyView(self)

        def values(self):
            """ Get a view object on member objects """
            return ValueView(self)

        def items(self):
            """ Get a view object on member items """
            return ItemView(self)

    else:
        def keys(self):
            """ Get a list containing member names """
            return list(self)

        def iterkeys(self):
            """ Get an iterator over member names """
            return iter(self)

        def values(self):
            """ Get a list containing member objects """
            return [self.get(x) for x in self]

        def itervalues(self):
            """ Get an iterator over member objects """
            for x in self:
                yield self.get(x)

        def items(self):
            """ Get a list of tuples containing (name, object) pairs """
            return [(x, self.get(x)) for x in self]

        def iteritems(self):
            """ Get an iterator over (name, object) pairs """
            for x in self:
                yield (x, self.get(x))


def isFaVar(obj):
    if not isinstance(obj, Group):
        return False
    for i,x in enumerate(obj):
        if x != str(i): return False
    return True
        
class FaVar(object):
    def __init__(self, grp):
        self.group = grp
        self.nfacet = len(grp)

    def __getitem__(self, idx):
        '''first index is facet index'''
        if not isinstance(idx, tuple):
            idx = (idx,)
        if idx[0] is Ellipsis:
            idx = (slice(None),) + idx
        fo = np.arange(self.nfacet)[idx[0]]
        idx = idx[1:]
        try:
            iter(fo)
        except TypeError:
            return self.group[str(fo)][idx]
        else:
            return fa.Facets(self.group[str(f)][idx] for f in fo)

    def get(self, idx=(), halo=0):
        obj = self.__getitem__(idx)
        if halo:
            obj = obj.addhalo(halo)
        return obj

    def __setitem__(self, idx, obj):
        '''first index is facet index'''
        if not isinstance(idx, tuple):
            idx = (idx,)
        if idx[0] is Ellipsis:
            idx = (slice(None),) + idx
        fo = np.arange(self.nfacet)[idx[0]]
        idx = idx[1:]
        for fi,f in enumerate(fo):
            self.group[str(f)][idx] = obj[fi]

    @property
    def shapes(self):
        shapes = [a.shape for a in self.group.values()]
        return shapes

    @property
    def shape(self):
        dims = zip(*self.shapes)
        return (self.nfacet,) + tuple( np.std(d) and d or d[0] for d in dims )

    @property
    def dtype(self):
        return self.group['0'].dtype

    def resize(self, shape, axis=None):
        try:
            iter(shape)
        except:
            if axis is None: axis = 0
            for f in range(self.nfacet):
                v = self.group[str(f)]
                sh = list(v.shape)
                sh[axis] = shape
                v.resize(sh)
        else:
            for f in range(self.nfacet):
                v = self.group[str(f)]
                v.resize(shape)


class FaFile(DictCompat):
    _align = (1024*768, 1048576)
    _cache = (0, 300, 300*1024*1024, 0.75)
    _meta = 1048576

    def __init__(self, *args, **kwargs):
        if len(args) < 2:
            kwargs.setdefault('mode', 'r')
        kwargs.setdefault('align', self._align)
        kwargs.setdefault('cache', self._cache)
        kwargs.setdefault('meta', self._meta)
        try:
            self.file = File(*args, **kwargs)
        except TypeError:
            del kwargs['align']
            del kwargs['cache']
            del kwargs['meta']
            self.file = File(*args, **kwargs)

    def __getitem__(self, name):
        obj = self.file.__getitem__(name)
        if isFaVar(obj):
            return FaVar(obj)
        else:
            return obj

    get = __getitem__

    def create_favar(self, name, shape=None, dtype=None, dims=None, halo=0,
            extrau=0, extrav=0, data=None, chunks=None, maxshape=None,
            chunkdims=None, maxdims=None, **kwds):
        kw = dict(kwds)
        g = self.file.create_group(name)
        if data is not None:
            if dtype is None: dtype = data.dtype
            shapes = data.shapes
            dims = data.dims
        else:
            shapes = fa.calc_shapes2(shape, dims, halo, extrau, extrav)
        if chunkdims is None: chunkdims = dims
        if maxdims is None: maxdims = dims
        if chunkdims is not None and chunks is None:
            chunks = (len(shapes[0])-2)*(1,)
        if chunks is not None:
            chunks = fa.calc_shapes2(chunks, chunkdims, halo, extrau, extrav)
        if maxshape is not None:
            maxshape = fa.calc_shapes2(maxshape, maxdims, halo, extrau, extrav)
        for f in range(len(shapes)):
            if data is not None: kw['data'] = data[f]
            if chunks is not None: kw['chunks'] = chunks[f]
            if maxshape is not None: kw['maxshape'] = maxshape[f]
            g.create_dataset(str(f), shapes[f], dtype, **kw)
        return FaVar(g)

    def __setitem__(self, name, obj):
        if isinstance(obj, fa.Facets):
            if not name in self.file:
                self.file.create_group(name)
            for f in range(obj.nfacet):
                self.file[name][str(f)] = obj[f]
        else:
            self.file.__setitem__(name, obj)

    def __delitem__(self, name):
        self.file.__delitem__(name)

    def __iter__(self):
        for x in self.file:
            yield x

    def __contains__(self, name):
        return name in self.file

    def id(self):
        return self.file.id

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.file:
            self.close()

    def __del__(self):
        if getattr(self, 'file', False):
            self.close()

    def flush(self):
        self.file.flush()

