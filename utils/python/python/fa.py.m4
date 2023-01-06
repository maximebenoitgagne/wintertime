import builtins
import operator
import itertools as itools
import numpy as np
import numpy.core.umath as umath
try:
    from fractions import gcd
except ImportError:
    try:
        import numpy.core._internal._gcd as gcd
    except ImportError:
        def gcd(a, b):
            """Calculate the greatest common divisor of a and b"""
            while b:
                a, b = b, a%b
            return a

class _FacetUnaryOperation(object):
    def __init__(self, func):
        self.f = func

    def __call__(self, a, *args, **kwargs):
        try:
            facets = a.facets
        except AttributeError:
            return self.f(a, *args, **kwargs)
        else:
            return Facets( self.f(f, *args, **kwargs) for f in facets )

    def __str__ (self):
        return "Facets version of %s" % str(self.f)

class _FacetUnaryOperation2(object):
    " Unary operation with 2 return values "
    def __init__(self, func):
        self.f = func

    def __call__(self, a, *args, **kwargs):
        try:
            facets = a.facets
        except AttributeError:
            return self.f(a, *args, **kwargs)
        else:
            listofpairs = [ self.f(f, *args, **kwargs) for f in facets ]
            facets1,facets2 = zip(*listofpairs)
            return Facets( f for f in facets1 ), Facets( f for f in facets2 )

    def __str__ (self):
        return "Facets version of %s" % str(self.f)

class _FacetBinaryOperation(object):
    def __init__(self, func):
        self.f = func

    def __call__(self, a, b, *args, **kwargs):
        try:
            afacets = a.facets
        except AttributeError:
            try:
                bfacets = b.facets
            except AttributeError:
                return self.f(a, b, *args, **kwargs)
            else:
                return Facets( self.f(a, bf, *args, **kwargs) for bf in bfacets )
        else:
            try:
                bfacets = b.facets
            except AttributeError:
                return Facets( self.f(af, b, *args, **kwargs) for af in afacets )
            else:
                return Facets( self.f(af, bf, *args, **kwargs) for af,bf in zip(afacets,bfacets) )

    def reduce(self, target, axis=None, dtype=None):
        try:
            facets = target.facets
        except AttributeError:
            if axis is None:
                axis = 0
            return self.f.reduce(target,axis,dtype)
        else:
            if axis is None:
                axis = 1
            if axis in [0,-len(facets)]:
                if dtype is None or dtype == facets[0].dtype:
                    res = facets[0].copy()
                else:
                    res = facets[0].astype(dtype)

                for f in facets[1:]:
                    res[:] = self.f(res, f)

                return res
            else:
                if axis > 0:
                    axis -= 1
                return Facets( self.f.reduce(f, axis, dtype) for f in facets )

    def accumulate(self, target, axis=None, dtype=None):
        try:
            facets = target.facets
        except AttributeError:
            if axis is None:
                axis = 0
            return self.f.accumulate(target, axis, dtype)
        else:
            if axis is None:
                axis = 1
            if axis in [0,-len(facets)]:
                if dtype is None:
                    dtype = facets[0].dtype

                tmp = facets[0].astype(dtype)
                res = [ tmp ]
                for f in facets[1:]:
                    tmp = self.f(tmp, f)
                    res.append( tmp )

                return Facets(res)
            else:
                if axis > 0:
                    axis -= 1
                return Facets( self.f.accumulate(f, axis, dtype) for f in facets )

    def outer (self, a, b):
        raise NotImplementedError('outer product operations for Facets')

    def __str__ (self):
        return "Facets version of %s" % str(self.f)

class _FacetBinaryOperation2(object):
    def __init__(self, func):
        self.f = func

    def __call__(self, a, b, *args, **kwargs):
        try:
            afacets = a.facets
        except AttributeError:
            try:
                bfacets = b.facets
            except AttributeError:
                return self.f(a, b, *args, **kwargs)
            else:
                listofpairs = [ self.f(a, bf, *args, **kwargs) for bf in bfacets ]
                facets1,facets2 = zip(*listofpairs)
                return Facets( f for f in facets1 ), Facets( f for f in facets2 )
        else:
            try:
                bfacets = b.facets
            except AttributeError:
                listofpairs = [ self.f(af, b, *args, **kwargs) for af in afacets ]
                facets1,facets2 = zip(*listofpairs)
                return Facets( f for f in facets1 ), Facets( f for f in facets2 )
            else:
                listofpairs = [ self.f(af, bf, *args, **kwargs) for af,bf in zip(afacets,bfacets) ]
                facets1,facets2 = zip(*listofpairs)
                return Facets( f for f in facets1 ), Facets( f for f in facets2 )

    def reduce(self, target, axis=None, dtype=None):
        try:
            facets = target.facets
        except AttributeError:
            if axis is None:
                axis = 0
            return self.f.reduce(target,axis,dtype)
        else:
            if axis is None:
                axis = 1
            if axis in [0,-len(facets)]:
                if dtype is None or dtype == facets[0].dtype:
                    res = facets[0].copy()
                else:
                    res = facets[0].astype(dtype)

                for f in facets[1:]:
                    res[:] = self.f(res, f)

                return res
            else:
                if axis > 0:
                    axis -= 1
                return Facets( self.f.reduce(f, axis, dtype) for f in facets )

    def accumulate(self, target, axis=None, dtype=None):
        try:
            facets = target.facets
        except AttributeError:
            if axis is None:
                axis = 0
            return self.f.accumulate(target, axis, dtype)
        else:
            if axis is None:
                axis = 1
            if axis in [0,-len(facets)]:
                if dtype is None:
                    dtype = facets[0].dtype

                tmp = facets[0].astype(dtype)
                res = [ tmp ]
                for f in facets[1:]:
                    tmp = self.f(tmp, f)
                    res.append( tmp )

                return Facets(res)
            else:
                if axis > 0:
                    axis -= 1
                return Facets( self.f.accumulate(f, axis, dtype) for f in facets )

    def outer (self, a, b):
        raise NotImplementedError('outer product operations for Facets')

    def __str__ (self):
        return "Facets version of %s" % str(self.f)


abs = absolute = _FacetUnaryOperation(umath.absolute)
arccos      = _FacetUnaryOperation(umath.arccos)
arccosh     = _FacetUnaryOperation(umath.arccosh)
arcsin      = _FacetUnaryOperation(umath.arcsin)
arcsinh     = _FacetUnaryOperation(umath.arcsinh)
arctan      = _FacetUnaryOperation(umath.arctan)
arctanh     = _FacetUnaryOperation(umath.arctanh)
around      = _FacetUnaryOperation(np.round_)
ceil        = _FacetUnaryOperation(umath.ceil)
conj = conjugate = _FacetUnaryOperation(umath.conjugate)
cos         = _FacetUnaryOperation(umath.cos)
cosh        = _FacetUnaryOperation(umath.cosh)
exp         = _FacetUnaryOperation(umath.exp)
fabs        = _FacetUnaryOperation(umath.fabs)
floor       = _FacetUnaryOperation(umath.floor)
log10       = _FacetUnaryOperation(umath.log10)
log2        = _FacetUnaryOperation(umath.log2)
log         = _FacetUnaryOperation(umath.log)
logical_not = _FacetUnaryOperation(umath.logical_not)
negative    = _FacetUnaryOperation(umath.negative)
sin         = _FacetUnaryOperation(umath.sin)
sinh        = _FacetUnaryOperation(umath.sinh)
sqrt        = _FacetUnaryOperation(umath.sqrt)
tan         = _FacetUnaryOperation(umath.tan)
tan         = _FacetUnaryOperation(umath.tan)
tanh        = _FacetUnaryOperation(umath.tanh)

deg2rad     = _FacetUnaryOperation(umath.deg2rad)
degrees     = _FacetUnaryOperation(umath.degrees)
exp2        = _FacetUnaryOperation(umath.exp2      )
expm1       = _FacetUnaryOperation(umath.expm1     )
invert      = _FacetUnaryOperation(umath.invert    )
isfinite    = _FacetUnaryOperation(umath.isfinite  )
isinf       = _FacetUnaryOperation(umath.isinf     )
isnan       = _FacetUnaryOperation(umath.isnan     )
log1p       = _FacetUnaryOperation(umath.log1p     )
ones_like   = np.ones_like
rad2deg     = _FacetUnaryOperation(umath.rad2deg   )
radians     = _FacetUnaryOperation(umath.radians   )
reciprocal  = _FacetUnaryOperation(umath.reciprocal)
rint        = _FacetUnaryOperation(umath.rint      )
sign        = _FacetUnaryOperation(umath.sign      )
signbit     = _FacetUnaryOperation(umath.signbit   )
spacing     = _FacetUnaryOperation(umath.spacing   )
square      = _FacetUnaryOperation(umath.square    )
trunc       = _FacetUnaryOperation(umath.trunc     )

# 2 return values
frexp       = _FacetUnaryOperation2(umath.frexp     )
modf        = _FacetUnaryOperation2(umath.modf      )
bitwise_not = invert
# Binary ufuncs ...............................................................
add                  = _FacetBinaryOperation(umath.add)
arctan2              = _FacetBinaryOperation(umath.arctan2)
bitwise_and          = _FacetBinaryOperation(umath.bitwise_and)
bitwise_or           = _FacetBinaryOperation(umath.bitwise_or)
bitwise_xor          = _FacetBinaryOperation(umath.bitwise_xor)
divide               = _FacetBinaryOperation(umath.divide)
equal                = _FacetBinaryOperation(umath.equal)
floor_divide         = _FacetBinaryOperation(umath.floor_divide)
fmod                 = _FacetBinaryOperation(umath.fmod)
greater_equal        = _FacetBinaryOperation(umath.greater_equal)
greater              = _FacetBinaryOperation(umath.greater)
hypot                = _FacetBinaryOperation(umath.hypot)
less_equal           = _FacetBinaryOperation(umath.less_equal)
less                 = _FacetBinaryOperation(umath.less)
logical_and          = _FacetBinaryOperation(umath.logical_and)
logical_or           = _FacetBinaryOperation(umath.logical_or)
logical_xor          = _FacetBinaryOperation(umath.logical_xor)
mod                  = _FacetBinaryOperation(umath.mod)
multiply             = _FacetBinaryOperation(umath.multiply)
not_equal            = _FacetBinaryOperation(umath.not_equal)
power                = _FacetBinaryOperation(umath.power)
remainder            = _FacetBinaryOperation(umath.remainder)
subtract             = _FacetBinaryOperation(umath.subtract)
true_divide          = _FacetBinaryOperation(umath.true_divide)

copysign    = _FacetBinaryOperation(umath.copysign   )
fmax        = _FacetBinaryOperation(umath.fmax       )
fmin        = _FacetBinaryOperation(umath.fmin       )
ldexp       = _FacetBinaryOperation(umath.ldexp      )
left_shift  = _FacetBinaryOperation(umath.left_shift )
maximum     = _FacetBinaryOperation(umath.maximum    )
minimum     = _FacetBinaryOperation(umath.minimum    )
nextafter   = _FacetBinaryOperation(umath.nextafter  )
right_shift = _FacetBinaryOperation(umath.right_shift)

divmod = _FacetBinaryOperation2(divmod)

equal.reduce = None
greater_equal.reduce = None
greater.reduce = None
less_equal.reduce = None
less.reduce = None
not_equal.reduce = None
alltrue = logical_and.reduce
sometrue = logical_or.reduce

def max(arr, axis=None, out=None):
    try:
        return arr.max(axis=axis, out=out)
    except AttributeError:
        return np.asanyarray(arr).max(axis=axis, out=out)

def min(arr, axis=None, out=None):
    try:
        return arr.min(axis=axis, out=out)
    except AttributeError:
        return np.asanyarray(arr).min(axis=axis, out=out)

def where(cnd, a=None, b=None):
    if isinstance(cnd, np.ndarray):
        return np.where(cnd, a, b)
    if a is not None:
        raise ImplementationError()
    nd = cnd.ndim
    idx = [[] for _ in range(nd)]
    for f in range(len(cnd)):
        idxf = np.where(cnd[f])
        idx[0].append(len(idxf[0])*[f])
        for i in range(1,nd):
            idx[i].append(idxf[i-1])
    return tuple(np.concatenate(ii) for ii in idx)


def calc_shapes(shape=(), dims=None, halo=0, extrau=0, extrav=0):
    if dims is None:
        # shape is a tuple of dimensions or lists of dimensions
        # simple dimension get repeated for all faces
        nfacet = builtins.max( np.iterable(d) and len(d) or 1 for d in shape )
        shape = [ np.iterable(d) and d or nfacet*[d] for d in shape ]
        shape[-1] = [ x+2*halo+extrau for x in shape[-1] ]
        shape[-2] = [ x+2*halo+extrav for x in shape[-2] ]
        shapes = zip(*shape)
    else:
        nxs = dims[0::2]
        nys = dims[1::2]
        shapes = [ shape+(ny+2*halo+extrav,nx+2*halo+extrau) for nx,ny in zip(nxs,nys) ]
    return shapes

def calc_shapes2(shape=(), dims=None, halo=0, extrau=0, extrav=0):
    if dims is None:
        # shape is a tuple of dimensions or lists of dimensions
        # first dimension is number of facets
        # simple dimension get repeated for all faces
        nfacet = shape[0]
        shape = [ np.iterable(d) and d or nfacet*[d] for d in shape[1:] ]
        shape[-1] = [ x+2*halo+extrau for x in shape[-1] ]
        shape[-2] = [ x+2*halo+extrav for x in shape[-2] ]
        shapes = zip(*shape)
    else:
        nxs = dims[0::2]
        nys = dims[1::2]
        shapes = [ shape+(ny+2*halo+extrav,nx+2*halo+extrau) for nx,ny in zip(nxs,nys) ]
    return shapes

class Facets(object):
    """
    an array with multiple facets (e.g. cubed-sphere faces).

    To create:

    1. from arrays for facets:

        f1 = [[1., 2.], [3., 4.]]
        f2 = [[5., 6.], [7., 8.]]
        a = Facets([f1, f2])   # 2 facets, each 2x2

    2. empty with given shape:

        u = Facets.empty((10,), 'f', dims=6*[510, 510], halo=1, extrau=1)

       creates a facet array with 6 facets of shape (10, 512, 513) (to be
       interpreted as 510x510 with a halo of 1 and an extra row in x appropriate
       for velocity along x).

    3. zeros:

        u = Facets.zeros((10,), 'f', dims=6*[510, 510], halo=1, extrau=1)

    4. from a global array (using one of MITgcm/exch2's maps):

        arr = np.zeros((10, 510, 3060))
        a = Facets.fromglobal(arr, dims=6*[510, 510], map=-1)

    5. from a binary file:

        a = Facets.fromfile('THETA.data', '>f4', (50,), dims=6*[510, 510])

    Facet arrays can mostly be used like normal numpy arrays.  The first dimension
    is the facet index,

        f1 = a[0]  # first facet
        f2 = a[1]  # second facet

    Slices in higher dimensions apply to all facets.  Use negative indices to refer
    to slices near the upper boundaries, e.g.

        b = a[..., :-1]

    drops the last row in x from all facets of a.

    """
    __array_priority__ = 20

    def __init__(self, arrs, masks=None):
        if masks is not None:
            self.facets = tuple(np.ma.MaskedArray(arr, mask) for arr,mask in zip(arrs,masks))
        else:
            self.facets = tuple(np.asanyarray(arr) for arr in arrs)
        if self.facets:
            self.dtype = self.facets[0].dtype
            for f in self.facets[1:]:
                if f.dtype != self.dtype:
                    raise TypeError('Facets must have same dtype')

    @classmethod
    def empty(cls, shape=(), dtype=None, dims=None, halo=0, extrau=0, extrav=0):
        shapes = calc_shapes2(shape, dims, halo, extrau, extrav)
        return cls( np.empty(sh, dtype) for sh in shapes )

    @classmethod
    def zeros(cls, shape=(), dtype=None, dims=None, halo=0, extrau=0, extrav=0, mask=None):
        shapes = calc_shapes2(shape, dims, halo, extrau, extrav)
        if mask is None:
            zeros = np.zeros
        else:
            zeros = np.ma.zeros
        obj = cls( zeros(sh, dtype) for sh in shapes )
        if mask:
            obj.mask = True
        return obj

    @classmethod
    def fromglobal(cls, arr, shape=None, dims=None, halo=0, extrau=0, extrav=0, missing=None, map=-1):
        if not (halo or extrau or extrav) and dims is None and missing is None:
            return view(arr, shape, map)
        arr = np.asanyarray(arr)
        if shape is None and dims is not None:
            shape = arr.shape[:-2]
        hasmask = hasattr(arr, 'mask') or None
        obj = cls.zeros(shape, arr.dtype, dims, halo, extrau, extrav, mask=hasmask)
        obj.set(arr, map, halo, extrau, extrav)
        if missing is not None:
            obj.mask = obj == missing
        return obj

    @classmethod
    def fromfile(cls, fname, dtype, shape=(), dims=None, halo=0, extrau=0, extrav=0, missing=None, map=-1):
        if not (halo or extrau or extrav) and dims is None and missing is None:
            return view(np.fromfile(fname, dtype), shape, map)
        obj = cls.zeros(shape, dtype, dims, halo, extrau, extrav, missing)
        obj.set(np.fromfile(fname, dtype), map, halo, extrau, extrav)
        if missing is not None:
            obj.mask = obj == missing
        return obj

    @classmethod
    def frombin(cls, fname, dims=None, halo=0, extrau=0, extrav=0, missing=None, map=-1):
        from oj.num import loadbin
        data = loadbin(fname)
        dtype = data.dtype
        shape = data.shape[:-2]
        if not (halo or extrau or extrav) and dims is None and missing is None:
            return view(data, shape, map)
        obj = cls.zeros(shape, dtype, dims, halo, extrau, extrav, missing)
        obj.set(data, map, halo, extrau, extrav)
        if missing is not None:
            obj.mask = obj == missing
        return obj

    def copy(self):
        return self.__class__( f.copy() for f in self.facets )

    def face(self, i):
        return self.facets[i]

    @property
    def data(self):
        return Facets(f.data for f in self.facets)

    @data.setter
    def data(self, value):
        if np.ndim(value) == self.ndim:
            for f in range(self.nfacet):
                self.facets[f].data[...] = value.facets[f]
        else:
            for f in self.facets:
                f.data[...] = value

    def withmask(self, mask=None):
        if mask is None:
            if self.facets and not hasattr(self.facets[0], 'mask'):
                return Facets(self.facets, [np.zeros(s, bool) for s in self.shapes])
            else:
                return self
        else:
            return Facets(self.facets, mask)

    @property
    def mask(self):
        return Facets(f.mask for f in self.facets)

    @mask.setter
    def mask(self, value):
        #if np.ndim(value) == self.ndim:
        vshape = np.shape(value)
        if vshape and vshape[0] == self.nfacet:
            for f in range(self.nfacet):
                self.facets[f].mask = value[f]
        else:
            for f in self.facets:
                f.mask = value

    def filled(self, fill_value=None):
        return self.__class__( f.filled(fill_value) for f in self.facets )

    def astype(self, newtype):
        """
        Returns a copy of the Facets cast to given newtype.

        Returns
        -------
        output : Facets
            A copy of self cast to input newtype.
            The returned record shape matches self.shape.

        """
        newtype = np.dtype(newtype)
        output = self.__class__( f.astype(newtype) for f in self.facets )
        return output

    def __call__(self, indx):
        facets = self.facets[indx]
        ## if indx selects several facets, turn into Facets
        if type(facets) == type(()):
            return Facets(facets)
        else:
            return facets

    def slice_facets(self, *args):
        return Facets( self.facets[f][idx] for f,idx in enumerate(args)
                                           if idx is not None )

    def __getitem__(self, indx):
        if type(indx) == type(()):
            if indx[0] is Ellipsis:
                indx = (self.ndim-len(indx)+1+indx.count(None))*np.s_[:,] + indx[1:]
            if np.iterable(indx[0]):
                # advanced indexing (only pure for now)
                #a = np.broadcast_arrays(*[a for a in indx if np.iterable(a)])
                a = np.broadcast_arrays(*indx)
                res = np.zeros(a[0].shape, self.dtype)
                for f in range(self.nfacet):
                    msk = a[0] == f
                    b = tuple(x[msk] for x in a[1:])
                    res[msk] = self.facets[f][b]
                return res
            else:
                facets = self.facets[indx[0]]
                if type(facets) == type(()):
                    return Facets( a[indx[1:]] for a in facets )
                else:
                    return facets[indx[1:]]
        elif hasattr(indx,'facets'):
            if indx.facets[0].dtype == bool:
                return Facets( f[i] for f,i in zip(self.facets,indx.facets) )
            else:
                return NotImplemented
        else:
            if indx is Ellipsis:
                indx = np.s_[:]
            facets = self.facets[indx]
            ## if indx selects several facets, turn into Facets
            if type(facets) == type(()):
                return Facets(facets)
            else:
                return facets

    def __getslice__(self,i,j):
        return Facets(self.facets[i:j])

    def __setitem__(self, indx, val):
        if isinstance(indx, Facets) and indx[0].dtype == bool:
            # boolean advanced indexing
            if np.ndim(val) == 0:
                for f in range(self.nfacet):
                    self.facets[f][indx[f]] = val
            elif isinstance(val, Facets):
                for f in range(self.nfacet):
                    self.facets[f][indx[f]] = val[f]
            else:
                off = 0
                for f in range(self.nfacet):
                    n = self.facets[f][indx[f]].size
                    self.facets[f][indx[f]] = val[off:off+n]
                    off += n
            return

        if type(indx) != type(()):
            try:
                dtype = indx[0].dtype
            except (TypeError,AttributeError):
                pass
            else:
                if dtype == bool:
                    if hasattr(val,'facets'):
                        for i,facet in enumerate(self.facets):
                            facet[indx[i]] = val[i]
                    else:
                        for i,facet in enumerate(self.facets):
                            facet[indx[i]] = val
                    return
                else:
                    raise NotImplementedError('integer indexing with Facets')

            indx = (indx,)

        if indx[0] is Ellipsis:
            indx = (self.ndim-len(indx)+1+indx.count(None))*np.s_[:,] + indx[1:]
        facets = self.facets[indx[0]]
        if type(facets) == type(()):
            if hasattr(val,'facets'):
                assert len(facets) == len(val.facets)
                for i,facet in enumerate(facets):
                    facet[indx[1:]] = val[i]
            else:
                for facet in facets:
                    facet[indx[1:]] = val
        else:
            # single facet
            facets[indx[1:]] = val

    @property
    def nfacet(self):
        return len(self.facets)

    def __len__(self):
        return len(self.facets)

    @property
    def ndimfacet(self):
        try:
            facet = self.facets[0]
        except IndexError:
            return 0

        return facet.ndim

    @property
    def ndim(self):
        return self.ndimfacet + 1

    @property
    def shapes(self):
        return [ f.shape for f in self.facets ]

    @property
    def dims(self):
        return [x for shape in self.shapes for x in shape[:-3:-1]]

    @property
    def shape(self):
        dims = zip(*self.shapes)
        return (self.nfacet,) + tuple( np.std(d) and d or d[0] for d in dims )

    def set(self, arr, map=-1, halo=0, extrau=0, extrav=0):
        arr = np.asanyarray(arr)

        # compute shape of and slices into global array
        gshape,slices = globalmap(self.shapes, map, halo, extrau, extrav)

        if arr.ndim == 1:
            try:
                arr = arr.reshape(gshape)
            except ValueError:
                raise ValueError('Cannot reshape array of {} elements to shape {}'.format(arr.size, gshape))
        else:
            # check for correct shape
            if map > 0:
                # stacked (and folded) in y -- flatten last 2 to make equiv to map==0
                arr = arr.reshape(arr.shape[:-2]+(-1,))
            if arr.shape != gshape:
                raise ValueError('Unexpected shape: ' + str(arr.shape) + ' ' + str(gshape))
        for i,s in enumerate(slices):
            ny,nx = self[i].shape[-2:]
            #nyi = ny-2*halo-extrav
            #nxi = nx-2*halo-extrau
            #self[i][..., halo:ny-halo-extrav, halo:nx-halo-extrau].flat = arr[s].reshape(nyi*nxi)
            self[i][..., halo:ny-halo-extrav, halo:nx-halo-extrau] = arr[s]

    def setfromfile(self, fname, dtype, map=-1, halo=0, extrau=0, extrav=0, offset=None):
        """set facet array from a global file

        if offset is given, start reading at this byte offset in the file
        and read only as much data as required

        """
        if offset is not None:
            gshape,slices = globalmap(self.shapes, map, halo, extrau, extrav)
            count = reduce(operator.mul, gshape)
            with open(fname) as f:
                f.seek(offset)
                data = np.fromfile(f, dtype, count=count)
                self.set(data, map, halo, extrau, extrav)
        else:
            self.set(np.fromfile(fname, dtype), map, halo, extrau, extrav)

    def __repr__(self):
        #return '<Facets(' + ', '.join( str(f.shape) for f in self.facets ) + ') at 0x{0:x}>'.format(id(self))
#        return 'Facets(' + ',\n\n       '.join( str(f).replace('\n','\n       ') for f in self.facets ) + ')'
        return 'Facets(' + ',\n\n       '.join( '(' + str(f).replace('\n','\n       ') + ",dtype='" + str(f.dtype) + "')" for f in self.facets ) + ')'

    def __str__(self):
        return '[' + '\n\n '.join( f.__str__().replace('\n','\n ') for f in self.facets ) + ']'

    def max(self, axis=None, out=None, fill_value=None):
        if hasattr(self, 'mask'):
            if fill_value is None:
                fill_value = np.ma.maximum_fill_value(self)
            arr = self.filled(fill_value)
        else:
            arr = self
        if axis is None:
            res = np.max([ np.max(f) for f in arr.facets if f.size ])
            if out is not None:
                out[:] = res
            return res
        elif axis in [0, -len(arr.facets) ]:
            # won't work unless facets all have same shape
            return np.maximum.reduce(arr.facets, out=out)
        else:
            if axis > 0:
                axis = axis - 1
            if out is None:
                return Facets( f.max(axis=axis) for f in arr.facets )
            else:
                for i,f in enumerate(arr.facets):
                    out[i] = f.max(axis=axis)
                return out

    def min(self, axis=None, out=None, fill_value=None):
        if hasattr(self, 'mask'):
            if fill_value is None:
                fill_value = np.ma.minimum_fill_value(self)
            arr = self.filled(fill_value)
        else:
            arr = self
        if axis is None:
            res = np.min([ np.min(f) for f in arr.facets if f.size ])
            if out is not None:
                out[:] = res
            return res
        elif axis in [0, -len(arr.facets) ]:
            # won't work unless facets all have same shape
            return np.minimum.reduce(arr.facets, out=out)
        else:
            if axis > 0:
                axis = axis - 1
            if out is None:
                return Facets( f.min(axis=axis) for f in arr.facets )
            else:
                for i,f in enumerate(arr.facets):
                    out[i] = f.min(axis=axis)
                return out

    def sum(self, axis=None, dtype=None, out=None):
        if axis is None:
            res = np.sum([ np.sum(f, dtype=dtype) for f in self.facets ])
            if out is not None:
                out[:] = res
            return res
        elif axis in [0, -len(self.facets) ]:
            # won't work unless facets all have same shape
            return np.add.reduce(self.facets, dtype=dtype, out=out)
        else:
            if axis > 0:
                axis = axis - 1
            if out is None:
                return Facets( f.sum(axis=axis, dtype=dtype) for f in self.facets )
            else:
                for i,f in enumerate(self.facets):
                    out[i] = f.sum(dtype=dtype, axis=axis)
                return out

    def __abs__(self): return abs(self)

    def __add__(self, other):
        "Add other to self."
        return add(self, other)
    #
    def __radd__(self, other):
        "Add self to other."
        return add(other, self)
    #
    def __sub__(self, other):
        "Subtract other from self."
        return subtract(self, other)
    #
    def __rsub__(self, other):
        "Subtract self from other."
        return subtract(other, self)
    #
    def __mul__(self, other):
        "Multiply self by other."
        return multiply(self, other)
    #
    def __rmul__(self, other):
        "Multiply other by self."
        return multiply(other, self)
    #
    def __div__(self, other):
        "Divide self by other."
        return divide(self, other)
    #
    def __rdiv__(self, other):
        "Divide other by self."
        return divide(other, self)
    #
    def __truediv__(self, other):
        "Divide self by other."
        return true_divide(self, other)
    #
    def __rtruediv__(self, other):
        "Divide other by self."
        return true_divide(other, self)
    #
    def __floordiv__(self, other):
        "Divide self by other."
        return floor_divide(self, other)
    #
    def __rfloordiv__(self, other):
        "Divide other by self."
        return floor_divide(other, self)
    #
    def __divmod__(self, other):
        "divmod self by other."
        return divmod(self, other)
    #
    def __rdivmod__(self, other):
        "divmod other by self."
        return divmod(other, self)
    #
    def __pow__(self, other):
        "Raise self to the power other."
        return power(self, other)
    #
    def __rpow__(self, other):
        "Raise other to the power self."
        return power(other, self)
    #
    def __eq__(self, other): return equal(self, other)
    def __ne__(self, other): return not_equal(self, other)
    def __lt__(self, other): return less (self, other)
    def __gt__(self, other): return greater(self, other)
    def __le__(self, other): return less_equal(self, other)
    def __ge__(self, other): return greater_equal(self, other)
    def __and__(self, other): return bitwise_and(self, other)
    def __rand__(self, other): return bitwise_and(self, other)
    def __or__(self, other): return bitwise_or(self, other)
    def __ror__(self, other): return bitwise_or(self, other)
    def __xor__(self, other): return bitwise_xor(self, other)
    def __rxor__(self, other): return bitwise_xor(self, other)
    def __mod__(self, other): return mod(self, other)
    def __rmod__(self, other): return mod(other, self)
    def __pos__(self): return self
    def __neg__(self): return negative(self)
    def __invert__(self): return invert(self)
    ## ............................................
m4_define(IOP,`#
    def $1(self, other):
        $2
        try:
            facets = other.facets
        except AttributeError:
            for f in self.facets:
                f.$1(other)
        else:
            for f,of in zip(self.facets, facets):
                f.$1(of)
        return self')m4_dnl
    IOP(__iadd__,"Add other to self in-place.")
    IOP(__isub__,"Subtract other from self in-place.")
    IOP(__imul__,"Multiply self by other in-place.")
    IOP(__idiv__,"Divide self by other in-place.")
    IOP(__ifloordiv__,"Floor divide self by other in-place.")
    IOP(__itruediv__,"True divide self by other in-place.")
    IOP(__ipow__,`"Raise self to the power other, in place."')
    IOP(__iand__,"And in-place.")
    IOP(__ior__,"Or in-place.")
    IOP(__ixor__,"Xor in-place.")
    IOP(__imod__,"Mod in-place.")

    def addhalo(self,extra=[]):
        try:
            iter(extra)
        except TypeError:
            # if just a number, assume equal halos in last 2 dimensions
            extra = 2*[extra]

        extra = np.r_[(self.ndimfacet-len(extra))*[0],  extra]
        res = self.__class__( np.ndarray.__new__(f.__class__, sh+2*extra, f.dtype) for sh,f in zip(self.shapes,self.facets) )

        s = tuple( e and np.s_[e:-e] or np.s_[:] for e in extra )
        for i in range(self.nfacet):
            res.facets[i][s] = self.facets[i]

        return res

    def toglobal(self,out=None,dtype=None,map=-1,halo=0,extrau=0,extrav=0):
        if halo or extrau or extrav:
            self = self[..., halo:(-halo-extrav or None), halo:(-halo-extrau or None)]

        if dtype is None:
            dtype = self.facets[0].dtype

        if map == -1:
            # stacked in x
            nflatdim = 1
        else:
            # facets concatenated
            nflatdim = 2

        facetshapes = self.shapes
        dimlists = zip(*facetshapes)
        gshape,slices = globalmap(facetshapes,map)
        if out is None:
            res = np.zeros(gshape,dtype).view(self.facets[0].__class__)
        else:
            if out.ndim == 1:
                res = out.reshape(gshape)
            else:
                if map == 0:
                    res = out
                else:
                    # stacked (and folded) in y -- flatten 2d
                    res = out.reshape(arr.shape[:-2]+(-1,))

                if res.shape != gshape:
#                    print maxdims
#                    print flatdims
#                    print flatbounds
                    raise ValueError('Unexpected shape: ' + str(res.shape) + ' ' + str(gshape))

        for s,f in zip(slices,self.facets):
            if map >= 0:
                f = f.reshape(f.shape[:-2] + (-1,))

            try:
                res[s] = f
            except ValueError:
                raise ValueError('shape mismatch: ' + str(res[s].shape) + ' ' + str(f.shape))

        if map > 0:
            nx = reduce(gcd, dimlists[-1])
            res = res.reshape(res.shape[:-1] + (-1,nx))

        return res


def globalmap(facetshapes, map=-1, halo=0, extrau=0, extrav=0):
    if map == -1:
        # stacked in x
        nflatdim = 1
    else:
        # facets concatenated
        nflatdim = 2

    if halo > 0 or extrau > 0 or extrav > 0:
        ndim = len(facetshapes[0])
        halo0 = (ndim-2)*[0] + [halo, halo]
        halo1 = (ndim-2)*[0] + [halo+extrav, halo+extrau]
        facetshapes = [
                [ d-h0-h1 for d,h0,h1 in zip(shape, halo0, halo1) ]
                for shape in facetshapes ]

    dimlists = zip(*facetshapes)
    maxdims = tuple( max(d) for d in dimlists[:-nflatdim] )
    flatdims = [ int(np.prod(sh[-nflatdim:])) for sh in facetshapes ]
    flatbounds = [0] + list(np.cumsum(flatdims))
    gshape = maxdims + (flatbounds[-1],)

#    starts = [ (ndim-nflatdim)*(0,) + (s,) for s in flatbounds[:-1] ]
#    ends = [ sh[:-nflatdim] + (e,) for sh,e in zip(facetshapes,flatbounds[1:]) ]
#    facetslices = [ tuple( np.s_[s:e] for s,e in zip(ss,ee) ) for ss,ee in zip(starts,ends) ]
    slices = [ tuple( np.s_[0:d] for d in sh[:-nflatdim] ) + np.s_[s:e,]
               for sh,s,e in zip(facetshapes,flatbounds[:-1],flatbounds[1:]) ]

    return gshape, slices


array = Facets
empty = Facets.empty
zeros = Facets.zeros
fromfile = Facets.fromfile
frombin = Facets.frombin
fromglobal = Facets.fromglobal

def view(arr, shape, map=-1):
    """return a view of a numpy array as a Facets array"""
#    dimlists = np.broadcast_arrays(*shape[1:])
#    facetshapes = zip(*dimlists)
#    # compute shape of and slices into global array
    facetshapes = calc_shapes2(shape)
    gshape,slices = globalmap(facetshapes,map)

    if arr.ndim == 1:
        arr = arr.reshape(gshape)
    else:
        # check for correct shape
        if map > 0:
            # stacked (and folded) in y -- flatten last 2 to make equiv to map==0
            arr = arr.reshape(arr.shape[:-2]+(-1,))

        if arr.shape != gshape:
            raise ValueError('Unexpected shape: ' + str(arr.shape) + ' ' + str(gshape))

    self = Facets( arr[s] for s in slices )
    return self


def apply(func, *args, **kwargs):
    def mkargs(args, f):
        for arg in args:
            if isinstance(arg, Facets):
                yield arg[f]
            else:
                yield arg

    fargs = [ arg for arg in args if isinstance(arg, Facets) ]
    if len(fargs):
        nfacet = fargs[0].nfacet
        res = [ func(*mkargs(args, f), **kwargs) for f in range(nfacet) ]
        if type(res[0]) == type(()):
            return tuple(Facets(arrs) for arrs in zip(*res))
        else:
            try:
                return Facets(res)
            except AttributeError:
                return res
    else:
        return func(*args, **kwargs)

def masked(arr, mask):
    return Facets(np.ma.MaskedArray(arr[f], mask[f]) for f in range(len(arr)))

def diff(a, n=1, axis=-1):
    if n == 0:
        return a
    if n < 0:
        raise ValueError(
                "order must be non-negative but got " + repr(n))
    if not isinstance(a, Facets):
        a = Facets(a)
    nd = len(a.shape)
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    if n > 1:
        return diff(a[slice1]-a[slice2], n-1, axis=axis)
    else:
        return a[slice1]-a[slice2]


class MITGrid(object):
    """Example:

    grid = MITGrid('tile{0:03d}.mitgrid', 6*[510, 510])
    xg = grid.xg
    yg = grid.yg
    ac = grid.load('ac')
    az = Facets.zeros((), 'f', dims=grid.dims, extrau=1, extrav=1, halo=1)
    grid.set(az, 'az', halo=1)

    """
    _fldnames = ['xc', 'yc', 'dxf', 'dyf', 'ac', 'xg', 'yg', 'dxv', 'dyu', 'az', 'dxc', 'dyc', 'aw', 'as', 'dxg', 'dyg', 'anglecs', 'anglesn']
    _smate    = {'dxc':'dyc', 'dyg':'dxg', 'aw':'as'}
    _wmate    = {'dyc':'dxc', 'dxg':'dyg', 'as':'aw'}
    _zfields  = ['xg','yg','dxv','dyu','az']
    _cfields  = ['xc','yc','dxf','dyf','ac','anglecs','anglesn']
    _end = dict.fromkeys(_cfields, (-1,-1))
    _end.update(dict.fromkeys(_smate, (-1,None)))
    _end.update(dict.fromkeys(_wmate, (None,-1)))
    _end.update(dict.fromkeys(_zfields, (None,None)))
    _extra = dict((k, tuple(e is None and 1 or e+1 for e in ee)) for k,ee in _end.items())

    def __init__(self, files, dims, dtype='>f8'):
        self.files = files
        self.file_dtype = np.dtype(dtype)
        self.dims = dims
        self.nx = dims[0::2]
        self.ny = dims[1::2]
        self.shapes = [(ny+1,nx+1) for nx,ny in zip(self.nx, self.ny)]
        self._count = [(nx+1)*(ny+1) for nx,ny in zip(self.nx, self.ny)]
        self.nfaces = len(self.nx)
        self._fields = dict()

        if len(self.files) != self.nfaces:
            self.files = [ files.format(i+1) for i in range(self.nfaces) ]

    def load(self, name, dtype=float):
        skip = self._fldnames.index(name)
        endy,endx = self._end[name]
        arrs = []
        for f in range(self.nfaces):
            if self.nx[f] > 0 and self.ny[f] > 0:
                count = self._count[f]
                with open(self.files[f]) as fid:
                    fid.seek(skip*count*self.file_dtype.itemsize)
                    arr = np.fromfile(fid, self.file_dtype, count=count)
                try:
                    arr = arr.reshape(self.shapes[f])
                except ValueError:
                    raise IOError("fa.MITGrid: could not read enough data for %s" % name)
                arrs.append(arr[:endy, :endx].astype(dtype))
        return Facets(arrs)

    def set(self, farr, name, halo=0):
        skip = self._fldnames.index(name)
        extray, extrax = self._extra[name]
        for f in range(self.nfaces):
            if self.nx[f] > 0 and self.ny[f] > 0:
                count = self._count[f]
                with open(self.files[f]) as fid:
                    fid.seek(skip*count*self.file_dtype.itemsize)
                    arr = np.fromfile(fid, self.file_dtype, count=count)
                try:
                    arr = arr.reshape(self.shapes[f])
                except ValueError:
                    raise IOError("fa.MITGrid: could not read enough data for %s" % name)
                nx = builtins.min(farr[f].shape[-1] - 2*halo, self.nx[f] + extrax)
                ny = builtins.min(farr[f].shape[-2] - 2*halo, self.ny[f] + extray)
#                print f, halo, ny, nx
                farr[f, halo:halo+ny, halo:halo+nx] = arr[:ny, :nx]
 
    def __getattr__(self, name):
        if name.lower() in self._fldnames:
            return self.load(name.lower())
        else:
            raise AttributeError("'MITGrid' object has no attribute '" + name + "'")


def pcolormesh(x, y, c, facets=None, axes=None, hold=None, pcolor=None, **pcolorargs):
    import matplotlib.pyplot as plt
    if axes is None: axes = plt.gca()
    if facets is None: facets = range(x.nfacet)
    washold = axes.ishold()
    if hold is not None:
        axes.hold(hold)
    if pcolor is None:
        pcolor = axes.__class__.pcolormesh
    try:
        qms = [ pcolor(axes, x[f], y[f], c[f], **pcolorargs) for f in facets ]
        plt.draw_if_interactive()
    finally:
        axes.hold(washold)
    # fix unequal autoscaling of facets unless rgb
    if c.ndim == 3:
        vmins = [qm.norm.vmin for qm in qms]
        vmin = min(vmins)
        if max(vmins) > vmin:
            for qm in qms:
                qm.norm.vmin = vmin
        vmaxs = [qm.norm.vmax for qm in qms]
        vmax = max(vmaxs)
        if min(vmaxs) < vmax:
            for qm in qms:
                qm.norm.vmax = vmax
    # good enough for connecting a colorbar
    axes._sci(qms[-1])
    return qms

