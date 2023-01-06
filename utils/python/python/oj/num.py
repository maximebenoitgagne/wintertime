#!/usr/bin/env python
from __future__ import print_function
import os
import sys, time, gzip
import re
import getopt as go
from glob import glob
import numpy as np
from numpy import frombuffer, fromfile
from numpy.ma import MaskedArray
from numpy import newaxis as N
try:
    from bottleneck import nanargmin, nanargmax
except ImportError:
    from numpy import nanargmin, nanargmax

_debug = False


def sumclasses(a, classes, axis=0):
    if axis < 0:
        axis += a.ndim
    oshape = (max(classes) + 1,) + a.shape[:axis] + a.shape[axis+1:]
    res = np.zeros(oshape, a.dtype)
    for i in range(a.shape[axis]):
        res[classes[i]] += a.take(i, axis=axis)

    return res

def getopt(shortopts=None, longopts=None, args=None, doc=None, narg=None):
    if shortopts is None:
        m = re.search(r'\[([^]]*)\]', doc)
        if m is None:
            sys.stderr.write('Warning: getopt: no options found in doc string\n')
            shortopts = ''
        else:
            shortopts = m.group(1).lstrip('-')
    if longopts is None:
        longopts = []
        if doc is not None:
            m = re.search(r'\[--([^]]*)\]', doc)
            if m:
                longopts = m.group(1).split()
    if args is None:
        args = sys.argv[1:]
    opts,args = go.gnu_getopt(args, shortopts, longopts)
    if narg is not None and doc is not None:
        if len(args) != narg:
            sys.exit(doc)
    return dict(opts), args

def pathcomps(path):
    comps = []
    path,name = os.path.split(path)
    comps[:0] = [name]
    while len(path):
        path,name = os.path.split(path)
        if len(name):
            comps[:0] = [name]
        else:
            comps[0] = path + comps[0]
            break
    return comps

def fromgzipfile(file, dtype=float, count=-1, offset=0):
    fid = gzip.open(file, 'rb')
    res = frombuffer(fid.read(), dtype, count, offset)
    fid.close()
    return res

#def myfromfile(file, dtype=float, count=-1):
#    if file.endswith('.gz'):
#        return fromgzipfile(file, dtype, count)
#    elif os.path.exists(file):
#        return fromfile(file, dtype, count)
#    elif os.path.exists(file + '.gz'):
#        print 'myfromfile: reading', file + '.gz'
#        return fromgzipfile(file + '.gz', dtype, count)
#    else:
#        # this will most likely raise an IOError
#        return fromfile(file, dtype, count)

def myfromfid(fid, dtype=float, shape=None, count=-1, skip=-1):
    size = np.dtype(dtype).itemsize
    if shape is not None:
        size *= np.prod(shape)
        if count >= 0:
            shape = (count,) + shape
        if count >= 0 or skip >= 0:
            count = np.prod(shape)
 
    if skip > 0:
        fid.seek(skip*size)

#    print count, shape
    a = fromfile(fid, dtype, count)

    if shape is not None:
        a = a.reshape(shape)

    return a


def myfromfile(filename, dtype=float, shape=None, count=-1, skip=-1, skipbytes=0):
    zipped = False
    if filename.endswith('.gz'):
        zipped = True
    elif os.path.exists(filename):
        zipped = False
    elif os.path.exists(filename + '.gz'):
        if _debug: print('myfromfile: reading', filename + '.gz')
        zipped = True
        filename = filename + '.gz'
    else:
        # this will most likely raise an IOError
        pass

    if zipped:
        openf = gzip.open
    else:
        openf = open

    countbytes = -1

    size = np.dtype(dtype).itemsize
    if shape is not None:
        size *= np.prod(shape)
        if count >= 0:
            shape = (count,) + shape
        if count >= 0 or skip >= 0 or skipbytes > 0:
            count = np.prod(shape)

    if skip > 0:
        skipbytes += skip*size
 
# gzip doesn't support the 'with', so we do it ourselves       
#    with openf(file, 'rb') as fid:
#        if skip > 0:
#            size = np.dtype(dtype).itemsize
#            if shape is not None:
#                size *= np.prod(shape)
#            fid.seek(skip*size)
#        a = fromfile(fid, dtype, count)

    fid = openf(filename, 'rb')
    exc = True
    try:
        try:
            if skipbytes > 0:
                fid.seek(skipbytes)
#            print dtype,count
            if zipped:
                a = frombuffer(fid.read(), dtype, count)
            else:
                a = fromfile(fid, dtype, count)
        except:
            exc = False
            fid.close()
            raise
    finally:
        if exc:
            fid.close()

    if shape is not None:
        try:
            a = a.reshape(shape)
        except ValueError:
            raise ValueError('{f}: read {r} items, need {n} for shape {s}'.format(
                f=filename, r=a.size, n=np.prod(shape), s=str(shape)))

    return a


_typemap = {'>f4':'R4', '>f8':'R8', '>c8':'C8', '>c16':'C16', '>i4':'I4', '>i2':'I2'}
_invtypemap = dict((v,k) for k, v in _typemap.items())

def str2type(type):
    if ':' in type:
        type = [ tuple(s.split(':')) for s in type.split(',') ]
        type = np.dtype([(k, str2type(v)) for k,v in type])
    else:
        try:
            type = _invtypemap[type]
        except KeyError:
            m = re.match(r'([A-Z])', type)
            if m:
                l = m.group(1)
                type = re.sub(r'^' + l, '>' + l.lower(), type)
    return type


def type2str(dt):
    dtp = np.dtype(dt)
    if dtp.fields:
        dtypes = ','.join('{0}:{1}'.format(k,type2str(dtp.fields[k][0])) for k in dtp.names)
    else:
        dtypes = str(dt)
        if '>' in dtypes:
            try:
                dtypes = _typemap[dtypes]
            except KeyError:
                m = re.match(r'>([a-z])', dtypes)
                if m:
                    l = m.group(1)
                    dtypes = re.sub(r'>' + l, l.upper(), dtypes)

    return dtypes


def binname(f, a=None, dtype=None, shape=None):
    if a is not None:
        a = np.asanyarray(a)
        if dtype is None:
            dtype = a.dtype
        else:
            try:
                dtype = str2type(dtype)
            except TypeError:
                dtype = np.dtype(dtype)
        shape = a.shape
    else:
        dtype = np.dtype(dtype)
    dtypes = type2str(dtype)
    shapes = 'x'.join([str(i) for i in shape[::-1]])
    fname = '{0}.{1}_{2}.bin'.format(f, dtypes, shapes)
    fsize = dtype.itemsize*np.prod(shape)
    return fname, fsize


def savebin(f, a, dtype=None, append=False, clobber=False):
    a = np.asanyarray(a)
    if dtype is not None:
        try:
            dtype = str2type(dtype)
        except TypeError:
            dtype = np.dtype(dtype)
        a = a.astype(dtype)
    dtypes = type2str(a.dtype)
    shapes = 'x'.join([str(i) for i in a.shape[::-1]])
    if append:
        shapepatt = re.sub(r'x[^x]*$', '*', shapes)
        oldname,type,oldshape = findbin('{0}.{1}_{2}.bin'.format(f, dtypes, shapepatt))
        with open(oldname, "a") as f:
            a.tofile(f)
        newshape = (oldshape[0]+a.shape[0],) + oldshape[1:]
        shapes = 'x'.join([str(i) for i in newshape[::-1]])
        fname = '{0}.{1}_{2}.bin'.format(f, dtypes, shapes)
        os.rename(oldname, fname)
    else:
        fname = '{0}.{1}_{2}.bin'.format(f, dtypes, shapes)
        a.tofile(fname)

    if clobber:
        for name in findbins(f):
            if name != fname:
                sys.stderr.write('savebin: removing ' + name + '\n')
                os.unlink(name)


binpatt = re.compile(r'\.(([<>a-zA-Z0-9:,]*)_)?([-0-9x]*)\.bin$')

def findbins(f):
    res = []
    # does f have grid info in it already?
    m = binpatt.search(f)
    if m:
        res.append(f)
    else:
        if not '*' in f and not '?' in f:
            fglob = f + '.[<>a-zA-Z0-9]*_[-0-9]*.bin'

        fnames = glob(fglob)
        for fname in fnames:
            m = binpatt.search(fname)
            if m:
                res.append(fname)

    return res

def findbin(f, exceptions=True):
    # does f have grid info in it already?
    m = binpatt.search(f)
    if m:
        fname = f
    else:
        if not '*' in f and not '?' in f:
            fglob = f + '.[<>a-zA-Z0-9]*_[-0-9]*.bin'

        fnames = glob(fglob)
        for fname in fnames:
            m = binpatt.search(fname)
            if m: break
        else:
            if exceptions:
                raise IOError('file not found: ' +  fglob)
            else:
                return None

    if not m:
        if exceptions:
            raise IOError('file not found: ' +  fglob)
        else:
            return None

    dims = [ int(s) for s in m.group(3).split('x')[::-1] ]
    type = str2type(m.group(2))

    return fname, type, dims

def binbase(f):
    # does f have grid info in it already?
    m = binpatt.search(f)
    if m:
        f,_ = os.path.splitext(f[:-4])
    return f


def mkidx(s, n):
    try:
        sss = s.indices(n)
    except AttributeError:
        try:
            iter(s)
        except TypeError:
            if s < 0:
                s += n
            assert 0 <= s and s < n
            return s
        else:
            return s
    else:
        return range(*sss)


def index(l, s):
    try:
        return l.index(s)
    except ValueError:
        raise ValueError("'" + s + "' is not in list:\n\n" + ' '.join(l))


def loadbin(f, rec=None, its=None, mask_val=None, astype=None, returnits=False, trunc=False,
            verbose=False, flds=None):
    '''
    a = loadbin(f, ...)
    a = loadbin(f, rec, ...)
    a,its = loadbin(f, ..., returnits=True, ...)

    Parameters:
    rec        which record to read; can be a (multi-)slice
    mask_val
    astype     convert to this type
    its        list of iterations to read; iteration number is appended to file name
               special values:
               nan ,'all'  :: all iterations found
               inf, 'last' :: the last iteration found
    returnits  whether to return a list of iterations read as a second return value
    trunc      truncate extra data at end (raises error if not set)
    verbose    how much info to report
    flds       a single field name, an inclusive range fld1-fld2 or a python-style
               range fld1:fld2 that excludes fld2.  names are read from f.fldList.txt
    '''

    if verbose:
        sys.stderr.write('Reading ' + f + ' ... ')
    if its is not None:
        if its in [np.nan, np.inf, 'all', 'last']:
            patt = f + '.' + 10*'[0-9]' + '.[<>a-zA-Z0-9:,]*_[-0-9]*.bin'
            fnames = glob(patt)
            sys.stderr.write('Found {0} files for {1}.\n'.format(len(fnames), f))
            i = len(f) + 1
            allits = sorted([ int(s[i:i+10]) for s in fnames ])
            if its in [np.inf, 'last']:
                its = allits[-1:]
            else:
                its = allits
        data = np.array([ loadbin('{0}.{1:010d}'.format(f, it),
                                  mask_val=mask_val, astype=astype, rec=rec, flds=flds)
                          for it in its ])
        if verbose:
            sys.stderr.write('OK\n')
        if returnits:
            return data,its
        else:
            return data

    # does f have grid info in it already?
    m = binpatt.search(f)
    if m:
        fname = f
    else:
        if not '*' in f and not '?' in f:
#            fglob = f + '.[<>a-zA-Z0-9:,]*_[-0-9]*.bin'
            fglob = f + '.[<>a-zA-Z0-9:,]*_*.bin'

        fnames = glob(fglob)
        patt = re.compile(r'\.(([<>a-zA-Z0-9:,]*)_)?([-0-9x]*)\.bin$')
        matches = []
        for fname in fnames:
            if not '*' in f and not '?' in f:
                m1 = patt.match(fname[len(f):])
            else:
                m1 = patt.search(fname)
            if m1:
                matches.append(fname)
                m = m1

        if len(matches) > 1:
            sys.stderr.write('Warning: loadbin: multiple matches for ' + fglob + '\n')
            sys.stderr.write('Warning: loadbin: using ' + matches[-1] + '\n')

        if m:
            fname = matches[-1]

    if not m:
        raise IOError('file not found: ' +  fglob)

    dims = m.group(3)
    if dims:
        dims = [ int(s) for s in dims.split('x')[::-1] ]
    else:
        dims = []
    tp = str2type(m.group(2))
    dtype = np.dtype(tp)

    if flds is not None:
        assert rec is None
        base = binbase(fname)
        lname = base + '.fldList.txt'
        if not os.path.exists(lname):
            base,_ = os.path.splitext(base)
            lname = base + '.fldList.txt'

        with open(lname) as f:
            fldlst = [l.strip() for l in f]

        if '-' in flds:
            sep = '-'
        elif ':' in flds:
            sep = ':'
        elif ',' in flds:
            sep = ','
        else:
            sep = None

        if sep is None:
            rec = index(fldlst, flds)
        elif sep == ',':
            flds = flds.split(sep)
            rec = [index(fldlst, fld) if fld else None for fld in flds]
        elif sep is not None:
            flds = flds.split(sep)
            rec = [index(fldlst, fld) if fld else None for fld in flds[:2]]
            rec[2:] = map(int, flds[2:])
            if sep == '-' and len(rec) > 1 and rec[1] is not None:
                rec[1] += 1

            rec = slice(*rec)

    if type(rec) == type(1):
        a = myfromfile(fname, dtype=dtype, shape=tuple(dims[1:]), skip=rec)
    elif rec is not None:
        if type(rec) != type(()):
            rec = (rec,)
        nrecd = len(rec)
        pdims = dims[nrecd:]
        nitem = np.prod(pdims, dtype=int)
        count = nitem
        idxs = [np.arange(dims[i])[rec[i]].tolist() for i in range(nrecd)]
        if type(rec[-1]) is slice and rec[-1].step in (1, None):
            idx = idxs.pop(-1)
            nrecd -= 1
            n = len(idx)
            pdims[:0] = [n]
            offset = idx[0]*nitem
            count = n*nitem
            nitem *= dims[nrecd]
        else:
            offset = 0

        idxl = map(np.atleast_1d, idxs)
        nidx = map(len, idxl)
        a = np.zeros(nidx + pdims, dtype)
        with open(fname) as f:
            if nrecd == 0:
                f.seek(offset*dtype.itemsize)
                a[:] = np.fromfile(f, dtype, count=count).reshape(pdims)
            else:
                for ii in np.ndindex(*nidx):
                    iii = [ idx[i] for idx,i in zip(idxl, ii) ]
                    if verbose:
                        sys.stderr.write('loadbin: loading ' + str(iii) + '\n')
                    i = np.ravel_multi_index(iii, dims[:nrecd])
                    f.seek((i*nitem+offset)*dtype.itemsize)
                    a[ii] = np.fromfile(f, dtype, count=count).reshape(pdims)

        shape = [len(i) for i in idxs if type(i) == type([])] + pdims
        a.shape = shape
    else:
        a = np.fromfile(fname, dtype=dtype)
        if trunc:
            sdim = np.prod(dims[1:])
            a = a[:a.size//sdim*sdim].reshape([-1]+dims[1:])
        else:
            try:
                a = a.reshape(dims)
            except ValueError:
                raise IOError('Wrong dimensions for file size: {} {} {}\n'.format(
                                   fname, dims, a.size))

    if mask_val is not None:
        a = np.ma.MaskedArray(a, a==mask_val)

    if astype is not None:
        a = a.astype(astype)

    if verbose:
        sys.stderr.write('OK\n')

    return a


class BinFileIter(object):
    def __init__(self, f, mask_val=None, astype=None, verbose=False):
        if verbose:
            sys.stderr.write('Reading ' + f + ' ... ')

        # does f have grid info in it already?
        m = binpatt.search(f)
        if m:
            fname = f
        else:
            if not '*' in f and not '?' in f:
                fglob = f + '.[<>a-zA-Z0-9:,]*_[-0-9]*.bin'

            fnames = glob(fglob)
            patt = re.compile(r'\.(([<>a-zA-Z0-9:,]*)_)?([-0-9x]*)\.bin$')
            for fname in fnames:
                if m:
                    sys.stderr.write('Warning: loadbin: multiple matches for ' + fglob + '\n')
                    sys.stderr.write('Warning: loadbin: using ' + fname + '\n')
                m = patt.search(fname)

        if not m:
            raise IOError('file not found: ' +  fglob)

        dims = [ int(s) for s in m.group(3).split('x')[::-1] ]
        type = str2type(m.group(2))

        self.mask_val = mask_val
        self.astype = astype
        self.dims = dims
        self.dtype = np.dtype(type)
        self.f = open(fname)

    def next(self):
        a = myfromfid(self.f, dtype=self.dtype, shape=self.dims[1:], skip=0)

        if self.mask_val is not None:
            a = np.ma.MaskedArray(a, a==self.mask_val)

        if self.astype is not None:
            a = a.astype(self.astype)

        return a

    def close(self):
        self.f.close()


biniter = BinFileIter


def toraw(a,f,dtype=None):
    a = np.asanyarray(a)
    if dtype is not None:
        try:
            dtype = str2type(dtype)
        except TypeError:
            dtype = np.dtype(dtype)
        a = a.astype(dtype)

    dtypes = type2str(a.dtype)
#    dtypes = str(a.dtype)
#    if '>' in dtypes:
#        try:
#            dtypes = _typemap[dtypes]
#        except KeyError:
#            m = re.match(r'>([a-z])', dtypes)
#            if m:
#                l = m.group(1)
#                dtypes = re.sub(r'>' + l, l.upper(), dtypes)

    fname = f + '.' + 'x'.join([str(i) for i in a.shape]) + '_' + dtypes + '.raw'
    return a.tofile(fname)


def rawname(shape,f,dtype):
    if dtype is not None:
        if not isinstance(dtype, np.dtype):
            dtype = str2type(dtype)

    dtypes = type2str(dtype)

    fname = f + '.' + 'x'.join([str(i) for i in shape]) + '_' + dtypes + '.raw'
    return fname


def rawsplit(fname):
    m = re.search(r'^(.*)[\._]([0-9x]*)(_([<>a-zA-Z0-9]*))?\.raw$', fname)
    if m is None:
        raise ValueError('Not a "raw" file name: {0}'.format(fname))
    base,dims,_,tp = m.groups()
    dims = [ int(s) for s in dims.split('x') ]
    tp = str2type(tp)
    return base, dims, tp


def rawbase(f):
    # does f have grid info in it already?
    m = re.search(r'^(.*)[\._]([0-9x]*)(_([<>a-zA-Z0-9]*))?\.raw$', f)
    if m:
        base,dims,_,tp = m.groups()
    else:
        base = f
    return base


def rawparams(f):
    # does f have grid info in it already?
    m = re.search(r'[\._]([0-9x]*)(_([<>a-zA-Z0-9]*))?\.raw$', f)
    if m:
        file = f
    else:
        if not '*' in f and not '?' in f:
            f = f + '.[0-9]*.raw'

        files = glob(f)
        for file in files:
            m = re.search(r'[\._]([0-9x]*)(_([<>a-zA-Z0-9]*))?\.raw$', file)
            if m: break
        else:
            sys.stderr.write('file not found: ' +  f + '\n')
            raise IOError

    if not m:
        raise IOError

    dims = [ int(s) for s in m.group(1).split('x') ]
    type = str2type(m.group(3))

    return dims, type


def fromraw(f, mask_val=None, astype=None, rec=None):
    # does f have grid info in it already?
    m = re.search(r'[\._]([-0-9x]*)(_([<>a-zA-Z0-9]*))?\.raw$', f)
    if m:
        file = f
    else:
        if not '*' in f and not '?' in f:
            f = f + '.[-0-9]*.raw'

        files = glob(f)
        for file in files:
            m = re.search(r'[\._]([-0-9x]*)(_([<>a-zA-Z0-9]*))?\.raw$', file)
            if m: break
        else:
            raise IOError('file not found: ' +  f)

    if not m:
        raise IOError('file not found: ' +  f)

    dims = [ int(s) for s in m.group(1).split('x') ]
    type = str2type(m.group(3))
#    try:
#        type = _invtypemap[type]
#    except KeyError:
#        m = re.match(r'([A-Z])', type)
#        if m:
#            l = m.group(1)
#            type = re.sub(r'^' + l, '>' + l.lower(), type)

    if rec is not None:
        a = myfromfile(file, dtype=np.dtype(type), shape=tuple(dims[1:]), skip=rec)
    else:
        a = np.fromfile(file, dtype=np.dtype(type)).reshape(dims)

    if mask_val is not None:
        a = np.ma.MaskedArray(a, a==mask_val)

    if astype is not None:
        a = a.astype(astype)

    return a


def globits(patt):
    files = glob(patt)
    pre,suf = patt.split('*')
    its = []
    for file in files:
        s = re.sub(r'^' + re.escape(pre), '', file)
        s = re.sub(re.escape(suf) + r'$', '', s)
        s = re.sub(r'^0*', '', s)
        its.append(int(s))

    its.sort()
    return its


def fromunformatted(file,dtype='float32', shape=None, skip=-1, count=-1):
    if skip >= 0:
        endcount = 1
    else:
        endcount = -1

    try:
        file.seek(0,1)
    except AttributeError:
        file = open(file)

    if skip > 0 or count >= 0:
        for i in range(skip):
            n1, = np.fromfile(file,'int32',count=1)
            file.seek(n1+4,1)

    if count > 0:
        res = np.empty((count,)+shape,dtype)
        for c in range(count):
            res[c,...] = fromunformatted(file,dtype,shape,skip=0)

        return res

    try:
        # skip header
        n1, = np.fromfile(file,'int32',count=1)
    except TypeError:
        raise
    else:
        n1 /= np.dtype(dtype).itemsize
        data = np.fromfile(file, dtype, count=n1)
        n2, = np.fromfile(file,'int32',count=endcount)

        if shape is not None:
            data = data.reshape(shape)

        return data


def grid2cell(x):
    return .5*(x[1:]+x[:-1])


def block1d(a,n=2,f=np.mean,axis=-1):
    dims = a.shape
    nx = dims[axis]
    dimsl = dims[:axis]
    if axis == -1:
        dimsr = ()
    else:
        dimsr = dims[axis+1:]

    if axis >= 0:
        axis += 1

    tmp = f(a.reshape(dimsl + (nx/n,n) + dimsr), axis=axis)
    return tmp


def block2d(a,n=2,f=np.mean):
    dims = a.shape
    ny,nx = dims[-2:]
    tmp = f(f(a.reshape(dims[:-2] + (ny/n,n,nx/n,n)), axis=-1), axis=-2)
    return tmp


def unblock2d(a,n=2):
    dims = a.shape
    ny,nx = dims[-2:]
    tmp = a.reshape(dims[:-2] + (ny,1,nx,1)
                   ) * np.ones(len(dims[:-2])*(1,) + (1,n,1,n))
    return tmp.reshape(dims[:-2] + (ny*n,nx*n))


def unblock1d(a,n=2,axis=-1):
    dims = a.shape
    nx = dims[axis]
    dimsl = dims[:axis]
    if axis == -1:
        dimsr = ()
    else:
        dimsr = dims[axis+1:]

    tmp = a.reshape(dimsl + (nx,1) + dimsr) * \
          np.ones(len(dimsl)*(1,) + (1,n) + len(dimsr)*(1,)) 
    return tmp.reshape(dimsl + (nx*n,) + dimsr)


def it2ymdhms(it, dt=1200, start=694224000):
    """
        step and start are in seconds since the epoch
        default for start is the equivalent of 19920101 UTC
    """
    date = time.gmtime(start+dt*it)
    return date[0:6]


def it2date(it, dt=1200, start=694224000):
    """
        step and start are in seconds since the epoch
        default for start is the equivalent of 19920101 UTC
    """
    return time.strftime('%Y%m%d %H%M%S', time.gmtime(start+dt*it))


def it2day(it, dt=1200, start=694224000, sep=''):
    """
        dt and start are in seconds since the epoch
        default for start is the equivalent of 19920101 UTC
    """
    return time.strftime('%Y' + sep + '%m' + sep + '%d', time.gmtime(start+dt*it-86400))


def it2dayl(it, dt=1200, start=694224000):
    """
        dt and start are in seconds since the epoch
        default for start is the equivalent of 19920101 UTC
    """
    return time.strftime('%Y-%m-%d', time.gmtime(start+dt*it-86400))
    

def it2mon(it, dt=1200, start=694224000):
    """
        dt and start are in seconds since the epoch
        default for start is the equivalent of 19920101 UTC
    """
    (y,m,d,H,M,S,xx,yy,zz) = time.gmtime(start+dt*it)
    m = m - 1
    if m == 0:
        y = y - 1
        m = 12
    return time.strftime('%Y-%m', (y,m,d,H,M,S,xx,yy,zz))
    

def mercatory(lat):
    """ y coordinate of Mercator projection (in degrees) """
    #return 180./pi*log(tan(lat*pi/180.) + 1./cos(lat*pi/180))
    return 180./np.pi*np.log(np.tan(np.pi/4.+lat*np.pi/360.))


def smax(x,axis=None):
    " signed maximum: max if max>|min|, min else "
    mx=np.max(x,axis)
    mn=np.min(x,axis)
    neg=abs(mn)>abs(mx)
    return (1-neg)*mx+neg*mn


def maxabs(x,axis=None):
    " maximum modulus "
    return np.max(abs(x),axis)


def indmin(a,axis=None):
    flatindex = np.argmin(a,axis)
    return np.unravel_index(flatindex, a.shape)


def indmax(a,axis=None):
    flatindex = np.argmax(a,axis)
    return np.unravel_index(flatindex, a.shape)


def nanindmin(a,axis=None):
    flatindex = nanargmin(a,axis)
    return np.unravel_index(flatindex, a.shape)


def nanindmax(a,axis=None):
    flatindex = nanargmax(a,axis)
    return np.unravel_index(flatindex, a.shape)


def indmaxabs(a,axis=None):
    flatindex = np.argmax(np.abs(a),axis)
    return np.unravel_index(flatindex, a.shape)


def max2(a):
    return max(max(a,axis=-1),axis=-1)


def maxabs2(a):
    return np.max(np.max(abs(a),axis=-1),axis=-1)


def imean(iterable, dtype=np.float):
    itr = iter(iterable)
    s = np.array(itr.next(), dtype)
    n = 1
    for a in itr:
        s += a
        n += 1
    s /= n
    return s


def fromgen(n, dtype=None):
    '''factory for array-from-generator contructors, e.g.,

    a = ai(5)( rand(10,10) for _ in range(5) )
    '''
    def fromarrayiter(aa, n=n, dtype=dtype):
        p = next(aa)
        if dtype is None: dtype = p.dtype
        a = np.empty((n,) + p.shape, dtype)
        a[0] = p
        for i in range(1, n):
            a[i] = next(aa)
        return a
    return fromarrayiter


def maskeval(msk, exp1, exp2, arrs, ns, dtype=None):
    '''res = maskeval(msk, exp1, exp2, arrs, ns)

    Like numpy.where but does not evaluate expressions outside mask.
    Example:

        a = maskeval(p>0, 'p*log(p)', '0.0', ['p'], vars())
    '''
    res = np.empty(msk.shape, dtype)

    d = dict(ns)
    for v in arrs:
        d[v] = ns[v][msk]
    res[msk] = eval(exp1, d)

    msk = np.logical_not(msk)
    for v in arrs:
        d[v] = ns[v][msk]
    res[msk] = eval(exp2, d)

    return res


# for netCDF3
def untile(a,nty,ntx):
    sh = a.shape[1:-2]
    ny,nx = a.shape[-2:]
    n = len(sh)
    if not hasattr(a,'reshape'):
        a = a[:]

    return a.reshape((nty,ntx)+sh+(ny,nx)).transpose(range(2,2+n)+[0,2+n,1,3+n]).reshape(sh+(nty*ny,ntx*nx))

from namespace import Namespace
class NetcdfVariables(Namespace):
    def __init__(self, fname):
        from scipy.io.netcdf import netcdf_file
        self.netcdf_file = netcdf_file(fname)
        self.__dict__.update(self.netcdf_file.variables)

    def __del__(self):
        self.netcdf_file.close()

    def __repr__(self):
        type_name = type(self).__name__
        arg_strings = []
        for arg in self._get_args():
            arg_strings.append(repr(arg))
        for name, value in self._get_kwargs():
            if name != 'netcdf_file':
                try:
                    value = value.shape
                except:
                    s = '%s=%r' % (name, value)
                else:
                    s = '%s%r' % (name, value)
                arg_strings.append(s)
        return '%s(%s)' % (type_name, ', '.join(arg_strings))

ncvars = NetcdfVariables

def g2c(a, axis=-1):
    a = np.asanyarray(a)
    nd = len(a.shape)
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(None, -1)
    slice2[axis] = slice(1, None)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    return .5*(a[slice1] + a[slice2])


def reshape_index(multi_index, idims, odims):
    '''
    multi_index : tuple of array_like
        A tuple of integer arrays, one array for each dimension.
    idims : original dims of multi_index
    odims : new dims for returned multi_index
    '''
    flat_index = np.ravel_multi_index(multi_index, idims)
    return np.unravel_index(flat_index, odims)

