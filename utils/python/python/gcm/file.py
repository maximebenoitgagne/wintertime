import sys, time, gzip
import os
import re
from glob import glob
import numpy as np

_debug = False

def fromgzipfile(file, dtype=float, count=-1, offset=0):
    fid = gzip.open(file, 'rb')
    res = np.frombuffer(fid.read(), dtype, count, offset)
    fid.close()
    return res

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

    a = np.fromfile(fid, dtype, count)

    if shape is not None:
        a = a.reshape(shape)

    return a


def myfromfile(file, dtype=float, shape=None, count=-1, skip=-1, skipbytes=0):
    zipped = False
    if file.endswith('.gz'):
        zipped = True
    elif os.path.exists(file):
        zipped = False
    elif os.path.exists(file + '.gz'):
        if _debug: print 'myfromfile: reading', file + '.gz'
        zipped = True
        file = file + '.gz'
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
 
    fid = openf(file, 'rb')
    exc = True
    try:
        try:
            if skipbytes > 0:
                fid.seek(skipbytes)
            if zipped:
                a = np.frombuffer(fid.read(), dtype, count)
            else:
                a = np.fromfile(fid, dtype, count)
        except:
            exc = False
            fid.close()
            raise
    finally:
        if exc:
            fid.close()

    if shape is not None:
        a = a.reshape(shape)

    return a


_typemap = {'>f4':'R4', '>f8':'R8', '>c8':'C8', '>c16':'C16', '>i2':'I2', '>i4':'I4'}
_invtypemap = dict((v,k) for k, v in _typemap.iteritems())

def str2type(type):
    try:
        type = _invtypemap[type]
    except KeyError:
        m = re.match(r'([A-Z])', type)
        if m:
            l = m.group(1)
            type = re.sub(r'^' + l, '>' + l.lower(), type)
    return type


def type2str(dt):
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


def dataextension(a,f,dtype=None):
    if dtype is not None:
        if type(dtype) == type(''):
            dtypes = dtype
            dtype = str2type(dtype)
        else:
            dtypes = type2str(dtype)
    else:
        dtype = a.dtype
        dtypes = type2str(dtype)

    ext = '.' + 'x'.join([str(i) for i in reversed(a.shape)]) + '_' + dtypes + '.data'
    return ext, dtype


def savedata(a,f,dtype=None):
    ext,npdtype = dataextension(a,f,dtype)
    if dtype is not None:
        a = a.astype(npdtype)
    return a.tofile(f+ext)


_datapattern = r'[\._]([-0-9x]*)(_([<>a-zA-Z0-9]*))?'

def globdata(f, ext=None, reversedims=None):
    if ext is None:
        if f.endswith('.raw'):
            ext = '.raw'
        else:
            ext = '.data'
    if reversedims is None:
        reversedims = ext != '.raw'
            
    # does f have grid info in it already?
    patt = _datapattern + re.sub(r'\.',r'\.',ext) + '$'
    m = re.search(patt,f)
    if m:
        file = f
    else:
        if not '*' in f and not '?' in f:
            f = f + '.[0-9]*' + ext

        files = glob(f)
        for file in files:
            m = re.search(patt,file)
            if m: break
        else:
            sys.stderr.write('file not found: ' +  f + '\n')
            raise IOError

    if not m:
        raise IOError

    dims = m.group(1).split('x')
    type = str2type(m.group(3))

    if reversedims:
        dims.reverse()

    shape = tuple( int(s) for s in dims )

    return file, shape, type


def loaddata(f, dtype=None, shape=None, mask_val=None, astype=None, rec=None, ext=None, reversedims=None):
    """fromdata(filename, dtype, shape, mask_val, astype, rec=None)

    read array from .data file.  If dtype is not specified, it is assumed to be
    encoded in the filename as BASENAME.NXxNY..._TYPE.data
    """
    if dtype is None:
        filename, shape, dtype = globdata(f, ext, reversedims)
    else:
        filename = f
        if shape is None:
            shape = (-1,)
            if rec is not None:
                sys.stderr.write('fromdata: have to specify shape if rec is given.\n')
                raise ValueError

    if rec is not None:
        a = myfromfile(filename, dtype=np.dtype(dtype), shape=shape[1:], skip=rec)
    else:
        a = np.fromfile(filename, dtype=np.dtype(dtype)).reshape(shape)

    if mask_val is not None:
        a = np.ma.MaskedArray(a, a==mask_val)

    if astype is not None:
        a = a.astype(astype)

    return a


def globits(patt):
    """ glob for files matching patt, which must contain one '*' wildcard,
        and return list of integer iteration numbers found for '*'
    """
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
    """ read unformatted fortran output file (with item count headers)

      dtype :: data type
      shape :: shape of one record
      skip  :: skip this many records
      count :: read this many records (requires shape)
    """
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


######################################################################
# rdmds
def ravel_index(ind, dims):
    skip = 0
    for i,d in zip(ind,dims):
        skip *= d
        skip += i
    return skip


def readmeta(fname):
    dims = []
    i1s = []
    i2s = []
    dtype = np.dtype('>f4')
    nrecords = 1
    flds = []
    with open(fname) as fid:
        group = ''
        for line in fid:
            # look for
            #          var = [ val ];
            m = re.match(r" ([a-zA-Z]*) = \[ ([^\]]*) \];", line)
            if m:
                var,val = m.groups()
                if var == 'dataprec':
                    if val == "'float64'" or val == "'real*8'":
                        dtype = np.dtype('>f8')
                elif var == 'nrecords':
                    nrecords = int(val.strip("'"))
            # look for 
            #          dimList = [
            #                     n, i1, i2,
            #                     ...
            #                         ];
            if re.match(r' *[\]}]', line):
                group = ''
            if group in ['dimList']:
                dim,i1,i2 = map(int, re.split(', *', line.strip(', \n')))
                dims.append(dim)
                i1s.append(i1)
                i2s.append(i2)
            elif group in ['fldList']:
#                v = map(lambda s:s.strip("' "), re.split("'  *'", line.strip("', \n")))
                v = map(lambda s:s.rstrip(), re.split("'  *'", line.strip("', \n")))
                flds += v
            if re.match(r' dimList ', line): 
                group = 'dimList'
            elif re.match(r' fldList ', line): 
                group = 'fldList'
    # covert to C order
    dims.reverse()
    i1s.reverse()
    i2s.reverse()
    return dims,i1s,i2s,dtype,nrecords,flds


def rdmds(baseglob, ind=[], fill=0, fields=False, astype=None, keepprec=False):
    """ a[,fldlst] = rdmds(baseglob, ind, fill, fields, astype, keepprec)

reads a meta-data file pair or set of tiled meta-data files.  Baseglob may
contain shell wildcards, but not the tile-number or .meta/.data extensions.

  ind      :: read only the slice with this indices in the leading dimensions
              (must be a list of integers), i.e.

    rdmds(b, [i,j,...]) == rdmds(b)[i,j,...]  but only reads necessary parts

  fill     :: fill value for missing tiles
  fields   :: if true, also return names of fields read
  astype   :: convert to this dtype (default float64)
  keepprec :: don't convert, keep same dtype as in file

By default rdmds converts to double precision in order to avoid surprises 
with inaccurate arithmetic.
"""
    metafiles = glob(baseglob + '.meta')
    if len(metafiles) == 0:
        sys.stderr.write('rdmds: file not found: ' + baseglob + '.meta\n')
        raise IOError
    dims,i1s,i2s,filedtype,nrec,flds = readmeta(metafiles[0])
    if nrec > 1:
        dims = [nrec] + dims
    if len(ind) > len(dims) - 2 and len(metafiles) > 1:
        raise ValueError('Cannot specify indices for last 2 dimensions with tile files')
    if keepprec:
        if astype is not None:
            raise ValueError('keepprec and astype arguments are mutually exclusive')
        astype = filedtype
    res = np.zeros(dims[len(ind):],astype)
    if fill != 0:
        res[:] = fill
    for metafile in metafiles:
        datafile = re.sub(r'\.meta$', '.data', metafile)
        dims,i1s,i2s,dtype,nrec,flds = readmeta(metafile)
        slc = map(lambda x,y: np.s_[x-1:y], i1s, i2s)
        tdims = map(lambda x,y:y-x+1, i1s, i2s)
        if nrec > 1:
            slc = [np.s_[:nrec]] + slc
            tdims = [nrec] + tdims
        if len(ind) > 0:
            count = np.prod(tdims[len(ind):])
            skip = ravel_index(ind, tdims[:len(ind)])*count
            size = np.dtype(dtype).itemsize
            if _debug: print datafile, dtype, tdims[len(ind):], skip/count
            with open(datafile) as fid:
                fid.seek(skip*size)
                res[slc[len(ind):]] = np.fromfile(fid, dtype, count=count).reshape(tdims[len(ind):])
        else:
            if _debug: print datafile, dtype, tdims
            res[slc] = np.fromfile(datafile, dtype).reshape(tdims)

    if fields:
        return res,flds
    else:
        return res


######################################################################
# for netCDF3
def untile(a,nty,ntx):
    sh = a.shape[1:-2]
    ny,nx = a.shape[-2:]
    n = len(sh)
    if not hasattr(a,'reshape'):
        a = a[:]

    return a.reshape((nty,ntx)+sh+(ny,nx)).transpose(range(2,2+n)+[0,2+n,1,3+n]).reshape(sh+(nty*ny,ntx*nx))


######################################################################
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
    

