import numpy as np

def writeunformatted(f, a, dtype=None):
    a = np.asarray(a, dtype)
    bytesize = np.array(a.size*a.itemsize, 'int32')
    bytesize.tofile(f)
    a.tofile(f)
    bytesize.tofile(f)


def fromunformatted(file,dtype='float32', shape=None, skip=-1, count=-1):
    '''
    Read one record from a fortran unformatted file

    file can be an open file or a file name.
    '''
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

def readunformatted(fname,dtype='float32', shape=None):
    '''
    Read all records from a fortran unformatted file

    Returns a list of arrays.
    '''
    if '>' in dtype:
        itype = '>i4'
    else:
        itype = 'int32'
    res = []
    with open(fname) as f:
        while True:
            # skip header
            nn = np.fromfile(f, itype, count=1)
            try:
                n1, = nn
            except ValueError:
                break
            n1 /= np.dtype(dtype).itemsize
#            print fname, n1
            data = np.fromfile(f, dtype, count=n1)
            n2, = np.fromfile(f, itype, count=1)

            if shape is not None:
                data = data.reshape(shape)

            res.append(data)

    if len(res) == 1: res = res[0]
    return res


def fromseawifs(fnam):
    from pyhdf.SD import SD
    sd = SD(fnam)
    res = sd.select('l3m_data').get()[::-1,:]

    return res

def fromseawifsspec(file,lam=[412,443,490,510,555,670]):
    res = np.empty((len(lam),2160,4320))
    for i,l in enumerate(lam):
        fnam = file.format(l)
#        print fnam
        sd = SD(fnam)
        res[i,:,:] = sd.select('l3m_data').get()[::-1,:]

    return res

