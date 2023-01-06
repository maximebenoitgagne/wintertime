import numpy as np

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


def mercatory(lat):
    """ y coordinate of Mercator projection (in degrees) """
    #return 180./pi*log(tan(lat*pi/180.) + 1./cos(lat*pi/180))
    return 180./np.pi*np.log(np.tan(np.pi/4.+lat*np.pi/360.))


def smax(x,axis=None):
    " signed maximum: max if max>|min|, min else "
    mx = np.max(x,axis)
    mn = np.min(x,axis)
    neg = np.abs(mn)>np.abs(mx)
    return (1-neg)*mx + neg*mn


def maxabs(x,axis=None):
    " maximum modulus "
    return np.max(np.abs(x),axis)


def indmin(a,axis=None):
    flatindex = np.argmin(a,axis)
    return np.unravel_index(flatindex, a.shape)


def indmax(a,axis=None):
    flatindex = np.argmax(a,axis)
    return np.unravel_index(flatindex, a.shape)


def nanindmin(a,axis=None):
    flatindex = np.nanargmin(a,axis)
    return np.unravel_index(flatindex, a.shape)


def nanindmax(a,axis=None):
    flatindex = np.nanargmax(a,axis)
    return np.unravel_index(flatindex, a.shape)


def indmaxabs(a,axis=None):
    flatindex = np.argmax(np.abs(a),axis)
    return np.unravel_index(flatindex, a.shape)


def max2(a):
    return np.amax(np.amax(a,axis=-1),axis=-1)


def maxabs2(a):
    return np.amax(np.amax(np.abs(a),axis=-1),axis=-1)


