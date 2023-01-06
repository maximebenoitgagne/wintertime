import numpy as np

class ValueException(Exception):
    pass

def agulhas(a,fill=0):
    """ extract agulhas region from 510 x 6 x 510 field """
    if a.shape[-2] == 6:
        ncs = a.shape[-1]
        extradims = a.shape[:-3]
    else:
        ncs = a.shape[-2]
        extradims = a.shape[:-2]

    nr = np.prod(a.shape)/ncs/ncs/6
    a = a.reshape((nr,ncs,6,ncs))

    if ncs%17:
        raise ValueException, 'ncs not divisible by 17: %d' % ncs

    tnx = ncs/17
    tny = ncs/17

    x10 = 6*tnx
    x2e = 5*tnx
    x60 = 6*tnx
    x61 = 15*tnx

    y1e = 5*tny
    y2e = 5*tny
    y60 = 12*tny
    y61 = 15*tny

    n1 = 11*tnx
    n2 = 5*tnx

    # extract regions and replace each point by an f x f block
    all = np.empty((nr,n2,n1+n2+n1),dtype=a.dtype)
    face1 = all[:,:,:n1]
    face2 = all[:,:,n1:n1+n2]
    face3 = all[:,:,n1+n2:]
    face1[:] = a[:,:y1e,0,x10:]
    face2[:] = a[:,:y2e,1,:x2e]
    face3[:] = a[:,y60:,5,x60:]
    # set points outside boundary to zero
    if fill is not None:
        face3[:,:y61-y60,:x61-x60] = fill

    return all.reshape(extradims + (n2,n1+n2+n1))


def refine2d(a,n):
    ny,nx = a.shape[-2:]
    extra = a.shape[:-2]
    return (a.reshape(extra+(ny,1,nx,1))*np.ones((1,n,1,n))).reshape(extra+(ny*n,nx*n))


def agulhas_ystack(a,fill=0):
    n2,n1 = a.shape
    n1 = (n1-n2)/2
    n3 = n2*2/5
    res = np.zeros((3*n2,n1),dtype=a.dtype)
    res[:n2,:] = a[:,:n1]
    res[n2:2*n2,:n2] = a[:,n1:n1+n2]
    res[2*n2:,] = a[:,n1+n2:]
    if fill is not None:
        res[n2:2*n2,n2:] = fill
        res[2*n2:-n3,:-n3] = fill
    return res


def agulhas2plot(a, cut='wedge', fill=np.nan):
    n2,n1 = a.shape
    n1 = (n1-n2)/2
    n3 = n2*2/5
    res = np.zeros((n2+n3,n1+n2))
    res[n3:,:n1+n2] = a[:,:n1+n2]
    res[:n3,:n1] = a[n2-n3:,n1+n2:]
    res[:n3,n1:] = a.T[-n3:,::-1]
    if cut == 'wedge':
        for i in range(n3-1):
            res[i,n1-n3+i+1:n1+n3-i-1] = fill
    elif cut == 'line':
        res[:n3-1,n1-1:n1+1] = fill
    return res


def agulhas2rect(a, fill=np.nan):
    n2,n1 = a.shape
    n1 = (n1-n2)/2
    n3 = n2*2/5
    res = np.zeros((n2+n2,n1+n2))
    res[n2:,:n1+n2] = a[:,:n1+n2]
    res[:n2,:n1] = a[:,n1+n2:]
    res[:n2-n3,:n1-n3] = fill
    res[:n2,n1:] = fill
    return res


def write_nml_vec(file, vec):
    i = 0
    while i < len(vec):
        ind = vec[i]
        n = 1
        while i+1 < len(vec) and vec[i+1] == ind:
            i += 1
            n += 1
        if n == 1:
            file.write(' %d,' % ind)
        else:
            file.write(' %d*%d,' % (n,ind))
        i += 1
    file.write('\n')
        
