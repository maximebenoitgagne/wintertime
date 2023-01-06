import numpy as np

def shannon(a,axis=0,thresh=0,dtype=None):
    a = np.rollaxis(np.asanyarray(a), axis)
    if dtype is None:
        dtype = a.dtype

    total = np.sum(a, 0, dtype)
    valid = total > thresh

    resshape = a.shape[1:]
    if len(resshape):
        shannon = np.zeros(resshape, dtype)
        simpson = np.zeros(resshape, dtype)
        for b in a:
            valid1 = valid & (b>0)
            tmp = b[valid1]/total[valid1]
            shannon[valid1] += tmp*np.log(tmp)
            simpson[valid1] += tmp*tmp
    else:
        if valid:
            tmp = a[a>0]/total
            shannon = np.sum(tmp*np.log(tmp), 0)
            simpson = np.sum(tmp*tmp, 0)

    return -shannon, simpson


def richness(a,axis=0,thresh=0):
    a = np.rollaxis(np.asanyarray(a), axis)

    richness = np.zeros(a.shape[1:], int)
    for b in a:
        richness[b>thresh] += 1

    return richness

