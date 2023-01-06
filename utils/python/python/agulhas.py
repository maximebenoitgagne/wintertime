import numpy as np
from oj.misc import myimshow
from haloarray import Exchange, FacetArray
from gridfacet import GridFacet, MITGridFacet

exch = Exchange([[ None , (2,0), (1,0), None ],
                 [ None , (2,1), None , (0,0)],
                 [ (0,0), None , (1,1), None ]])

dims510 = (330,150, 150,150, 330,150)
n1,n2 = 1320,600
dims = (n1,n2, n2,n2, n1,n2)
grid17 = GridFacet('/scratch/jahn/grid/ap0003.17', 50, dims, exch)
gridSm2 = GridFacet('/scratch/jahn/grid/ap0003.39', 50, dims, exch)
grid42 = GridFacet('/scratch/jahn/grid/ap0003.42', 50, dims, exch)
grid48 = GridFacet('/scratch/jahn/grid/ap0003.48', 50, dims, exch)
mitgrid = MITGridFacet('/scratch/jahn/grid/agulhas2040/tile{0:03d}.mitgrid', dims, exch)


class ValueException(Exception):
    pass


def unblock(a,n=4):
    dims = a.shape
    ny,nx = dims[-2:]
    tmp = a.reshape(dims[:-2] + (ny,1,nx,1)
                   ) * np.ones(len(dims[:-2])*(1,) + (1,n,1,n))
    return tmp.reshape(dims[:-2] + (ny*n,nx*n))


def unblock1d(a,n=4):
    dims = a.shape
    nx = dims[-1]
    tmp = a.reshape(dims[:-1] + (nx,1)
                   ) * np.ones(len(dims[:-1])*(1,) + (1,n))
    return tmp.reshape(dims[:-1] + (nx*n,))


def agulhasFacet(a, s=None, fill=0, halo=1, exch=None):
    """ extract agulhas region from 510 x 6 x 510 field """
    ncs = a.shape[-2]
    extradims = a.shape[:-2]

    if s is None or s is Ellipsis:
        s = len(extradims)*np.s_[:]

    outdims = ()
    for sl,d in zip(s,extradims):
        if isinstance(sl,slice):
            start,stop,step = sl.indices(d)
            n = (stop - start)//step
            outdims += (n,)

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
    fa = FacetArray.zeros(outdims, (n1,n2,n2,n2,n1,n2), halo=halo, dtype=a.dtype)
    fa.face(0).i[...] = a[s+np.s_[:y1e,x10:ncs]]
    fa.face(1).i[...] = a[s+np.s_[:y2e,ncs:ncs+x2e]]
    fa.face(2).i[...] = a[s+np.s_[y60:,5*ncs+x60:]]
    # set points outside boundary to zero
    if fill is not None:
        fa.face(2).i[...,:y61-y60,:x61-x60] = fill

    if exch is not None:
        exch(fa)

    return fa


def agulhas(a,fill=0,perpindex=np.s_[...]):
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


def agushow(a, *args, **kwargs):
    if 'mask' in kwargs:
        kwargs['mask'] = agulhas2plot(kwargs['mask'])
    fill = kwargs.pop('fill', 0.)
    return myimshow(agulhas2plot(a,fill=fill), *args, **kwargs)


def rectshow(a, *args, **kwargs):
    if 'mask' in kwargs:
        kwargs['mask'] = agulhas2rect(kwargs['mask'])
    fill = kwargs.pop('fill', np.nan)
    return myimshow(agulhas2rect(a, fill=fill), *args, **kwargs)


def rect2plot(a, cut='wedge', fill=np.nan):
    n2,n1 = a.shape
    n2 /= 2
    n1 = n1-n2
    n3 = n2*2/5
    res = np.zeros((n2+n3,n1+n2))
    res[n3:,:n1+n2] = a[n2:,:n1+n2]
    res[:n3,:n1] = a[n2-n3:n2,:n1]
    res[:n3,n1:] = a.T[n1-n3:n1,n2-1::-1]
    if cut == 'rect':
        res[:n3,n1:n1+n3] = a[n2-n3:n2,n1:n1+n3]
        res[:n3,n1+n3-1] = fill
    if cut == 'wedge':
        for i in range(n3-1):
            res[i,n1-n3+i+1:n1+n3-i-1] = fill
    elif cut == 'line':
        res[:n3-1,n1-1:n1+1] = fill
    return res


def rect2agulhas(a):
    n2,n1 = a.shape
    n2 /= 2
    n1 -= n2
    res = np.zeros((n2,n1+n2+n1))
    res[:,:n1+n2] = a[n2:,:n1+n2]
    res[:,n1+n2:] = a[:n2,:n1]
    return res


def agulhas2rect(a, halo=0, fill=0):
    n2,n1 = a.shape
    n1 = (n1-n2)/2
    n3 = n2*2/5
    res = np.zeros((n2+n2,n1+n2))
    res[n2:,:n1+n2] = a[:,:n1+n2]
    res[:n2,:n1] = a[:,n1+n2:]
    res[:n2-n3,:n1-n3] = fill
    res[:n2,n1:] = fill
    if halo > 0:
        copyhalo(res,halo)
    return res


def copyhalo(res,halo,diag=None):
    n2,n1 = res.shape
    n2 /= 2
    n1 -= n2
    for i in range(halo):
        for j in range(n2-i-1):
            res[j,n1+i] = res[n2+i,-1-j]
            res[n2-1-i,-1-j] = res[j,n1-1-i]
        if diag is not None:
            res[n2-1-i,n1+i] = diag
        else:
            res[n2-1-i,n1+i] = .5*(res[n2+i,n1+i]+res[n2-1-i,n1-1-i])


def copyhalouv(u,v,halo,diag=None):
    n2,n1 = u.shape
    n2 /= 2
    n1 -= n2
    for i in range(halo):
        for j in range(n2-i):
            u[j,n1+i] = v[n2+i,-1-j]
            v[n2-1-i,-1-j] = u[j,n1-1-i]
        for j in range(1,n2-i):
            v[j,n1+i] = -u[n2+i,-j]
            u[n2-1-i,-j] = -v[j,n1-1-i]
        if diag is not None:
            v[n2-1-i,n1+i] = diag
        else:
            v[n2-1-i,n1+i] = .5*(-u[n2+i,n1+1+i]+u[n2-1-i,n1-1-i])


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
        

def mapglob(pg, ind, blankval=np.nan):
    pl = np.empty((150,480+330))
    pl[:,:480] = pg[ind+np.s_[:5*30,6*30:22*30]]
    pl[:,480:] = pg[ind+np.s_[12*30:,5*510+6*30:]]
    pl[:3*30,480:480+9*30] = blankval
    return pl


def clearboundary(pin, blankval=np.nan):
    pl = pin
    pl[:,0] = blankval
    pl[-1,:480] = blankval
    pl[:,479] = blankval
    pl[3*30:,480] = blankval
    pl[:3*30,-60] = blankval
    pl[3*30,480:-59] = blankval
    pl[0,-60:] = blankval
    return pl


def maxabs(a):
    return abs(a).max()


def argmaxabs(a):
    return abs(a).argmax()


def loadgrid(fdims=(),fpatt='tile%03d.mitgrid'):
    class grid():
        pass

    nface = len(fdims)/2
    grid.xc  = nface*[None]
    grid.yc  = nface*[None]
    grid.dxf = nface*[None]
    grid.dyf = nface*[None]
    grid.ra  = nface*[None]
    grid.xg  = nface*[None]
    grid.yg  = nface*[None]
    grid.dxv = nface*[None]
    grid.dyu = nface*[None]
    grid.raz = nface*[None]
    grid.dxc = nface*[None]
    grid.dyc = nface*[None]
    grid.raw = nface*[None]
    grid.ras = nface*[None]
    grid.dxg = nface*[None]
    grid.dyg = nface*[None]
    for i,face in enumerate([1,2,6]):
        with open(fpatt%face) as f:
            shape = (fdims[2*i]+1,fdims[2*i+1]+1)
            grid.xc [i] = np.fromfile(f,'>f8',count=np.prod(shape)).reshape(shape)
            grid.yc [i] = np.fromfile(f,'>f8',count=np.prod(shape)).reshape(shape)
            grid.dxf[i] = np.fromfile(f,'>f8',count=np.prod(shape)).reshape(shape)
            grid.dyf[i] = np.fromfile(f,'>f8',count=np.prod(shape)).reshape(shape)
            grid.ra [i] = np.fromfile(f,'>f8',count=np.prod(shape)).reshape(shape)
            grid.xg [i] = np.fromfile(f,'>f8',count=np.prod(shape)).reshape(shape)
            grid.yg [i] = np.fromfile(f,'>f8',count=np.prod(shape)).reshape(shape)
            grid.dxv[i] = np.fromfile(f,'>f8',count=np.prod(shape)).reshape(shape)
            grid.dyu[i] = np.fromfile(f,'>f8',count=np.prod(shape)).reshape(shape)
            grid.raz[i] = np.fromfile(f,'>f8',count=np.prod(shape)).reshape(shape)
            grid.dxc[i] = np.fromfile(f,'>f8',count=np.prod(shape)).reshape(shape)
            grid.dyc[i] = np.fromfile(f,'>f8',count=np.prod(shape)).reshape(shape)
            grid.raw[i] = np.fromfile(f,'>f8',count=np.prod(shape)).reshape(shape)
            grid.ras[i] = np.fromfile(f,'>f8',count=np.prod(shape)).reshape(shape)
            grid.dxg[i] = np.fromfile(f,'>f8',count=np.prod(shape)).reshape(shape)
            grid.dyg[i] = np.fromfile(f,'>f8',count=np.prod(shape)).reshape(shape)

    return grid


def delrectborder(a,w=3,fill=0):
    """ operates in place! (and returns) """
    nxt,nyt = a.shape[-2:]
    d2 = a.shape[:-2]
    d2slots = len(d2)*np.s_[:,]
    n2 = nyt//2
    n1 = nxt - n2
    n3 = n2*2//5
    x1 = n1-n3
    y1 = n2-n3
    a[d2slots+np.s_[:y1+w,:x1+w]] = fill
    a[d2slots+np.s_[:w,:]] = fill
    a[d2slots+np.s_[-w:,:]] = fill
    a[d2slots+np.s_[:,:w]] = fill
    a[d2slots+np.s_[:,-w:]] = fill
    return a


