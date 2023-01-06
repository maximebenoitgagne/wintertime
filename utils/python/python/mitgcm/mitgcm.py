import os.path
import numpy as np

class ValueException(Exception):
    pass


def loadgrid(fdims=(),fpatt='.'):
    if os.path.isdir(fpatt):
        fpatt = os.path.join(fpatt, 'tile%03d.mitgrid')

    class grid():
        pass

    if isinstance(fdims,int):
        fdims = 12*[fdims]
    nface = len(fdims)/2
    grid.fdims = fdims
    grid.nx  = fdims[1::2]
    grid.ny  = fdims[0::2]
    grid.xc  = nface*[None]
    grid.yc  = nface*[None]
    grid.dxf = nface*[None]
    grid.dyf = nface*[None]
    grid.rac = nface*[None]
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
    for i,face in enumerate(range(1,1+nface)):
        ny,nx = fdims[2*i:2*i+2]
        if ny*nx > 0:
            with open(fpatt%face) as f:
                shape = (ny+1,nx+1)
                grid.xc [i] = np.fromfile(f,'>f8',count=np.prod(shape)).reshape(shape)
                grid.yc [i] = np.fromfile(f,'>f8',count=np.prod(shape)).reshape(shape)
                grid.dxf[i] = np.fromfile(f,'>f8',count=np.prod(shape)).reshape(shape)
                grid.dyf[i] = np.fromfile(f,'>f8',count=np.prod(shape)).reshape(shape)
                grid.rac[i] = np.fromfile(f,'>f8',count=np.prod(shape)).reshape(shape)
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


#def faces2global(faces, ncs=None):
#    if ncs is None:
#        ncs = faces[0].shape[0]
#    inonempty = np.where([face is not None for face in faces])[0].tolist()
#
#    return np.hstack([faces[i][:ncs,:ncs] for i in inonempty])


def faces2global(faces, fdims):
    ny = max(fdims[::2])
    nx = sum(fdims[1::2])
    res = np.zeros((ny,nx))
    off = 0
    for i,face in enumerate(faces):
        ny,nx = fdims[2*i:2*i+2]
        if ny*nx > 0:
            res[:ny,off:off+nx] = face[:ny,:nx]
        off += nx

    return res


