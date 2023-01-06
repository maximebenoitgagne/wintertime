import sys
import numpy as np
from oj.num import myfromfile
from tiled import rdmds

debug=False

class HaloArray(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

#    def __array_finalize__(self, obj):
#        if obj is None: return

    # vorticity grid and shifted
    @property
    def z(self):
        return self[...,1:,1:]

    @z.setter
    def z(self,v):
        self[...,1:,1:] = v

    @property
    def zs(self):
        return self[...,:-1,1:]

    @zs.setter
    def zs(self,v):
        self[...,:-1,1:] = v

    @property
    def zw(self):
        return self[...,1:,:-1]

    @zw.setter
    def zw(self,v):
        self[...,1:,:-1] = v

    # u grid and shifted
    @property
    def u(self):
        return self[...,1:-1,1:]

    @z.setter
    def u(self,v):
        self[...,1:-1,1:] = v

    @property
    def un(self):
        return self[...,2:,1:]

    @zs.setter
    def un(self,v):
        self[...,2:,1:] = v

    @property
    def us(self):
        return self[...,:-2,1:]

    @zs.setter
    def us(self,v):
        self[...,:-2,1:] = v

    @property
    def uw(self):
        return self[...,1:-1,:-1]

    @zw.setter
    def uw(self,v):
        self[...,1:-1,:-1] = v

    # v grid and shifted
    @property
    def v(self):
        return self[...,1:,1:-1]

    @z.setter
    def v(self,v):
        self[...,1:,1:-1] = v

    @property
    def vs(self):
        return self[...,:-1,1:-1]

    @zs.setter
    def vs(self,v):
        self[...,:-1,1:-1] = v

    @property
    def ve(self):
        return self[...,1:,2:]

    @zs.setter
    def ve(self,v):
        self[...,1:,2:] = v

    @property
    def vw(self):
        return self[...,1:,:-2]

    @zw.setter
    def vw(self,v):
        self[...,1:,:-2] = v

    # tracer grid and shifted
    @property
    def i(self):
        return self[...,1:-1,1:-1]

    @i.setter
    def i(self,v):
        self[...,1:-1,1:-1] = v

    @property
    def n(self):
        return self[...,2:,1:-1]

    @n.setter
    def n(self,v):
        self[...,2:,1:-1] = v

    @property
    def s(self):
        return self[...,:-2,1:-1]

    @s.setter
    def s(self,v):
        self[...,:-2,1:-1] = v

    @property
    def e(self):
        return self[...,1:-1,2:]

    @e.setter
    def e(self,v):
        self[...,1:-1,2:] = v

    @property
    def w(self):
        return self[...,1:-1,:-2]

    @w.setter
    def w(self,v):
        self[...,1:-1,:-2] = v

    @property
    def ne(self):
        return self[...,2:,2:]

    @ne.setter
    def ne(self,v):
        self[...,2:,2:] = v

    @property
    def se(self):
        return self[...,:-2,2:]

    @se.setter
    def se(self,v):
        self[...,:-2,2:] = v

    @property
    def nw(self):
        return self[...,2:,:-2]

    @nw.setter
    def nw(self,v):
        self[...,2:,:-2] = v

    @property
    def sw(self):
        return self[...,:-2,:-2]

    @sw.setter
    def sw(self,v):
        self[...,:-2,:-2] = v

    def N(self):
        obj = self.copy()
        obj.i = self.n
        return obj

    def S(self):
        obj = self.copy()
        obj.i = self.s
        return obj

    def E(self):
        obj = self.copy()
        obj.i = self.e
        return obj

    def W(self):
        obj = self.copy()
        obj.i = self.w
        return obj



def _calcshape(dims,halo):
    nf = len(dims)//2
    return (max(dims[1::2])+2*halo, sum(dims[::2])+2*nf*halo)

class FacetArray(HaloArray):
    def __new__(cls, input_array, dims=None, halo=1, exch=None, dtype=None):
        if debug: print 'FacetArray.__new__'
        arr = np.asarray(input_array)
        if dims is not None and halo is not None:
            shape2 = _calcshape(dims,halo)

        if halo is None or dims is None or arr.shape[-2:] == shape2:
            # arr already includes halos
            obj = np.asarray(arr,dtype).view(cls)
            obj._calc(dims,halo)
        else:
            if dtype is None:
                dtype = arr.dtype
            obj = np.ndarray.__new__(cls, arr.shape[:-2] + shape2, dtype)
            obj._calc(dims,halo)
            origie = np.cumsum(dims[::2]).tolist()
            origi0 = [0] + origie[:-1]
            for i in range(obj.nf):
                obj.face(i)[...] = 0
                try:
                    obj.face(i).i[...] = arr[...,:dims[2*i+1],origi0[i]:origie[i]]
                except ValueError:
                    print 'shape mismatch:', obj.face(i).i.shape, arr[...,:dims[2*i+1],origi0[i]:origie[i]].shape, arr.shape
                    raise
        if exch is not None:
            exch(obj)
        return obj

    @classmethod
    def zeros(cls, pshape, dims, halo=1, dtype=float):
        shape = tuple(pshape) + _calcshape(dims,halo)
        data = np.zeros(shape, dtype)
        return cls.__new__(cls, data, dims, halo)

    @classmethod
    def zeros_like(cls, a):
        data = np.zeros_like(a)
        return cls.__new__(cls, data, a.dims, a.halo)

    @classmethod
    def fromfile(cls, fname, dtype, pshape, dims, halo=1, count=-1, skip=-1, exch=None):
        shape = tuple(pshape) + (max(dims[1::2]), sum(dims[::2]))
        data = myfromfile(fname, dtype=dtype, shape=shape, count=count, skip=skip)
        return cls.__new__(cls, data, dims, halo, exch)

    @classmethod
    def frommds(cls, fname, dtype, dims, halo=1, ind=[], fill=0):
        data = rdmds(fname, ind=ind, fill=fill)
        return cls.__new__(cls, data, dims, halo)

    def _calc(self, dims=None, halo=None):
        if debug: print 'FacetArray._calc'
        if halo is None: halo = 1
        if dims is None:
            dims = [ d-2*halo for d in self.shape[-1:-3:-1] ]

        self.dims = dims
        self.nx = self.dims[::2]
        self.ny = self.dims[1::2]
        self.halo = halo
        self.hdims = [ d+2*self.halo for d in self.dims ]
        self.hnx = self.hdims[::2]
        self.hny = self.hdims[1::2]
        self.nf = len(self.hnx)
        self.hie = np.cumsum(self.hnx).tolist()
        self.hi0 = [0]+self.hie[:-1]
        self.hje = self.hny
        self.hj0 = self.nf*[0]

    def __array_finalize__(self, obj):
        if debug: print 'FacetArray.__array_finalize__'
        if obj is None: return
#        print 'obj is',type(obj)
#        print 'obj has dims =',hasattr(obj,'dims')
#        print 'obj has halo =',hasattr(obj,'halo')
#        print 'obj has nf   =',hasattr(obj,'nf')

#        if getattr(self, 'dims', None) is None and obj is not None:
#            self.dims = getattr(obj, 'dims', None)
#        if getattr(self, 'halo', None) is None and obj is not None:
#            self.halo = getattr(obj, 'halo', None)
        self.dims  = getattr(obj, 'dims', None)
        self.nx    = getattr(obj, 'nx', None)
        self.ny    = getattr(obj, 'ny', None)
        self.halo  = getattr(obj, 'halo', None)
        self.hdims = getattr(obj, 'hdims', None)
        self.hnx   = getattr(obj, 'hnx', None)
        self.hny   = getattr(obj, 'hny', None)
        self.nf    = getattr(obj, 'nf', None)
        self.hie   = getattr(obj, 'hie', None)
        self.hi0   = getattr(obj, 'hi0', None)
        self.hje   = getattr(obj, 'hje', None)
        self.hj0   = getattr(obj, 'hj0', None)
#        self._calc(self.dims,self.halo)

#        try:
#            self._calc(self.dims,self.halo)
#        except AttributeError:
#            pass

    def face(self, i):
        return self[...,self.hj0[i]:self.hje[i],self.hi0[i]:self.hie[i]].view(HaloArray)

    def toglobal(self, dtype=None, out=None):
        gy = max(self.ny)
        gx = sum(self.nx)
        hy = (self.shape[-2] - gy)//2
        hx = (self.shape[-1] - gx)//self.nf//2
        if out is None:
            out = np.empty(self.shape[:-2]+(gy,gx),dtype)
        x0 = 0
        for i,nx in enumerate(self.nx):
            ny = self.ny[i]
            out[...,x0:x0+nx] = self.face(i)[...,hy:hy+ny,hx:hx+nx]
            x0 += nx
#        out = np.concatenate([ self.face(i)[...,hy:hy+ny,hx:hx+nx] for i,(nx,ny) in enumerate(zip(self.nx,self.ny)) ], -1)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def setglobal(self, arr):
        gy = max(self.ny)
        gx = sum(self.nx)
        hy = (self.shape[-2] - gy)//2
        hx = (self.shape[-1] - gx)//self.nf//2
        x0 = 0
        for i,nx in enumerate(self.nx):
            ny = self.ny[i]
            self.face(i)[...,hy:hy+ny,hx:hx+nx] = arr[...,x0:x0+nx]
            x0 += nx

    def mult(self, arr):
        gy = max(self.ny)
        gx = sum(self.nx)
        hy = (self.shape[-2] - gy)//2
        hx = (self.shape[-1] - gx)//self.nf//2
        x0 = 0
        for i,nx in enumerate(self.nx):
            ny = self.ny[i]
            self.face(i)[...,hy:hy+ny,hx:hx+nx] *= arr[...,x0:x0+nx]
            x0 += nx

    def tohstack(self, dtype=None):
        ag = np.concatenate([ self.face(i).i for i in range(self.nf) ], -1)
        if dtype is not None:
            ag = ag.astype(dtype)
        return ag


    def tovstack(self, dtype=None):
        if dtype is not None:
            dtype = self.dtype
        ag = np.zeros(self.shape[:-2] + (np.sum(self.ny),np.max(self.nx)), dtype)
        je = np.cumsum(self.ny).tolist()
        j0 = [0] + je[:-1]
        for f in range(self.nf):
            ag[...,j0[f]:je[f],:self.nx[f]] = self.face(f).i
        return ag

def toFacetArray(insidearr):
    res = FacetArray.zeros(insidearr.shape[:-2], insidearr.dims, insidearr.halo, insidearr.dtype)
    res.i[...] = insidearr
    return res

tofacet = toFacetArray

cardinal = ['N','S','E','W']
_N,_S,_E,_W = 0,1,2,3

_sl_o = [ np.s_[...,-1,:],
          np.s_[...,0,:],
          np.s_[...,:,-1],
          np.s_[...,:,0],
        ]
_sl_i = [ np.s_[...,-2,:],
          np.s_[...,1,:],
          np.s_[...,:,-2],
          np.s_[...,:,1],
        ]
_NE,_SE,_NW,_SW = 0,1,2,3
_sl_corner = [ np.s_[...,-1:,-1:],
               np.s_[...,:1,-1:],
               np.s_[...,-1:,:1],
               np.s_[...,:1,:1] 
             ]
_opp = [[1,0,3,2],
        [3,2,1,0]]

class TracerArray(FacetArray):
    def __new__(cls, a, dims, link, halo=None, corner=0):
        obj = FacetArray.__new__(cls,a,dims,halo)
        obj.link = link
        obj.exch(corner)
        return obj

    def __array_finalize__(self, obj):
        if debug: print 'TracerArray.__array_finalize__'
        if obj is None: return
        self.link  = getattr(obj, 'link', None)
        super(TracerArray, self).__array_finalize__(obj)

    def exch(self,corner=None):
        for d in range(4):
            for f in range(self.nf):
                lk = self.link[f][d]
                if lk is None:
                    edge = 0
                else:
                    nn,rot = lk
                    if debug: print cardinal[d], 'neighbor of', f, 'is', nn, 'rot', rot
                    dd = _opp[rot][d]
                    edge = self.face(nn)[_sl_i[dd]]
                    if rot:
                        edge = edge[::-1]
                self.face(f)[_sl_o[d]] = edge
        if corner is not None:
            for f in range(self.nf):
                for s in _sl_corner:
                    self.face(f)[s] = corner


#          N  S  E  W
_redge = [_E,_W,_S,_N]

class Edge(object):
    def __init__(self,face,d):
        self.face = face
        self.d = d

    def right(self):
        dright = _redge[self.d]
        return self.face.edge[dright]

    def __str__(self):
        return "Edge(" + str(self.face.f) + "," + str(self.d) + ")"

    def __repr__(self):
        return "Edge(" + str(self.face.f) + "," + str(self.d) + ")"


class Face(object):
    def __init__(self, f, link):
        self.f = f
        self.nn = [ l and l[0] for l in link ]
        self.rot = [ l and l[1] or 0 for l in link ]
        self.d = [ _opp[rot][d] for d,rot in enumerate(self.rot) ]
        self.edge = [ Edge(self,d) for d in range(4) ]
        self.cycles = 4*[0]

class Exchange(object):
    def __init__(self, links):
        self.links = links
        self.nf = len(links)
        self.faces = [ Face(f,l) for f,l in enumerate(links) ]
        for face in self.faces:
            for d in range(4):
                if face.nn[d] is not None:
                    face.edge[d].opp = self.faces[face.nn[d]].edge[face.d[d]]
                else:
                    face.edge[d].opp = None

        for face in self.faces:
            for d1,d2 in [(_E,_N),(_N,_W),(_W,_S),(_S,_E)]:
                e = face.edge[d1].opp
                n = 1
                while e and e.face != face:
                    e = e.right().opp
                    n += 1

                face.cycles[d1] = e and n or 0


    def find_color(self,f,dir):
        faces = []
        dirs = []
        while f not in faces:
            faces.append(f)
            dirs.append(dir)
            f = self.face[f]

    def init_colors(self):
        self.udir = [[] for f in range(self.nf)]
        self.vdir = [[] for f in range(self.nf)]
        while 1:
            for f in range(self.nf):
                if True not in self.udir[f]:
                    self.find_color(f,0)
                if True not in self.vdir[f]:
                    self.find_color(f,1)
                
                
    def __getitem__(self,ind):
        f,d = ind
        return self.link[f,d]

    def tr(self,a,corner=None,outside=0):
        for d in range(4):
            for f in range(self.nf):
                lk = self.links[f][d]
                if lk is None:
                    edge = outside
                else:
                    nn,rot = lk
                    if debug: print cardinal[d], 'neighbor of', f, 'is', nn, 'rot', rot
                    dd = _opp[rot][d]
                    edge = a.face(nn)[_sl_i[dd]]
                    if rot:
                        edge = edge[...,::-1]
                a.face(f)[_sl_o[d]] = edge
        if corner is not None:
            for f in range(self.nf):
                for s in _sl_corner:
                    a.face(f)[s] = corner

    __call__ = tr
    c = tr


    def vort(self,a,corner=None):
        sys.stderr.write('WARNING: exch.vort is noop\n')
        pass

    z = vort


    def uv(self,u,v,corner=None,sign=-1,halo=None):
        nf = self.nf
        link = self.links
        if halo is None:
            halo = u.halo
        # normal velocities first
        for d,tgt in zip(range(4),[v,v,u,u]):
            for f in range(nf):
                lk = link[f][d]
                if lk is None:
                    edge = 0
                else:
                    nn,rot = lk
                    dd = _opp[rot][d]
                    if dd < 2:
                        src = v
                    else:
                        src = u
                    edge = src.face(nn)[_sl_i[dd]]
                    if rot:
                        edge = edge[...,::-1]
                tgt.face(f)[_sl_o[d]] = edge
        # tangential velocities next
        for d,tgt in zip(range(4),[u,u,v,v]):
            for f in range(nf):
                lk = link[f][d]
                if lk is None:
                    edge = 0
                    rot = 0
                else:
                    nn,rot = lk
                    dd = _opp[rot][d]
                    if dd < 2:
                        src = u
                    else:
                        src = v
                    edge = src.face(nn)[_sl_i[dd]]
                if rot:
                    tgt.face(f)[_sl_o[d]][...,1:] = sign*edge[...,1:][...,::-1]
                else:
                    tgt.face(f)[_sl_o[d]] = edge

        if corner is not None:
            for f,face in enumerate(self.faces):
                if face.cycles[_NE] < 4:
                    u.face(f)[_sl_corner[_NE]][...,:,1:] = corner
                    v.face(f)[_sl_corner[_NE]][...,1:,:] = corner
                    v.face(f)[...,-halo,-halo:] += u.face(f)[...,-2*halo:-halo,-halo]
                    v.face(f)[...,-halo,-halo:] *= .5
                    u.face(f)[...,-halo:,-halo] += v.face(f)[...,-halo,-2*halo:-halo]
                    u.face(f)[...,-halo:,-halo] *= .5
                if face.cycles[_SE] < 4:
                    u.face(f)[_sl_corner[_SE]][...,:,1:] = corner
                    v.face(f)[_sl_corner[_SE]] = corner
                    v.face(f)[...,halo,-halo:] += sign*u.face(f)[...,halo:2*halo,-halo]
                    v.face(f)[...,halo,-halo:] *= .5
                    u.face(f)[...,:halo,-halo] += sign*v.face(f)[...,halo,-2*halo:-halo]
                    u.face(f)[...,:halo,-halo] *= .5
                if face.cycles[_NW] < 4:
                    u.face(f)[_sl_corner[_NW]] = corner
                    v.face(f)[_sl_corner[_NW]][...,1:,:] = corner
                    v.face(f)[...,-halo,:halo] += sign*u.face(f)[...,-2*halo:-halo,halo]
                    v.face(f)[...,-halo,:halo] *= .5
                    u.face(f)[...,-halo:,halo] += sign*v.face(f)[...,-halo,halo:2*halo]
                    u.face(f)[...,-halo:,halo] *= .5
                if face.cycles[_SW] < 4:
                    u.face(f)[_sl_corner[_SW]] = corner
                    v.face(f)[_sl_corner[_SW]] = corner
                    v.face(f)[...,halo,:halo] += u.face(f)[...,halo:2*halo,halo]
                    v.face(f)[...,halo,:halo] *= .5
                    u.face(f)[...,:halo,halo] += v.face(f)[...,halo,halo:2*halo]
                    u.face(f)[...,:halo,halo] *= .5

    def ws(self,w,s,corner=None,halo=None):
        return self.uv(w,s,corner,+1,halo)


def trplot(u):
    from oj.plot import myimshow

    dims = u.hdims
    a = np.ma.MaskedArray(np.zeros((u.shape[-2],u.shape[-1]+len(dims)/2-1)), True)
    ie = (np.cumsum([d+1 for d in dims[::2]])).tolist()
    i0 = [0] + ie[:-1]
    ie = [ x-1 for x in ie ]
    je = [d for d in dims[1::2]]
    j0 = [0 for d in dims[1::2]]
    for f in range(len(i0)):
        a[ :je[f], i0[f]:ie[f] ] = u.face(f)

    myimshow(a)


def velplot(u,v,dims):
    from oj.plot import myimshow

    a = np.ma.MaskedArray(np.zeros((u.shape[-2]*4,u.shape[-1]*4+len(dims)/2-1)), True)
    ie = (np.cumsum([4*d+1 for d in dims[::2]])).tolist()
    i0 = [0] + ie[:-1]
    ie = [ x-1 for x in ie ]
    je = [4*d for d in dims[1::2]]
    j0 = [0 for d in dims[1::2]]
    for f in range(len(i0)):
        a[ 1:4*je[f]:4, i0[f]:ie[f]:4 ] = u.face(f)
        a[ 2:4*je[f]:4, i0[f]:ie[f]:4 ] = u.face(f)
        a[ 3:4*je[f]:4, i0[f]:ie[f]:4 ] = u.face(f)
        a[ :4*je[f]:4, i0[f]+1:ie[f]:4 ] = v.face(f)
        a[ :4*je[f]:4, i0[f]+2:ie[f]:4 ] = v.face(f)
        a[ :4*je[f]:4, i0[f]+3:ie[f]:4 ] = v.face(f)

    myimshow(a)


if __name__ == '__main__':
    link = [[ None , (2,0), (1,0), None ],
            [ None , (2,1), None , (0,0)],
            [ (0,0), None , (1,1), None ]]

    exch = Exchange(link)

    fa = FacetArray([[3,4,7,8,11,12],[1,2,5,6,9,10]], (2,2, 2,2, 2,2), 1) + 0.
    u = fa.astype(FacetArray)
    v = fa.astype(FacetArray)
    for f in [0,2]: u.face(f).i[:,0]=0
    for f in [2]: v.face(f).i[0,:]=0
    fa.i += 1
    #fa.exch(corner=-1)
    exch.tr(fa,corner=-1)

    exch.uv(u,v,-1)

