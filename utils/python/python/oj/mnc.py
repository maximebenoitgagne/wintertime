import numpy as np
from glob import glob
try:
    import netCDF3 as nc
except ImportError:
    import netCDF4 as nc

def untile(a,nty,ntx):
    ntot = a.shape[0]
    npre = ntot/nty/ntx
    sh = a.shape[1:-2]
    ny,nx = a.shape[-2:]
    n = len(sh)
    if not hasattr(a,'reshape'):
        a = a[:]

    res = a.reshape((npre,nty,ntx)+sh+(ny,nx)).transpose([0]+range(3,3+n)+[1,3+n,2,4+n]).reshape((npre,)+sh+(nty*ny,ntx*nx))
    if npre == 1:
        res = res[0]

    return res


def sufcmp(a,b):
    return cmp(a[-18:],b[-18:])
    

class MncVariable(object):
    def __init__(self, var, ds):
        self.var = var
        self.ds = ds
        npre = var.shape[0]/ds.nTx/ds.nTy
        if npre > 1:
            preshape = (npre,)
        else:
            preshape = ()

        self.shape = preshape + var.shape[1:-2] + (ds.Ny,ds.Nx)
        self.prelen = len(preshape)
        self.dtype = var.dtype

    def __getitem__(self, i):
        if type(i) != type(()):
            i = (i,)

        mylen = len(self.shape)

        # fill all slots
        if Ellipsis in i:
            ii = i.index(Ellipsis)
            i = i[:ii] + (mylen-len(i)+1)*(slice(None),) + i[ii+1:]
        elif len(i) < mylen:
            i = i + (mylen-len(i))*(slice(None),)

        # compute shape of result
        ressh = []
        for ii,n in zip(i,self.shape):
            if isinstance(ii,slice):
                ressh.append(len(range(n)[ii]))

        # have to do horizontal slices after assembly
        j = (Ellipsis,)
        if len(i) > mylen - 2:
            j = j + i[mylen-2:]
            i = i[:mylen-2]

        if self.prelen > 0:
            ntile = self.ds.nTy*self.ds.nTx
            if isinstance(i[0], slice):
                sh = ressh
                i0s = range(self.shape[0])[i[0]]
#                if type(i0s) != type([]):
#                    sh = [1] + ressh
#                    i0s = [i0s]

                res = np.empty(sh, self.dtype)
                for ii0,i0 in enumerate(i0s):
                    sl0 = slice(ntile*i0,ntile*(i0+1))
                    res[ii0,...] = untile(self.var[(sl0,) + i[1:]], self.ds.nTy, self.ds.nTx)[j]

                return res.reshape(ressh)
            else:
                if i[0] < 0 or i[0] >= self.shape[0]:
                    raise IndexError

                return untile(self.var[(slice(ntile*i[0],ntile*(i[0]+1)),)+i[1:]], self.ds.nTy, self.ds.nTx)[j]
        else:
            # netCDF3 doesn't like length-1 slices...
            imod = ()
            for ii,s in zip(i,ressh):
                if isinstance(ii,slice) and s == 1:
                    imod = imod + (ii.indices(1)[0],)
                else:
                    imod = imod + (ii,)

            return untile(self.var[(slice(None),)+imod], self.ds.nTy, self.ds.nTx)[j].reshape(ressh)

    def __array__(self):
        return self[...]

    def __repr__(self):
        return 'MncVariable(' + ','.join([ str(i) for i in self.shape ]) + ')'


class MncVariableVector(object):
    def __init__(self, vars, ds):
        self.vars = vars
        self.ds = ds

    def __getitem__(self, i):
        if type(i) != type(()):
            if isinstance(i,slice):
                return np.array([ self.ds.vars[v][:] for v in self.vars[i] ])
            else:
                return self.ds.vars[self.vars[i]][:]
        else:
            i0 = i[0]
            if isinstance(i0,slice):
                return np.array([ self.ds.vars[v][i[1:]] for v in self.vars[i0] ])
            else:
                return self.ds.vars[self.vars[i0]][i[1:]]


class MFDataset(nc.MFDataset):
    def __new__(cls, patt, check=False, exclude=[]):
        files = glob(patt)
        files.sort(sufcmp)
        if len(files) == 0:
            raise IOError('File not found: %s\n' % patt)
        else:
            print patt, len(files), 'files'

        self = nc.MFDataset.__new__(cls)
        self.__init__(files, check=check, exclude=exclude)

        self.nTx = self.Nx / self.sNx
        self.nTy = self.Ny / self.sNy

        self.vars = dict((k,MncVariable(v,self)) for k,v in self.variables.items())

        return self

    def vec(self, v, i0, ie=None):
        if ie == None:
            ie = i0+1
            i0 = 1

        vars = [ v%d for d in range(i0,ie) if v%d in self.vars ]
        return MncVariableVector(vars,self)

    def __getitem__(self, i):
        return self.vars[i]



