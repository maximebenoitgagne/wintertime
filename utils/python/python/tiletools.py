__all__ = ['readmeta', 'rdmds', 'sliceshape', 'tilemap', 'tiledfile', 'tiledfiles']

import re
import numpy as np
from numpy import prod, zeros, array
from glob import glob
from numpy import memmap, nan, empty
from ojmisc import myfromfile

debug = True

def readmeta(fname):
    dims = []
    i1s = []
    i2s = []
    dtype = '>f4'
    nrecords = 1
    with open(fname) as fid:
        ready = False
        for line in fid:
            # look for
            #          var = [ val ];
            m = re.match(r" ([a-zA-Z]*) = \[ ([^\]]*) \];", line)
            if m:
                var,val = m.groups()
                if var == 'dataprec':
                    if val == "'float64'" or val == "'real*8'":
                        dtype = '>f8'
                elif var == 'nrecords':
                    nrecords = int(val.strip("'"))
            # look for 
            #          dimList = [
            #                     n, i1, i2,
            #                     ...
            #                         ];
            if re.match(r' \]', line):
                ready = False
            if ready:
                dim,i1,i2 = map(int, re.split(', *', line.strip(', \n')))
                dims.append(dim)
                i1s.append(i1)
                i2s.append(i2)
            if re.match(r' dimList ', line): 
                ready = True
    # covert to C order
    dims.reverse()
    i1s.reverse()
    i2s.reverse()
    return dims,i1s,i2s,dtype,nrecords


def rdmds(baseglob, fill=0):
    metafiles = glob(baseglob + '.meta')
    dims,i1s,i2s,dtype,nrec = readmeta(metafiles[0])
    if nrec > 1:
        dims = [nrec] + dims
    res = np.zeros(dims)
    if fill != 0:
        res = fill
    for metafile in metafiles:
        datafile = re.sub(r'\.meta$', '.data', metafile)
        dims,i1s,i2s,dtype,nrec = readmeta(metafile)
        slc = map(lambda x,y: np.s_[x-1:y], i1s, i2s)
        tdims = map(lambda x,y:y-x+1, i1s, i2s)
        if nrec > 1:
            slc = [np.s_[:nrec]] + slc
            tdims = [nrec] + tdims
        if debug: print datafile, dtype, tdims
        res[slc] = np.fromfile(datafile, dtype).reshape(tdims)
    return res


def sliceshape(slicetuple, totshape):
    """ compute shape that results when slicing a totshape array with slicetuple """
    res = []
    for i,s in enumerate(slicetuple):
        if isinstance(s,int):
            #n = 1
            pass
        else:
            i0,i1,istep = s.indices(totshape[i])
            n = (i1-i0)//istep
            res.append(n)
    return res

class tilemap:
    def __init__(self, filepatt, tshape, itile=None, dtype='>f4', mode='r', offset=0, order=None, ncs=None, blankval=nan):
        """
            tm = tilemap('res_????/PP/PP_day.0000052704.%03d.001.data',(30,51,102),itile,'f4','r')
            tm = tilemap('res_????/PP/PP_day.0000052704.%03d.001.data',(30,51,102),ncs=510)
            tm = tilemap('res_%04d/PP/PP_day.0000052704.%03d.001.data',tshape=(30,51,102),ncs=510,dtype='>f4',mode='r')
        """
        self.filepatt = filepatt
        self.dtype = dtype
        self.mode = mode
        self.offset = offset
        self.order = order
        self.tshape = tuple(tshape)
        self.tnx = tshape[-1]
        self.tny = tshape[-2]
        self.glob = False
        if '*' in self.filepatt or '?' in self.filepatt:
            self.glob = True
        # value for blank tiles
        self.blankval = blankval
        if itile is not None:
            self.itile = itile
        elif ncs is not None:
            # standard cube sphere layout from ncs, tshape
            nface = 6
            ntx = ncs/self.tnx
            nty = ncs/self.tny
            self.itile = zeros((nty, nface*ntx), int)
            for yt in range(nty):
                for iface in range(nface):
                    for xt in range(ntx):
                        self.itile[yt,xt] = 1+(iface*nty+yt)*ntx+xt
        else:
            print 'error: tilemap: need either itile or ncs'
            return

        self.nty, self.ntx = itile.shape
        self.ntile = self.nty*self.ntx
        self.nx = self.ntx*self.tnx
        self.ny = self.nty*self.tny
        self.shape = self.tshape[:-2] + (self.ny,self.nx)
        self.memmap = [None for i in range(self.ntile)]

    def __getitem__(self, ind):
        """ val = tilemap[...,y,x] """
        if isinstance(ind[-1],int) and isinstance(ind[-2],int):
            # just one point in i,j plane
            x = ind[-1]
            y = ind[-2]
            xt,i = divmod(x, self.tnx)
            yt,j = divmod(y, self.tny)
            it = self.itile[yt,xt]
            if it == 0:
                return self.blankval
            else:
                if self.memmap[it-1] is None:
                    if self.glob:
                        # filepatt ~ 'res_*/field.%03d.data'
                        filename, = glob(self.filepatt % it)
                    else:
                        # filepatt ~ 'res_%04d/field.%03d.data'
                        filename = self.filepatt % (it-1,it)
                    self.memmap[it-1] = memmap(filename, self.dtype, self.mode, self.offset, self.tshape, self.order)
                tileind = tuple(ind[:-2]) + (j,i)
                return self.memmap[it-1][tileind]
        else:
            # i,j slices: we call ourselves for all index values (inefficient...)
            dims = sliceshape(ind, self.shape)
            i0,i1,istep = ind[-1].indices(self.nx)
            j0,j1,jstep = ind[-2].indices(self.ny)
            res = empty(dims)
            # tuple with as many trivial slices as non-i,j indices
            resind = (len(dims)-2)*(slice(None),)
            for sj,j in enumerate(range(j0,j1,jstep)):
                for si,i in enumerate(range(i0,i1,istep)):
                    res[resind+(sj,si)] = self[ind[:-2]+(j,i)]
            return res

    def __del__(self):
        # close files
        for mm in self.memmap:
            if mm is not None:
                del mm

class tiledfile:
    def __init__(self, filepatt, tshape, itile=None, dtype='>f4', ncs=None, blankval=nan):
        """
            tm = tiledfile('res_????/PP/PP_day.0000052704.%03d.001.data',(30,51,102),itile,'f4','r')
            tm = tiledfile('res_????/PP/PP_day.0000052704.%03d.001.data',(30,51,102),ncs=510)
            tm = tiledfile('res_%04d/PP/PP_day.0000052704.%03d.001.data',tshape=(30,51,102),ncs=510,dtype='>f4',mode='r')
        """
        self.filepatt = filepatt
        self.dtype = dtype
        self.tshape = tuple(tshape)
        self.tnx = tshape[-1]
        self.tny = tshape[-2]
        self.tsize = prod(self.tshape)
        self.glob = False
        if '*' in self.filepatt or '?' in self.filepatt:
            self.glob = True
        # value for blank tiles
        self.blankval = blankval
        if itile is not None:
            self.itile = array(itile, int)
        elif ncs is not None:
            # standard cube sphere layout from ncs, tshape
            nface = 6
            ntx = ncs/self.tnx
            nty = ncs/self.tny
            self.itile = zeros((nty, nface*ntx), int)
            for yt in range(nty):
                for iface in range(nface):
                    for xt in range(ntx):
                        self.itile[yt,iface*ntx+xt] = 1+(iface*nty+yt)*ntx+xt
        else:
            print 'error: tiledfile: need either itile or ncs'
            return

        self.nty, self.ntx = self.itile.shape
        self.ntile = self.nty*self.ntx
        self.nx = self.ntx*self.tnx
        self.ny = self.nty*self.tny
        self.shape = self.tshape[:-2] + (self.ny,self.nx)
        self.tiles = [None for i in range(self.ntile)]

    def __getitem__(self, ind):
        """ val = tiledfile[...,y,x] """
        if isinstance(ind[-1],int) and isinstance(ind[-2],int):
            # just one point in i,j plane
            x = ind[-1]
            y = ind[-2]
            xt,i = divmod(x, self.tnx)
            yt,j = divmod(y, self.tny)
            it = self.itile[yt,xt]
            if it == 0:
                return self.blankval
            else:
                if self.tiles[it-1] is None:
                    if self.glob:
                        # filepatt ~ 'res_*/field.%03d.data'
                        filename, = glob(self.filepatt % it)
                    else:
                        # filepatt ~ 'res_%04d/field.%03d.data'
                        filename = self.filepatt % (it-1,it)
                    self.tiles[it-1] = myfromfile(filename, self.dtype, count=self.tsize).reshape(self.tshape)
                tileind = tuple(ind[:-2]) + (j,i)
                return self.tiles[it-1][tileind]
        else:
            # i,j slices: we call ourselves for all index values (inefficient...)
            dims = sliceshape(ind, self.shape)
            i0,i1,istep = ind[-1].indices(self.nx)
            j0,j1,jstep = ind[-2].indices(self.ny)
            res = empty(dims)
            # tuple with as many trivial slices as non-i,j indices
            resind = (len(dims)-2)*(slice(None),)
            for sj,j in enumerate(range(j0,j1,jstep)):
                for si,i in enumerate(range(i0,i1,istep)):
                    res[resind+(sj,si)] = self[ind[:-2]+(j,i)]
            return res

    def __del__(self):
        # close files
        for tile in self.tiles:
            if tile is not None:
                del tile


class tiledfiles:
    def __init__(self, filepatt, tshape=None, itile=None, dtype='>f4', ncs=None, blankval=nan, its=None, cache=False,rot=None):
        """
            tm = tiledfiles('res_%04d/PP/PP_day.%010d.%03d.001.data',(30,51,102),itile,'f4','r')
            tm = tiledfiles('res_????/PP/PP_day.%010d.%03d.001.data',(30,51,102),ncs=510,dtype='f4')

            filepatt :: either printf format '..%d..%d..%d..' with itile-1, iter and itile slots
                        or printf format with shell glob '.*..%d..%d..' with iter and itile slots
            tshape   :: shape of single tile
            itile    :: 2d list or int array of tile indices (>= 1 for existing tiles, 0 for blank tiles)
            ncs      :: extent of cube face, used to compute itile (no blank tiles!)
            blankval :: value used to fill blank tiles
            its      :: list of iteration numbers to be assigned to time indices 0,1,...

            result tm can be indexed tm[t,k,j,i] where i and j may be both slices or both integers,
            k and t may be slice or integer.
        """
        self.filepatt = filepatt
        self.dtype = dtype
        if tshape is not None:
            self.tshape = tuple(tshape)
        else:
            metapatt = re.sub(r'\.data$', '.meta', filepatt)
            if re.search(r'%010d', metapatt):
                metapatt = re.sub(r'%010d', '%010d'%its[0], metpatt)
            metaglob = re.sub(r'%[0-9]*d', '*', metapatt)
            metafiles = glob(metaglob)
            dims,i1s,i2s = readmeta(metafiles[0])
            self.tshape = dims

        self.tnx = tshape[-1]
        self.tny = tshape[-2]
        self.tsize = prod(self.tshape)
        self.glob = False
        self.rot = rot
        if '*' in self.filepatt or '?' in self.filepatt:
            self.glob = True
        # value for blank tiles
        self.blankval = blankval
        self.its = its
        if itile is not None:
            self.itile = array(itile, int)
        elif ncs is not None:
            # standard cube sphere layout from ncs, tshape
            nface = 6
            ntx = ncs/self.tnx
            nty = ncs/self.tny
            self.itile = zeros((nty, nface*ntx), int)
            for yt in range(nty):
                for iface in range(nface):
                    for xt in range(ntx):
                        self.itile[yt,iface*ntx+xt] = 1+(iface*nty+yt)*ntx+xt
        else:
            print 'error: tiledfiles: need either itile or ncs'
            return

        self.nty, self.ntx = self.itile.shape
        self.ntile = self.nty*self.ntx
        self.nx = self.ntx*self.tnx
        self.ny = self.nty*self.tny
        self.cache = cache
        if its:  #  and cache:
            self.nt = len(its)
        else:
            self.nt = 1
        self.shape = (len(its),) + self.tshape[:-2] + (self.ny,self.nx)
        self.tiles = [[None for i in range(self.ntile)] for it in range(self.nt)]
        # current it (if not chache)
        self.it = None

    def __getitem__(self, ind):
        """ val = tiledfiles[...,y,x] """
        if isinstance(ind[0],int) and isinstance(ind[-1],int) and isinstance(ind[-2],int):
            # just one point in i,j plane
            x = ind[-1]
            y = ind[-2]
            if self.cache:
                t = ind[0]
            else:
                t = 0
            xt,i = divmod(x, self.tnx)
            yt,j = divmod(y, self.tny)
            itl = self.itile[yt,xt]
            if itl == 0:
                return self.blankval
            else:
                it = self.its[ind[0]]
                if (not self.cache) and it != self.it:
                    for ii,tile in enumerate(self.tiles[t]):
                        if tile is not None:
                            del tile
                            self.tiles[t][ii] = None
                    self.it = it
                if self.tiles[t][itl-1] is None:
                    if self.glob:
                        # filepatt ~ 'res_*/field.%010d.%03d.data'
                        patt = self.filepatt % (it,itl)
                        filenames = glob(patt)
                        if len(filenames) != 1:
                            filenames = glob(patt + '.gz')
                            if len(filenames) != 1:
                                print 'error: tiledfiles:', len(filenames), ' matches for ', patt, ' and ', patt + '.gz'
                                return None
                            else:
                                filename = filenames[0]
                        else:
                            filename = filenames[0]
                    else:
                        # filepatt ~ 'res_%04d/field.%010d.%03d.data'
                        filename = self.filepatt % (itl-1,it,itl)
                    print 'tiledfiles:', filename
                    self.tiles[t][itl-1] = myfromfile(filename, self.dtype, count=self.tsize).reshape(self.tshape)
                if self.rot is not None and self.rot[yt,xt]:
                    i,j = j,self.tnx-1-i
                tileind = tuple(ind[1:-2]) + (j,i)
                return self.tiles[t][itl-1][tileind]
        else:
            # i,j slices: we call ourselves for all index values (inefficient...)
            outdims = sliceshape(ind, self.shape)
            # ignore t,x,y for resind
            extradims = sliceshape(ind[1:-2], self.shape)
            # tuple with as many trivial slices as non-i,j indices
            extraind = (len(extradims))*(slice(None),)
            if isinstance(ind[0],int):
                t0,t1,tstep = ind[0],ind[0]+1,1
            else:
                t0,t1,tstep = ind[0].indices(self.nt)
            if isinstance(ind[-1], int):
                i0,i1,istep = ind[-1],ind[-1]+1,1
            else:
                i0,i1,istep = ind[-1].indices(self.nx)
            if isinstance(ind[-2], int):
                j0,j1,jstep = ind[-2],ind[-2]+1,1
            else:
                j0,j1,jstep = ind[-2].indices(self.ny)
            tmpind = (slice(t0,t1,tstep),) + ind[1:-2] + (slice(j0,j1,jstep), slice(i0,i1,istep))
            tmpdims = sliceshape(tmpind, self.shape)
            tmp = empty(tmpdims)
            print tmp.shape, outdims
            for st,t in enumerate(range(t0,t1,tstep)):
                for sj,j in enumerate(range(j0,j1,jstep)):
                    for si,i in enumerate(range(i0,i1,istep)):
                        tmp[(st,)+extraind+(sj,si)] = self[(t,)+ind[1:-2]+(j,i)]
            return tmp.reshape(outdims)

    def __del__(self):
        # close files
        for t in range(self.nt):
            for tile in self.tiles[t]:
                if tile is not None:
                    del tile

