#!/usr/bin/env/python
import sys,re
import pickle
import numpy as np
from numpy import prod, zeros, array
from glob import glob
from numpy import memmap, nan, empty
from mds import Metadata
from oj.num import myfromfile
import fnmatch

debug = True

def ravel_index(ind, dims):
    skip = 0
    for i,d in zip(ind,dims):
        skip *= d
        skip += i
    return skip


def asslice(ind):
    if isinstance(ind,int):
        return np.s_[ind:ind+1]
    else:
        return ind


def startstopstep(ind,nt):
    return asslice(ind).indices(nt)


def sliceindex(ind, starts, stops, steps):
    return tuple( min(stop-1,start+step*i) for i,start, stop, step in zip(ind, starts, stops, steps) )


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
            if re.match(r' [\]}]', line):
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


def rdmds(baseglob, ind=[], fill=0, fields=False):
    """ a = rdmds(baseglob, ind, fill)

reads baseglob.data using baseglob.meta to find shape.
Fill missing tiles with <fill>.

rdmds(b, ind) == rdmds(b)[ind+np.s_[...,]]  but only reads necessary bits
"""
    metafiles = glob(baseglob + '.meta')
    if len(metafiles) == 0:
        sys.stderr.write('rdmds: file not found: ' + baseglob + '.meta\n')
        raise IOError
    dims,i1s,i2s,dtype,nrec,flds = readmeta(metafiles[0])
    if nrec > 1:
        dims = [nrec] + dims
    res = np.zeros(dims[len(ind):])
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
            if debug: print datafile, dtype, tdims[len(ind):], skip/count
            with open(datafile) as fid:
                fid.seek(skip*size)
                res[slc[len(ind):]] = np.fromfile(fid, dtype, count=count).reshape(tdims[len(ind):])
        else:
            if debug: print datafile, dtype, tdims
            res[slc] = np.fromfile(datafile, dtype).reshape(tdims)

    if fields:
        return res,flds
    else:
        return res


class Mds(np.ndarray):
    def __new__(subtype, *args, **kwargs):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        arr,flds = rdmds(*args,fields=True,**kwargs)
        obj = arr.view(subtype)
        obj.fields = flds
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).
        self.fields = getattr(obj, 'fields', [])
        # We do not need to return anything


def gettiledata(pre, fieldit):
    """ tiledata = gettiledata('res_*/', 'ETAN.0000052704') """

    pres = re.split(r'\*', pre)
    preres = map(lambda s:fnmatch.translate(s)[:-1], pres)

    # dict of 6 empty lists
    tiledata = {}
    for key in ['tilepre', 'tilesuf', 'i0s', 'ies', 'j0s', 'jes']:
        tiledata[key] = []

    globpatt = pre + fieldit + '*.meta'
    print 'looking for meta files:', globpatt
    metafiles = glob(globpatt)
    metare = re.compile('(' + '.*'.join(preres) + ')' + fieldit + '(.*)$')

    maxdims = []
    for itile,metafile in enumerate(metafiles):
        m = metare.match(metafile)
        tilepre = m.group(1)
        tilesuf = m.group(2)
        dims,i1s,i2s,dtype,nrec,flds = readmeta(metafile)

        tiledata['dtype'] = dtype
        tiledata['tilepre'].append(tilepre)
        tiledata['tilesuf'].append(re.sub(r'\.meta$', '.data', tilesuf))
        i0 = i1s[-1]-1
        ie = i2s[-1]
        j0 = i1s[-2]-1
        je = i2s[-2]
        tiledata['i0s'].append(i0)
        tiledata['ies'].append(ie)
        tiledata['j0s'].append(j0)
        tiledata['jes'].append(je)

        if nrec > 1:
            dims = [nrec] + dims
        if len(dims) > len(maxdims):
            maxdims[0:0] = (len(dims)-len(maxdims))*[1]
        for i in range(len(dims)):
            maxdims[-1-i] = max(maxdims[-1-i], dims[-1-i])

    tiledata['filelayers'] = maxdims[:-2]

    return tiledata


def savetiledata(fname, tiledata):
    i0s = [ str(i) for i in tiledata['i0s'] ]
    ies = [ str(i) for i in tiledata['ies'] ]
    j0s = [ str(i) for i in tiledata['j0s'] ]
    jes = [ str(i) for i in tiledata['jes'] ]
    np.savetxt(fname, np.c_[i0s,ies,j0s,jes,tiledata['tilepre'],tiledata['tilesuf']], fmt='%s')


def loadtiledata(fname):
    ij0es = np.loadtxt(fname,'S')
    tiledata = {}
    tiledata['i0s'] = ij0es[:,0].astype(int).tolist()
    tiledata['ies'] = ij0es[:,1].astype(int).tolist()
    tiledata['j0s'] = ij0es[:,2].astype(int).tolist()
    tiledata['jes'] = ij0es[:,3].astype(int).tolist()
    tiledata['tilepre'] = ij0es[:,4].tolist()
    tiledata['tilesuf'] = ij0es[:,5].tolist()
    return tiledata


class TiledView:
    def __init__(self, tiled, index):
        self.tiled = tiled
        # self.start[]
        # self.stop[]
        # self.step[]
        # self.shape[]
        # self.ndim

        # Fix index, handling ellipsis and incomplete slices.
        if type(index) != type(()): index = (index,)
        fixed = []
        collapse = []
        length, dims = len(index), len(tiled.shape)
        for slice_ in index:
            if slice_ is Ellipsis:
                fixed.extend([slice(None)] * (dims-length+1))
                collapse.extend([False] * (dims-length+1))
                length = len(fixed)
            elif isinstance(slice_, (int, long)):
                fixed.append(slice(slice_, slice_+1, 1))
                collapse.append(True)
            else:
                fixed.append(slice_)
                collapse.append(False)
        index = tuple(fixed)
        if len(index) < dims:
            index += (slice(None),) * (dims-len(index))
            collapse += [False] * (dims-len(index))

        # Return a new arrayterator object.
#        out = self.__class__(self.var, self.buf_size)
        self.start = dims*[0]
        self.stop  = dims*[0]
        self.step  = dims*[0]
        for i, (stop, slice_) in enumerate(
                zip(tiled.shape, index)):
            start = 0
            step = 1
            self.start[i] = start + (slice_.start or 0)
            self.step[i] = step * (slice_.step or 1)
            self.stop[i] = start + (slice_.stop or stop-start)
            self.stop[i] = min(stop, self.stop[i])

        self.shape = tuple(((stop-start-1)//step+1) for start, stop, step in
                           zip(self.start, self.stop, self.step))
        self.ndim = dims


    def __getitem__(self, index):
        # Make sure index has entries for all dimensions
        if type(index) != type(()): index = (index,)
        fixed = []
        length, dims = len(index), len(self.shape)
        for slice_ in index:
            if slice_ is Ellipsis:
                fixed.extend([slice(None)] * (dims-length+1))
                length = len(fixed)
            else:
                fixed.append(slice_)
        index = tuple(fixed)
        if len(index) < dims:
            index += (slice(None),) * (dims-len(index))

        # map index through start,stop,step
        mind = dims*[0]
        for i, (ind, start, stop, step) in enumerate(
                        zip(index, self.start, self.stop, self.step)):
            if isinstance(ind,slice):
                a = start + (ind.start or 0)
                c = step * (ind.step or 1)
                b = start + (ind.stop or stop-start)
                b = min(stop, b)
                mind[i] = slice(a,b,c)
            else:
                mind[i] = min(stop-1, start+step*ind)

        print mind
        return self.tiled[mind]



class TiledFiles:
    def __init__(self,ncache=0):
        self.nameparts = []
        self.tilepre = []
        self.tilesuf = []
        self.dims = []
        self.i0s = []
        self.ies = []
        self.j0s = []
        self.jes = []
        self.ni = []
        self.nj = []
        self.itile = None
        self.buf = None
        self.bufind = []
        self.bufitile = -1
        self.ncache = ncache
        self.icache = 0
        self.cachedict = {}
        self.cache = self.ncache*[None]
        self.cachekey = self.ncache*[None]
        self.fill = 0
        self.notfound = 0
        self.fields = []
        self.nameparts = []
        self.indnames = []
        self.ntiledim = 0
        self.shape = []
        self.ndim = 0
        self.nfileind = 0
        self.tshape = []
        self.dtype = None

    @classmethod
    def fromtiledata(cls, basepatt, its=[], fields=None, tilepre=[], tilesuf=[], i0s=[],ies=[],j0s=[],jes=[], filelayers=None, dtype='>f4', fill=0, ncache=None, memmax=None, notfound=np.nan):
        self = cls(0)

        self.dtype = np.dtype(dtype)

        self.fill = fill
        self.notfound = notfound

        try:
            pre,suf = re.split(r'\*\*', basepatt)
        except ValueError:
            sys.stderr.write('TiledFiles: basepatt must include "**"\n')
            return None

        pres = re.split(r'\*', pre)
        nprestar = len(pres)-1
        preres = map(lambda s:fnmatch.translate(s)[:-1], pres)

        if fields is None:
            if isinstance(its,(int,long)):
                it = its
            else:
                it = its[0]
            globpatt = pre + '*' + suf + '.%010d'%it + tilesuf[0]
            print "Looking for fields in", globpatt
            datafiles = glob(globpatt)
            datare = re.compile('.*'.join(preres) + '(.*)' + suf + r'\.[0-9]{10}')
            fields = []
            for datafile in datafiles:
                m = datare.match(datafile)
                field = ''
                if m:
                    field = m.group(1)
                fields.append(field)
            fields.sort()
            print fields

        if isinstance(its, (int,long)) or len(its) == 0:
            globpatt = pre + fields[0] + suf + '.*' + tilesuf[0]
            print "Looking for its in", globpatt
            datafiles = glob(globpatt)
            if len(datafiles) == 0:
                datafiles = glob(globpatt + '.gz')
            datare = re.compile('.*'.join(preres) + fields[0] + suf + r'\.([0-9]{10})')
            its = []
            for datafile in datafiles:
                m = datare.match(datafile)
                it = 0
                if m:
                    it = int(m.group(1))
                its.append(it)
            its.sort()
            print its

        self.its = list(its)
        self.fields = fields

        self.nameparts = [pre, suf]
        self.indnames = [ fields, [ '.%010d'%i for i in its ] ]
        self.labels = [fields, its]

        if filelayers is None:
            filelayers = []

        tnx = max([ie-i0 for i0,ie in zip(i0s,ies)])
        tny = max([je-j0 for j0,je in zip(j0s,jes)])
        tshape = list(filelayers) + [tny,tnx]

        nx = max(ies)
        ny = max(jes)
        maxdims = list(tshape[:-2]) + [ny,nx]
        self.dims = [maxdims for field in fields]
            
        self.ntiledim = len(maxdims)
        self.shape = [len(self.fields),len(its)] + maxdims
        self.ndim = len(self.shape)
        self.nfileind = self.ndim - self.ntiledim

        self.tshape = tshape
        tsize2d = np.prod(self.tshape[-2:])*8
        if ncache is None:
            if memmax is not None:
                ncache = memmax//tsize2d
            else:
                ncache = 1000
        if memmax is None: memmax = 2*1024*1024*1024
        if ncache*tsize2d > memmax: ncache = memmax//tsize2d
        self.ncache = ncache
        self.cache = ncache*[None]
        self.cachekey = ncache*[None]

        if tilepre is None:
            tilepre = ['' for i in tilesuf]
            fieldit = self.indnames[0][0] + suf + self.indnames[1][0]
            globpatt = pre + fieldit + '.*.data'
            print "Looking for tile prefixes in", globpatt
            datafiles = glob(globpatt)
            if len(datafiles) == 0:
                print "Looking for tile prefixes in", globpatt
                datafiles = glob(globpatt + '.gz')
            datare = re.compile('(' + '.*'.join(preres) + ')' + fieldit + '(.*)$')
            if len(datafiles) == 0:
                print 'file not found'
            else:
                for datafile in datafiles:
                    m = datare.match(datafile)
                    mypre,mysuf = m.groups()
                    mysuf = re.sub('\.gz$','',mysuf)
#                    print mysuf
                    tilepre[tilesuf.index(mysuf)] = mypre

        self.tilepre = tilepre
        self.tilesuf = tilesuf
        self.i0s = i0s
        self.ies = ies
        self.j0s = j0s
        self.jes = jes
        self.itile = np.zeros(maxdims[-2:], np.integer) - 1
        self.ni = [ ie-i0 for i0,ie in zip(i0s,ies) ]
        self.nj = [ je-j0 for j0,je in zip(j0s,jes) ]

        for itile,(i0,ie,j0,je) in enumerate(zip(i0s,ies,j0s,jes)):
            self.itile[j0:je,i0:ie] = itile

        return self


    @classmethod
    def fromglobalfiles(cls, basepatt, dtype, shape, its=[], fields=None, fill=0, ncache=None, memmax=None, notfound=np.nan):
        
        self = cls(0)

        try:
            pre,suf = re.split(r'\*\*', basepatt)
        except ValueError:
            sys.stderr.write('TiledFiles: basepatt must include "**"\n')
            return None

        tilepre = [ pre ]
        tilesuf = [ '.data' ]
        
        pres = re.split(r'\*', pre)
        nprestar = len(pres)-1
        preres = map(lambda s:fnmatch.translate(s)[:-1], pres)

        if fields is None:
            if isinstance(its,(int,long)):
                it = its
            else:
                it = its[0]
            globpatt = pre + '*' + suf + '.%010d'%it + tilesuf[0]
            print "Looking for fields in", globpatt
            datafiles = glob(globpatt)
            datare = re.compile('.*'.join(preres) + '(.*)' + suf + r'\.[0-9]{10}')
            fields = []
            for datafile in datafiles:
                m = datare.match(datafile)
                field = ''
                if m:
                    field = m.group(1)
                fields.append(field)
            fields.sort()
            print fields

        if isinstance(its, (int,long)) or len(its) == 0:
            globpatt = pre + fields[0] + suf + '.*' + tilesuf[0]
            print "Looking for its in", globpatt
            datafiles = glob(globpatt)
            datare = re.compile('.*'.join(preres) + fields[0] + suf + r'\.([0-9]{10})')
            its = []
            for datafile in datafiles:
                m = datare.match(datafile)
                it = 0
                if m:
                    it = int(m.group(1))
                its.append(it)
            its.sort()
            print its

        #its = list(its)

        filelayers = shape[:-2]
        i0s = [ 0 ]
        ies = [ shape[-1] ]
        j0s = [ 0 ]
        jes = [ shape[-2] ]

        obj = cls.fromtiledata(basepatt, its, fields, tilepre, tilesuf, i0s,ies,j0s,jes, filelayers, dtype, fill, ncache, memmax, notfound)
        return obj


    @classmethod
    def cs(cls, basepatt, ncs, tshape, dtype='>f4', its=None, fields=None, fill=0, ncache=None, memmax=None):
        self = cls(0)

        self.fields = fields or []
        self.fill = fill

        try:
            pre,suf = re.split(r'\*\*', basepatt)
        except ValueError:
            sys.stderr.write('TiledFiles: basepatt must include "**"\n')
            return None

        pres = re.split(r'\*', pre)
        nprestar = len(pres)-1
        preres = map(lambda s:fnmatch.translate(s)[:-1], pres)

        if its is None and fields is not None:
            # get its from first field and tile
            base = pre + fields[0] + suf
            print 'looking for its in ' + base
            datafiles = glob(base + '.*.001.001.data')
            if len(datafiles) == 0:
                datafiles = glob(base + '.*.001.001.data.gz')
            if len(datafiles) == 0:
                datafiles = glob(base + '.*.data')

            datare = re.compile('.*'.join(preres) + fields[0] + suf + r'\.([0-9]{10})')
            its = []
            for datafile in datafiles:
                m = datare.match(datafile)
                if m:
                    its.append(int(m.group(1)))
            its.sort()

        self.its = its

        self.nameparts = [pre, suf]
        self.indnames = [[], map(lambda i:'.%010d'%i, its)]

        if fields is None:
            # get fields from first it and tile
            globpre = pre + '*' + suf + '.%010d' % its[0]
            print 'looking for dimensions in ' + globpre
            datafiles = glob(globpre + '.data')
            if len(datafiles) == 0:
                datafiles = glob(globpre + '.data.gz')
            if len(datafiles) == 0:
                datafiles = glob(globpre + '.001.001.data')
            if len(datafiles) == 0:
                datafiles = glob(globpre + '.001.001.data.gz')
            if len(datafiles) == 0:
                print 'nofiles found: ', globpre + '.data'

            datare = re.compile('.*'.join(preres) + '(.*)' + suf + r'\.[0-9]{10}')
            maxdims = []
            for datafile in datafiles:
                m = datare.match(datafile)
                field = ''
                if m:
                    field = m.group(1)
                self.indnames[0] += [field]

                field = re.sub(r'^'+pre, '', datafile)
                field = re.sub(suf+r'\.[0-9]*$', '', field)
                if fields is None:
                    self.fields.append(field)
        else:
            self.indnames[0] = fields

        maxdims = list(tshape[:-2]) + [ncs,ncs*6]
        self.dims = [maxdims for field in fields]
            
        self.ntiledim = len(maxdims)
        self.shape = [len(self.fields),len(its)] + maxdims
        self.ndim = len(self.shape)
        self.nfileind = self.ndim - self.ntiledim

        self.tshape = tshape
        tsize2d = np.prod(self.tshape[-2:])*8
        if ncache is None:
            if memmax is not None:
                ncache = memmax//tsize2d
            else:
                ncache = 1000
        if memmax is None: memmax = 2*1024*1024*1024
        if ncache*tsize2d > memmax: ncache = memmax//tsize2d
        self.ncache = ncache
        self.cache = ncache*[None]
        self.cachekey = ncache*[None]

        # get tile info from first field
        self.tilepre = []
        self.tilesuf = []
        self.i0s = []
        self.ies = []
        self.j0s = []
        self.jes = []
        self.itile = np.zeros(maxdims[-2:], np.integer) - 1

        tnx = tshape[-1]
        tny = tshape[-2]
        ntx = 6*ncs/tnx
        nty = ncs/tny

        self.dtype = np.dtype(dtype)

        if 0:
            for itile in range(ntx*nty):
                tilenum = itile + 1
                tilepre = pres[0] + '%04d' % itile + pres[1]
                tilesuf = '.%03d.001.data' % tilenum

                self.tilepre.append(tilepre)
                self.tilesuf.append(re.sub(r'\.data.*$', '.data', tilesuf))

                tmp,itx = divmod(tilenum-1,ntx/6)
                ifc,ity = divmod(tmp,nty)

                i0 = ifc*ncs + itx*tnx
                ie = ifc*ncs + (itx+1)*tnx
                j0 = ity*tny
                je = (ity+1)*tny

                self.i0s.append(i0)
                self.ies.append(ie)
                self.j0s.append(j0)
                self.jes.append(je)
                self.ni.append(ie-i0)
                self.nj.append(je-j0)
                self.itile[j0:je,i0:ie] = itile

        else:
            fielditname = self.indnames[0][0] + suf + self.indnames[1][0]
            datafiles = glob(pre + fielditname + '*.data')
            if len(datafiles) == 0:
                datafiles = glob(pre + fielditname + '*.data.gz')
            datare = re.compile('(' + '.*'.join(preres) + ')' + fielditname + '(.*)$')
            for itile,datafile in enumerate(datafiles):
                m = datare.match(datafile)
                tilepre = m.group(1)
                tilesuf = m.group(2)

                self.tilepre.append(tilepre)
                self.tilesuf.append(re.sub(r'\.data.*$', '.data', tilesuf))

                tilenum = int(re.sub(r'\.001\.data.*$', '', tilesuf)[-3:])
                tmp,itx = divmod(tilenum-1,ntx/6)
                ifc,ity = divmod(tmp,nty)

                i0 = ifc*ncs + itx*tnx
                ie = ifc*ncs + (itx+1)*tnx
                j0 = ity*tny
                je = (ity+1)*tny

                self.i0s.append(i0)
                self.ies.append(ie)
                self.j0s.append(j0)
                self.jes.append(je)
                self.ni.append(ie-i0)
                self.nj.append(je-j0)
                self.itile[j0:je,i0:ie] = itile

        return self


    @classmethod
    def mds(cls, basepatt, its=None, fields=None, fill=0, ncache=None, memmax=None):
        self = cls(0)

        self.fields = fields or []
        self.fill = fill

        try:
            pre,suf = re.split(r'\*\*', basepatt)
        except ValueError:
            sys.stderr.write('TiledFiles.mds: basepatt must include "**"\n')
            return None

        pres = re.split(r'\*', pre)
        nprestar = len(pres)-1
        preres = map(lambda s:fnmatch.translate(s)[:-1], pres)

        if its is None and fields is not None:
            # get its from first field and tile
            base = pre + fields[0] + suf
            print 'looking for its in ' + base
            metafiles = glob(base + '.*.001.001.meta')
            if len(metafiles) == 0:
                metafiles = glob(base + '.*.meta')

            metare = re.compile('.*'.join(preres) + fields[0] + suf + r'\.([0-9]{10})')
            its = []
            for metafile in metafiles:
                m = metare.match(metafile)
                if m:
                    its.append(int(m.group(1)))
            its.sort()

        self.its = its

        self.nameparts = [pre, suf]
        self.indnames = [[], map(lambda i:'.%010d'%i, its)]

        # get fields from first it and tile
        globpre = pre + '*' + suf + '.%010d' % its[0]
        print 'looking for dimensions in ' + globpre
        metafiles = glob(globpre + '.meta')
        if len(metafiles) == 0:
            metafiles = glob(globpre + '.001.001.meta')

        metare = re.compile('.*'.join(preres) + '(.*)' + suf + r'\.[0-9]{10}')
        maxdims = []
        for metafile in metafiles:
            m = metare.match(metafile)
            field = ''
            if m:
                field = m.group(1)
            self.indnames[0] += [field]

            dims,i1s,i2s,dtype,nrec,flds = readmeta(metafile)
            if nrec > 1:
                dims = [nrec] + dims
            self.dims += [dims]

            if len(flds) > 0:
                field = flds[0]
            else:
                field = re.sub(r'^'+pre, '', metafile)
                field = re.sub(suf+r'\.[0-9]*$', '', field)
            if fields is None:
                self.fields.append(field)

            if len(dims) > len(maxdims):
                maxdims = (len(dims)-len(maxdims))*[1] + maxdims
            for i in range(len(dims)):
                maxdims[-1-i] = max(maxdims[-1-i], dims[-1-i])
            
        self.ntiledim = len(maxdims)
        self.shape = [len(self.fields),len(its)] + maxdims
        self.ndim = len(self.shape)
        self.nfileind = self.ndim - self.ntiledim

        self.tshape = self.shape[:-2] + [i2-i1 for i1,i2 in zip(i1s[-2:],i2s[-2:])]
        tsize2d = np.prod(self.tshape[-2:])*8

        # get tile info from first field
        self.tilepre = []
        self.tilesuf = []
        self.i0s = []
        self.ies = []
        self.j0s = []
        self.jes = []
        self.itile = np.zeros(maxdims[-2:], np.integer) - 1

        fielditname = self.indnames[0][0] + suf + self.indnames[1][0]
        metafiles = glob(pre + fielditname + '*.meta')
        metare = re.compile('(' + '.*'.join(preres) + ')' + fielditname + '(.*)$')
        for itile,metafile in enumerate(metafiles):
            m = metare.match(metafile)
            tilepre = m.group(1)
            tilesuf = m.group(2)
            dims,i1s,i2s,dtype,nrec,flds = readmeta(metafile)

            self.dtype = dtype
            self.tilepre.append(tilepre)
            self.tilesuf.append(re.sub(r'\.meta$', '.data', tilesuf))
            i0 = i1s[-1]-1
            ie = i2s[-1]
            j0 = i1s[-2]-1
            je = i2s[-2]
            self.i0s.append(i0)
            self.ies.append(ie)
            self.j0s.append(j0)
            self.jes.append(je)
            self.ni.append(ie-i0)
            self.nj.append(je-j0)
            for j in range(j0,je):
                for i in range(i0,ie):
                    self.itile[j,i] = itile

        self.metadata = Metadata.read(metafiles[0])
        self.metadata.makeglobal()

	ntile = len(self.tilepre)
	if ncache is None:
            if memmax is not None:
                ncache = memmax//tsize2d
            else:
                ncache = ntile
        if memmax is None: memmax = 2*1024*1024*1024
        if ncache*tsize2d > memmax: ncache = memmax//tsize2d
        self.ncache = ncache
        self.cache = ncache*[None]
        self.cachekey = ncache*[None]

        return self


    def update_its(self, its=None):
        if its is None:
            # get its from first field and tile
            base = self.nameparts[0] + self.fields[0] + self.nameparts[1]
            pres = re.split(r'\*', self.nameparts[0])
            preres = map(lambda s:fnmatch.translate(s)[:-1], pres)
            metare = re.compile('.*'.join(preres) + self.fields[0] + self.nameparts[1] + r'\.([0-9]{10})')

            print 'looking for its in ' + base
            metafiles = glob(base + '.*.001.001.meta')
            if len(metafiles) == 0:
                metafiles = glob(base + '.*.meta')

            its = []
            for metafile in metafiles:
                m = metare.match(metafile)
                if m:
                    its.append(int(m.group(1)))
            its.sort()

        print its

        self.its = its

        self.indnames[1] = map(lambda i:'.%010d'%i, its)
        self.labels[1] = its

        self.shape[1] = len(its)


    def gettile(self, ind, itile):
        """ ind does not include j,i """
        key = tuple(ind) + (itile,)
        if key in self.cachedict:
            buf = self.cache[self.cachedict[key]]
        else:
            tilepre = self.tilepre[itile]
            dataname = tilepre
            for i in range(len(self.indnames)):
                # tilepre replaces nameparts[0]
                if i > 0:
                    dataname += self.nameparts[i]
                dataname += self.indnames[i][ind[i]]

            dataname += self.tilesuf[itile]

            ni = self.ni[itile]
            nj = self.nj[itile]
#            dims = list(self.shape[self.nfileind:-2]) + [nj,ni]
            size = nj*ni*self.dtype.itemsize
            dims = self.dims[ind[0]]
            ndim = len(dims)
            myind = ind[self.nfileind+self.ntiledim-ndim:self.nfileind+self.ntiledim-2]
            for i in range(ndim-2):
                im = -1-i
                im2 = -3-i
                if myind[im] >= dims[im2] and myind[im] < self.shape[im2]:
                    myind[im] = dims[im2]-1
            skip = 0
            for i in range(ndim-2):
                skip *= dims[i]
                skip += myind[i]
            
            if debug: print dataname, self.dtype, (nj,ni), skip
            try:
                buf = myfromfile(dataname, self.dtype, shape=(nj,ni), skip=skip)
            except IOError as err:
                #print 'IOError: ', dataname, err
                print 'IOError: ', err
                buf = np.empty((nj,ni))
                buf[:] = self.notfound

            oldkey = self.cachekey[self.icache]
            if oldkey in self.cachedict:
                del self.cachedict[oldkey]
            self.cache[self.icache] = buf
            self.cachekey[self.icache] = key
            self.cachedict[key] = self.icache
            self.icache = (self.icache+1) % self.ncache

        return buf


    def __getitem__(self, ind):
        """ val = mds[...,y,x] """
        if type(ind) not in [type(()), type([])]: ind = (ind,)
        singleslice = True
        if len(ind) < self.ndim - 2:
            singleslice = False
        else:
            for i in ind[:self.ndim-2]:
                if not isinstance(i, (int,long)):
                    singleslice = False
        if singleslice:
            if len(ind) < self.ndim or ind[self.ndim-1] is Ellipsis:
                indx = slice(None)
            else:
                indx = ind[self.ndim-1]
            if len(ind) < self.ndim-1 or ind[self.ndim-2] is Ellipsis:
                indy = slice(None)
            else:
                indy = ind[self.ndim-2]
            eind = ind[:self.ndim-2]

            j0,je,dj = startstopstep(indy,self.shape[-2])
            i0,ie,di = startstopstep(indx,self.shape[-1])

            nny, nnx = (je-j0)//dj, (ie-i0)//di
            res = np.empty((nny,nnx))
            res[:] = self.fill

            if nny*nnx <= 100:

                tiles = set(self.itile[indy,indx].flat)
                for itile in tiles:
                  if itile >= 0:
                    buf = self.gettile(eind,itile)

                    for jj,j in enumerate(range(j0,je,dj)):
                      if self.j0s[itile] <= j < self.jes[itile]:
                        for ii,i in enumerate(range(i0,ie,di)):
                          if self.i0s[itile] <= i < self.ies[itile]:
                            res[jj,ii] = buf[j-self.j0s[itile],i-self.i0s[itile]]

            else:

                for itile in range(len(self.i0s)):
#                    print 'buf ', eind, itile
                    
                    j0t = self.j0s[itile]
                    jet = self.jes[itile]
                    i0t = self.i0s[itile]
                    iet = self.ies[itile]
                    jj0 = max(0,   -(j0 - j0t)//dj)
                    jje = min(nny, -(j0 - jet)//dj)
                    ii0 = max(0,   -(i0 - i0t)//di)
                    iie = min(nnx, -(i0 - iet)//di)
                    if jje > jj0 and iie > ii0:
                        buf = self.gettile(eind,itile)
                        res[jj0:jje,ii0:iie] = buf[j0+dj*jj0-j0t:j0+dj*jje-j0t:dj, i0+di*ii0-i0t:i0+di*iie-i0t:di]

            return res
        else:
            return TiledView(self, ind)


    def dumptiledata(self, f):
        tiledata = {}
        tiledata['i0s'] = self.i0s
        tiledata['ies'] = self.ies
        tiledata['j0s'] = self.j0s
        tiledata['jes'] = self.jes
        tiledata['tilepre'] = self.tilepre
        tiledata['tilesuf'] = self.tilesuf
        try:
            pickle.dump(tiledata, f)
        except AttributeError:
            with open(f,'w') as fid:
                pickle.dump(tiledata, fid)


class mds:
    def __init__(self, basepatt, its=[], fields=None, fill=0):
        """ mds('res_????/#/day', [52632], fill) 
        
            reads   res_0000/ETAN/day.0000052632.001.001.meta, ...
        """
        self.tiled = False
        self.fill = fill
        self.its = its
        self.fields = []
        self.shape = ()
        self.buf = None
        self.bufind = []
        self.bufnind = 0
        self.bufdims = []

        if fields is not None:
            self.fields = fields

        pre,suf = re.split(r'#', basepatt)
        self.pre = pre
        self.suf = suf

        metafiles = glob(pre + '*' + suf + '.%010d' % its[0] + '.meta')
        if len(metafiles) == 0:
            metafiles = glob(pre + '*' + suf + '.%010d' % its[0] + '.001.001.meta')
            self.tiled = True
            
        maxdims = []
        for metafile in metafiles:
            dims,i1s,i2s,dtype,nrec,flds = readmeta(metafile)
            if len(flds) > 0:
                field = flds[0]
            else:
                field = re.sub(r'^'+pre, '', metafile)
                field = re.sub(suf+r'\.[0-9]*$', '', field)
            if fields is None:
                self.fields.append(field)

            if nrec > 1:
                dims = [nrec] + dims
            if len(dims) > len(maxdims):
                maxdims = (len(dims)-len(maxdims))*[1] + maxdims
            for i in range(len(dims)):
                maxdims[-1-i] = max(maxdims[-1-i], dims[-1-i])
            
        self.shape = [len(self.fields),len(its)] + maxdims
        self.bufnind = 2
        self.bufdims = maxdims
        self.bufind = [-1,-1]

    def __getitem__(self, ind):
        """ val = mds[...,y,x] """
        if ind[:self.bufnind] != self.bufind:
            field = self.fields[ind[0]]
            it = self.its[ind[1]]
            baseglob = self.pre + field + self.suf + '.%010d' % it + '*'
            self.buf = rdmds(baseglob, fill=self.fill)
            self.bufind = ind[:self.bufnind]
            self.bufdims = self.buf.shape
            if len(self.bufdims) < len(self.shape)-2:
                self.bufdims = (len(self.shape)-2-len(self.bufdims))*(1,) + self.bufdims
                self.buf = self.buf.reshape(self.bufdims)
        ndim = len(self.bufdims)
        myind = list(ind[2:])
        for i in range(ndim):
            im = -1-i
            if myind[im] >= self.buf.shape[im] and myind[im] < self.shape[im]:
                myind[im] = self.buf.shape[im]-1
        return self.buf[myind]
    

