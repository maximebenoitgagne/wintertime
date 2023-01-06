#!/usr/bin/env python
import sys,os,re
import copy
import fnmatch
from myfnmatch import re_parts, interleave #, format2re
from myformatter import fmt,fstring,template_replace,template_replace_all,template_fields,format2re
import numpy as np
from glob import glob
from mds import Metadata
from mslice import MSlice
import oj.num

debug = False

#def template2re(tmpl):
#    """ given a template with both {...} and glob wildcards,
#    generates a regexp that will match a result of glob and return
#    a group for each {...}
#    """
#    parts = re.split(r'\{[^\}]*\}', tmpl)
#    # turn into regexps without trailing '$'
#    regexps = [ fnmatch.translate(s)[:-1] for s in parts ]
#    regexp = re.compile(r'(.*)'.join(regexps)) + '$'
#    return regexp


class VariableData(object):
    def __init__(self, tmpl, its=None, astype=None, fill=0, dtype=None, shape=None, tshape=None, ncs=None, dt=0, mmap=True, debug=False, quiet=False, nblank=0):
        self.usemmap = mmap
        self.hasits = re.search(r'\{it[:\}]', tmpl) is not None
        if self.hasits and its is None:
            parts = re.split(r'\{it[^\}]*\}', tmpl)
            base = '*'.join(parts)
            for suf in ['.001.001.data', '.data', '.001.001.data.gz', '.data.gz']:
                print base + suf
                datafiles = glob(base + suf)
                if len(datafiles) != 0:
                    break
            else:
                sys.stderr.write('file not found: ' + base + suf + '\n')
                raise IOError

            its = []

            parts[-1] += suf
            # turn into regexps without trailing '$'
            regexps = [ re.sub(r'(\\Z\(\?ms\))?\$?$','',fnmatch.translate(s)) for s in parts ]
            try:
                regexp = re.compile(r'(.*)'.join(regexps) + '$')
            except:
                print 're:' + r'(.*)'.join(regexps) + '$'
                raise
            for datafile in datafiles:
                m = regexp.match(datafile)
                if m:
                    it = m.group(1)
                    its.append(it)
                else:
                    print 'file does not match:', datafile, r', (.*)'.join(regexps) + '$'

            its.sort()

            self.filedicts = np.array([ {'it':it} for it in its ])

            # choose first for tile search
            base = tmpl.format(it=its[0])
        else:
            self.filedicts = np.array([{}])
            base = tmpl

        self.its = its
        if debug: print 'its:',its

        if dtype is not None and shape is not None and tshape is None:
            # global files
            self.tiled = False
            self.metadata = None
            self.shape = shape
            self.nrec = 1
            self.tdtype = np.dtype(dtype)
            self.tny,self.tnx = shape[-2:]
        elif dtype is not None and tshape is not None and shape is None:
            # cubed sphere
            self.tiled = True
            self.metadata = None
            self.nrec = 1
            self.tdtype = np.dtype(dtype)
            self.tny,self.tnx = tshape[-2:]
            if glob(base + '.001.001.data'):
                suf = ''
            else:
                suf = '.gz'
            globpatt = base + '.*.*.data' + suf
            tmplre,tmplparts = re_parts(tmpl + '.*.*.data' + suf)
            basere,baseparts = re_parts(base + '.*.*.data' + suf)
            datafiles = glob(globpatt)
            datafiles.sort()
            self.ntile = len(datafiles)
            ny = int(np.sqrt(self.tnx*self.tny*self.ntile/6))
            nx = 6*ny
            self.shape = tshape[:-2] + (ny,nx)
            self.ntx = nx/self.tnx
            self.nty = ny/self.tny
            self.files = []
            # blank tiles have index -1
            self.itile = np.zeros((self.nty,self.ntx), int) - 1
            for itile,datafile in enumerate(datafiles):
                m = basere.match(datafile)
                g = m.groups()
                datatmpl = interleave(g,tmplparts)
                # in case some files are not gzipped
                datatmpl = re.sub(r'\.gz$','',datatmpl)
                self.files.append(datatmpl)
                _,it = divmod(itile,self.ntx/6)
                f,jt = divmod(_,self.nty)
                it += self.ntx/6*f
                self.itile[jt,it] = itile
        else:
            # metafiles
            globpatt = base + '.001.001.meta'
            if debug: print 'looking for metafiles:', globpatt
            metafiles = glob(globpatt)
            if len(metafiles) > 0:
                self.tiled = True
            else:
                globpatt = base + '.meta'
                if debug: print 'looking for metafiles:', globpatt
                metafiles = glob(base + '.meta')
                self.tiled = False

            if len(metafiles) == 0:
                sys.stderr.write('File not found: ' + base + '.meta\n')
                raise IOError

            # get global info from first metafile
            self.metadata = Metadata.read(metafiles[0])
            self.shape = tuple(self.metadata.dims)
            self.nrec = self.metadata.nrecords
            self.tdtype = np.dtype(self.metadata.dtype)
            self.tnx = self.metadata.ends[-1] - self.metadata.starts[-1]
            self.tny = self.metadata.ends[-2] - self.metadata.starts[-2]
            self.metadata.makeglobal()
            if self.hasits:
                if 'timeStepNumber' in self.metadata:
                    it = int(self.metadata['timeStepNumber'])
                    self.timeStepNumber = map(int,self.its)
                    self.metadata['timeStepNumber'] = []
                    if 'timeInterval' in self.metadata:
                        ts = self.metadata['timeInterval'].split()
                        dt = float(ts[-1])/it
                        if len(ts) == 1:
                            ival = [ [dt*int(it)] for it in self.its ]
                        else:
                            period = float(ts[1]) - float(ts[0])
                            ival = [ [dt*int(it)-period, dt*int(it)] for it in self.its ]

                        self.timeInterval = ival
                        self.metadata['timeInterval'] = []

        if self.nrec > 1:
            self.shape = (self.nrec,) + self.shape

        self.tshape = self.shape[:-2] + (self.tny, self.tnx)

        if self.hasits:
            self.shape = (len(its),) + self.shape

        self.ndim = len(self.shape)
        self.ndimtile = len(self.tshape)
        self.ndimfiles = self.ndim - self.ndimtile

        # dependend stuff
        self.nx = self.shape[-1]
        self.ny = self.shape[-2]
        self.ntx = self.nx // self.tnx
        self.nty = self.ny // self.tny

        if self.tiled:
            if tshape is None:
                # read other metafiles to get tile locations
                globpatt = base + '.*.*.meta'
                tmplre,tmplparts = re_parts(tmpl + '.*.*.meta')
                basere,baseparts = re_parts(base + '.*.*.meta')
                metafiles = glob(globpatt)
                metafiles.sort()
                self.ntile = len(metafiles)
                self.files = []
                # blank tiles have index -1
                self.itile = np.zeros((self.nty,self.ntx), int) - 1
                for itile,metafile in enumerate(metafiles):
                    m = basere.match(metafile)
                    g = m.groups()
                    datatmpl = interleave(g,tmplparts)[:-5] + '.data'
                    self.files.append(datatmpl)

                    meta = Metadata.read(metafile)
                    it = meta.starts[-1] // self.tnx
                    jt = meta.starts[-2] // self.tny
                    self.itile[jt,it] = itile
        else:
            self.ntile = 1
            self.files = [ tmpl + '.data' ]
            self.itile = np.array([[0]],int)

        nmissing = (self.itile < 0).sum()
        if nmissing != nblank and not quiet:
            sys.stderr.write('WARNING: {} tiles missing (expected {}).\n'.format(nmissing, nblank))

        # tile boundaries along axes
        self.i0s = range(0, self.nx, self.tnx)
        self.ies = range(self.tnx, self.nx+1, self.tnx)
        self.j0s = range(0, self.ny, self.tny)
        self.jes = range(self.tny, self.ny+1, self.tny)

        if astype is not None:
            self.dtype = np.dtype(astype)
        else:
            self.dtype = self.tdtype

        self.itemsize = self.dtype.itemsize
        self.fill = self.dtype.type(fill)
        self.dt = dt
        if self.its is not None and dt > 0:
            self.times = [ int(re.sub(r'^0*','',s))*dt for s in self.its ]

    def savetiledata(self, fname):
        if self.tiled:
            it = [ int(f[-12:-9]) for f in self.files ]
            jt = [ int(f[ -8:-5]) for f in self.files ]
        else:
            it = [ 0 ]
            jt = [ 0 ]

        np.savetxt(fname, np.array([it,jt,self.i0s,self.ies,self.j0s,self.jes]))


def unknowndimvals(unknowntmpl, sufs=['']):
    globpatt = template_replace_all(unknowntmpl, '*')
    for suf in sufs:
        print "Trying", globpatt + suf
        files = glob(globpatt + suf)
        if len(files):
            break
    else:
        raise IOError(globpatt + suf)

    vals = dict((k,set()) for k in template_fields(unknowntmpl))
    regexp,parts,keys = format2re(unknowntmpl + suf)
    for name in files:
        m = re.match(regexp,name)
        if m:
            g = m.groups()
            for k,v in zip(keys,g):
                vals[k].add(v)
        else:
            print name,'does not match',regexp

    return dict((k,sorted(list(s))) for k,s in vals.items())


def globdimvals(tmpl, valuesdict, sufs=['.001.001.meta','.meta'], fast=1):
#    # remove formats: {xx:yy} -> {xx}
#    tmpl = re.sub(r'{([^:}]*)(:[^}]*)?}', r'{\1}', tmpl)
    superfast = fast >= 2

    fieldsinorder = template_fields(tmpl)
    fields = sorted(list(set(fieldsinorder)))

    # just pick actual fields
    known = dict((k,v) for k,v in valuesdict.items() if k in fields)
    # first known value for each field
    first = dict((k,v[0]) for k,v in known.items())

    unknown = set(fields) - set(known)
    if unknown:
        if fast:
            dimvals = {}
            while True:
                tmpl = template_replace(tmpl, first)
                if 0 == len(template_fields(tmpl)):
                    break
                if superfast:
                    while True:
                        t = fmt.parse(tmpl).next()[0]
                        if re.search(r'\*.*/',t):
                            trunc = re.sub(r'^([^\*]*)\*([^/]*)/(.*)$',r'\1*\2/',t)
                            print trunc
                            dirs = glob(trunc)
                            tmpl = re.sub(r'^([^\*]*)\*([^/]*)/',dirs[0],tmpl)
                        else:
                            break
    #            s = ''
                trunc = ''
                # replace fields until hitting a '/', cut rest
                for t,n,f,c in fmt.parse(tmpl):
                    if '/' in t and trunc != '':
                        pre,post = t.split('/',1)
    #                    s = s + pre + '/'
                        trunc = trunc + pre + '/'
                        break
    #                s = s + t + '*'
                    trunc = trunc + fstring(t,n,f,c)

                if trunc.endswith('/'):
                    dvs = unknowndimvals(trunc)
                    dimvals.update(dvs)
                    first = dict((k,v[0]) for k,v in dvs.items())
                else:
                    dvs = unknowndimvals(trunc,sufs)
                    dimvals.update(dvs)
                    break
        else:
            dimvals = unknowndimvals(trunc,sufs)
    else:
        dimvals = {}

    dimvals.update(known)

    return dimvals,fields


def findvarfiles(tmpl, **kwargs):
    i = 0
    dimnames = []
    while re.search(r'{d%d[}:]' % i, tmpl):
        dimnames.append('d%d' % i)
        i += 1

    ndims = i

    i = 0
    vardimnames = []
    while re.search(r'{v%d[}:]' % i, tmpl):
        vardimnames.append('v%d' % i)
        i += 1

    nvardims = i

    dimvals = dict( (k,kwargs.get(k, [])) for k in dimnames+vardimnames )
    first = dict( (k,kwargs.get(k, ['*'])[0]) for k in dimnames+vardimnames )
    replfirst = dict( (k,kwargs.get(k, ['{'+k+'}'])[0]) for k in dimnames+vardimnames )

    base = tmpl.format(**first)

    if '*' in first.values():
        suf = '.001.001.meta'
        metafiles = glob(base + suf)
        if len(metafiles) == 0:
            suf = '.meta'
            metafiles = glob(base + suf)

        if len(metafiles) == 0:
            raise IOError(base + suf)

        regexp,parts,keys = format2re(tmpl.format(**replfirst) + suf)
        for metafile in metafiles:
            g = re.match(regexp,metafile).groups()
            for k,v in zip(keys,g):
                if v not in dimvals[k]:
                    dimvals[k].append(v)

        for k in keys:
            dimvals[k].sort()

    # have: dimnames, vardimnames, dimvals (both), ndims, nvardims

    dimvalslist = [ dimvals[d] for d in dimnames ]
    filedims = [ len(dimvals[d]) for d in dimnames ]
    inds = np.indices(filedims)

    def tmp(*ii):
        return dict((n,dimvals[n][i]) for n,i in zip(dimnames,ii))

    vtmp = np.frompyfunc(tmp, ndims, 1)
    filedicts = vtmp(*inds)

    return dimvals,dimvalslist,filedims,filedicts


def findfiles(tmpl, **kwargs):
    i = 0
    dimnames = []
    while re.search(r'{d%d[}:]' % i, tmpl):
        dimnames.append('d%d' % i)
        i += 1

    ndims = i

    dimvals = dict( (k,kwargs.get(k, [])) for k in dimnames )
    first = dict( (k,kwargs.get(k, ['*'])[0]) for k in dimnames )
    replfirst = dict( (k,kwargs.get(k, ['{'+k+'}'])[0]) for k in dimnames )

    base = tmpl.format(**first)

    if '*' in first.values():
        suf = '.001.001.meta'
        metafiles = glob(base + suf)
        if len(metafiles) == 0:
            suf = '.meta'
            metafiles = glob(base + suf)

        if len(metafiles) == 0:
            raise IOError(base + suf)

        regexp,parts,keys = format2re(tmpl.format(**replfirst) + suf)
        for metafile in metafiles:
            g = re.match(regexp,metafile).groups()
            for k,v in zip(keys,g):
                if v not in dimvals[k]:
                    dimvals[k].append(v)

        for k in keys:
            dimvals[k].sort()

    dimvalslist = [ dimvals[d] for d in dimnames ]
    filedims = [ len(dimvals[d]) for d in dimnames ]
    inds = np.indices(filedims)

    def tmp(*ii):
        return dict((n,dimvals[n][i]) for n,i in zip(dimnames,ii))

    vtmp = np.frompyfunc(tmp, ndims, 1)
    filedicts = vtmp(*inds)

    return dimvals,dimvalslist,filedims,filedicts


class VariableDataMulti(VariableData):
    def __init__(self, tmpl, astype=None, fill=0, dtype=None, shape=None, tshape=None, fast=1, ncs=None, mmap=True, **kwargs):
#        dimvals,self.dimvals,self.filedims,self.filedicts = findfiles(tmpl,**kwargs)
        if tshape is None:
            sufs = ['.001.001.meta','.meta']
        else:
            sufs = ['.001.001.data','.data','.001.001.data.gz','.data.gz']

        if type(tmpl) == type([]):
            files = tmpl
            tmpl = re.sub(r'\.001\.001\.data$','',tmpl[0])
        else:
            files = None

        self.usemmap = mmap
        dimvals,dimnames = globdimvals(tmpl,kwargs,sufs,fast=fast)
        self.dimvals = [ dimvals[d] for d in dimnames ]
        self.filedims = [ len(v) for v in self.dimvals ]
        self.filedicts = np.empty(self.filedims, object)
        inds = zip(*[x.flat for x in np.indices(self.filedims)])
        for ifile,ind in enumerate(inds):
            self.filedicts.flat[ifile] = dict((n,dimvals[n][i]) for n,i in zip(dimnames,ind))

        first = dict( (k,v[0]) for k,v in dimvals.items() )
        base = tmpl.format(**first)

        if dtype is not None and shape is not None and tshape is None:
            # global files
            self.tiled = False
            self.metadata = None
            self.shape = shape
            self.nrec = 1
            self.tdtype = np.dtype(dtype)
            self.tny,self.tnx = shape[-2:]
            tmpl = re.sub(r'\.data$','',tmpl)
        elif tshape is not None:
            # cubed sphere
            self.tiled = True
            self.metadata = None
            self.nrec = 1
            self.tdtype = np.dtype(dtype)
            self.tny,self.tnx = tshape[-2:]
            if files is not None:
                self.files = files
                self.ntile = len(files)
                ny = int(np.sqrt(self.tnx*self.tny*self.ntile/6))
                nx = 6*ny
                self.shape = tshape[:-2] + (ny,nx)
                self.ntx = nx/self.tnx
                self.nty = ny/self.tny
                # blank tiles have index -1
                self.itile = np.zeros((self.nty,self.ntx), int) - 1
                for itile in range(self.ntile):
                    _,it = divmod(itile,self.ntx/6)
                    f,jt = divmod(_,self.nty)
                    it += self.ntx/6*f
                    self.itile[jt,it] = itile
            else:
                if glob(base + '.001.001.data'):
                    suf = ''
                else:
                    suf = '.gz'
                globpatt = base + '.*.*.data' + suf
                tmplre,tmplparts = re_parts(tmpl + '.*.*.data' + suf)
                basere,baseparts = re_parts(base + '.*.*.data' + suf)
                print globpatt
                datafiles = glob(globpatt)
                print "Found",len(datafiles),'for'
                datafiles.sort()
                self.ntile = len(datafiles)
                ny = int(np.sqrt(self.tnx*self.tny*self.ntile/6))
                nx = 6*ny
                self.shape = tshape[:-2] + (ny,nx)
                self.ntx = nx/self.tnx
                self.nty = ny/self.tny
                self.files = []
                # blank tiles have index -1
                self.itile = np.zeros((self.nty,self.ntx), int) - 1
                for itile,datafile in enumerate(datafiles):
                    m = basere.match(datafile)
                    g = m.groups()
                    datatmpl = interleave(g,tmplparts)
                    # in case some files are not gzipped
                    datatmpl = re.sub(r'\.gz$','',datatmpl)
                    self.files.append(datatmpl)
                    _,it = divmod(itile,self.ntx/6)
                    f,jt = divmod(_,self.nty)
                    it += self.ntx/6*f
                    self.itile[jt,it] = itile
        else:
            metafiles = glob(base + '.001.001.meta')
            if len(metafiles) > 0:
                self.tiled = True
            else:
                metafiles = glob(base + '.meta')
                self.tiled = False

            if len(metafiles) == 0:
                raise IOError('File not found: {}.meta'.format(base))

            # get global info from first metafile
            self.metadata = Metadata.read(metafiles[0])
            self.shape = tuple(self.metadata.dims)
            self.nrec = self.metadata.nrecords
            self.tdtype = np.dtype(self.metadata.dtype)
            self.tnx = self.metadata.ends[-1] - self.metadata.starts[-1]
            self.tny = self.metadata.ends[-2] - self.metadata.starts[-2]
            self.metadata.makeglobal()

        if self.nrec > 1:
            self.shape = (self.nrec,) + self.shape

        self.tshape = self.shape[:-2] + (self.tny, self.tnx)

        self.shape = tuple(self.filedims) + self.shape

        self.ndim = len(self.shape)
        self.ndimtile = len(self.tshape)
        self.ndimfiles = self.ndim - self.ndimtile

        # dependend stuff
        self.nx = self.shape[-1]
        self.ny = self.shape[-2]
        self.ntx = self.nx // self.tnx
        self.nty = self.ny // self.tny

        if self.tiled:
            if tshape is None:
                # read other metafiles to get tile locations
                globpatt = base + '.*.*.meta'
                tmplre,tmplparts = re_parts(tmpl + '.*.*.meta')
                basere,baseparts = re_parts(base + '.*.*.meta')
                metafiles = glob(globpatt)
                metafiles.sort()
                self.ntile = len(metafiles)
                self.files = []
                # blank tiles have index -1
                self.itile = np.zeros((self.nty,self.ntx), int) - 1
                for itile,metafile in enumerate(metafiles):
                    m = basere.match(metafile)
                    g = m.groups()
                    datatmpl = interleave(g,tmplparts)[:-5] + '.data'
                    self.files.append(datatmpl)

                    meta = Metadata.read(metafile)
                    it = meta.starts[-1] // self.tnx
                    jt = meta.starts[-2] // self.tny
                    self.itile[jt,it] = itile
        else:
            self.ntile = 1
            self.files = [ tmpl + '.data' ]
            self.itile = np.array([[0]],int)

        # tile boundaries along axes
        self.i0s = range(0, self.nx, self.tnx)
        self.ies = range(self.tnx, self.nx+1, self.tnx)
        self.j0s = range(0, self.ny, self.tny)
        self.jes = range(self.tny, self.ny+1, self.tny)

        if astype is not None:
            self.dtype = np.dtype(astype)
        else:
            self.dtype = self.tdtype

        self.itemsize = self.dtype.itemsize
        self.fill = self.dtype.type(fill)


def calc(s,tnx,i0s,ies):
    """ compute tiles and slices needed for assignment

    res[tgtslice] = tile[srcslice]

    to emulate

    res = global[s]

    for tiled files
    """

    # round down
    tile0 = s.start // tnx
    # round up
    tilee = -(-s.stop // tnx)

    tiles = []
    srcslices = []
    tgtslices = []
    for tile in range(tile0,tilee):
        ii0 = max(0, -((s.start - i0s[tile]) // s.step))
        iie = -((s.start - min(s.stop,ies[tile])) // s.step)
        if iie > ii0:
            tiles.append(tile)
            myi0 = s.start + ii0*s.step - i0s[tile]
            myie = s.start + iie*s.step - i0s[tile]
            srcslices.append(slice(myi0,myie,s.step))
            tgtslices.append(slice(ii0,iie))

    return tiles, srcslices, tgtslices


class Variable(object):
    """ v = Variable.mds(tmpl, fill=0)

    tiled memory-mapped variable object

    v[...] slices without reading, but reads upon conversion to array
    v() reads

    tmpl may contain '*' as in 'res_*/ETAN/day.0000000000'
    """

    def __init__(self, data, slice, template=''):
        if not isinstance(data, VariableData):
            raise ValueError

        self.data = data
        self.slice = slice
        self.template = template

    @classmethod
    def mds(cls, tmpl, its=None, astype=None, fill=0, dt=0, debug=False, nblank=0):
        data = VariableData(tmpl, its=its, fill=fill, astype=astype, dt=dt, debug=debug, nblank=nblank)
        shape = data.shape
        mslice = MSlice(shape)
        return cls(data,mslice)

    @classmethod
    def data(cls, tmpl, dtype, shape, its=None, astype=None, fill=0, dt=0):
        data = VariableData(tmpl, its=its, fill=fill, astype=astype, dtype=dtype, shape=shape, dt=dt)
        shape = data.shape
        mslice = MSlice(shape)
        return cls(data,mslice)

    @classmethod
    def cs(cls, tmpl, dtype, tshape, its=None, astype=None, fill=0, dt=0):
        data = VariableData(tmpl, its=its, fill=fill, astype=astype, dtype=dtype, tshape=tshape, dt=dt)
        shape = data.shape
        mslice = MSlice(shape)
        return cls(data,mslice)

    @classmethod
    def mcs(cls, tmpl, dtype, tshape, astype=None, fill=0, fast=1, ncs=None, **kwargs):
        data = VariableDataMulti(tmpl, astype=astype, fill=fill, dtype=dtype, tshape=tshape, fast=fast, ncs=ncs, **kwargs)
        shape = data.shape
        mslice = MSlice(shape)
        return cls(data,mslice,tmpl)

    @classmethod
    def mmds(cls, tmpl, astype=None, fill=0, fast=1, **kwargs):
        """ tmpl = '.../*/{d0}/{d1}' without .001.001.data
            astype :: convertible to dtype
            fill   :: not used
            kwargs :: values for d0, ... (strings)
        """
        data = VariableDataMulti(tmpl, astype=astype, fill=fill, fast=fast, **kwargs)
        shape = data.shape
        mslice = MSlice(shape)
        return cls(data,mslice)

    @classmethod
    def mdata(cls, tmpl, dtype, shape, astype=None, fill=0, fast=1, **kwargs):
        data = VariableDataMulti(tmpl, astype=astype, fill=fill, dtype=dtype, shape=shape, fast=fast, **kwargs)
        shape = data.shape
        mslice = MSlice(shape)
        return cls(data,mslice,tmpl)

    def __getitem__(self,indices):
        # slice
        if not type(indices) == type(()):
            indices = (indices,)
        if hasattr(self.data,'dimvals'):
            inds = ()
            myindices = indices
            if Ellipsis in indices:
                i = indices.index(Ellipsis)
                myindices = indices[:i] + (len(self.slice.shape)-len(indices)+1)*np.s_[:,] + indices[i+1:]
            for ii,i in enumerate(myindices):
                iact = self.slice.active[ii]
                s = self.slice.s[iact]
#                if iact < len(self.data.dimvals) and i in self.data.dimvals[iact]:
                if type(i) == type(''):
                    inds = inds + (self.data.dimvals[iact][s].index(i),)
                else:
                    inds = inds + (i,)
            indices = inds
        slice = self.slice[indices]
        var = Variable(self.data, slice)
        if len(slice.active) == 0:
            return var()
        else:
            return var

    def __call__(self,*indices):
        if len(indices):
            if hasattr(self.data,'dimvals'):
                inds = ()
                for ii,i in enumerate(indices):
                    iact = self.slice.active[ii]
                    s = self.slice.s[iact]
#                    if iact < len(self.data.dimvals) and i in self.data.dimvals[iact]:
                    if iact < len(self.data.dimvals):
                        inds = inds + (self.data.dimvals[iact][s].index(i),)
                    else:
                        inds = inds + (i,)
                indices = inds
            return self[indices]

        # read data from files according to current slice
        data = self.data
        mslice = self.slice
        shape = mslice.shape

        augshape = mslice.augshape
        shfiles = augshape[:data.ndimfiles]
        shtile  = augshape[data.ndimfiles:]
        augshape = (np.prod(shfiles),) + shtile

        indfiles = mslice.s[:data.ndimfiles]
        indperp  = mslice.s[data.ndimfiles:-2]
        indy,indx = mslice.s[-2:]

        filedicts = data.filedicts[indfiles].flat
#        print indfiles, filedicts[0]

        if mslice.lens[-1] == 1:
            # quicker way for single index
            it,ti0 = divmod(indx.start, data.tnx)
            itxs = [it]
            islices = [ti0]
            iislices = [0]
        else:
            itxs,islices,iislices = calc(indx, data.tnx, data.i0s, data.ies)

        if mslice.lens[-2] == 1:
            # quicker way for single index
            jt,tj0 = divmod(indy.start, data.tny)
            jtys = [jt]
            jslices = [tj0]
            jjslices = [0]
        else:
            jtys,jslices,jjslices = calc(indy, data.tny, data.j0s, data.jes)

        if len(shape) == 0:
            # quicker for single element
            itile = data.itile[jt,it]
            if itile >= 0:
                dict = filedicts[0]
                file = data.files[itile].format(**dict)
                if debug: print file
                if not file.endswith('.gz') and os.path.exists(file):
                    mm = np.memmap(file, data.tdtype, 'r', shape=data.tshape)
                else:
                    # can handle .gz
                    mm = oj.num.myfromfile(file, data.tdtype, data.tshape)
                ind = tuple(s.start for s in indperp) + (tj0,ti0)
                return data.dtype.type(mm[ind])
            else:
                return data.fill
        else:
#            print shape,augshape
            res = np.empty(augshape, data.dtype)
            res[:] = data.fill

            for iifile,dict in enumerate(filedicts):
                for jty,jslice,jjslice in zip(jtys,jslices,jjslices):
                    for itx,islice,iislice in zip(itxs,islices,iislices):
                        itile = data.itile[jty,itx]
                        if itile >= 0:
                            file = data.files[itile].format(**dict)
                            if debug: print file
                            if data.usemmap and not file.endswith('.gz') and os.path.exists(file):
                                mm = np.memmap(file, data.tdtype, 'r', shape=data.tshape)
                            else:
                                # can handle .gz
                                mm = oj.num.myfromfile(file, data.tdtype, data.tshape)
                            ind = indperp + (jslice,islice)
                            tmp = mm[ind]
#                            print res[(iifile,Ellipsis,jjslice,iislice)].shape, tmp.shape
                            res[iifile,...,jjslice,iislice] = tmp

#            print res.shape, shape
            return res.reshape(shape)

    def __array__(self,*args):
        return self().__array__(*args)

    def __str__(self):
        return 'Variable' + str(self.data.shape) + '[' + str(self.slice) + '] ' + self.template

    def __repr__(self):
        return 'Variable' + str(self.slice.shape)

    def __getattr__(self,attr):
        return getattr(self.data,attr)

    def __dir__(self):
        res = dir(self.__class__) + self.__dict__.keys() + dir(self.data)
        res = list(set(res))
        res.sort()
        return res

    def writemeta(self,filename,**override):
        # read data from files according to current slice
        data = self.data
        mslice = self.slice
        shape = mslice.shape
        m = copy.deepcopy(self.metadata)

        indfiles = mslice.s[:data.ndimfiles]

        ndim = self.data.ndim
        ndimtile = len(m['dimList'])
        active = [ i for i in mslice.active if i >= ndim-ndimtile ]
        tileshape = shape[-len(active):]

#        tileshape = mslice.augshape[-ndimtile:]
        if self.nrec != 1:
            nrec = mslice.augshape[-ndimtile-1]
            if nrec != data.nrec:
                if 'nFlds' in m:
                    del m['nFlds']
                if 'fldList' in m and len(m['fldList']) > 1:
                    del m['fldList']
            m['nrecords'] = '%5d'%(mslice.augshape[-ndimtile-1])

        m['dimList'] = [ ['%5d'%i for i in [n,1,n]] for n in reversed(tileshape) ]
        m['nDims'] = '%3d'%len(tileshape)

        filedicts = data.filedicts[indfiles].flat
        if len(filedicts) != 1:
            print 'filedicts:', list(filedicts)
            raise AssertionError('len(filedicts) != 1')

        if self.data.hasits:
            iit = indfiles[-1]
            try:
                m['timeStepNumber'] = '%10d'%data.timeStepNumber[iit][0]
            except AttributeError:
                pass

            try:
                m['timeInterval'] = ' '.join('%19.12E'%f for f in data.timeInterval[iit][0])
            except AttributeError:
                pass

        for k,v in override.items():
            m[k] = v

        m.write(filename)

    @property
    def shape(self):
        return self.slice.shape

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def itemsize(self):
        return self.data.itemsize

    @property
    def nbytes(self):
        return self.itemsize*self.size

    @property
    def metadata(self):
        return self.data.metadata

#    def dimvals(self,d):
#        d = self.slice.active[d]
#        return self.data.dimvals[d][self.slice.s[d]]
    @property
    def dimvals(self):
        return [ self.data.dimvals[d][self.slice.s[d]] for d in self.slice.active if d < len(self.data.dimvals) ]

    @property
    def variables(self):
        return [np.array(_) for _ in self.dimvals]

    @property
    def fields(self):
        return self.dimvals[0]

    def dimnums(self,d):
        l = self.dimvals[d]
        return [ int(v) for v in l ]
#        try:
#            return [ int(re.sub(r'^0*','',v)) for v in l ]
#        except (TypeError,ValueError):
#            return int(re.sub(r'^0*','',l))

    def todict(self):
        return dict((k, self[k]) for k in self.dimvals[0])



# for multi-variable
def globvardimvals(tmpl, valuesdict,sufs=['.001.001.meta', '.meta']):
    """
    given a glob template like "base/{v0}/res_*/{v1}/{d0}_day.{d1}"
    go looking for files that match this template and find values for v0,
    ... and d0, ... and return a dictionary of dictionaries
    
        (v0,v1) -> {'d0':['a','b','c'], 'd1':[0,1,2], ...}

    values provided in valuesdict are substituted before searching for files.
    For given v0,v1,... all combinations of d0,d1,... are expected to be
    present.  v0,v1,... may take only selected combinations.
    """
    # remove formats: {xx:yy} -> {xx}
    tmpl = re.sub(r'{([^:}]*)(:[^}]*)?}', r'{\1}', tmpl)

    fields = list(set(re.findall(r'{([^}]*)}', tmpl)))
    vardims = [k for k in fields if k.startswith('v')]
    vardims.sort()
    knownvars = dict((k,v) for k,v in valuesdict.items() if k in vardims)
    knownvardims = [ k for k in vardims if k in knownvars ]
    knownvarvals = [ knownvars[k] for k in knownvardims ]
    knownvarlens = [ len(v) for v in knownvarvals ]
    unknownvardims = [ k for k in vardims if not k in knownvars ]

    fixdims = [k for k in fields if not k.startswith('v')]
    fixdims.sort()

    # just pick actual fields
    known = dict((k,v) for k,v in valuesdict.items() if k in fields)
    knowndims = dict((k,v) for k,v in known.items() if k not in vardims)
    # first known value for each field
    firstdims = dict((k,v[0]) for k,v in knowndims.items())

    if 'vars' in valuesdict:
        # list of variable value tuples
        # must be all variables; will ignore other v0=... settings
        varvals = valuesdict['vars']
    else:
        knownvarindices = np.indices(knownvarlens)
        varvals = []
        for vi in zip(*[x.flat for x in knownvarindices]):
            varval = tuple(v[i] for v,i in zip(knownvarvals,vi))
            varvals.append(varval)

    dimvals = {}

    unknown = set(fields) - set(known)
    if unknown:
        replaceknown = dict((k,'{'+k+'}') for k in fields)
        for k,v in firstdims.items():
            replaceknown[k] = v

        for knownvarval in varvals:
            vars = dict(zip(knownvardims, knownvarval))
            replaceknown.update(vars)

            unknowntmpl = tmpl.format(**replaceknown)

            globpatt = re.sub(r'{[^}]*}', '*', unknowntmpl)
            for suf in sufs:
                metafiles = glob(globpatt + suf)
                if len(metafiles):
                    break
            else:
                raise IOError(globpatt + suf)

            unknowndims = [k for k in unknown if not k.startswith('v')]
            regexp,parts,keys = format2re(unknowntmpl + suf)
            vals = {}
            for metafile in metafiles:
                g = re.match(regexp,metafile).groups()
                d = dict(zip(keys,g))
                varval = tuple(d[k] for k in unknownvardims)
                if varval not in vals:
                    vals[varval] = dict((k,set()) for k in unknowndims)
                for k,v in zip(keys,g):
                    if not k.startswith('v'):
                        vals[varval][k].add(v)

            for unknownvarvals,vs in vals.items():
                unknownvars = dict(zip(unknownvardims,unknownvarvals))
                vars.update(unknownvars)
                varval = tuple(vars[k] for k in vardims)
                dimvals[varval] = dict((k,sorted(list(s))) for k,s in vs.items())
                dimvals[varval].update(knowndims)
    else:
        dimvals = dict.fromkeys(varvals, knowndims)
            
    # res: (v0,v1) -> {'d0':['a','b','c'], 'd1':[0,1,2], ...}
    return vardims,fixdims,dimvals


class MultiVariable(object):
    def __init__(self,vars=[],vdim=()):
        self.vars = vars
        self.vdim = vdim
        
    @classmethod
    def mmds(cls,tmpl,**kwargs):
        varnames,dimnames,dimvals = globvardimvals(tmpl,kwargs)
        vars = {}
        for vs,ds in dimvals.items():
            vardict = dict((k,v) for k,v in zip(varnames,vs))
            vartmpl = template_replace(tmpl, vardict)
            vars[vs] = Variable.mmds(vartmpl,**kwargs)
            
        return cls(vars,len(vs))

    @classmethod
    def mcs(cls,tmpl,dtype,tshape,**kwargs):
        try:
            kwargs['vars'] = tshape.keys()
        except AttributeError:
            tshapes = None
        else:
            tshapes = tshape

        print tmpl
        print kwargs
        varnames,dimnames,dimvals = globvardimvals(tmpl,kwargs,['.001.001.data','.data','.001.001.data.gz','.data.gz'])
        print dimvals
        vars = {}
        for vs,ds in dimvals.items():
            vardict = dict((k,v) for k,v in zip(varnames,vs))
            vartmpl = template_replace(tmpl, vardict)
            if tshapes:
                tshape = tshapes[vs]
            vars[vs] = Variable.mcs(vartmpl,dtype,tshape,**kwargs)
            
        return cls(vars,len(vs))

    def __getitem__(self,i):
        if type(i) != type(()):
            i = (i,)
        n = len(i)
        if n < self.vdim:
            vars = dict((k[n:],v) for k,v in self.vars.items() if k[:n] == i)
            vdim = self.vdim - n
            return MultiVariable(vars,vdim)
        if len(i) > self.vdim:
            return self.vars[i[:self.vdim]][i[self.vdim:]]
        else:
            return self.vars[i]


