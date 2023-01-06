#!/usr/bin/env python
from __future__ import print_function
import sys,re
import numpy as np
from glob import glob

debug = True

def ravel_index(ind, dims):
    skip = 0
    for i,d in zip(ind,dims):
        skip *= d
        skip += i

    return skip


meta_type_dict = { np.dtype('>f4') : 'float32',
                   np.dtype('>f8') : 'float64',
                 }
mds_type_dict = { 'float32' : '>f4',
                  'float64' : '>f8',
                }


class Metadata(object):
    def __init__(self, metadata={}, comments={}, variables=[]):
        self.metadata = metadata
        self.comments = comments
        self.variables = variables
    
    @classmethod
    def read(cls, fname):
        meta = {}
        comments = {}
        variables = []
        with open(fname) as fid:
            group = ''
            for line in fid:
                line = line.rstrip()
                iscomment = re.match(r' */\*.*\*/ *$', line) is not None
                if iscomment:
                    # strip comment characters
                    line = re.sub(r'^ */\*(.*)\*/', r'\1', line)

                if group != '':
                    # we are inside a multi-line variable
                    if re.match(r' [\]}]', line):
                        meta[group] = groupval
                        group = ''
                    else:
                        if grouptype == '[':
                            vals = re.split(',', line[1:].rstrip(' ,'))
                            groupval.append(vals)
                        elif grouptype == '{':
                            v = map(lambda s:s, re.split("'  *'", line.strip(' ').strip("',")))
                            groupval += v
                        else:
                            sys.stderr.write('Unknown group delimiter: %s\n' % grouptype)
                            raise ValueError
                else:
                    # we are looking for a new variable
                    m = re.match(r" ([a-zA-Z]*) = ([\[{])( [^\]]*)?([\]}];)?", line)
                    if m:
                        var,delim,val,delimr = m.groups()
                        if delimr is not None:
                            # strip one space each end
                            val = val[1:-1]
                            if iscomment:
                                comments[var] = val
                            else:
                                meta[var] = val
                        else:
                            group = var
                            grouptype = delim
                            groupval = []

                        variables.append(var)
                    else:
                        sys.stderr.write('Syntax error in meta file:\n'+line)
                        raise ValueError

        return cls(meta, comments, variables)

    def write(self, fname):
        meta = self.metadata
        comments = self.comments
        variables = self.variables
        nmax = 20
        with open(fname,"w") as f:
            for var in variables:
                if var in comments:
                    f.write(" /* %s = [ %s ];*/\n" % (var,comments[var]))
                else:
                    if var in [ 'dimList' ]:
                        f.write(" %s = [\n" % var)
#                        for r in meta[var]:
#                            f.write(" " + ",".join(r) + ",\n")
                        f.write(',\n'.join(" " + ",".join(r) for r in meta[var]) + '\n')

                        f.write(" ];\n")
                    elif var in [ 'fldList' ]:
                        f.write(" %s = {\n" % var)
                        nrow = (len(meta[var]) + nmax-1)//nmax
                        for i in range(nrow):
                            r = meta[var][i*nmax:(i+1)*nmax]
                            r = [ '%-8s'%s for s in r ]
                            f.write(" '" + "' '".join(r) + "'\n")

                        f.write(" };\n")
                    else:
                        f.write(" %s = [ %s ];\n" % (var,meta[var]))
        
    def __getitem__(self,i):
        return self.metadata[i]

    def __setitem__(self,i,v):
        self.metadata[i] = v
        if i not in self.variables:
            self.variables.append(i)

    def __delitem__(self,i):
        del self.metadata[i]
        self.variables.remove(i)

    def __contains__(self,i):
        return i in self.variables

    def get(self,k,d=None):
        return self.metadata.get(k,d)

    @property
    def dtype(self):
        return mds_type_dict[self['dataprec'].strip("'")]

    @dtype.setter
    def dtype(self,tp):
        tp = meta_type_dict[tp]
        self['dataprec'] = "'" + tp + "'"
        
    @property
    def dims(self):
        return [ int(r[0]) for r in self['dimList'][::-1] ]

    @property
    def starts(self):
        return [ int(r[1])-1 for r in self['dimList'][::-1] ]

    @property
    def ends(self):
        return [ int(r[2]) for r in self['dimList'][::-1] ]

    @property
    def nrecords(self):
        return int(self['nrecords'])

    def makeglobal(self):
        self['dimList'] = [ [r[0],'    1',r[0]] for r in self['dimList'] ]


class Tiledata(object):
    def __new__(cls, itiles, jtiles, i0s, ies, j0s, jes):
        obj = object.__new__(cls)
        obj.itiles = itiles
        obj.jtiles = jtiles
        obj.i0s = i0s
        obj.ies = ies
        obj.j0s = j0s
        obj.jes = jes
        return obj

    @classmethod
    def fromfile(cls, baseglob, ind=[], fill=0):
        """ a = fromfile(baseglob, ind, fill)

    reads baseglob.data using baseglob.meta to find shape.
    Fill missing tiles with <fill>.

    fromfile(b, ind) == fromfile(b)[ind+np.s_[...,]]  but only reads necessary bits
    """
        metafiles = glob(baseglob + '.meta')
        if len(metafiles) == 0:
            sys.stderr.write('MdsArray.fromfile: file not found: ' + baseglob + '.meta\n')
            raise IOError

    #    dims,i1s,i2s,dtype,nrec,flds = readmeta(metafiles[0])
        metadata = Metadata.read(metafiles[0])
        metadata.makeglobal()
        dims = metadata.dims
        nrec = metadata.nrecords
        if nrec > 1:
            dims = [nrec] + dims

        dtype = metadata.dtype

        itiles = []
        jtiles = []
        i0s = []
        ies = []
        j0s = []
        jes = []
        for metafile in metafiles:
            itiles.append( int(metafile[-12:-9]) )
            jtiles.append( int(metafile[ -8:-5]) )

            meta = Metadata.read(metafile)
            i0s.append( meta.starts[-1] )
            j0s.append( meta.starts[-2] )
            ies.append( meta.ends[-1] )
            jes.append( meta.ends[-2] )

        return cls.__new__(cls, itiles, jtiles, i0s, ies, j0s, jes)

    def write(self, fname):
        np.savetxt(fname, np.array([self.itiles,self.jtiles,self.i0s,self.ies,self.j0s,self.jes]).T,
                   '%d')


class MdsArray(np.ndarray):
    def __new__(cls, input_array, metadata=None):
        obj = np.asarray(input_array).view(cls)
        obj.metadata = metadata
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.metadata = getattr(obj, 'metadata', None)

    @classmethod
    def fromfile(cls, baseglob, ind=[], fill=0):
        """ a = fromfile(baseglob, ind, fill)

    reads baseglob.data using baseglob.meta to find shape.
    Fill missing tiles with <fill>.

    fromfile(b, ind) == fromfile(b)[ind+np.s_[...,]]  but only reads necessary bits
    """
        metafiles = glob(baseglob + '.meta')
        if len(metafiles) == 0:
            sys.stderr.write('MdsArray.fromfile: file not found: ' + baseglob + '.meta\n')
            raise IOError

    #    dims,i1s,i2s,dtype,nrec,flds = readmeta(metafiles[0])
        metadata = Metadata.read(metafiles[0])
        metadata.makeglobal()
        dims = metadata.dims
        nrec = metadata.nrecords
        if nrec > 1:
            dims = [nrec] + dims

        dtype = metadata.dtype

        data = np.zeros(dims[len(ind):],dtype)
        if fill != 0:
            data[:] = fill

        for metafile in metafiles:
            datafile = re.sub(r'\.meta$', '.data', metafile)
            #dims,i1s,i2s,dtype,nrec,flds = readmeta(metafile)
            meta = Metadata.read(metafile)
            i0s  = meta.starts
            ies  = meta.ends

            slc = [ np.s_[i0:ie] for i0,ie in zip(i0s,ies) ]
            tdims = [ ie-i0 for i0,ie in zip(i0s,ies) ]
            if nrec > 1:
                slc = [np.s_[:nrec]] + slc
                tdims = [nrec] + tdims

            if len(ind) > 0:
                count = np.prod(tdims[len(ind):])
                skip = ravel_index(ind, tdims[:len(ind)])*count
                size = np.dtype(dtype).itemsize
                if debug: print(datafile, dtype, tdims[len(ind):], skip/count)
                with open(datafile) as fid:
                    fid.seek(skip*size)
                    data[slc[len(ind):]] = np.fromfile(fid, dtype, count=count).reshape(tdims[len(ind):])
            else:
                if debug: print(datafile, dtype, tdims)
                data[slc] = np.fromfile(datafile, dtype).reshape(tdims)

        return cls.__new__(cls, data, metadata)

    def writemds(self, fbase, dtype=None):
        if dtype is not None:
            self = self.astype(dtype)

        self.tofile(fbase + '.data')

        # make sure type is current
        meta = self.metadata
        meta.dtype = self.dtype
        meta.write(fbase + '.meta')


if __name__ == '__main__':
    from getopt import gnu_getopt as getopt
    me = sys.argv[0]
    opts,args = getopt(sys.argv[1:],'',['writetiling='])
    opts = dict(opts)
    tiledata = opts.get('--writetiling','')
    if tiledata:
        inname, = args
        Tiledata.fromfile(inname).write(tiledata)
    else:
        try:
            inname,outname = args
        except ValueError:
            sys.exit(
    """Usage: """ + me + """ inglob outbase
       """ + me + """ --writetiling=tilingfile inglob
            
    inglob may contain '*' which must be escaped.
    It may not contain the final .data
    Example:

        """ + me + """ 'res_*/THETA.0000000000.*' THETA.0000000000

    Requires the whole file to fit in memory!"""
            )

        a = MdsArray.fromfile(inname)
    #    print(a.metadata.metadata)
    #    print(a.metadata.comments)
        a.writemds(outname)

