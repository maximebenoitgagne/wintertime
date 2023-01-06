#!/usr/bin/env python
"""gluemnc.py [<options>] -o <outfile> <files>

 -x xnames  names of 'x' dimensions, default 'X,Xp1'
 -y ynames  names of 'y' dimensions, default 'Y,Yp1'
 -t tname   name of 't' dimension, default 'T'
 -v vars    comma-separated list of variable names or glob patterns
 -c         check global attributes do not change across files
 -q         don't check correct number of tiles read
 --scipy    always use scipy's netcdf module (requires scipy 0.8.0)
            default is to try netCDF3/4 first
 --verbose

all files must have the same variables.

Example:

gluemnc.py -o ptr.nc -v 'BIO_*' mnc_*/ptr_tave.*.nc
"""
import sys
import re
import glob
import fnmatch
from getopt import gnu_getopt as getopt
try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict
import numpy as np

# bridge mismatches between netCDF3/4 and scipy APIs
def getncattrs(obj):
    try:
        return obj.ncattrs()
    except AttributeError:
        return obj._attributes

def isunlimited(dim):
    try:
        return dim.isunlimited()
    except AttributeError:
        return dim is None

def getdtype(obj):
    try:
        return obj.dtype
    except AttributeError:
        return obj.typecode()

# NB: fancy indexing is also different, which should not matter as all tiles
# and times should be contiguous, so no fancy indexing should ever be used...


tilepatt = re.compile(r'(\.t[0-9]{3}\.nc)$')
iterpatt = re.compile(r'(\.[0-9]{10})$')

# parse command-line arguments
try:
    optlist,args = getopt(sys.argv[1:], 'v:o:x:y:t:cqs', ['verbose', 'scipy'])
    opts = dict(optlist)
    assert '-o' in opts
    fnames = args
    assert len(fnames) > 0
except (ValueError, AssertionError):
    sys.exit(__doc__)

outname = opts.get('-o')
xnames = opts.get('-x', 'X,Xp1').split(',')
xnames += opts.get('-y', 'Y,Yp1').split(',')
tname = opts.get('-t', 'T')
verbose = '--verbose' in opts
usescipy = '--scipy' in opts or '-s' in opts
checkattributes = '-c' in opts
checkcount = '-q' not in opts

if usescipy:
    import scipy
    assert scipy.__version__.split('.') >= '0.8.0'.split('.')
    from scipy.io.netcdf import netcdf_file as Dataset
else:
    try:
        from netCDF3 import Dataset
        print 'Using netCDF3'
    except ImportError:
        try:
            from netCDF4 import Dataset
            print 'Using netCDF4'
        except ImportError:
            import scipy
            assert scipy.__version__.split('.') >= '0.8.0'.split('.')
            from scipy.io.netcdf import netcdf_file as Dataset
            print 'Using scipy.io.netcdf'

# API mismatches now handles above
#try:
#    Dataset.ncattrs
#except AttributeError:
#    def getncattrs(obj):
#        return obj._attributes
#    def dimisunlimited(dim):
#        return dim is None
#    def getdtype(obj):
#        return obj.typecode()
#else:
#    def getncattrs(obj):
#        return obj.ncattrs()
#    def dimisunlimited(dim):
#        return dim.isunlimited()
#    def getdtype(obj):
#        return obj.dtype

# turn into list of compiled regular expressions
varpatt = opts.get('-v', '').split(',')
#def makeregexp(patt):
#    if any(s in patt for s in '^?*+$'):
#        patt = '^{}$'.format(patt)
#    return re.compile(patt)
varpatt = [ re.compile(fnmatch.translate(patt.strip())) for patt in varpatt ]

if len(fnames) == 1 and any(s in fnames[0] for s in '*?[]'):
    fnames = glob.glob(fnames[0])
#
fnames.sort()

# Get list of tiles and iterations
its = set()
tiles = set()
for fname in fnames:
    tile = tilepatt.search(fname).group(1)
    tiles.add(tile)
    base = tilepatt.sub('', fname)
    m = iterpatt.search(base)
    if m:
        its.add(m.group(1))

its = list(its)
its.sort()
tiles = list(tiles)
tiles.sort()

sys.stderr.write('{0} tile{1}\n'.format(len(tiles), len(tiles)!=1 and 's' or ''))

# Get all file names for first iteration
if len(its) > 0:
    sys.stderr.write('{0} file{1} per tile\n'.format(len(its), len(its)>1 and 's' or ''))
    def getit(myit):
        for fname in fnames:
            base = tilepatt.sub('', fname)
            it = iterpatt.search(base).group(1)
            if it == myit:
                yield fname

    fnames0 = list(getit(its[0]))
else:
    fnames0 = fnames
    
# get dimensions, variables, ... from these
Xtile = OrderedDict()
dims = {}
varprops = {}
deleted_attrs = []
for fname in fnames0:
    m = tilepatt.search(fname)
    tile = m.group(1)
    nc = Dataset(fname, 'r')
    if dims:
        for name in getncattrs(nc):
            if name not in attrs:
                if name not in deleted_attrs:
                    sys.stderr.write('Warning: new global attribute: {0}\n'.format(name))
            elif getattr(nc,name) != attrs[name]:
                deleted_attrs.append(name)
                del attrs[name]
                if verbose:
                    sys.stderr.write('Deleting variable global attribute: {0}\n'.format(name))
    else:
        attrs = OrderedDict((k,getattr(nc,k)) for k in getncattrs(nc))
        alldims = nc.dimensions.keys()
        for name in xnames:
            if name in nc.dimensions:
                Xtile[name] = {}

        for name,dim in nc.dimensions.items():
            if name not in xnames:
                if isunlimited(dim):
                    dims[name] = None
                else:
                    dims[name] = nc.variables[name][:]

        for name,var in nc.variables.items():
            if name in alldims or any(patt.search(name) for patt in varpatt) or len(varpatt) == 0:
                varprops[name] = {}
                varprops[name]['dtype'] = getdtype(var)
                varprops[name]['dimensions'] = var.dimensions
                varprops[name]['ncattrs'] = dict((k,getattr(var,k)) for k in getncattrs(var))

    for name in Xtile:
        Xtile[name][tile] = nc.variables[name][:]

    nc.close()

def combinelists(lists):
    flat = list(set(np.concatenate(lists)))
    flat.sort()
    return flat
    
Xs = OrderedDict( (name, combinelists(Xtile[name].values())) for name in Xtile )

havetime = tname in dims
assert havetime or len(its) <= 1
if havetime:
    assert dims[tname] is None

# create global netcdf file
ncout = Dataset(outname, 'w')
for name in Xs:
    ncout.createDimension(name, len(Xs[name]))

for name,dim in dims.items():
    if dim is not None:
        dim = len(dim)
    ncout.createDimension(name, dim)

vars = {}
for name,var in varprops.items():
#    if name not in dims.keys() + [xname, yname] and any(dim not in var['dimensions'] for dim in [tname, xname, yname]):
#        sys.stderr.write('Skipping {} (lacks T, X or Y).\n'.format(name))
#        continue
    vars[name] = ncout.createVariable(name, var['dtype'], var['dimensions'])
    for attname,att in var['ncattrs'].items():
        # workaround for scipy bug with zero-length attributes
        if len(att) == 0: att = ' '
        setattr(vars[name], attname, att)
    if name in dims and dims[name] is not None:
        vars[name][:] = dims[name]

print 'Tiled dimensions:'
for name in Xs:
    vars[name][:] = Xs[name]
    print '  {0}({1})'.format(name, len(Xs[name]))

for name,att in attrs.items():
    setattr(ncout, name, att)

print 'Variables:'
indstrings = {}
for name in sorted(varprops):
    if name not in alldims:
        prop = varprops[name]
        vardims = prop['dimensions']
        indx = [ ':' for _ in vardims ]
        if tname in vardims: indx[vardims.index(tname)] = tname
        for xname in Xs:
            if xname in vardims:
                indx[vardims.index(xname)] = xname
        indx = ','.join(indx)
        indstrings[name] = indx
        print '  {0:<12s}({1})'.format(name, indx)

# keep a list of times encountered, so we know where to insert
T = []

def makecount(dims):
    sh = []
    for name in dims:
        if name in Xs:
            sh.append(len(Xs[name]))
    return np.zeros(sh, int)

if checkcount:
    count = dict((k, makecount(prop['dimensions'])) for k,prop in varprops.items() if k not in alldims)

# read all files and fill in variables
for fname in fnames:
    if verbose:
        print fname
    nc = Dataset(fname)

    if checkattributes:
        for name in getncattrs(nc):
            if name not in attrs:
                if name not in deleted_attrs:
                    sys.stderr.write('Warning: new global attribute: {0}\n'.format(name))
            elif getattr(nc,name) != attrs.get(name):
                sys.stderr.write('Warning: global attribute changed: {0}\n'.format(name))
                delattr(ncout, name)

    # set up indexing arrays/slices for inserting tile
    if havetime:
        myts = nc.variables[tname][:]
        for t in myts:
            if t not in T:
                T.append(t)

        # where to insert this tile/iteration
        it = [ T.index(t) for t in myts ]
        # if contiguous, turn into slice
        if np.all(np.diff(it)==1): it = np.s_[it[0]:it[0]+len(it)]

    # dito for x and y
    ixs = {}
    for name in Xs:
        ix = [ Xs[name].index(x) for x in nc.variables[name][:] ]
        if np.all(np.diff(ix)==1):
            ix = np.s_[ix[0]:ix[0]+len(ix)]
        ixs[name] = ix

    for name,var in vars.items():
        if name not in alldims:
            vardims = varprops[name]['dimensions']
            indx = [ np.s_[:] for _ in vardims ]
            if tname in vardims: indx[vardims.index(tname)] = it
            for xname in Xs:
                if xname in vardims:
                    indx[vardims.index(xname)] = ixs[xname]
            var[tuple(indx)] = nc.variables[name][:]

            if checkcount:
                #count[name] += 1
                indx = []
                for xname in vardims:
                    if xname in ixs:
                        indx.append(ixs[xname])
                try:
                    sh = [len(i) for i in indx]
                except TypeError:
                    count[name][indx] += 1
                else:
                    for ii in np.ndindex(*sh):
                        iii = tuple(ind[i] for ind,i in zip(indx,ii))
                        count[name][iii] += 1

    nc.close()

ncout.close()

if havetime:
    print '(T) :: ({0})'.format(len(T))

if checkcount:
    n = 1
    if havetime: n *= len(its)

    any = False
    for name,cnt in count.items():
        fullname = '{0}({1})'.format(name, indstrings[name])
        if cnt.ndim:
            mn = cnt.min()
            mx = cnt.max()
            if mx != n:
                c = (cnt>n).sum()
                sys.stderr.write(
                    'Warning: read {name:<16s} up to {mx} instead of {n} times for {c} points\n'.format(
                        c=c, mx=mx, name=fullname, n=n))
                any = True

            if mn != n:
                c = (cnt<n).sum()
                sys.stderr.write(
                    'Warning: read {name:<16s} down to {mn} instead of {n} times for {c} points\n'.format(
                        c=c, mn=mn, name=fullname, n=n))
        else:
            if cnt != n:
                sys.stderr.write(
                    'Warning: read {name:<16s} {c} instead of {n} times\n'.format(
                        c=cnt, name=fullname, n=n))
                any = True

    if any:
        sys.stderr.write('Multiple readings may occur for non-tracer points\n')

