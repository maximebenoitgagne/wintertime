from warnings import warn
import operator
import string
import re
import os
import glob
import numpy as np
from scipy.io import mmread
from scipy.sparse import coo_matrix
import oj.num

__all__ = ['SparseMap', 'savemap']

def savemap(basename, m, shape, formats=['i4','i4','f8']):
    coo = m.tocoo()
    a = np.rec.fromarrays([coo.row, coo.col, coo.data], formats=formats, names=['i','j','w'])
    fmts = ''.join(formats)
    shapes = 'x'.join(str(i) for i in shape[::-1])
    a.tofile('{}.{}_{}x{}.map'.format(basename, fmts, shapes, m.shape[1]))

namepatt = re.compile(r'.*[_\.]([0-9x]+)\.mtx$')
binpatt = re.compile(r'.*[_\.]([0-9x]+)(?:\.[^\.]*\.bin)?$')
mappatt = re.compile(r'.*\.([a-zA-Z0-9]+)_([0-9x]+)\.map$')
typepatt = re.compile(r'([a-zA-Z])')
mapglob = '.[a-zA-Z][0-9][a-zA-Z][0-9][a-zA-Z][0-9]*_[0-9]*[0-9].map'

class SparseMap:
    def __init__(self, mat, shape):
        self.shape = tuple(shape)
        self.mat = mat

    def __repr__(self):
        return '<SparseMap with destination shape ({}), matrix\n  {}>'.format(', '.join([str(d) for d in self.shape]), repr(self.mat))

    def __call__(self, x, axis=None):
        '''Apply SparseMap to as many slots as necessary, starting from axis.
        Resulting new axis will be on the left and reshaped to self.shape.
        '''
        x = np.asanyarray(x)
        if x.size == self.mat.shape[1]:
            return (self.mat*x.flat).reshape(self.shape)
        elif x.ndim == 2 and x.shape[0] == self.mat.shape[1] and axis in [0, None]:
            return (self.mat*x).reshape(self.shape + (x.shape[1],))
        elif axis is None:
            shape = list(x.shape)
            n = 1
            while n < self.mat.shape[1] and shape:
                n *= shape.pop()
            res = np.empty(tuple(shape) + self.mat.shape[:1], x.dtype)
            for idx in np.ndindex(*shape):
                res[idx] = self.mat*x[idx].ravel()
            return res.reshape(tuple(shape) + self.shape)
        else:
            if axis < 0: axis += x.ndim
            shape = list(x.shape)
            n = 1
            while n < self.mat.shape[1]:
                n *= shape.pop(axis)
            n = np.prod(shape)
            n1 = axis
            n2 = x.ndim - len(shape) + n1
            ii = range(x.ndim)
            x1 = x.transpose(ii[n1:n2] + ii[:n1] + ii[n2:]).reshape(self.mat.shape[1], n)
            res = self.mat*x1
            return res.reshape(self.shape + tuple(shape))

    @classmethod
    def frommm(cls, fname):
        m = namepatt.match(fname)
        shape = map(int, m.group(1).split('x'))[::-1]
        mat = mmread(fname).tocsr()
        return cls(mat, shape)

    @classmethod
    def frombin(cls, fname):
        m = binpatt.match(fname)
        shape = map(int, m.group(1).split('x'))[::-1]
        nrow = reduce(operator.mul, shape[1:])
        ncol = shape[0]
        ijw = oj.num.loadbin(fname)
        mat = coo_matrix((ijw['w'], (ijw['i'], ijw['j'])), (nrow, ncol)).tocsr()
        return cls(mat, shape[1:])

    @classmethod
    def fromscrip(cls, fname, shape):
        from scipy.io.netcdf import netcdf_file
        nc = netcdf_file(fname)
        nrow = nc.dimensions['dst_grid_size']
        ncol = nc.dimensions['src_grid_size']
        w = nc.variables['remap_matrix'][:,0]
        i = nc.variables['dst_address'][:] - 1
        j = nc.variables['src_address'][:] - 1
        mat = coo_matrix((w, (i, j)), (nrow, ncol)).tocsr()
        return cls(mat, shape)

    @classmethod
    def frommap(cls, fname):
        m = mappatt.match(fname)
        if m is None:
            l = glob.glob(fname + mapglob)
            if len(l) > 1:
                warn('More than 1 match:\n' + '\n'.join(l))
            elif len(l) == 0:
                raise IOError('SparseMap: ' + fname)
            fname = l[0]
            m = mappatt.match(fname)
        tp,dims = m.groups()
        l = typepatt.split(tp)[1:]
        formats = ['{}{}{}'.format(c in string.uppercase and '>' or '', c, n)
                   for i in range(0, len(l), 2) for c,n in [l[i:i+2]]]
        tp = dict(names='ijwpabcdefgh'[:len(formats)], formats=formats)
        shape = map(int, dims.split('x'))[::-1]
        nrow = reduce(operator.mul, shape[1:])
        ncol = shape[0]
        ijw = np.fromfile(fname, tp).view(np.recarray)
        mat = coo_matrix((ijw['w'], (ijw['i'], ijw['j'])), (nrow, ncol)).tocsr()
        obj = cls(mat, shape[1:])
        obj.ra = ijw
        return obj

    def tomap(self, basename, formats=['i4','i4','f8']):
        savemap(basename, self.mat, self.shape, formats)

fromscrip = SparseMap.fromscrip
frommap   = SparseMap.frommap
