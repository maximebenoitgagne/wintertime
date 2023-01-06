__all__ = ['netcdf_file', 'netcdf_mmap']

from operator import mul
from collections import OrderedDict
from mmap import mmap
from numpy.compat import asbytes
from numpy import ndarray, dtype, array
from scipy.io.netcdf import netcdf_file, netcdf_variable

ABSENT       = asbytes('\x00\x00\x00\x00\x00\x00\x00\x00')
ZERO         = asbytes('\x00\x00\x00\x00')
NC_BYTE      = asbytes('\x00\x00\x00\x01')
NC_CHAR      = asbytes('\x00\x00\x00\x02')
NC_SHORT     = asbytes('\x00\x00\x00\x03')
NC_INT       = asbytes('\x00\x00\x00\x04')
NC_FLOAT     = asbytes('\x00\x00\x00\x05')
NC_DOUBLE    = asbytes('\x00\x00\x00\x06')
NC_DIMENSION = asbytes('\x00\x00\x00\n')
NC_VARIABLE  = asbytes('\x00\x00\x00\x0b')
NC_ATTRIBUTE = asbytes('\x00\x00\x00\x0c')

TYPEMAP = { NC_BYTE:   ('b', 1),
            NC_CHAR:   ('c', 1),
            NC_SHORT:  ('h', 2),
            NC_INT:    ('i', 4),
            NC_FLOAT:  ('f', 4),
            NC_DOUBLE: ('d', 8) }

REVERSE = { ('b', 1): NC_BYTE,
            ('B', 1): NC_CHAR,
            ('c', 1): NC_CHAR,
            ('h', 2): NC_SHORT,
            ('i', 4): NC_INT,
            ('f', 4): NC_FLOAT,
            ('d', 8): NC_DOUBLE,

            # these come from asarray(1).dtype.char and asarray('foo').dtype.char,
            # used when getting the types from generic attributes.
            ('l', 4): NC_INT,
            ('S', 1): NC_CHAR }

class unmapped_array(object):
    def __init__(self, shape, dtype_):
        self.shape = shape
        self.dtype = dtype(dtype_)

    @property
    def itemsize(self):
        return self.dtype.itemsize

    @property
    def size(self):
        return reduce(mul, self.shape, 1)

    @property
    def nbytes(self):
        return self.size * self.itemsize

    def __len__(self):
        return self.shape[0]


class netcdf_mmap(netcdf_file):
    """
    A version of netcdf_file that creates the netcdf file on disk before values
    are assigned to variables.  Variables are created as mmaps, so little RAM
    is used on assigning.

    Call write_metadata after defining all attributes, dimensions and variables,
    but before assigning any values to variables.
    """
    def __init__(self, filename, mode='r', mmap=True, version=1):
        if mmap and mode == 'w':
            fmode = 'w+'
        else:
            fmode = mode
        fp = open(filename, fmode)
        netcdf_file.__init__(self, fp, mode, mmap, version)
        self.__dict__['_variables'] = OrderedDict()
        self.__dict__['_mapped'] = False
        self.__dict__['_rec_array'] = None

    def createVariable(self, name, type, dimensions):
        """
        Create an empty variable for the `netcdf_file` object, specifying its data
        type and the dimensions it uses.

        Parameters
        ----------
        name : str
            Name of the new variable.
        type : dtype or str
            Data type of the variable.
        dimensions : sequence of str
            List of the dimension names used by the variable, in the desired order.

        Returns
        -------
        variable : netcdf_variable
            The newly created ``netcdf_variable`` object.
            This object has also been added to the `netcdf_file` object as well.

        See Also
        --------
        createDimension

        Notes
        -----
        Any dimensions to be used by the variable should already exist in the
        NetCDF data structure or should be created by `createDimension` prior to
        creating the NetCDF variable.

        """
        shape = tuple([self.dimensions[dim] for dim in dimensions])
        shape_ = tuple([dim or 0 for dim in shape])  # replace None with 0 for numpy

        if isinstance(type, basestring): type = dtype(type)
        typecode, size = type.char, type.itemsize
        dtype_ = '>%s' % typecode
        if size > 1: dtype_ += str(size)

        data = unmapped_array(shape_, dtype_)
        self.variables[name] = netcdf_variable(data, typecode, shape, dimensions)
        return self.variables[name]

    def flush(self):
        '''Flush and close mmaps'''
        if getattr(self, 'mode', None) is 'w':
            if self.use_mmap:
                if not self._mapped:
                    self._map()
                for var in self.variables.values():
                    if var.isrec:
                        var.data = None
                    else:
                        var.data.base.close()
                if self._rec_array is not None:
                    self._rec_array.base.close()
            else:
                self._write()
    sync = flush

    def write_metadata(self):
        '''This needs to be called before assigning any data to variables!'''
        self._map()

    def _map(self):
        self.fp.seek(0)
        self.fp.write(asbytes('CDF'))
        self.fp.write(array(self.version_byte, '>b').tostring())

        # Write headers and data.
        self._write_numrecs()
        self._write_dim_array()
        self._write_gatt_array()
        self._map_var_array()
        self._mapped = True

    def _map_var_array(self):
        if self.variables:
            self.fp.write(NC_VARIABLE)
            self._pack_int(len(self.variables))

            # Sort variables non-recs first, then recs.
            # keep order from variable creation otherwise
            nonrec_vars = [ k for k,v in self.variables.items() if not v.isrec ]
            rec_vars = [ k for k,v in self.variables.items() if v.isrec ]

            # Set the metadata for all variables.
            for name in nonrec_vars + rec_vars:
                self._map_var_metadata(name)
            # Now that we have the metadata, we know the vsize of
            # each record variable, so we can calculate recsize.
            nonrecsize = sum([
                    var._vsize for var in self.variables.values()
                    if not var.isrec])
            self.__dict__['_recsize'] = sum([
                    var._vsize for var in self.variables.values()
                    if var.isrec])

            # fill file
            pos0 = self.fp.tell()
            end = pos0 + nonrecsize + self._recs*self._recsize
            self.fp.seek(end-1)
            self.fp.write('\x00')
            self.fp.flush()
            self.fp.seek(pos0)

            # Create mmaps for all variables.
            for name in nonrec_vars:
                self._map_var_data(name)
            if rec_vars:
                self._map_recvar_data(rec_vars)
        else:
            self.fp.write(ABSENT)

    def _map_var_metadata(self, name):
        var = self.variables[name]

        self._pack_string(name)
        self._pack_int(len(var.dimensions))
        for dimname in var.dimensions:
            dimid = self._dims.index(dimname)
            self._pack_int(dimid)

        self._write_att_array(var._attributes)

        nc_type = REVERSE[var.typecode(), var.data.itemsize]
        self.fp.write(asbytes(nc_type))

        if not var.isrec:
            vsize = var.data.size * var.data.itemsize
            vsize += -vsize % 4
        else:  # record variable
            if 1:  #var.data.shape[0]:
                size = reduce(mul, var.data.shape[1:], 1)
                vsize = size * var.data.itemsize
            else:
                vsize = 0
            rec_vars = len([var for var in self.variables.values()
                    if var.isrec])
            if rec_vars > 1:
                vsize += -vsize % 4
        self.variables[name].__dict__['_vsize'] = vsize
        self._pack_int(vsize)

        # Pack a bogus begin, and set the real value later.
        self.variables[name].__dict__['_begin'] = self.fp.tell()
        self._pack_begin(0)

    def _map_var_data(self, name):
        var = self.variables[name]

        # Set begin in file header.
        the_beguine = self.fp.tell()
        self.fp.seek(var._begin)
        self._pack_begin(the_beguine)
        # and seek to beginning of next var
        self.fp.seek(the_beguine + var._vsize)

        # Create mmaps
        mm = mmap(self.fp.fileno(), the_beguine + var.data.nbytes)
        var.data = ndarray.__new__(ndarray, var.shape, dtype=var.data.dtype,
                                   buffer=mm, offset=the_beguine, order=0)

    def _map_recvar_data(self, rec_vars):
        dtypes = {'names': [], 'formats': []}
        for ivar,name in enumerate(rec_vars):
            var = self.variables[name]
            dtypes['names'].append(name)
            dtype_ = '>%s' % var.typecode()
            dtypes['formats'].append(str(var.data.shape[1:]) + dtype_)
            padding = -var.data.nbytes % 4
            if padding:
                dtypes['names'].append('_padding_%d' % ivar)
                dtypes['formats'].append('(%d,)>b' % padding)

        # Remove padding when only one record variable.
        if len(rec_vars) == 1:
            dtypes['names'] = dtypes['names'][:1]
            dtypes['formats'] = dtypes['formats'][:1]

        # Create mmap
        pos0 = pos = self.fp.tell()
        mm = mmap(self.fp.fileno(), pos0+self._recs*self._recsize)
        self._rec_array = ndarray.__new__(ndarray, (self._recs,), dtype=dtypes,
                                          buffer=mm, offset=pos0, order=0)

        for name in rec_vars:
            var = self.variables[name]
            self.variables[name].__dict__['data'] = self._rec_array[name]

            # Set begin in file header.
            self.fp.seek(var._begin)
            self._pack_begin(pos)
            pos += self._recsize

        # eof
        self.fp.seek(pos)

