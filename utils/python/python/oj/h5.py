import sys
import h5py
from h5py import h5i, _objects
from h5py._hl.group import Group
from h5py._hl.files import make_fapl, make_fid

class File(h5py.File):

    def __init__(self, name, mode=None, driver=None,
                 libver=None, userblock_size=None,
                 align=None, cache=None, **kwds):
        """Create a new file object.

        See the h5py user guide for a detailed explanation of the options.

        name       Name of the file on disk. Note: for files created with the 'core'
                   driver, HDF5 still requires this be non-empty.
        mode       one of r, r+, w, w-, a (read, r/w, truncate, create, r/w/create) [a]
        driver     Name of the driver to use. Legal values are None (default,
                   recommended), 'core', 'sec2', 'stdio', 'mpio'.
        libver     Library version bounds. Currently only the strings 'earliest'
                   and 'latest' are defined.
        userblock  Desired size of user block. Only allowed when creating a new
                   file (mode w or w-).
        align      (threshold, alignment) chunk alignment in bytes [(1L, 1L)]
        cache      (*, nchunk, rawsize, preemptthresh) [(0, 521, 1048576, 0.75)]

        Additional keywords passed on to the selected file driver.
        """
        if isinstance(name, _objects.ObjectID):
            fid = h5i.get_file_id(name)
        else:
            try:
                # If the byte string doesn't match the default
                # encoding, just pass it on as-is. Note Unicode
                # objects can always be encoded.
                name = name.encode(sys.getfilesystemencoding())
            except (UnicodeError, LookupError):
                pass

            if driver is None and kwds:
                raise UserWarning('myh5.File: extra argument: ' + ' '.join(kwds.keys()))

            fapl = make_fapl(driver, libver, **kwds)
            if align is not None:
                fapl.set_alignment(*align)
            if cache is not None:
                fapl.set_cache(*cache)
            fid = make_fid(name, mode, userblock_size, fapl)

        Group.__init__(self, fid)

