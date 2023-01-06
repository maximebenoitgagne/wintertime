#!/usr/bin/env python2

"""Functions to read and write NetCDF4 files.
"""

# author: Maxime Benoit-Gagne - Takuvik - Canada.
# date of creation: November 12, 2015.
#
# Python from Anaconda.
# maximebenoit-gagne$ python
# Python 2.7.15 |Anaconda custom (x86_64)| (default, May  1 2018, 18:37:05) 
# [GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)] on darwin
# Type "help", "copyright", "credits" or "license" for more information.
#
# I installed netCDF4 with
# conda install netcdf4

########### Importing modules ###########

import netCDF4
import numpy as np
import sys

########### functions ###########

"""
Read a variable from a NetCDF4 file.

Args:
    ncfile(str):
        The netCDF4 file.
    variable(str):
        The variable name.

Returns:
    numpy.array:
    Array with the same shape as in the NetCDF4 file.

Raises:
    IOError: If ncfile is not found or doesn't contain a variable named
    variable.

"""
def read_netcdf(ncfile, variable):
    try:
        # open the netCDF file for reading.
        fh = netCDF4.Dataset(ncfile,'r')
    except:
        raise IOError("File not found: {}.".format(ncfile))
    try:
        # read the data in variable named v
        v = fh.variables[variable][:]
    except:
        raise IOError("Variable {} not found in {}.".format(variable, ncfile))
    fh.close()
    return v
