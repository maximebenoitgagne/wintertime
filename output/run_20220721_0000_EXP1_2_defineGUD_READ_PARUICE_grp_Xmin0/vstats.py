#!/usr/bin/env python2

"""calculate statistics from a water column
"""

# author: Maxime Benoit-Gagne - ULaval - Canada.
# date of creation: December 16, 2019.
#
# Python from Anaconda$ python
# Python 3.7.3 | packaged by conda-forge | (default, Jul  1 2019, 14:38:56) 
# [Clang 4.0.1 (tags/RELEASE_401/final)] :: Anaconda, Inc. on darwin
# Type "help", "copyright", "credits" or "license" for more information.

########### Importing modules ###########

import numpy as np

########### function ###########

"""
find index of element nearest to value

Args:
    array(array-like):
        Array of 1 dimension.
        The first dimension is the indices.
        The values are the values.
    value(float):
        Target value.

Returns:
    idx(integer):
        index of element nearest to value
"""
# see
# https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
# answer of unutbu
def find_idx_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

"""
surface mean

Args:
    array2d_idepth_iT_tracer(numpy.array):
        Array of 2 dimensions.
        The first dimension is the indices of the depths.
        The second dimension is the indices of the time steps.
        The values are the tracer.
    array1d_idepth_delR(numpy.array):
        Array of 1 dimension.
        The first dimension is the indices of the depths.
        The values are the the r cell face separations, meaning the thickness 
        of each depth layer (in m).
        it corresponds to delR on
        https://mitgcm.readthedocs.io/en/latest/getting_started/getting_started.html#grid
    depth_end(integer):
        depth to which calculate the surface mean

Returns:
    array1d_iT_tracer(numpy.array):
    Array of 1 dimension.
    The first dimension is the indices of the time steps.
    The values are the surface means.
"""

def mean(array2d_idepth_iT_tracer,
         array1d_idepth_delR,
         depth_end):
    nT=array2d_idepth_iT_tracer.shape[1]
    array1d_iT_tracer=np.zeros(nT)
    ndepth=array1d_idepth_delR.size
    RF2=np.zeros(ndepth+1)
    for idepth in range(0,ndepth):
        RF2[idepth+1]=-np.sum(array1d_idepth_delR[0:idepth+1])
    idepth_start=0
    idepth_end=np.argwhere(RF2<depth_end)[0][0]
    array1d_idepth_weight=(array1d_idepth_delR[idepth_start:idepth_end]).copy()
    array1d_idepth_weight[-1]=RF2[idepth_end-1]-(depth_end)
    for iT in range(0,nT):
        array1d_idepth_tracer \
            =array2d_idepth_iT_tracer[idepth_start:idepth_end,iT]
        mean=np.average(array1d_idepth_tracer,weights=array1d_idepth_weight)
        array1d_iT_tracer[iT]=mean
    return array1d_iT_tracer

########### function ###########

"""
vertically integrate one tracer from surface to depth_end

Args:
    array2d_idepth_iT_tracer(numpy.array):
        Array of 2 dimensions.
        The first dimension is the indices of the depths.
        The second dimension is the indices of the time steps.
        The values are the tracer.
    array1d_idepth_delR(numpy.array):
        Array of 1 dimension.
        The first dimension is the indices of the depths.
        The values are the the r cell face separations, meaning the thickness 
        of each depth layer (in m).
        it corresponds to delR on
        https://mitgcm.readthedocs.io/en/latest/getting_started/getting_started.html#grid
        it corresponds to drF in grid.nc
    depth_end(float):
        depth to which integration is made.

Returns:
    array1d_iT_vint(numpy.array):
    Array of 1 dimension.
    The first dimension is the indices of the time steps.
    The values are the tracer vertically integrated from surface to depth_end.
"""
def vint(array2d_idepth_iT_tracer,
         array1d_idepth_delR,
         depth_end):
    assert depth_end<0, \
        "depth_end(%r) is positive or nul" % depth_end
    array1d_idepth_delRcopy=array1d_idepth_delR.copy()
    RF=np.zeros(array1d_idepth_delRcopy.size+1)
    for iRF in range(1,RF.size):
        RF[iRF]=-np.sum(array1d_idepth_delRcopy[0:iRF])
    nT=array2d_idepth_iT_tracer.shape[1]
    array1d_iT_vint=np.zeros(nT)
    for iT in range(0,nT):
        nlayers=(RF[RF>depth_end]).size
        tracertarget=array2d_idepth_iT_tracer[0:nlayers,iT]
        drFtarget=array1d_idepth_delRcopy[0:nlayers]
        drFtarget[-1]=RF[nlayers-1]-depth_end
        vint=np.dot(tracertarget,drFtarget)
        array1d_iT_vint[iT]=vint
    return array1d_iT_vint

########### function ###########

"""
vertically integrate one tracer using depth indices

Args:
    array2d_idepth_iT_tracer(numpy.array):
        Array of 2 dimensions.
        The first dimension is the indices of the depths.
        The second dimension is the indices of the time steps.
        The values are the tracer.
    array1d_idepth_delR(numpy.array):
        Array of 1 dimension.
        The first dimension is the indices of the depths.
        The values are the the r cell face separations, meaning the thickness 
        of each depth layer (in m).
        it corresponds to delR on
        https://mitgcm.readthedocs.io/en/latest/getting_started/getting_started.html#grid
        it corresponds to drF in grid.nc
    idepth_start(integer):
        Index of the first depth from which a vertical integration will be done.
        0 is the index of the upper layer.
    idepth_end(integer):
        Index of the last depth from which a vertical integration will be done
        (exclusive).
    dropna(boolean):
        replace NaNs with zeros if and only if dropna is True before 
        vertically integrate.
        The vertical integration for a specific iT will still be a NaN if there
        is only 1 value to vertically integrate from.

Returns:
    array1d_iT_tracer(numpy.array):
    Array of 1 dimension.
    The first dimension is the indices of the time steps.
    The values are the tracer vertically integrated from idepth_start 
    (inclusive) to idepth_end (exclusive).
"""
def vintegrate(array2d_idepth_iT_tracer,
               array1d_idepth_delR,
               idepth_start,
               idepth_end,
               dropna=False):
    ndepth=array2d_idepth_iT_tracer.shape[0]
    assert idepth_start>=0, \
        "idepth_start is lesser than 0: %r" % idepth_start
    assert idepth_start<idepth_end, \
        "idepth_start(%r) is equal or greater than idepth_end(%r)" \
        % (idepth_start,idepth_end)
    assert idepth_end<=ndepth, \
        "idepth_end (%r) is greater than ndepth(%r)" % (idepth_end, ndepth)
    if dropna:
        array2d_idepth_iT_tracercopy=array2d_idepth_iT_tracer.copy()
        array2d_idepth_iT_tracercopy[np.isnan(array2d_idepth_iT_tracercopy)]=0
        array2d_idepth_iT_tracer=array2d_idepth_iT_tracercopy
    nT=array2d_idepth_iT_tracer.shape[1]
    array1d_iT_tracer=np.zeros(nT)
    for iT in range(0,nT):
        if not dropna or np.count_nonzero(array2d_idepth_iT_tracer[:,iT])>=2:
            tracer=0; # mmol C m^-2
            for idepth in range(idepth_start,idepth_end):
                tracerm3=array2d_idepth_iT_tracer[idepth,iT] # mmol C m^-3
                delR=array1d_idepth_delR[idepth]
                tracertempo=tracerm3*delR # mmol C m^-2
                tracer=tracer+tracertempo # mmol C m^-2
        else: # dropna and 0 or 1 not nan
            tracer=np.nan
        array1d_iT_tracer[iT]=tracer
    return array1d_iT_tracer
