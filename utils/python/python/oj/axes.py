#!/usr/bin/env python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
#from matplotlib.axes import *

_debug = False

def _pcolorargs(ax, funcname, *args):
    if len(args)==1:
        C = args[0]
        numRows, numCols = C.shape[:2]
        X, Y = np.meshgrid(np.arange(numCols+1), np.arange(numRows+1) )
    elif len(args)==3:
        X, Y, C = args
    else:
        raise TypeError(
            'Illegal arguments to %s; see help(%s)' % (funcname, funcname))

    Nx = X.shape[-1]
    Ny = Y.shape[0]
    if len(X.shape) != 2 or X.shape[0] == 1:
        x = X.reshape(1,Nx)
        X = x.repeat(Ny, axis=0)
    if len(Y.shape) != 2 or Y.shape[1] == 1:
        y = Y.reshape(Ny, 1)
        Y = y.repeat(Nx, axis=1)
    if X.shape != Y.shape:
        raise TypeError(
            'Incompatible X, Y inputs to %s; see help(%s)' % (
            funcname, funcname))
    return X, Y, C

def pcolormesh(ax, *args, **kwargs):
    """
    call signatures::

      pcolormesh(C)
      pcolormesh(X, Y, C)
      pcolormesh(C, **kwargs)

    *C* may be a masked array, but *X* and *Y* may not.  Masked
    array support is implemented via *cmap* and *norm*; in
    contrast, :func:`~matplotlib.pyplot.pcolor` simply does not
    draw quadrilaterals with masked colors or vertices.

    Keyword arguments:

      *cmap*: [ None | Colormap ]
        A :class:`matplotlib.cm.Colormap` instance. If None, use
        rc settings.

      *norm*: [ None | Normalize ]
        A :class:`matplotlib.colors.Normalize` instance is used to
        scale luminance data to 0,1. If None, defaults to
        :func:`normalize`.

      *vmin*/*vmax*: [ None | scalar ]
        *vmin* and *vmax* are used in conjunction with *norm* to
        normalize luminance data.  If either are *None*, the min
        and max of the color array *C* is used.  If you pass a
        *norm* instance, *vmin* and *vmax* will be ignored.

      *shading*: [ 'flat' | 'faceted' | 'gouraud' ]
        If 'faceted', a black grid is drawn around each rectangle; if
        'flat', edges are not drawn. Default is 'flat', contrary to
        Matlab(TM).

        This kwarg is deprecated; please use 'edgecolors' instead:
          * shading='flat' -- edgecolors='None'
          * shading='faceted  -- edgecolors='k'

      *edgecolors*: [ None | 'None' | color | color sequence]
        If None, the rc setting is used by default.

        If 'None', edges will not be visible.

        An mpl color or sequence of colors will set the edge color

      *alpha*: 0 <= scalar <= 1
        the alpha blending value

    Return value is a :class:`matplotlib.collection.QuadMesh`
    object.

    kwargs can be used to control the
    :class:`matplotlib.collections.QuadMesh`
    properties:

    %(QuadMesh)s

    .. seealso::

        :func:`~matplotlib.pyplot.pcolor`
            For an explanation of the grid orientation and the
            expansion of 1-D *X* and/or *Y* to 2-D arrays.
    """
    if ax is None: ax = plt.gca()
    if not ax._hold: ax.cla()

    alpha = kwargs.pop('alpha', None)
    norm = kwargs.pop('norm', None)
    cmap = kwargs.pop('cmap', None)
    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    shading = kwargs.pop('shading', 'flat').lower()
    edgecolors = kwargs.pop('edgecolors', 'None')
    antialiased = kwargs.pop('antialiased', False)

    X, Y, C = _pcolorargs(ax, 'pcolormesh', *args)
    Ny, Nx = X.shape

    # convert to one dimensional arrays
    if shading != 'gouraud':
        C = C[0:Ny-1, 0:Nx-1, ...].reshape(((Nx-1)*(Ny-1),) + C.shape[2:])
                                        # data point in each cell is value at
                                        # lower left corner
    else:
        C = C.reshape((-1,) + C.shape[2:])
    if C.shape[-1] == 1:
        C = C.reshape((-1,))
    X = X.ravel()
    Y = Y.ravel()

    coords = np.zeros(((Nx * Ny), 2), dtype=float)
    coords[:, 0] = X
    coords[:, 1] = Y

    if shading == 'faceted' or edgecolors != 'None':
        showedges = 1
    else:
        showedges = 0

    collection = mcoll.QuadMesh(
        Nx - 1, Ny - 1, coords, showedges,
        shading=shading)  # kwargs are not used
    collection.set_antialiased(antialiased)
    collection.set_alpha(alpha)
    if len(C.shape) == 1:
        # use scalar data
        collection.set_array(C)
        if norm is not None: assert(isinstance(norm, mcolors.Normalize))
        collection.set_cmap(cmap)
        collection.set_norm(norm)
        if vmin is not None or vmax is not None:
            collection.set_clim(vmin, vmax)
        else:
            collection.autoscale_None()
    else:
        # C is already rgb(a)
        collection.set_facecolors(C)

    ax.grid(False)

    minx = np.amin(X)
    maxx = np.amax(X)
    miny = np.amin(Y)
    maxy = np.amax(Y)

    corners = (minx, miny), (maxx, maxy)
    ax.update_datalim( corners)
    ax.autoscale_view()
    ax.add_collection(collection)
    return collection


