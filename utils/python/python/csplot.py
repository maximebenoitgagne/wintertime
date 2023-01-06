#!/usr/bin/env python
from pylab import prod, mod, pcolormesh

def plotcube(x, y, c, xlim=None, ylim=None, **kwargs):
    """
        plotcube(x,y,cube,xlim,ylim)
     
        x[6,ny+1,nx+1]               :: x of cell corners for each face
        y[6,ny+1,nx+1]               :: y of cell corners for each face
        c[ny,6,nx] or c[ny,6*nx]     :: pseudocolor data (to be mapped by pcolormesh)
        c[3,ny,6,nx] or c[3,ny,6*nx] :: RGB data (supported by pcolormesh???)
        xlim                         :: range (or max.) of x to plot (default [0,360])
        ylim                         :: range (or max.) of y to plot (default [-90,90])
        kwargs                       :: passed on to pcolormesh

        x is assumed to be 360-periodic (i.e. longitude)
    """
    if xlim is None:
        xlim = [0, 360]
    elif len(xlim)<2:
        xlim = [0, xlim]
    if ylim is None:
        ylim = [-90, 90]
    elif len(ylim)<2:
        ylim = [-ylim, ylim]

    nf,nyp,nxp = x.shape
    if y.shape != (nf,nyp,nxp):
        return
    nx = nxp - 1
    ny = nyp - 1

    dims = c.shape
    #print nx,ny,dims
    if dims[-2:] == (ny,nf*nx):
        c = c.reshape((prod(dims[:-2]), ny, nf, nx))
    elif dims[-3:] == (ny,nf,nx):
        c = c.reshape((prod(dims[:-3]), ny, nf, nx))
    else:
        print 'plotcube: invalid dimensions of c:', dims
        return
    nc, ny, nf, nx = c.shape

    x = mod(x, 360)
    xp = mod(x-.001, 360)+.001
    xm = mod(x+179.999, 360)-179.999

    # 0 to 45E
    fc=0; i0=nx/2; i1=nx  ; j0=0   ; j1=ny  ; pcolormesh(x [fc,j0:j1+1,i0:i1+1].T, y[fc,j0:j1+1,i0:i1+1].T, c[:,j0:j1,fc,i0:i1].T, **kwargs)
    # 45E to 135E
    fc=1; i0=0   ; i1=nx  ; j0=0   ; j1=ny  ; pcolormesh(x [fc,j0:j1+1,i0:i1+1].T, y[fc,j0:j1+1,i0:i1+1].T, c[:,j0:j1,fc,i0:i1].T, **kwargs)
    # 135E to 135W
    fc=3; i0=0   ; i1=nx  ; j0=0   ; j1=ny  ; pcolormesh(x [fc,j0:j1+1,i0:i1+1].T, y[fc,j0:j1+1,i0:i1+1].T, c[:,j0:j1,fc,i0:i1].T, **kwargs)
    # 135W to 45W
    fc=4; i0=0   ; i1=nx  ; j0=0   ; j1=ny  ; pcolormesh(x [fc,j0:j1+1,i0:i1+1].T, y[fc,j0:j1+1,i0:i1+1].T, c[:,j0:j1,fc,i0:i1].T, **kwargs)
    # 45W to 0
    fc=0; i0=0   ; i1=nx/2; j0=0   ; j1=ny  ; pcolormesh(xp[fc,j0:j1+1,i0:i1+1].T, y[fc,j0:j1+1,i0:i1+1].T, c[:,j0:j1,fc,i0:i1].T, **kwargs)

    # arctic face 0 to 180E
    fc=2; i0=0   ; i1=nx  ; j0=0   ; j1=ny/2; pcolormesh(x [fc,j0:j1+1,i0:i1+1].T, y[fc,j0:j1+1,i0:i1+1].T, c[:,j0:j1,fc,i0:i1].T, **kwargs)
    # arctic face 180W to 0                                                                                                
    fc=2; i0=0   ; i1=nx  ; j0=ny/2; j1=ny  ; pcolormesh(xp[fc,j0:j1+1,i0:i1+1].T, y[fc,j0:j1+1,i0:i1+1].T, c[:,j0:j1,fc,i0:i1].T, **kwargs)

    # antarctic face 180W to 0
    fc=5; i0=0   ; i1=nx/2; j0=0   ; j1=ny  ; pcolormesh(xp[fc,j0:j1+1,i0:i1+1].T, y[fc,j0:j1+1,i0:i1+1].T, c[:,j0:j1,fc,i0:i1].T, **kwargs)
    # antarctic face 0 to 180W                                                                                                 
    fc=5; i0=nx/2; i1=nx  ; j0=0   ; j1=ny  ; im=pcolormesh(x [fc,j0:j1+1,i0:i1+1].T, y[fc,j0:j1+1,i0:i1+1].T, c[:,j0:j1,fc,i0:i1].T, **kwargs)

    # useful for colorbar, ...
    return im


