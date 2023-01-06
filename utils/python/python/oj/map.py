"""
    mapping tools (mainly Mollweide projection)

    example usage:

    landinsidemoll = logical_not(mapmask)&(lmmap)
    
    bmmoll = bm_moll(nx,ny,bor,landinsidemoll,-90)

    # plot ...
    # mycb ...

    imbm = myimshow(bmmoll)
    axis([xoff,nx+xoff,yoff,ny+yoff])
"""
import os
import numpy as np
from matplotlib.image import imread
from mpl_toolkits.basemap import Basemap,basemap_datadir
from oj.misc import grid2cell

def getbmxyg(lon0=-180):
    # make it match getbm
    lon0 = int(15*lon0)/15.
    latgbm,longbm = np.mgrid[-90.:90.:2701j,lon0:lon0+360.:5401j]
    return latgbm,longbm


def getbm(lon0=-180):
    """
        bm =getbm()
    """
    bm = np.ones((2700,5400,4),np.uint8)*255
    bm[:,:,:-1] = imread(os.path.join(basemap_datadir, 'bmng.jpg'))
    # make lakes, gaps along coast black
    sea = (bm[:,:,0]<.7*bm[:,:,2]) & (bm[:,:,1]<.9*bm[:,:,2])
    for c in range(3):
        bm[:,:,c][sea] = 0.

    bm[:,:,3] = 255*(1-sea.astype(np.uint8))

    roll = (2700-int(15*lon0)+5400)%5400
    return np.roll(bm,roll,axis=1)

 
def bm_moll(nx,ny,bor=0,landmask=1,lon_0=-90):
    """
        bmmoll = bm_moll(nx,ny,bor,landinsidemoll,lon_0)
    """
    moll = Basemap(projection='moll', lon_0=lon_0)

    bmlatc = grid2cell(np.linspace(-90,90,2700))
    bmlonc = grid2cell(np.linspace(-180,180,5400))

    bm = np.ones((2700,5400,4),np.uint8)*255
    bm[:,:,:-1] = imread(os.path.join(basemap_datadir, 'bmng.jpg'))
    # make lakes, gaps along coast black
    sea = (bm[:,:,0]<.7*bm[:,:,2]) & (bm[:,:,1]<.9*bm[:,:,2])
    for c in range(3):
        bm[:,:,c][sea] = 0.
    del sea
    bmmoll = np.zeros((ny,nx,4),np.uint8)
    for c in range(3):
        bmmoll[bor:-bor,bor:-bor,c] = moll.transform_scalar(bm[:,:,c], bmlonc, bmlatc, nx-2*bor, ny-2*bor, order=0, masked=False)
    # make my sea transparent
    bmmoll[:,:,3] = 255*landmask

    return bmmoll

 
