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
from matplotlib.image import imread
from mpl_toolkits.basemap import Basemap,basemap_datadir,os,np
from ojmisc import grid2cell

def getbm():
    """
        bm =getbm()
    """
    bmlatc = grid2cell(np.linspace(-90,90,2700))
    bmlonc = grid2cell(np.linspace(-180,180,5400))

    bm = np.ones((2700,5400,4),np.uint8)*255
    bm[:,:,:-1] = imread(os.path.join(basemap_datadir, 'bmng.jpg'))
    # make lakes, gaps along coast black
    sea = (bm[:,:,0]<.7*bm[:,:,2]) & (bm[:,:,1]<.9*bm[:,:,2])
    for c in range(3):
        bm[:,:,c][sea] = 0.

    bm[:,:,3] = 255*(1-sea.astype(np.uint8))

    return bm

 
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

 
