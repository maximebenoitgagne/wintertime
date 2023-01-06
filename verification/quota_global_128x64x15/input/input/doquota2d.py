#!/usr/bin/env python
import numpy as np
import ESMF
import MITgcmutils as mit
from scipy.io.netcdf import netcdf_file

def create_grid(bounds, domask):
    '''
    PRECONDITIONS: ESMPy has been initialized, 'bounds' contains the 
                   number of indices required for the first two 
                   dimensions of a Grid.  'domask' is a boolean value 
                   that gives the option to put a mask on this Grid.\n
    POSTCONDITIONS: An Grid has been created.\n
    RETURN VALUES: \n Grid :: grid \n
    '''

    nx = float(bounds[0])
    ny = float(bounds[1])

    dx = 360.0/nx
    dy = 180.0/ny

    DEG2RAD = 3.141592653589793/180.0

    max_index = np.array([nx,ny], dtype=np.int32)

    staggerLocs = [ESMF.StaggerLoc.CORNER, ESMF.StaggerLoc.CENTER]
    grid = ESMF.Grid(max_index, num_peri_dims=1, staggerlocs=staggerLocs)

    # VM
    vm = ESMF.ESMP_VMGetGlobal()
    localPet, petCount = ESMF.ESMP_VMGet(vm)

 # get the coordinate pointers and set the coordinates
    [x,y] = [0, 1]
    gridXCorner = grid.get_coords(x, ESMF.StaggerLoc.CORNER)
    gridYCorner = grid.get_coords(y, ESMF.StaggerLoc.CORNER)

    # make an array that holds indices from lower_bounds to upper_bounds
    bnd2indX = np.arange(grid.lower_bounds[ESMF.StaggerLoc.CORNER][x],
                         grid.upper_bounds[ESMF.StaggerLoc.CORNER][x], 1)
    bnd2indY = np.arange(grid.lower_bounds[ESMF.StaggerLoc.CORNER][y],
                         grid.upper_bounds[ESMF.StaggerLoc.CORNER][y], 1)

    for i in xrange(gridXCorner.shape[x]):
        gridXCorner[i, :] = float(bnd2indX[i])*dx - 180.0

    for j in xrange(gridYCorner.shape[y]):
        gridYCorner[:, j] = float(bnd2indY[j])*dy - 90.0

    ##     CENTERS

    # get the coordinate pointers and set the coordinates
    [x,y] = [0, 1]
    gridXCenter = grid.get_coords(x, ESMF.StaggerLoc.CENTER)
    gridYCenter = grid.get_coords(y, ESMF.StaggerLoc.CENTER)

    # make an array that holds indices from lower_bounds to upper_bounds
    bnd2indX = np.arange(grid.lower_bounds[ESMF.StaggerLoc.CENTER][x],
                         grid.upper_bounds[ESMF.StaggerLoc.CENTER][x], 1)
    bnd2indY = np.arange(grid.lower_bounds[ESMF.StaggerLoc.CENTER][y],
                         grid.upper_bounds[ESMF.StaggerLoc.CENTER][y], 1)

    for i in xrange(gridXCenter.shape[x]):
        gridXCenter[i, :] = float(bnd2indX[i])*dx + 0.5*dx - 180.0

    for j in xrange(gridYCenter.shape[y]):
        y = (float(bnd2indY[j])*dy - 90.0)
        yp1 = (float(bnd2indY[j]+1)*dy - 90.0)
        gridYCenter[:, j] = (y+yp1)/2.0

    '''
    # use mpi4py to collect values
    try:
        from mpi4py import MPI
    except:
        raise ImportError("mpi4py is not available, cannot compare \
                           global regridding error")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    print "PROC: "+str(rank)
    print "grid bounds: "+str(gridXCoord.shape)
    print "and    bounds: "+str(exLB)+str(exUB)
    '''

    [x,y] = [0, 1]
    mask = 0
    if domask:
        # set up the grid mask
        mask = grid.add_item(ESMF.GridItem.MASK)

        maskregionX = [175.,185.]
        maskregionY = [-5.,5.]

        for i in range(mask.shape[x]):
            for j in range(mask.shape[y]):
                if (maskregionX[0] < gridXCenter[i,j] < maskregionX[1] and
                    maskregionY[0] < gridYCenter[i,j] < maskregionY[1]):
                    mask[i, j] = 1
                else:
                    mask[i, j] = 0

    return grid

def create_grid_3d(bounds, rf):
    '''
    PRECONDITIONS: ESMPy has been initialized, 'bounds' contains the 
                   number of indices required for the first two 
                   dimensions of a Grid.  'domask' is a boolean value 
                   that gives the option to put a mask on this Grid.\n
    POSTCONDITIONS: An Grid has been created.\n
    RETURN VALUES: \n Grid :: grid \n
    '''

    nx = float(bounds[0])
    ny = float(bounds[1])
    nz = float(len(rf)-1)

    dx = 360.0/nx
    dy = 180.0/ny

    DEG2RAD = 3.141592653589793/180.0

    max_index = np.array([nx,ny,nz], dtype=np.int32)

    staggerLocs = [ESMF.StaggerLoc.CORNER_VFACE, ESMF.StaggerLoc.CENTER_VCENTER]
    grid = ESMF.Grid(max_index, num_peri_dims=0, coord_sys=ESMF.CoordSys.CART, staggerlocs=staggerLocs)

    # VM
    vm = ESMF.ESMP_VMGetGlobal()
    localPet, petCount = ESMF.ESMP_VMGet(vm)

    # get the coordinate pointers and set the coordinates
    [x,y,z] = [0, 1, 2]
    gridXCorner = grid.get_coords(x, ESMF.StaggerLoc.CORNER_VFACE)
    gridYCorner = grid.get_coords(y, ESMF.StaggerLoc.CORNER_VFACE)
    gridZCorner = grid.get_coords(z, ESMF.StaggerLoc.CORNER_VFACE)

    # make an array that holds indices from lower_bounds to upper_bounds
    bnd2indX = np.arange(grid.lower_bounds[ESMF.StaggerLoc.CORNER_VFACE][x],
                         grid.upper_bounds[ESMF.StaggerLoc.CORNER_VFACE][x], 1)
    bnd2indY = np.arange(grid.lower_bounds[ESMF.StaggerLoc.CORNER_VFACE][y],
                         grid.upper_bounds[ESMF.StaggerLoc.CORNER_VFACE][y], 1)

    for i in xrange(gridXCorner.shape[x]):
        gridXCorner[i, :, :] = float(bnd2indX[i])*dx - 180.0

    for j in xrange(gridYCorner.shape[y]):
        gridYCorner[:, j, :] = float(bnd2indY[j])*dy - 90.0

    for k in xrange(gridZCorner.shape[z]):
        gridZCorner[:, :, k] = float(rf[k])

    # get the coordinate pointers and set the coordinates
    gridXCenter = grid.get_coords(x, ESMF.StaggerLoc.CENTER_VCENTER)
    gridYCenter = grid.get_coords(y, ESMF.StaggerLoc.CENTER_VCENTER)
    gridZCenter = grid.get_coords(z, ESMF.StaggerLoc.CENTER_VCENTER)

    # make an array that holds indices from lower_bounds to upper_bounds
    bnd2indX = np.arange(grid.lower_bounds[ESMF.StaggerLoc.CENTER][x],
                         grid.upper_bounds[ESMF.StaggerLoc.CENTER][x], 1)
    bnd2indY = np.arange(grid.lower_bounds[ESMF.StaggerLoc.CENTER][y],
                         grid.upper_bounds[ESMF.StaggerLoc.CENTER][y], 1)

    for i in xrange(gridXCenter.shape[x]):
        gridXCenter[i, :, :] = float(bnd2indX[i])*dx + 0.5*dx - 180.0

    for j in xrange(gridYCenter.shape[y]):
        y = (float(bnd2indY[j])*dy - 90.0)
        yp1 = (float(bnd2indY[j]+1)*dy - 90.0)
        gridYCenter[:, j, :] = (y+yp1)/2.0

    for k in xrange(gridZCenter.shape[z]):
        gridZCenter[:, :, k] = .5*(rf[k]+rf[k+1])

    return grid

def fill_masked(f0, msk, n=10):
    f0.mask[:] = msk
    for i in range(n):
        f = f0.filled(0)
        w = 1.0 - f0.mask
        ww = np.zeros_like(f)
        ww[1:-1,1:-1] = (w[1:-1,2:]+w[1:-1,:-2]+w[2:,1:-1]+w[:-2,1:-1])
        ww[0   ,1:-1] = (w[0   ,2:]+w[0   ,:-2]+w[1 ,1:-1]+w[-1 ,1:-1])
        ww[-1  ,1:-1] = (w[-1  ,2:]+w[-1  ,:-2]+w[0 ,1:-1]+w[-2 ,1:-1])
        ww[1:-1,0   ] = (w[1:-1,1 ]            +w[2:,0   ]+w[:-2,0   ])
        ww[1:-1,-1  ] = (w[1:-1,-2]            +w[2:,-1  ]+w[:-2,-1  ])
        sm = np.zeros_like(f)
        sm[1:-1,1:-1] = (f[1:-1,2:]+f[1:-1,:-2]+f[2:,1:-1]+f[:-2,1:-1])
        sm[0   ,1:-1] = (f[0   ,2:]+f[0   ,:-2]+f[1 ,1:-1]+f[-1 ,1:-1])
        sm[-1  ,1:-1] = (f[-1  ,2:]+f[-1  ,:-2]+f[0 ,1:-1]+f[-2 ,1:-1])
        sm[1:-1,0   ] = (f[1:-1,1 ]            +f[2:,0   ]+f[:-2,0   ])
        sm[1:-1,-1  ] = (f[1:-1,-2]            +f[2:,-1  ]+f[:-2,-1  ])
        msk = f0.mask & (ww!=0)
        f0[msk] = sm[msk]/ww[msk]


#b0 = np.fromfile('bathy_fl.bin','>f4',count=160*360).reshape((160,360))
#b1 = np.fromfile('depth_g77.bin','>f4').reshape((64,128))

g = netcdf_file('grid1x1.nc')
rf0 = g.variables['RF'][:]
h0 = g.variables['HFacC'][:]
g.close()

g = netcdf_file('grid28.nc')
rf1 = g.variables['RF'][:]
h1 = g.variables['HFacC'][:]
g.close()


# initialize MPI
manager = ESMF.Manager(logkind = ESMF.LogKind.SINGLE, debug = True)  

g0 = create_grid((360,180), False)
g1 = create_grid((128,64), False)

m0 = g0.add_item(ESMF.GridItem.MASK)
m1 = g1.add_item(ESMF.GridItem.MASK)

m0[:] = 0
#m0[:] = 1
#m0[:,10:-10] = b0.T==0
m1[:] = 0
#m1[:,:] = b1.T==0

a0 = ESMF.Field(g0, 'srcarea')
a1 = ESMF.Field(g1, 'dstarea')

a0.get_area()
a1.get_area()

f0 = ESMF.Field(g0, 'src')
f1 = ESMF.Field(g1, 'dst')

lm0 = np.ones(f0.mask.shape + h0.shape[-3::-1], bool)
lm0[:, 10:-10] = h0.T == 0

ff0 = ESMF.Field(g0, 'srcfrac')
ff1 = ESMF.Field(g1, 'dstfrac')

regridSrc2Dst = ESMF.Regrid(f0, f1,
                            src_mask_values=np.array([1], dtype=np.int32),
                            dst_mask_values=np.array([1], dtype=np.int32),
                            regrid_method=ESMF.RegridMethod.CONSERVE,
                            unmapped_action=ESMF.UnmappedAction.IGNORE,
                            src_frac_field=ff0,
                            dst_frac_field=ff1)

p0 = np.zeros((12, 160, 360))
p1 = np.ma.zeros(p0.shape[:-2] + f1.shape[::-1], '>f4')

k = 0
for i in range(p0.shape[0]):
    f0[:,10:-10] = p0[i].T
    fill_masked(f0, lm0[:,:,k])
    f0[f0.mask] = np.nan
    regridSrc2Dst(f0, f1)
    p1[i] = f1.T


for fname in [
'mahowald2009_solubile_current_smooth_oce_mth-2d.bin',
'nasa_icefraction_mth-2d.bin',
'par_ecco.bin_mth-2d.bin',
]:
    p0 = np.fromfile('quota1x1/{}'.format(fname), '>f4').reshape((-1,160,360))

    for i in range(p0.shape[0]):
        f0[:,10:-10] = p0[i].T
        fill_masked(f0, lm0[:,:,k])
        f0[f0.mask] = np.nan
        regridSrc2Dst(f0, f1)
        p1[i] = f1.T
        p1[i][h1[k]==0] = 0

    p1.filled(0).astype('>f4').tofile('quota28x28/{}'.format(fname))

