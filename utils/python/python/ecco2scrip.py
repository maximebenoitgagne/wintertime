from scipy.io.netcdf import netcdf_file as nc
import scipy.sparse

_datadir = '/nobackup1/jahn/grid/ecco2/'

def load_e2_to_11():
    hrmp = nc(_datadir + 'rmp_e2_to_11_conserv.nc', 'r', mmap=False)
    remap_matrix = hrmp.variables['remap_matrix'][:,0]
    src_address = hrmp.variables['src_address']
    dst_address = hrmp.variables['dst_address']
#    src_grid_area = hrmp.variables['src_grid_area']
#    dst_grid_area = hrmp.variables['dst_grid_area']

    nout,nin = 180*360,6*510*510
    M = scipy.sparse.coo_matrix((remap_matrix,(dst_address[:]-1,src_address[:]-1)),shape=(nout,nin)).tocsr()
    return M

def load_11_to_e2():
    hrmp = nc(_datadir + 'rmp_11_to_e2_conserv.nc', 'r', mmap=False)
    remap_matrix = hrmp.variables['remap_matrix'][:,0]
    src_address = hrmp.variables['src_address']
    dst_address = hrmp.variables['dst_address']
#    src_grid_area = hrmp.variables['src_grid_area']
#    dst_grid_area = hrmp.variables['dst_grid_area']

    nin,nout = 180*360,6*510*510
    M = scipy.sparse.coo_matrix((remap_matrix,(dst_address[:]-1,src_address[:]-1)),shape=(nout,nin)).tocsr()
    return M

