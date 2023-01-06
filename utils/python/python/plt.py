import MITgcmutils as mit
import oj.plt
import xarray as xr
from pylab import *
from fortranio import *
from MITgcmutils import *
from oj.plt import *
from gapgrid import GapGrid, gapgrid, gapaxes
import facets as fa
import facetplot as fp
import exchange
from h5fa import FaFile
import cartopy.crs as ccrs
#from cartopy.crs import *
from os.path import splitext, split, join, exists

def ccax(proj=None, *args, **kwargs):
    kwargs['projection'] = proj or ccrs.PlateCarree()
    return plt.axes(*args, **kwargs)

