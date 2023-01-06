import sys
from .num import *
from .plot import *
from .axes import *
from .axesgrid import *
from .cal import *
try:
    from basemap import Basemap
except:
    sys.stderr.write('basemap could not be imported\n')
import oj.colors as mycolors
from .colors import colorbar as mycb
from .colors import SymNorm
from .cm import coolwarm, lBuRd, pm, pm2, CMRmap, CMRmap_r, coolwarm256, coolwarmblack, cubehelixmap, jeti, parula
from .cm import CMRmap_d, CMRmap_l, CMRmap_r_d, CMRmap_r_l

from matplotlib.colors import LogNorm
try:
    from matplotlib.colors import SymLogNorm
except ImportError:
    from colors import SymLogNorm

try:
    from cmocean import cm as cmo
except ImportError:
    sys.stderr.write('could not import cmocean\n')
else:
    import matplotlib.cm as mcm
    mcm.register_cmap('balance', cmo.balance)
    mcm.register_cmap('delta', cmo.delta)
    mcm.register_cmap('thermal', cmo.thermal)

def fapcm(*args, **kwargs):
    from cartopy import crs as ccrs
    kwargs.setdefault('transform', ccrs.PlateCarree())
    kwargs.setdefault('maskback', True)
    import faplot
    return faplot.pcolormeshes(*args, **kwargs)


def zoomify(*cbs, **kw):
    for cb in cbs:
        cb.zoom = mycolors.zoomify(cb, **kw)


#from mytext import mytext
#from m2it import m2it, date2endit, itrange
#from mycb import MyLogFormatter, MyMathLogFormatter, makecmap, cmap2xrgb, make_cmap_seawifs, cat_cmaps, scale_segmentdata
#from addcolors import addcolorssqrt, rgbcolorbar, rgbcb
##from lonlat import *
#from complex import z2rgb, hsv2rgb, zshow, fshow, zplot, zplotgrid
#from misc import fromgzipfile, myfromfile, toraw, fromraw, grid2cell, it2ymdhms, it2date, it2day, it2dayl, it2mon, rel2abs, xzeroaxis, yzeroaxis, myimshow, mercatory, axes_size_inches, im_dpi, im_size_pixels, im_size_axes, im_scale_factor, im_fig_size, im_axes_size, calc_fig_size, subaxes, smax, maxabs, indmin, indmax, nanindmax, nanindmin, indmaxabs, max2, maxabs2, pause, LatitudeFormatter, LongitudeFormatter, myfromfid, globits, block2d, unblock2d, unblock1d, block1d, myfig, pos_borders_inches, pos_borders_fs, axes_borders_inches, axes_borders_fs, mysavefig, rawparams, rawname
#from stat import shannon
#from axes import pcolormesh
#    
##if __name__ == '__main__':
##    args = sys.argv[1:]
##
##    if args is not None:
##        print it2date(float(args[0]))
#
#
