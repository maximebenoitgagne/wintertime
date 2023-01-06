import os
import math
import numpy as np
from numpy import ma
import mpl_toolkits.basemap as mbm

class Basemap(mbm.Basemap):
    def __init__(self, *args, **kwargs):
        mbm.Basemap.__init__(self, *args, **kwargs)

    # set __init__'s docstring
    __init__.__doc__ = mbm._Basemap_init_doc

    def makecgrid(self, nx, ny, returnxy=False):
        """
        return arrays of shape (ny,nx) containing lon,lat coordinates of
        an equally spaced, cell-centered native projection grid.

        If ``returnxy = True``, the x,y values of the grid are returned also.
        """
        dx = (self.urcrnrx-self.llcrnrx)/nx
        dy = (self.urcrnry-self.llcrnry)/ny
        j,i = np.indices((ny,nx), np.float32)
        x = self.llcrnrx+dx*(.5+i)
        y = self.llcrnry+dy*(.5+j)
        lons, lats = self(x, y, inverse=True)
        if returnxy:
            return lons, lats, x, y
        else:
            return lons, lats

    def transform_scalar(self,datin,lons,lats,nx,ny,returnxy=False,checkbounds=False,order=1,masked=False):
        """
        Interpolate a scalar field (``datin``) from a lat/lon grid with
        longitudes = ``lons`` and latitudes = ``lats`` to a ``ny`` by ``nx``
        map projection grid.  Typically used to transform data to
        map projection coordinates for plotting on a map with
        the :meth:`imshow`.

        .. tabularcolumns:: |l|L|

        ==============   ====================================================
        Argument         Description
        ==============   ====================================================
        datin            input data on a lat/lon grid.
        lons, lats       rank-1 arrays containing longitudes and latitudes
                         (in degrees) of input data in increasing order.
                         For non-cylindrical projections (those other than
                         ``cyl``, ``merc``, ``gall`` and ``mill``) lons must
                         fit within range -180 to 180.
        nx, ny           The size of the output regular grid in map
                         projection coordinates
        ==============   ====================================================

        .. tabularcolumns:: |l|L|

        ==============   ====================================================
        Keyword          Description
        ==============   ====================================================
        returnxy         If True, the x and y values of the map
                         projection grid are also returned (Default False).
        checkbounds      If True, values of lons and lats are checked to see
                         that they lie within the map projection region.
                         Default is False, and data outside map projection
                         region is clipped to values on boundary.
        masked           If True, interpolated data is returned as a masked
                         array with values outside map projection region
                         masked (Default False).
        order            0 for nearest-neighbor interpolation, 1 for
                         bilinear, 3 for cubic spline (Default 1).
                         Cubic spline interpolation requires scipy.ndimage.
        ==============   ====================================================

        Returns ``datout`` (data on map projection grid).
        If returnxy=True, returns ``data,x,y``.
        """
        # check that lons, lats increasing
        delon = lons[1:]-lons[0:-1]
        delat = lats[1:]-lats[0:-1]
        if min(delon) < 0. or min(delat) < 0.:
            raise ValueError('lons and lats must be increasing!')
        # check that lons in -180,180 for non-cylindrical projections.
        if self.projection not in mbm._cylproj:
            lonsa = np.array(lons)
            count = np.sum(lonsa < -180.00001) + np.sum(lonsa > 180.00001)
            if count > 1:
                raise ValueError('grid must be shifted so that lons are monotonically increasing and fit in range -180,+180 (see shiftgrid function)')
            # allow for wraparound point to be outside.
            elif count == 1 and math.fabs(lons[-1]-lons[0]-360.) > 1.e-4:
                raise ValueError('grid must be shifted so that lons are monotonically increasing and fit in range -180,+180 (see shiftgrid function)')
        if returnxy:
            lonsout, latsout, x, y = self.makecgrid(nx,ny,returnxy=True)
        else:
            lonsout, latsout = self.makecgrid(nx,ny)
        datout = mbm.interp(datin,lons,lats,lonsout,latsout,checkbounds=checkbounds,order=order,masked=masked)
        if returnxy:
            return datout, x, y
        else:
            return datout

    def warpimage(self,image="bluemarble",scale=None,**kwargs):
        """
        Display an image (filename given by ``image`` keyword) as a map background.
        If image is a URL (starts with 'http'), it is downloaded to a temp
        file using urllib.urlretrieve.

        Default (if ``image`` not specified) is to display
        'blue marble next generation' image from http://visibleearth.nasa.gov/.

        Specified image must have pixels covering the whole globe in a regular
        lat/lon grid, starting and -180W and the South Pole.
        Works with the global images from
        http://earthobservatory.nasa.gov/Features/BlueMarble/BlueMarble_monthlies.php.

        The ``scale`` keyword can be used to downsample (rescale) the image.
        Values less than 1.0 will speed things up at the expense of image
        resolution.

        Extra keyword ``ax`` can be used to override the default axis instance.

        \**kwargs passed on to :meth:`imshow`.

        returns a matplotlib.image.AxesImage instance.
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError('warpimage method requires PIL (http://www.pythonware.com/products/pil)')
        from matplotlib.image import pil_to_array
        ax = kwargs.pop('ax', None) or self._check_ax()
        # default image file is blue marble next generation
        # from NASA (http://visibleearth.nasa.gov).
        if image == "bluemarble":
            file = os.path.join(basemap_datadir,'bmng.jpg')
        # display shaded relief image (from
        # http://www.shadedreliefdata.com)
        elif image == "shadedrelief":
            file = os.path.join(basemap_datadir,'shadedrelief.jpg')
        # display etopo image (from
        # http://www.ngdc.noaa.gov/mgg/image/globalimages.html)
        elif image == "etopo":
            file = os.path.join(basemap_datadir,'etopo1.jpg')
        else:
            file = image
        # if image is same as previous invocation, used cached data.
        # if not, regenerate rgba data.
        if not hasattr(self,'_bm_file') or self._bm_file != file:
            newfile = True
        else:
            newfile = False
        if file.startswith('http'):
            from urllib import urlretrieve
            self._bm_file, headers = urlretrieve(file)
        else:
            self._bm_file = file
        # bmproj is True if map projection region is same as
        # image region.
        bmproj = self.projection == 'cyl' and \
                 self.llcrnrlon == -180 and self.urcrnrlon == 180 and \
                 self.llcrnrlat == -90 and self.urcrnrlat == 90
        # read in jpeg image to rgba array of normalized floats.
        if not hasattr(self,'_bm_rgba') or newfile:
            pilImage = Image.open(self._bm_file)
            if scale is not None:
                w, h = pilImage.size
                width = int(np.round(w*scale))
                height = int(np.round(h*scale))
                pilImage = pilImage.resize((width,height),Image.ANTIALIAS)
            self._bm_rgba = pil_to_array(pilImage)
            # define lat/lon grid that image spans.
            nlons = self._bm_rgba.shape[1]; nlats = self._bm_rgba.shape[0]
            delta = 360./float(nlons)
            self._bm_lons = np.arange(-180.+0.5*delta,180.,delta)
            self._bm_lats = np.arange(-90.+0.5*delta,90.,delta)
            # is it a cylindrical projection whose limits lie
            # outside the limits of the image?
            cylproj =  self.projection in mbm._cylproj and \
                      (self.urcrnrlon > self._bm_lons[-1] or \
                       self.llcrnrlon < self._bm_lons[0])
            # if pil_to_array returns a 2D array, it's a grayscale image.
            # create an RGB image, with R==G==B.
            if self._bm_rgba.ndim == 2:
                tmp = np.empty(self._bm_rgba.shape+(3,),np.uint8)
                for k in range(3):
                    tmp[:,:,k] = self._bm_rgba
                self._bm_rgba = tmp
            if cylproj and not bmproj:
                # stack grids side-by-side (in longitiudinal direction), so
                # any range of longitudes may be plotted on a world map.
                self._bm_lons = \
                np.concatenate((self._bm_lons,self._bm_lons+360),1)
                self._bm_rgba = \
                np.concatenate((self._bm_rgba,self._bm_rgba),1)
            # convert to normalized floats.
            self._bm_rgba = self._bm_rgba.astype(np.float32)/255.
        if not bmproj: # interpolation necessary.
            if newfile or not hasattr(self,'_bm_rgba_warped'):
                # transform to nx x ny regularly spaced native
                # projection grid.
                # nx and ny chosen to have roughly the
                # same horizontal res as original image.
                if self.projection != 'cyl':
                    dx = 2.*np.pi*self.rmajor/float(nlons)
                    nx = int((self.xmax-self.xmin)/dx+.5)
                    ny = int((self.ymax-self.ymin)/dx+.5)
                else:
                    dx = 360./float(nlons)
                    nx = int((self.urcrnrlon-self.llcrnrlon)/dx+.5)
                    ny = int((self.urcrnrlat-self.llcrnrlat)/dx+.5)
                self._bm_rgba_warped = np.ones((ny,nx,4),np.float64)
                # interpolate rgba values from geographic coords (proj='cyl')
                # to map projection coords.
                # if masked=True, values outside of
                # projection limb will be masked.
                for k in range(3):
                    self._bm_rgba_warped[:,:,k],x,y = \
                    self.transform_scalar(self._bm_rgba[:,:,k],\
                                          self._bm_lons,self._bm_lats,
                                          nx,ny,returnxy=True)
                # for ortho,geos mask pixels outside projection limb.
                if self.projection in ['geos','ortho','nsper'] or \
                   (self.projection == 'aeqd' and self._fulldisk):
                    lonsr,latsr = self(x,y,inverse=True)
                    mask = ma.zeros((ny,nx,4),np.int8)
                    mask[:,:,0] = np.logical_or(lonsr>1.e20,latsr>1.e30)
                    for k in range(1,4):
                        mask[:,:,k] = mask[:,:,0]
                    self._bm_rgba_warped = \
                    ma.masked_array(self._bm_rgba_warped,mask=mask)
                    # make points outside projection limb transparent.
                    self._bm_rgba_warped = self._bm_rgba_warped.filled(0.)
                # treat pseudo-cyl projections such as mollweide, robinson and sinusoidal.
                elif self.projection in mbm._pseudocyl:
                    lonsr,latsr = self(x,y,inverse=True)
                    mask = ma.zeros((ny,nx,4),np.int8)
                    lon_0 = self.projparams['lon_0']
                    lonright = lon_0+180.
                    lonleft = lon_0-180.
                    x1 = np.array(ny*[0.5*(self.xmax + self.xmin)],np.float)
                    # put lats on cgrid
                    dy = (self.ymax - self.ymin)/ny
                    y1 = np.linspace(self.ymin+.5*dy, self.ymax-.5*dy, ny)
                    lons1, lats1 = self(x1,y1,inverse=True)
#                    lats1 = np.where(lats1 < -89.999999, -89.999999, lats1)
#                    lats1 = np.where(lats1 > 89.999999, 89.999999, lats1)
                    for j,lat in enumerate(lats1):
                        xmax,ymax = self(lonright,lat)
                        xmin,ymin = self(lonleft,lat)
                        mask[j,:,0] = np.logical_or(x[j,:]>xmax,x[j,:]<xmin)
                    for k in range(1,4):
                        mask[:,:,k] = mask[:,:,0]
                    self._bm_rgba_warped = \
                    ma.masked_array(self._bm_rgba_warped,mask=mask)
                    # make points outside projection limb transparent.
                    self._bm_rgba_warped = self._bm_rgba_warped.filled(0.)
            # plot warped rgba image.
            im = self.imshow(self._bm_rgba_warped,ax=ax,**kwargs)
        else:
            # bmproj True, no interpolation necessary.
            im = self.imshow(self._bm_rgba,ax=ax,**kwargs)
        return im

