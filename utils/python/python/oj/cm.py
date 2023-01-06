import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from oj.interp import stineman
import oj.cie as cie
from oj.colors import cat_cmaps

#s = cm.RdBu_r._segmentdata
#_lBuRd_list = [(s['red'][i][2],s['green'][i][2],s['blue'][i][2]) for i in range(1,10)]

# RdBu_r without the really dark shades near the ends
_lBuRd_list = [(0.12941177189350128, 0.4000000059604645, 0.6745098233222961),
               (0.26274511218070984, 0.5764706134796143, 0.7647058963775635),
               (0.572549045085907, 0.772549033164978, 0.8705882430076599),
               (0.8196078538894653, 0.8980392217636108, 0.9411764740943909),
               (0.9686274528503418, 0.9686274528503418, 0.9686274528503418),
               (0.9921568632125854, 0.8588235378265381, 0.7803921699523926),
               (0.95686274766922, 0.6470588445663452, 0.5098039507865906),
               (0.8392156958580017, 0.3764705955982208, 0.3019607961177826),
               (0.6980392336845398, 0.0941176488995552, 0.16862745583057404),
              ]

# RdBu_r without the really dark shades near the ends
_pm_list = [(0.00941177189350128, 0.9900000059604645, 0.7647098233222961),
            (0.26274511218070984, 0.5764706134796143, 0.7647058963775635),
            (0.572549045085907, 0.772549033164978, 0.8705882430076599),
            (0.8196078538894653, 0.8980392217636108, 0.9411764740943909),
            (0.9686274528503418, 0.9686274528503418, 0.9686274528503418),
            (0.9921568632125854, 0.8588235378265381, 0.7803921699523926),
            (0.95686274766922, 0.6470588445663452, 0.5098039507865906),
            (0.8392156958580017, 0.3764705955982208, 0.3019607961177826),
            (0.8392392336845398, 0.9941176488995552, 0.00862745583057404),
           ]

lBuRd = mcolors.LinearSegmentedColormap.from_list('lBuRd',_lBuRd_list)
cm.register_cmap('lBuRd', lBuRd)

pm    = mcolors.LinearSegmentedColormap.from_list('pm',_pm_list)
cm.register_cmap('pm', pm)

_pm2_data = {'blue': [(0.0  , 0.7647, 0.5000),
                      (0.125, 0.7647, 0.7647),
                      (0.25 , 0.8705, 0.8705),
                      (0.375, 0.9411, 0.9411),
                      (0.5  , 0.9686, 0.9686),
                      (0.625, 0.7803, 0.7803),
                      (0.75 , 0.5098, 0.5098),
                      (0.875, 0.3019, 0.3019),
                      (1.0  , 0.0086, 0.0086)],
             'green': [(0.0  , 1.0000, 1.0000),
                       (0.125, 0.5764, 0.5764),
                       (0.25 , 0.7725, 0.7725),
                       (0.375, 0.8980, 0.8980),
                       (0.5  , 0.9686, 0.9686),
                       (0.625, 0.8588, 0.8588),
                       (0.75 , 0.6470, 0.6470),
                       (0.875, 0.3764, 0.3764),
                       (1.0  , 1.0000, 1.0000)],
             'red': [(0.0  , 0.0094, 0.0000),
                     (0.125, 0.2627, 0.2627),
                     (0.25 , 0.5725, 0.5725),
                     (0.375, 0.8196, 0.8196),
                     (0.5  , 0.9686, 0.9686),
                     (0.625, 0.9922, 0.9921),
                     (0.75 , 0.9568, 0.9568),
                     (0.875, 0.8392, 0.8392),
                     (1.0  , 0.5600, 0.8392)]}

pm2   = mcolors.LinearSegmentedColormap('pm',_pm2_data)
cm.register_cmap('pm2', pm2)


def makecubehelix(gamma=1.0, s=0.5, r=-1.5, h=1.0, name=None, N=256):
    if name is None:
        name = 'cubehelix(gamma={}, s={}, r={}, h={})'.format(gamma,s,r,h)
    return mcolors.LinearSegmentedColormap(name, cubehelix(gamma, s, r, h), N)

def makecubehelixpart(gamma=1.0, s=0.5, r=-1.5, h=1.0, f=1.0, name=None, N=256):
    if name is None:
        name = 'cubehelix(gamma={}, s={}, r={}, h={}, f={})'.format(gamma,s,r,h,f)
    return mcolors.LinearSegmentedColormap(name, cubehelixpart(gamma, s, r, h, f), N)

def part(cmap, x):
    name = cmap.name + 'part'
    sd = cmap._segmentdata
    sd = dict((k, lambda y: f(x*y)) for k,f in sd.items())
    return mcolors.LinearSegmentedColormap(name, sd, N=cmap.N, gamma=cmap._gamma)

def cubehelixpart(gamma=1.0, s=0.5, r=-1.5, h=1.0, f=1.0):
    """Return custom data dictionary of (r,g,b) conversion functions, which
    can be used with :func:`register_cmap`, for the cubehelix color scheme.

    Unlike most other color schemes cubehelix was designed by D.A. Green to
    be monotonically increasing in terms of perceived brightness.
    Also, when printed on a black and white postscript printer, the scheme
    results in a greyscale with monotonically increasing brightness.
    This color scheme is named cubehelix because the r,g,b values produced
    can be visualised as a squashed helix around the diagonal in the
    r,g,b color cube.

    For a unit color cube (i.e. 3-D coordinates for r,g,b each in the
    range 0 to 1) the color scheme starts at (r,g,b) = (0,0,0), i.e. black,
    and finishes at (r,g,b) = (1,1,1), i.e. white. For some fraction *x*,
    between 0 and 1, the color is the corresponding grey value at that
    fraction along the black to white diagonal (x,x,x) plus a color
    element. This color element is calculated in a plane of constant
    perceived intensity and controlled by the following parameters.

    Optional keyword arguments:

      =========   =======================================================
      Keyword     Description
      =========   =======================================================
      gamma       gamma factor to emphasise either low intensity values
                  (gamma < 1), or high intensity values (gamma > 1);
                  defaults to 1.0.
      s           the start color; defaults to 0.5 (i.e. purple).
      r           the number of r,g,b rotations in color that are made
                  from the start to the end of the color scheme; defaults
                  to -1.5 (i.e. -> B -> G -> R -> B).
      h           the hue parameter which controls how saturated the
                  colors are. If this parameter is zero then the color
                  scheme is purely a greyscale; defaults to 1.0.
      =========   =======================================================

    """

    def get_color_function(p0, p1, f):
        def color(x):
            # Apply gamma factor to emphasise low or high intensity values
            xg = (f*x)**gamma

            # Calculate amplitude and angle of deviation from the black
            # to white diagonal in the plane of constant
            # perceived intensity.
            a = h * xg * (1 - xg) / 2

            phi = 2 * np.pi * (s / 3 + r * f * x)

            return xg + a * (p0 * np.cos(phi) + p1 * np.sin(phi))
        return color

    return {
            'red': get_color_function(-0.14861, 1.78277, f),
            'green': get_color_function(-0.29227, -0.90649, f),
            'blue': get_color_function(1.97294, 0.0, f),
    }

def cubehelix(gamma=1.0, s=0.5, r=-1.5, h=1.0):
    """Return custom data dictionary of (r,g,b) conversion functions, which
    can be used with :func:`register_cmap`, for the cubehelix color scheme.

    Unlike most other color schemes cubehelix was designed by D.A. Green to
    be monotonically increasing in terms of perceived brightness.
    Also, when printed on a black and white postscript printer, the scheme
    results in a greyscale with monotonically increasing brightness.
    This color scheme is named cubehelix because the r,g,b values produced
    can be visualised as a squashed helix around the diagonal in the
    r,g,b color cube.

    For a unit color cube (i.e. 3-D coordinates for r,g,b each in the
    range 0 to 1) the color scheme starts at (r,g,b) = (0,0,0), i.e. black,
    and finishes at (r,g,b) = (1,1,1), i.e. white. For some fraction *x*,
    between 0 and 1, the color is the corresponding grey value at that
    fraction along the black to white diagonal (x,x,x) plus a color
    element. This color element is calculated in a plane of constant
    perceived intensity and controlled by the following parameters.

    Optional keyword arguments:

      =========   =======================================================
      Keyword     Description
      =========   =======================================================
      gamma       gamma factor to emphasise either low intensity values
                  (gamma < 1), or high intensity values (gamma > 1);
                  defaults to 1.0.
      s           the start color; defaults to 0.5 (i.e. purple).
      r           the number of r,g,b rotations in color that are made
                  from the start to the end of the color scheme; defaults
                  to -1.5 (i.e. -> B -> G -> R -> B).
      h           the hue parameter which controls how saturated the
                  colors are. If this parameter is zero then the color
                  scheme is purely a greyscale; defaults to 1.0.
      =========   =======================================================

    """

    def get_color_function(p0, p1):
        def color(x):
            # Apply gamma factor to emphasise low or high intensity values
            xg = x**gamma

            # Calculate amplitude and angle of deviation from the black
            # to white diagonal in the plane of constant
            # perceived intensity.
            a = h * xg * (1 - xg) / 2

            phi = 2 * np.pi * (s / 3 + r * x)

            return xg + a * (p0 * np.cos(phi) + p1 * np.sin(phi))
        return color

    return {
            'red': get_color_function(-0.14861, 1.78277),
            'green': get_color_function(-0.29227, -0.90649),
            'blue': get_color_function(1.97294, 0.0),
    }

_cubehelix_data = cubehelix()

# This bipolar color map was generated from CoolWarmFloat33.csv of
# "Diverging Color Maps for Scientific Visualization" by Kenneth Moreland. 
# <http://www.cs.unm.edu/~kmorel/documents/ColorMaps/>
_coolwarm_data = {
    'red': [
        (0.0, 0.2298057, 0.2298057),
        (0.03125, 0.26623388, 0.26623388),
        (0.0625, 0.30386891, 0.30386891),
        (0.09375, 0.342804478, 0.342804478),
        (0.125, 0.38301334, 0.38301334),
        (0.15625, 0.424369608, 0.424369608),
        (0.1875, 0.46666708, 0.46666708),
        (0.21875, 0.509635204, 0.509635204),
        (0.25, 0.552953156, 0.552953156),
        (0.28125, 0.596262162, 0.596262162),
        (0.3125, 0.639176211, 0.639176211),
        (0.34375, 0.681291281, 0.681291281),
        (0.375, 0.722193294, 0.722193294),
        (0.40625, 0.761464949, 0.761464949),
        (0.4375, 0.798691636, 0.798691636),
        (0.46875, 0.833466556, 0.833466556),
        (0.5, 0.865395197, 0.865395197),
        (0.53125, 0.897787179, 0.897787179),
        (0.5625, 0.924127593, 0.924127593),
        (0.59375, 0.944468518, 0.944468518),
        (0.625, 0.958852946, 0.958852946),
        (0.65625, 0.96732803, 0.96732803),
        (0.6875, 0.969954137, 0.969954137),
        (0.71875, 0.966811177, 0.966811177),
        (0.75, 0.958003065, 0.958003065),
        (0.78125, 0.943660866, 0.943660866),
        (0.8125, 0.923944917, 0.923944917),
        (0.84375, 0.89904617, 0.89904617),
        (0.875, 0.869186849, 0.869186849),
        (0.90625, 0.834620542, 0.834620542),
        (0.9375, 0.795631745, 0.795631745),
        (0.96875, 0.752534934, 0.752534934),
        (1.0, 0.705673158, 0.705673158)],
    'green': [
        (0.0, 0.298717966, 0.298717966),
        (0.03125, 0.353094838, 0.353094838),
        (0.0625, 0.406535296, 0.406535296),
        (0.09375, 0.458757618, 0.458757618),
        (0.125, 0.50941904, 0.50941904),
        (0.15625, 0.558148092, 0.558148092),
        (0.1875, 0.604562568, 0.604562568),
        (0.21875, 0.648280772, 0.648280772),
        (0.25, 0.688929332, 0.688929332),
        (0.28125, 0.726149107, 0.726149107),
        (0.3125, 0.759599947, 0.759599947),
        (0.34375, 0.788964712, 0.788964712),
        (0.375, 0.813952739, 0.813952739),
        (0.40625, 0.834302879, 0.834302879),
        (0.4375, 0.849786142, 0.849786142),
        (0.46875, 0.860207984, 0.860207984),
        (0.5, 0.86541021, 0.86541021),
        (0.53125, 0.848937047, 0.848937047),
        (0.5625, 0.827384882, 0.827384882),
        (0.59375, 0.800927443, 0.800927443),
        (0.625, 0.769767752, 0.769767752),
        (0.65625, 0.734132809, 0.734132809),
        (0.6875, 0.694266682, 0.694266682),
        (0.71875, 0.650421156, 0.650421156),
        (0.75, 0.602842431, 0.602842431),
        (0.78125, 0.551750968, 0.551750968),
        (0.8125, 0.49730856, 0.49730856),
        (0.84375, 0.439559467, 0.439559467),
        (0.875, 0.378313092, 0.378313092),
        (0.90625, 0.312874446, 0.312874446),
        (0.9375, 0.24128379, 0.24128379),
        (0.96875, 0.157246067, 0.157246067),
        (1.0, 0.01555616, 0.01555616)],
    'blue': [
        (0.0, 0.753683153, 0.753683153),
        (0.03125, 0.801466763, 0.801466763),
        (0.0625, 0.84495867, 0.84495867),
        (0.09375, 0.883725899, 0.883725899),
        (0.125, 0.917387822, 0.917387822),
        (0.15625, 0.945619588, 0.945619588),
        (0.1875, 0.968154911, 0.968154911),
        (0.21875, 0.98478814, 0.98478814),
        (0.25, 0.995375608, 0.995375608),
        (0.28125, 0.999836203, 0.999836203),
        (0.3125, 0.998151185, 0.998151185),
        (0.34375, 0.990363227, 0.990363227),
        (0.375, 0.976574709, 0.976574709),
        (0.40625, 0.956945269, 0.956945269),
        (0.4375, 0.931688648, 0.931688648),
        (0.46875, 0.901068838, 0.901068838),
        (0.5, 0.865395561, 0.865395561),
        (0.53125, 0.820880546, 0.820880546),
        (0.5625, 0.774508472, 0.774508472),
        (0.59375, 0.726736146, 0.726736146),
        (0.625, 0.678007945, 0.678007945),
        (0.65625, 0.628751763, 0.628751763),
        (0.6875, 0.579375448, 0.579375448),
        (0.71875, 0.530263762, 0.530263762),
        (0.75, 0.481775914, 0.481775914),
        (0.78125, 0.434243684, 0.434243684),
        (0.8125, 0.387970225, 0.387970225),
        (0.84375, 0.343229596, 0.343229596),
        (0.875, 0.300267182, 0.300267182),
        (0.90625, 0.259301199, 0.259301199),
        (0.9375, 0.220525627, 0.220525627),
        (0.96875, 0.184115123, 0.184115123),
        (1.0, 0.150232812, 0.150232812)]
    }

# Implementation of Carey Rappaport's CMRmap.
# See `A Color Map for Effective Black-and-White Rendering of Color-Scale Images' by Carey Rappaport
# http://www.mathworks.com/matlabcentral/fileexchange/2662-cmrmap-m
_CMRmap_data = {'red'   : ( (0.000, 0.00, 0.00),
                            (0.125, 0.15, 0.15),
                            (0.250, 0.30, 0.30),
                            (0.375, 0.60, 0.60),
                            (0.500, 1.00, 1.00),
                            (0.625, 0.90, 0.90),
                            (0.750, 0.90, 0.90),
                            (0.875, 0.90, 0.90),
                            (1.000, 1.00, 1.00) ),
                'green' : ( (0.000, 0.00, 0.00),
                            (0.125, 0.15, 0.15),
                            (0.250, 0.15, 0.15),
                            (0.375, 0.20, 0.20),
                            (0.500, 0.25, 0.25),
                            (0.625, 0.50, 0.50),
                            (0.750, 0.75, 0.75),
                            (0.875, 0.90, 0.90),
                            (1.000, 1.00, 1.00) ),
                'blue':   ( (0.000, 0.00, 0.00),
                            (0.125, 0.50, 0.50),
                            (0.250, 0.75, 0.75),
                            (0.375, 0.50, 0.50),
                            (0.500, 0.15, 0.15),
                            (0.625, 0.00, 0.00),
                            (0.750, 0.10, 0.10),
                            (0.875, 0.50, 0.50),
                            (1.000, 1.00, 1.00) )}
_CMRmap_r_data = cm._reverse_cmap_spec(_CMRmap_data)

coolwarm = mcolors.LinearSegmentedColormap('coolwarm', _coolwarm_data)
CMRmap = mcolors.LinearSegmentedColormap('CMRmap', _CMRmap_data)
CMRmap_d = mcolors.LinearSegmentedColormap('CMRmap_d', _CMRmap_data)
CMRmap_d.set_bad((.15,.12,.1,1.))
CMRmap_l = mcolors.LinearSegmentedColormap('CMRmap_l', _CMRmap_data)
CMRmap_l.set_bad((.89,.93,.95,1.))
CMRmap_r = mcolors.LinearSegmentedColormap('CMRmap_r', _CMRmap_r_data)
CMRmap_r_d = mcolors.LinearSegmentedColormap('CMRmap_r_d', _CMRmap_r_data)
CMRmap_r_d.set_bad((.15,.12,.1,1.))
CMRmap_r_l = mcolors.LinearSegmentedColormap('CMRmap_r_l', _CMRmap_r_data)
CMRmap_r_l.set_bad((.89,.93,.95,1.))
cubehelixmap = mcolors.LinearSegmentedColormap('cubehelix', _cubehelix_data)

# diverging colormaps a la K. Moreland

def adjust_hue(Msh, Munsat, axis=-1):
    M,s,h = cie.rollaxis(Msh, axis)
    hnew = h
    if M < Munsat:
        hspin = (s*np.sqrt(Munsat*Munsat - M**2)
                )/(M*np.sin(s))
        if h > -np.pi/3.:
            hnew += hspin
        else:
            hnew -= hspin
    return hnew

def diverging_colors_Msh(x, sRGB1, sRGB2, Mmid=88., mid=.5, adjust=True):
    try:
        iter(x)
    except:
        # x is number of colors
        x = np.linspace(0., 1., x)
    M1,s1,h1 = cie.sRGB2Msh(sRGB1)
    M2,s2,h2 = cie.sRGB2Msh(sRGB2)
    if Mmid > 33:
        Mmid = np.maximum(M1, np.maximum(M2, Mmid))
    else:
        Mmid = np.minimum(M1, np.minimum(M2, Mmid))
    if adjust:
        # adjust hue of unsaturated colors
        hm1 = adjust_hue((M1,s1,h1), Mmid)
        hm2 = adjust_hue((M2,s2,h2), Mmid)
    else:
        hm1 = h1
        hm2 = h2
    M = np.interp(x, [0., mid, 1.], [M1, Mmid, M2])
    s = np.interp(x, [0., mid, 1.], [s1, 0.0 , s2])
    h = np.interp(x, [0., mid, mid, 1.], [h1, hm1, hm2, h2])
    return cie.Msh2sRGB((M,s,h), axis=0).transpose([1,0])

def diverging_colormap_Msh(name, sRGB1, sRGB2, N=256, Mmid=88., mid=.5, adjust=True):
    colors = diverging_colors_Msh(N, sRGB1, sRGB2, Mmid, mid, adjust)
    return mcolors.ListedColormap(colors, name)

def diverging_colors_Lab(x, sRGB1, sRGB2, Lmid=0., mid=.5):
    try:
        iter(x)
    except:
        # x is number of colors
        x = np.linspace(0., 1., x)
    L1,a1,b1 = cie.sRGB2Lab(sRGB1)
    L2,a2,b2 = cie.sRGB2Lab(sRGB2)
    if Lmid > 33:
        Lmid = np.maximum(L1, np.maximum(L2, Lmid))
    else:
        Lmid = np.minimum(L1, np.minimum(L2, Lmid))
    L = np.interp(x, [0., mid, 1.], [L1, Lmid, L2])
    a = np.interp(x, [0., mid, 1.], [a1, 0., a2])
    b = np.interp(x, [0., mid, 1.], [b1, 0., b2])
    return cie.Lab2sRGB((L,a,b), axis=0).transpose([1,0])

def diverging_colormap_Lab(name, sRGB1, sRGB2, N=256, Lmid=0., mid=.5):
    colors = diverging_colors_Lab(N, sRGB1, sRGB2, Lmid, mid)
    return mcolors.ListedColormap(colors, name)

def diverging_colors_Lab_stineman(x, sRGB1, sRGB2, Lmid=0., mid=.5, s=1.2, sL=None, sa=None, sb=None):
    try:
        iter(x)
    except:
        # x is number of colors
        x = np.linspace(0., 1., x)
    L1,a1,b1 = cie.sRGB2Lab(sRGB1)
    L2,a2,b2 = cie.sRGB2Lab(sRGB2)
    if Lmid > 33:
        Lmid = np.maximum(L1, np.maximum(L2, Lmid))
    else:
        Lmid = np.minimum(L1, np.minimum(L2, Lmid))
    if sL is None: sL = s
    if sa is None: sa = s
    if sb is None: sb = s
    L = stineman(x, [0., mid, 1.], [L1, Lmid, L2], [-sL*L1/mid, 0., sL*L2/(1.-mid)])
    a = stineman(x, [0., mid, 1.], [a1, 0., a2], [-sa*a1/mid, 0., sa*a2/(1.-mid)])
    b = stineman(x, [0., mid, 1.], [b1, 0., b2], [-sb*b1/mid, 0., sb*b2/(1.-mid)])
    return cie.Lab2sRGB((L,a,b), axis=0).transpose([1,0])

def diverging_colormap_Lab_stineman(name, sRGB1, sRGB2, N=256, Lmid=0., mid=.5, s=1.2, sL=None, sa=None, sb=None):
    colors = diverging_colors_Lab_stineman(N, sRGB1, sRGB2, Lmid, mid, s, sL, sa, sb)
    return mcolors.ListedColormap(colors, name)

rgbcool = cie.Msh2sRGB((80., 1.08, -1.1))
rgbwarm = cie.Msh2sRGB((80., 1.08, .5))
coolwarm256 = diverging_colormap_Msh('coolwarm256', rgbcool, rgbwarm)
coolwarmblack = diverging_colormap_Lab('coolwarmblack', rgbcool, rgbwarm)

cm.register_cmap('coolwarm256', coolwarm256)
cm.register_cmap('coolwarmblack', coolwarmblack)

# extend gnuplot rainbow beyond purple
_sd = {'red':   lambda x: abs(2.5*x-1),
       'green': lambda x: np.sin((x*1.25-.25)*np.pi),
       'blue':  lambda x: np.where(x>=.2, np.cos((x-.2)*1.25/2*np.pi), np.cos((x-.2)*2.5*np.pi))}
fullrainbow = mcolors.LinearSegmentedColormap('fullrainbow', _sd)
cm.register_cmap('fullrainbow', fullrainbow)

# extend gnuplot rainbow beyond purple
_sd = {'red':   lambda x: np.minimum(abs(2.5*x-.5), abs(2.5*x-3.)),
       'green': lambda x: np.sin((x*1.25)*np.pi),
       'blue':  lambda x: np.where(x<=.8, np.cos(x*1.25/2*np.pi), np.sin(x*2.5*np.pi))}
purplerainbow = mcolors.LinearSegmentedColormap('purplerainbow', _sd)
cm.register_cmap('purplerainbow', purplerainbow)

from ._cm_data import (_viridisSqrtL1_list, _viridisSqrtL0_list,
        _viridisSqrtL05_list, _viridisSqrtL_list, _jeti_list,
        _myparula_list, _fake_parula_list, _parula_list, _parula256_list,
        _viridiswhite_list, _viridiswhite284_list, _cam02light_list)

viridisSqrtL = mcolors.ListedColormap(_viridisSqrtL_list, 'viridisSqrtL')
cm.register_cmap('viridisSqrtL', viridisSqrtL)

viridisSqrtL0 = mcolors.ListedColormap(_viridisSqrtL0_list, 'viridisSqrtL0')
cm.register_cmap('viridisSqrtL0', viridisSqrtL0)

viridisSqrtL05 = mcolors.ListedColormap(_viridisSqrtL05_list, 'viridisSqrtL05')
cm.register_cmap('viridisSqrtL05', viridisSqrtL05)

viridisSqrtL1 = mcolors.ListedColormap(_viridisSqrtL1_list, 'viridisSqrtL1')
cm.register_cmap('viridisSqrtL1', viridisSqrtL1)

jeti = mcolors.LinearSegmentedColormap.from_list('jeti',_jeti_list)
cm.register_cmap('jeti', jeti)

myparula = mcolors.LinearSegmentedColormap.from_list('myparula',_myparula_list)
cm.register_cmap('myparula', myparula)

fake_parula = mcolors.ListedColormap(_fake_parula_list, 'fake_parula')
cm.register_cmap('fake_parula', fake_parula)

parula = mcolors.ListedColormap(_parula_list, 'parula')
cm.register_cmap('parula', parula)

parula256 = mcolors.ListedColormap(_parula256_list, 'parula256')
cm.register_cmap('parula256', parula256)

parulaseg = mcolors.LinearSegmentedColormap.from_list('parulaseg',_parula_list)
cm.register_cmap('parulaseg', parulaseg)

viridiswhite = mcolors.ListedColormap(_viridiswhite_list, 'viridiswhite')
cm.register_cmap('viridiswhite', viridiswhite)

viridiswhite284 = mcolors.ListedColormap(_viridiswhite284_list, 'viridiswhite284')
cm.register_cmap('viridiswhite284', viridiswhite284)

cam02light = mcolors.ListedColormap(_cam02light_list, 'cam02light')
cm.register_cmap('cam02light', cam02light)

#from scipy.interpolate import CubicSpline
#_l = np.array(_parula_list)
#_splines = [CubicSpline(np.linspace(0., 1., 64), _l[:,i]) for i in range(3)]
#_rgb = np.array([_splines[i](np.linspace(0., 1., 256)) for i in range(3)]).T
#parula256 = mcolors.ListedColormap(_rgb, 'parula256')
#cm.register_cmap('parula256', parula256)


def boost(base, f=0., p=.5):
    from colorspacious import cspace_converter
    rgb2uni = cspace_converter("sRGB1", "CAM02-UCS")
    uni2rgb = cspace_converter("CAM02-UCS", "sRGB1")
    if p == .5:
        name = base + 'SqrtL'
    else:
        name = base + 'p{:g}'.format(p)
    name += '{:g}'.format(f)

    cmap = cm.get_cmap(base)
    cmap._init()
    rgb = cmap._lut[:-3, :3]
    Lab = rgb2uni(rgb)

    l = Lab[:,0]/100.
    lo = l.min()
    hi = l.max()
    y0 = lo*f
    p1 = 1./p
    x0 = ((hi - y0)**p1*lo - (lo - y0)**p1*hi)/((hi - y0)**p1 - (lo - y0)**p1)
    lf = y0 + (hi - y0)*((l - x0)/(hi - x0))**p
    Lab[:, 0] = 100*lf

    rgbf = uni2rgb(Lab)

    cmap = mcolors.ListedColormap(rgbf, name)

    return cmap

from .colors import make_seawifs_blue
seawifsblue = make_seawifs_blue()
cm.register_cmap('seawifsblue', seawifsblue)
seawifsblue02 = make_seawifs_blue(.2)
cm.register_cmap('seawifsblue02', seawifsblue02)

#sd = {
#    'red':[(0.,.9769,.9769),(1.,.9769,.9769)],
#    'green':[(0.,.9839,.9839),(.5,.9839,.9839),(1.,0.,0.)],
#    'blue':[(0.,.0805,.0805),(.5,1.,1.),(1.,1.,1.)],
#    }
x = [0., .333333, .666667, 1.]
_sd_ywp = {
    'red':   [(x[0],.9769,.9769),(x[2],.9769,.9769),(x[3],.9769/2,.9769/2)],
    'green': [(x[0],.9839,.9839),(x[1],.9839,.9839),(x[2],0.,0.),(x[3],0.,0.)],
    'blue':  [(x[0],.0805,.0805),(x[1],1.,1.),(x[2],1.,1.),(x[3],.5,.5)],
    }

ywp64 = mcolors.LinearSegmentedColormap('ywp',_sd_ywp, N=64)
cm.register_cmap('ywp64', ywp64)
ywp64._init()
lut = np.concatenate([_parula_list,ywp64._lut[1:-3,:3]], axis=0)
parulawp128 = mcolors.ListedColormap(lut, 'parula-w-p')
cm.register_cmap('parula-w-p', parulawp128)

def makeparulaWP(x):
    Nywp = int(np.round(64/x*(1-x)))
    ywp = mcolors.LinearSegmentedColormap('ywp', _sd_ywp, N=Nywp)
    ywp._init()
    lut = np.concatenate([_parula_list,ywp._lut[1:-3,:3]], axis=0)
    name = 'parula-w-p-{:g}'.format(x)
    cmap = mcolors.ListedColormap(lut, name)
    cm.register_cmap(name, cmap)
    return cmap

