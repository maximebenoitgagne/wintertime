import numpy as np
from scipy.interpolate import PchipInterpolator
from matplotlib.colors import ListedColormap

def isoL(N=256):
    cols = IsoL
    f = PchipInterpolator(np.linspace(0,1,len(cols)), cols, 0)
    return ListedColormap(f(np.linspace(0,1,N)), 'IsoL')

def isoaz(N=256):
    cols = np.r_[IsoAZ, IsoAZ[:1]]
    f = PchipInterpolator(np.linspace(0,1,len(cols)), cols, 0)
    return ListedColormap(f(np.linspace(0,1,N)), 'IsoAZ')

def isoazPuRd(N=256):
    cols = IsoAZ[:-4]
    f = PchipInterpolator(np.linspace(0,1,len(cols)), cols, 0)
    return ListedColormap(f(np.linspace(0,1,N)), 'IsoAZPuRd')

IsoL =      [[0.9102,    0.2236,    0.8997],
             [0.4027,    0.3711,    1.0000],
             [0.0422,    0.5904,    0.5899],
             [0.0386,    0.6206,    0.0201],
             [0.5441,    0.5428,    0.0110],
             [1.0000,    0.2288,    0.1631]]
 
CubicL =    [[0.4706,         0,    0.5216],
             [0.5137,    0.0527,    0.7096],
             [0.4942,    0.2507,    0.8781],
             [0.4296,    0.3858,    0.9922],
             [0.3691,    0.5172,    0.9495],
             [0.2963,    0.6191,    0.8515],
             [0.2199,    0.7134,    0.7225],
             [0.2643,    0.7836,    0.5756],
             [0.3094,    0.8388,    0.4248],
             [0.3623,    0.8917,    0.2858],
             [0.5200,    0.9210,    0.3137],
             [0.6800,    0.9255,    0.3386],
             [0.8000,    0.9255,    0.3529],
             [0.8706,    0.8549,    0.3608],
             [0.9514,    0.7466,    0.3686],
             [0.9765,    0.5887,    0.3569]]
         
CubicYF =   [[0.5151,    0.0482,    0.6697],
             [0.5199,    0.1762,    0.8083],
             [0.4884,    0.2912,    0.9234],
             [0.4297,    0.3855,    0.9921],
             [0.3893,    0.4792,    0.9775],
             [0.3337,    0.5650,    0.9056],
             [0.2795,    0.6419,    0.8287],
             [0.2210,    0.7123,    0.7258],
             [0.2468,    0.7612,    0.6248],
             [0.2833,    0.8125,    0.5069],
             [0.3198,    0.8492,    0.3956],
             [0.3602,    0.8896,    0.2919],
             [0.4568,    0.9136,    0.3018],
             [0.6033,    0.9255,    0.3295],
             [0.7066,    0.9255,    0.3414],
             [0.8000,    0.9255,    0.3529]]


LinearL =   [[0.0143,     0.0143,  0.0143],
             [0.1413,     0.0555,  0.1256],
             [0.1761,     0.0911,  0.2782],
             [0.1710,     0.1314,  0.4540],
             [0.1074,     0.2234,  0.4984],
             [0.0686,     0.3044,  0.5068],
             [0.0008,     0.3927,  0.4267],
             [0.0000,     0.4763,  0.3464],
             [0.0000,     0.5565,  0.2469],
             [0.0000,     0.6381,  0.1638],
             [0.2167,     0.6966,  0.0000],
             [0.3898,     0.7563,  0.0000],
             [0.6912,     0.7795,  0.0000],
             [0.8548,     0.8041,  0.4555],
             [0.9712,     0.8429,  0.7287],
             [0.9692,     0.9273,  0.8961]]


LinLhot =   [[0.0225,     0.0121,  0.0121],
             [0.1927,     0.0225,  0.0311],
             [0.3243,     0.0106,  0.0000],
             [0.4463,     0.0000,  0.0091],
             [0.5706,     0.0000,  0.0737],
             [0.6969,     0.0000,  0.1337],
             [0.8213,     0.0000,  0.1792],
             [0.8636,     0.0000,  0.0565],
             [0.8821,     0.2555,  0.0000],
             [0.8720,     0.4182,  0.0000],
             [0.8424,     0.5552,  0.0000],
             [0.8031,     0.6776,  0.0000],
             [0.7659,     0.7870,  0.0000],
             [0.8170,     0.8296,  0.0000],
             [0.8853,     0.8896,  0.4113],
             [0.9481,     0.9486,  0.7165]]
         
IsoAZ =     [[1.0000,     0.2627,  1.0000],
             [0.9765,     0.2941,  1.0000],
             [0.9373,     0.3255,  1.0000],
             [0.8824,     0.3647,  1.0000],
             [0.8157,     0.4078,  1.0000],
             [0.7451,     0.4549,  1.0000],
             [0.6471,     0.5137,  0.9961],
             [0.4902,     0.5882,  0.9765],
             [0.3020,     0.6745,  0.9412],
             [0.1333,     0.7490,  0.9020],
             [0.0235,     0.8000,  0.8510],
             [0.0000,     0.8196,  0.7961],
             [0.0000,     0.8275,  0.6980],
             [0.0000,     0.8314,  0.5725],
             [0.0000,     0.8353,  0.4353],
             [0.0000,     0.8392,  0.3137],
             [0.0000,     0.8392,  0.2275],
             [0.0588,     0.8353,  0.1647],
             [0.1961,     0.8196,  0.1059],
             [0.3725,     0.7961,  0.0549],
             [0.5490,     0.7686,  0.0196],
             [0.6824,     0.7412,  0.0000],
             [0.7647,     0.6941,  0.0039],
             [0.8431,     0.6157,  0.0275],
             [0.9098,     0.5176,  0.0627],
             [0.9647,     0.4275,  0.1098],
             [0.9961,     0.3569,  0.1647],
             [1.0000,     0.3255,  0.2275],
             [1.0000,     0.3059,  0.3294],
             [1.0000,     0.2863,  0.4667],
             [1.0000,     0.2745,  0.6314],
             [1.0000,     0.2667,  0.8235]]
         
IsoAZ180 =  [[0.8658,     0.5133,  0.6237],
             [0.8122,     0.5287,  0.7241],
             [0.7156,     0.5599,  0.8091],
             [0.5800,     0.5973,  0.8653],
             [0.4109,     0.6327,  0.8834],
             [0.2041,     0.6607,  0.8603],
             [0.0000,     0.6887,  0.8071],
             [0.0000,     0.6938,  0.7158],
             [0.2144,     0.6885,  0.6074],
             [0.3702,     0.6803,  0.5052],
             [0.4984,     0.6637,  0.4192],
             [0.6123,     0.6391,  0.3635],
             [0.7130,     0.6074,  0.3492],
             [0.7958,     0.5719,  0.3787],
             [0.8532,     0.5389,  0.4445],
             [0.8773,     0.5170,  0.5348],
             [0.8658,     0.5133,  0.6237],
             [0.8122,     0.5287,  0.7241],
             [0.7156,     0.5599,  0.8091],
             [0.5800,     0.5973,  0.8653],
             [0.4109,     0.6327,  0.8834],
             [0.2041,     0.6607,  0.8603],
             [0.0000,     0.6887,  0.8071],
             [0.0000,     0.6938,  0.7158],
             [0.2144,     0.6885,  0.6074],
             [0.3702,     0.6803,  0.5052],
             [0.4984,     0.6637,  0.4192],
             [0.6123,     0.6391,  0.3635],
             [0.7130,     0.6074,  0.3492],
             [0.7958,     0.5719,  0.3787],
             [0.8532,     0.5389,  0.4445],
             [0.8773,     0.5170,  0.5348]]
          
Swtth =     [[1.0000,     0.5395,  1.0000],
             [1.0000,     0.5060,  1.0000],
             [1.0000,     0.4721,  1.0000],
             [1.0000,     0.4377,  1.0000],
             [0.9746,     0.4026,  1.0000],
             [0.8759,     0.3666,  1.0000],
             [0.7774,     0.3294,  1.0000],
             [0.6789,     0.2906,  1.0000],
             [0.5802,     0.2499,  1.0000],
             [0.4803,     0.2065,  1.0000],
             [0.3772,     0.1589,  1.0000],
             [0.2644,     0.1033,  1.0000],
             [0.1100,     0.0220,  1.0000],
             [0.0000,     0.0868,  0.9879],
             [0.1235,     0.1246,  1.0000],
             [0.1917,     0.2207,  1.0000],
             [0.2187,     0.3086,  1.0000],
             [0.2246,     0.3914,  1.0000],
             [0.2179,     0.4698,  1.0000],
             [0.2037,     0.5446,  1.0000],
             [0.1847,     0.6166,  1.0000],
             [0.1618,     0.6864,  1.0000],
             [0.1342,     0.7546,  1.0000],
             [0.0988,     0.8218,  1.0000],
             [0.0421,     0.8882,  1.0000],
             [0.0000,     0.9560,  0.9951],
             [0.0000,     0.9724,  0.9345],
             [0.0000,     0.9348,  0.8244],
             [0.0000,     0.8956,  0.7181],
             [0.0000,     0.8551,  0.6170],
             [0.0000,     0.8137,  0.5236],
             [0.0000,     0.7718,  0.4409],
             [0.0000,     0.7294,  0.3730],
             [0.0000,     0.6868,  0.3235],
             [0.0000,     0.6438,  0.2933],
             [0.0000,     0.5996,  0.2752],
             [0.0000,     0.5517,  0.2474],
             [0.0000,     0.5003,  0.2065],
             [0.0000,     0.4455,  0.1476],
             [0.0000,     0.4723,  0.1742],
             [0.0000,     0.5231,  0.2118],
             [0.0000,     0.5684,  0.2279],
             [0.0000,     0.6074,  0.2202],
             [0.0000,     0.6389,  0.1747],
             [0.0374,     0.6634,  0.0000],
             [0.2443,     0.7077,  0.0000],
             [0.3707,     0.7499,  0.0000],
             [0.4848,     0.7901,  0.0000],
             [0.5951,     0.8281,  0.0000],
             [0.7044,     0.8642,  0.0000],
             [0.8139,     0.8982,  0.0000],
             [0.9237,     0.9305,  0.0000],
             [0.9273,     0.8577,  0.0000],
             [0.9299,     0.7840,  0.0000],
             [0.9311,     0.7089,  0.0000],
             [0.9303,     0.6322,  0.0000],
             [0.9268,     0.5533,  0.0000],
             [0.9197,     0.4714,  0.0000],
             [0.9077,     0.3853,  0.0000],
             [0.8897,     0.2921,  0.0000],
             [0.8643,     0.1826,  0.0000],
             [0.8319,     0.0000,  0.0159],
             [0.8020,     0.0000,  0.1461],
             [0.7606,     0.0000,  0.1769]]
