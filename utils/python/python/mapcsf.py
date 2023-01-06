import numpy as np
import scipy.special as sf

# map sphere to cube
#
# ll -> xyz -> f,stereo -> f,halfplane -> f,square -> f,tansquare == fxy
#                                                  -> jfi -> csflat
#
########################################################################
# cs face -> square
#
# stereo -> half plane -> square

_2o3 = 2./3
_tan2o3 = np.tan(2./3)
_r4 = np.sqrt(1j)
_rt3 = np.sqrt(3)
_2rt3 = 2*_rt3
_2prt3 = 2 + _rt3
_2mrt3 = 2 - _rt3
_Agcs = 2*np.sqrt(3j*_rt3)
_Afsquare = sf.gamma(.75)/(sf.gamma(1.25)*sf.gamma(.5))

#def stereo2half(x,y):
#    """ map cs face (symmetric to 0) to half plane
#        (+1+i)*xd -> -1
#        (-1+i)*xd -> 0
#        (-1-i)*xd -> 1
#        (+1-i)*xd -> inf
#        xd = (sqrt(3)-1)/2
#    """
#    z = y-1j*x
#    z2 = z*z
#    app = (2+_rt3+1j*z2)**1.5
#    apm = (2+_rt3-1j*z2)**1.5
#    amp = (2-_rt3+1j*z2)**1.5
#    amm = (2-_rt3-1j*z2)**1.5
#    if -x > y:
#        return 1j*app*amm/(apm*amp+_Agcs*z*(z2*z2-1))
#    else:
#        return 1j*(apm*amp-_Agcs*z*(z2*z2-1))/(app*amm)

def stereo2half(x,y):
    """ map cs face (symmetric to 0) to half plane
        (+1+i)*xd -> inf
        (-1+i)*xd -> -1
        (-1-i)*xd -> 0
        (+1-i)*xd -> 1
        xd = (sqrt(3)-1)/2
    """
    z = y - 1j*x
    z2j = 1j*z*z
    mz4 = z2j*z2j
    appamm = ((_2prt3+z2j)*(_2mrt3-z2j))**1.5
    apmamp = ((_2prt3-z2j)*(_2mrt3+z2j))**1.5
#    appamm = (1 - mz4 - _2rt3*z2j)**1.5
#    apmzmp = (1 - mz4 + _2rt3*z2j)**1.5
    res = np.where(-x > y, 1j*appamm/(apmamp-_Agcs*z*(mz4+1)),
                         1j*(apmamp+_Agcs*z*(mz4+1))/(appamm))
    return res


def half2square(z):
    """ map from half plane to square 0,1,1+i,i using 2F1
        (could use ellipticF)
        maps -1,0,1,inf -> i,0,1,1+i
        don't use near unit circle (away from -1+)!
    """
#    print np.abs(z), np.angle(z)/np.pi*2
    z2 = z*z
    res = 1j**.5*_Afsquare*(z/1j)**.5*sf.hyp2f1(.25,.5,1.25,z2)
    x,y = res.real, res.imag
    msk = (z.real < 0) & (y < x)
    X = np.where(msk, y, x)
    Y = np.where(msk, x, y)
    return X,Y


def stereo2square(x,y):
    """ map cs face in stereographic coordinates to square
        (+1+i)*xd -> i
        (-1+i)*xd -> 0
        (-1-i)*xd -> 1
        (+1-i)*xd -> 1+i
        xd = (sqrt(3)-1)/2
    """
    # make sure we stay where half2square is fast and reliable
    msky = y <= 0.
    mskx = x <= 0.
    x = np.where(mskx, x, -x)
    y = np.where(msky, y, -y)
    x,y = half2square(stereo2half(x,y))
    x = np.where(mskx, x, 1.-x)
    y = np.where(msky, y, 1.-y)
    return x,y

#####################################################################

def _tanunscale(t):
    tpm = 2*t-1
    return .5*(1+1.5*np.arctan(tpm*_tan2o3))

tanunscale = np.frompyfunc(_tanunscale,1,1)


def stereo2tansquare(x,y):
    x,y = stereo2square(x,y)
    x = tanunscale(x)
    y = tanunscale(y)
    return x,y

#####################################################################
#
# xyz -> facestereo -> square -> tansquare == fxy
#

def stereo2xyz(X,Y):
    """ inverse of stereographic projection: X,Y -> x,y,z """
    r2 = X*X + Y*Y
    f = 1./(1. + r2)
    x = 2*X*f
    y = 2*Y*f
    z = (1.-r2)*f
    # fix normalization, just in case
    r3 = np.sqrt(x*x+y*y+z*z)
    return x/r3, y/r3, z/r3


def xyz2stereo(x,y,z):
    """ stereographic projection: x,y,z -> X,Y """
    f = 1./(1. + z)
    return (f*x,f*y)


def xyz2facestereo(x,y,z):
    """ f,X,Y = xyz2facestereo(x,y,z)
    
    find face of point on sphere and project stereographically
    """
    ax = np.abs(x)
    ay = np.abs(y)
    az = np.abs(z)
    mskx = (y != x) & (z != x)
    mskyz = z != y
    msk0 = ( x >= ay) & ( x >= az) & mskx
    msk3 = (-x >= ay) & (-x >= az) & mskx
    msk1 = ( y >= az) & mskyz
    msk4 = (-y >= az) & mskyz
    msk2 = z > 0
    f = (1-msk0)*(msk3*3 + (1-msk3)*(msk1 + (1-msk1)*(msk4*4 + (1-msk4)*(msk2*2 + (1-msk2)*5))))
    xnew = np.choose(f, ( y, -x, -x, -z, -z,  y))
    ynew = np.choose(f, ( z,  z, -y, -y,  x,  x))
    znew = np.choose(f, ( x,  y,  z, -x, -y, -z))
    X,Y = xyz2stereo(xnew, ynew, znew)

    return f,X,Y


def xyz2fxy(x,y,z):
    """ f,zcs = xyz2fxy(x,y,z)

    map 3d hom. coordinates on sphere to face number (0,..,5) and x,y in [0,1] on cube
    """
    f,X,Y = xyz2facestereo(x,y,z)
    X,Y = stereo2tansquare(X,Y)
    return np.asarray(f,int),np.asfarray(X),np.asfarray(Y)


######################################################################
#
# ll -> xyz -> fxy -> jfi -> csflat
#

def ll2xyz(lon,lat):
    z = np.sin(np.pi*lat/180)
    r2 = np.cos(np.pi*lat/180)
    x = r2*np.cos(np.pi*lon/180)
    y = r2*np.sin(np.pi*lon/180)
    return x,y,z


def ll2fxy(lon,lat):
    """ f,x,y = ll2fxy(lon,lat)

    map lon-lat coordinates to face number (0,..,5) and x,y in [0,1] on cube
    """
    return xyz2fxy(*ll2xyz(lon,lat))


def ll2jfi(lon,lat,ncs):
    """ j,f,i = ll2jfi(lon,lat)

    map lon-lat coordinates to indices j,face,i on ncs cube
    """
    f,x,y = ll2fxy(lon,lat)
    i = np.clip(np.asarray(np.floor(ncs*x), int), 0, ncs-1) + 0
    j = np.clip(np.asarray(np.floor(ncs*y), int), 0, ncs-1) + 0
    f = f + 0
    return j,f,i


def ll2csflat(lon,lat,ncs):
    """ csind = ll2csflat(lon,lat)

    map lon-lat coordinates to flat index on cube with sides ncs
    """
    j,f,i = ll2jfi(lon,lat,ncs)
    return (j*6+f)*ncs+i

