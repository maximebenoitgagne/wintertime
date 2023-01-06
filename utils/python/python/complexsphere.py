from mpmath import mp
import numpy as np

def _sphdist(z1,z2):
    """ geodesic distance (angle) between two points on the sphere
        z1, z2 are complex numbers from stereographic projection
        (with equator on unit circle)
    """
    # rotate z1 to 0: 
    #                     z - z1                       
    #   z -> rot(z) = --------------
    #                 conj(z1)*z + 1
    #
    # then angle is obtained from radius of image of z2 by inverse
    # sterographic projection 
    #
    #   tan(angle/2) = |rot(z2)|
    #
    # south pole requires special treatment
    if mp.isinf(z1):
        return 2*mp.atan2(1,mp.fabs(z2))
    elif mp.isinf(z2):
        return 2*mp.atan2(1,mp.fabs(z1))
    else:
        return 2*mp.atan2(mp.fabs((z2-z1)), mp.fabs(mp.conj(z1)*z2+1))

sphdist = np.frompyfunc(_sphdist,2,1)


def arcangle(z,z1,z2):
    """ angle between great arcs z..z1 and z..z2 """
    # rotate z to 0, then angle is difference between args of z2 and z1
    #
    #                    z2 - z     conj(z)*z1 + 1
    #   angle = arg( -------------- -------------- )
    #                conj(z)*z2 + 1     z1 - z
    # 
    # avoid division by zero (arg(1/z)=arg(conj(z))),
    #
    #   angle = arg((z2 - z)*(z*conj(z2) + 1))
    #               *conj(z1 - z)*(conj(z)*z1 + 1))
    # 
    # south pole requires special treatment
    if mp.isinf(z):
        return mp.arg(z1*mp.conj(z2))
    elif mp.isinf(z1):
        return mp.arg((z2-z)*(z*mp.conj(z2)+1)*mp.conj(z))
    elif mp.isinf(z2):
        return mp.arg(z*mp.conj(z1-z)*(mp.conj(z)*z1+1))
    else:
        return mp.arg((z2-z)*(z*mp.conj(z2)+1)*mp.conj(z1-z)*(mp.conj(z)*z1+1))


def excessslow(*z):
    """ excess angle of great-arc polygon """
    # here we just add up the interior angles and subtract (n-2)*pi
    n = len(z)
    # append first couple points for cyclic indexing
    z = z + z[:2]
    s = 0
    with mp.workprec(2*mp.prec):
        for i in range(n):
            a = mp.modf(arcangle(z[i+1],z[i+2],z[i]), 2*mp.pi)
            s += a

        e = s - (n-2)*mp.pi

    return e


def excess(*z):
    """ excess angle of great-arc polygon """
    # simplified formula with common factors removed
    # the mod 4pi value comes out of the arg of the sqrt 
    # kind of serendipitously

    n = len(z)
    # append first point for cyclic indexing
    z = z + z[:1]
    p = 1
    with mp.workprec(2*mp.prec):
        for i in range(n):
            if mp.isinf(z[i]):
                p *= z[i+1]
            elif mp.isinf(z[i+1]):
                p *= mp.conj(z[i])
            else:
                p *= 1 + mp.conj(z[i])*z[i+1]

        # put in interval (0,4pi)
        #e = 2*mp.modf(mp.arg(p), 2*mp.pi)
        e = 2*(mp.arg(p)%(2*mp.pi))

    return e

excessquad = np.frompyfunc(excess,4,1)


def _z2lat(z):
    return 90 - 360*mp.atan(mp.fabs(z))/mp.pi

z2lat = np.frompyfunc(_z2lat,1,1)


def _z2lon(z):
    return 180*mp.arg(z)/mp.pi

z2lon = np.frompyfunc(_z2lon,1,1)


def _z2ll(z):
    lat = 90 - 360*mp.atan(mp.fabs(z))/mp.pi
    lon = 180*mp.arg(z)/mp.pi
    return lon,lat

z2ll = np.frompyfunc(_z2ll,1,2)


def _z2xyz(z):
    r = mp.fabs(z)
    f = 1/(1 + r)
    return 2*z.real*f, 2*z.imag*f, (1-r)*f

z2xyz = np.frompyfunc(_z2xyz,1,3)


def ll2xyz(lon,lat):
    z = mp.sinpi(lat/180.)
    r2 = mp.cospi(lat/180.)
    x = r2*mp.cospi(lon/180.)
    y = r2*mp.sinpi(lon/180.)
    return x,y,z


def xyz2stereo(x,y,z):
    """ stereographic projection: x,y,z -> X,Y """
    f = 1./(1. + z)
    return (f*x,f*y)


def _ll2z(lon,lat):
    z = mp.sinpi(lat/180.)
    r2 = mp.cospi(lat/180.)
    x = mp.cospi(lon/180.)
    y = mp.sinpi(lon/180.)
    z = np.where(np.asanyarray(z==-1.), -2., z)
    f = np.where(z!=-2., r2/(1. + z), mp.inf)
    return mp.mpc(f*x,f*y)

ll2z = np.frompyfunc(_ll2z,2,1)


