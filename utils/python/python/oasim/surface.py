from numpy import exp, log, cos, sin, tan, arcsin, pi, clip, zeros
from .data import water

_rad = 180.0/pi
rn = 1.341        # index of refraction of pure seawater
roair = 1.2E3     # density of air g/m3

nlt = len(water)

a0,a1,a2,a3 = 0.9976,0.2194,5.554E-2,6.7E-3
b0,b1,b2,b3 = 5.026,-0.01138,9.552E-6,-2.698E-9

def compute_wfac():
    wfac = zeros((nlt,))
    for nl in range(nlt):
        rlam = float(water.lam[nl])
        if water.lam[nl] < 900:
            t = exp(-(water.a[nl]+0.5*water.b[nl]))
            tlog = log(1.0E-36+t)
            fac = a0 + a1*tlog + a2*tlog*tlog + a3*tlog*tlog*tlog
            wfac[nl] = min(fac, 1.0)
            wfac[nl] = max(fac, 0.0)
        else:
            fac = b0 + b1*rlam + b2*rlam*rlam + b3*rlam*rlam*rlam
            wfac[nl] = max(fac, 0.0)
    return wfac

wfac = compute_wfac()


def refract(solz):
    ''' compute inverse cosine of below-water zenith angle from
    above-water zenith angle in degrees
    '''
    rsolzw = arcsin(sin(solz/_rad)/rn)
    rmud = clip(1.0/cos(rsolzw), 0.0, 1.5)
    return rmud


def reflectance(theta,ws):
    '''
    Computes surface reflectance for direct (rod) and diffuse (ros)
    components separately, as a function of theta, wind speed or
    stress.
    Includes spectral dependence of foam reflectance derived from Frouin
    et al., 1996 (JGR)
    
    Arguments

    theta :: solar zenith angle
    ws    :: wind speed

    Returns

    rod :: reflectance for collimated (direct/beam) light
    ros :: reflectance for diffuse light
    '''
    #  Foam and diffuse reflectance
    if ws > 4.0:
        if ws <= 7.0:
            cn = 6.2E-4 + 1.56E-3/ws
            rof = roair*cn*2.2E-5*ws*ws - 4.0E-4
        else:
            cn = 0.49E-3 + 0.065E-3*ws
            rof = (roair*cn*4.5E-5 - 4.0E-5)*ws*ws
        rosps = 0.057
    else:
        rof = 0.0
        rosps = 0.066

    #  Direct
    #   Fresnel reflectance for theta < 40, ws < 2 m/s
    if theta < 40.0 or ws < 2.0:
        if theta == 0.0:
            rospd = 0.0211
        else:
            rtheta = theta/_rad
            sintr = sin(rtheta)/rn
            rthetar = arcsin(sintr)
            rmin = rtheta - rthetar
            rpls = rtheta + rthetar
            sinrmin = sin(rmin)
            sinrpls = sin(rpls)
            tanrmin = tan(rmin)
            tanrpls = tan(rpls)
            sinp = (sinrmin*sinrmin)/(sinrpls*sinrpls)
            tanp = (tanrmin*tanrmin)/(tanrpls*tanrpls)
            rospd = 0.5*(sinp + tanp)
    else:
        # Empirical fit otherwise
        a = 0.0253
        b = -7.14E-4*ws + 0.0618
        rospd = a*exp(b*(theta-40.0))

    #  Reflectance totals
    rod = rospd + rof*wfac
    ros = rosps + rof*wfac

    return rod,ros

