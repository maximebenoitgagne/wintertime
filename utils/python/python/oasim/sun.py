from numpy import exp, cos, sin, mod, sqrt, pi, zeros
from .util import julian_day

_radeg = 180.0/pi
_xk    = 0.0056932  # Constant of aberration

def sun_vector(iday, iyr, gmt, imon=1):
    '''
    compute sun vector in earth-fixed coordinate system

    Together with a local zenith-pointing vector, this can be used
    to compute the solar zenith angle.

    Original comment:
    Given year, day of year, time in hours (GMT) and latitude and
    longitude, returns an accurate solar zenith and azimuth angle.
    Based on IAU 1976 Earth ellipsoid.  Method for computing solar
    vector and local vertical from Patt and Gregg, 1994, Int. J.
    Remote Sensing.  Only outputs solar zenith angle.  This version
    utilizes a pre-calculation of the local up, north, and east
    vectors, since the locations where the solar zenith angle are
    calculated in the model are fixed.

    iday :: yearday (1..366)
    iyr  :: year (e.g., 1970)
    gmt  :: time of day in hours (e.g., 12.5)

    Returns

    sunvec(3) :: sun vector in earth-fixed coordinate system
    rs        :: correction to earth-sun distance for given time
    '''
    sec = gmt*3600.0

    #  Compute floating point days since Jan 1.5, 2000
    #   Note that the Julian day starts at noon on the specified date
    rjd = float(julian_day(iyr,imon,iday))
    t = rjd - 2451545.0 + (sec-43200.0)/86400.0
    xls,gs,xlm,omega = ephparms(t)
    dpsi,eps = nutate(t,xls,gs,xlm,omega)

    #  Compute sun vector
    #   Compute unit sun vector in geocentric inertial coordinates
    suni,rs = sun2000(t,sec,xls,gs,xlm,omega,dpsi,eps)

    #   Get Greenwich mean sidereal angle
    fday = sec/86400.0
    gha = gha2000(t, fday, dpsi, eps)
    ghar = gha/_radeg

    #  Transform Sun vector into geocentric rotating frame
    v = zeros((3,))
    v[0] = suni[0]*cos(ghar) + suni[1]*sin(ghar)
    v[1] = suni[1]*cos(ghar) - suni[0]*sin(ghar)
    v[2] = suni[2]

    return v, rs


def sun2000(t, sec, xls, gs, xlm, omega, dpsi, eps):
    '''
    This subroutine computes the Sun vector in geocentric inertial
    (equatorial) coodinates.  It uses the model referenced in The
    Astronomical Almanac for 1984, Section S (Supplement) and documented
    in Exact closed-form geolocation algorithm for Earth survey
    sensors, by F.S. Patt and W.W. Gregg, Int. Journal of Remote
    Sensing, 1993.  The accuracy of the Sun vector is approximately 0.1
    arcminute.

    Arguments:
 
    Name    Type    I/O     Description
    --------------------------------------------------------
    IYR     I*4      I      Year, four digits (i.e, 1993)
    IDAY    I*4      I      Day of year (1-366)
    SEC     R*8      I      Seconds of day
    SUN(3)  R*8      O      Unit Sun vector in geocentric inertial
                             coordinates of date
    RS      R*8      O      Magnitude of the Sun vector (AU)
 
    Subprograms referenced:
 
    JD              Computes Julian day from calendar date
    EPHPARMS        Computes mean solar longitude and anomaly and
                     mean lunar lontitude and ascending node
    NUTATE          Compute nutation corrections to lontitude and
                     obliquity
 
    Coded by:  Frederick S. Patt, GSC, November 2, 1992
    Modified to include Earth constants subroutine by W. Gregg,
    May 11, 1993.
    '''
    #  Compute planet mean anomalies
    #   Venus Mean Anomaly
    g2 = 50.40828 + 1.60213022*t
    g2 = mod(g2,360.0)

    #   Mars Mean Anomaly
    g4 = 19.38816 + 0.52402078*t
    g4 = mod(g4,360.0)

    #  Jupiter Mean Anomaly
    g5 = 20.35116 + 0.08309121*t
    g5 = mod(g5,360.0)

    #  Compute solar distance (AU)
    rs = 1.00014 - 0.01671*cos(gs/_radeg) - 0.00014*cos(2.0*gs/_radeg)

    #  Compute Geometric Solar Longitude
    dls = ( (6893.0 - 4.6543463E-4*t)*sin(gs/_radeg)
            + 72.0*sin(2.0*gs/_radeg)
            -  7.0*cos((gs - g5)/_radeg)
            +  6.0*sin((xlm - xls)/_radeg)
            +  5.0*sin((4.0*gs - 8.0*g4 + 3.0*g5)/_radeg)
            -  5.0*cos((2.0*gs - 2.0*g2)/_radeg)
            -  4.0*sin((gs - g2)/_radeg)
            +  4.0*cos((4.0*gs - 8.0*g4 + 3.0*g5)/_radeg)
            +  3.0*sin((2.0*gs - 2.0*g2)/_radeg)
            -  3.0*sin(g5/_radeg)
            -  3.0*sin((2.0*gs - 2.0*g5)/_radeg)  #arcseconds
        )

    xlsg = xls + dls/3600.0

    #  Compute Apparent Solar Longitude; includes corrections for nutation
    #   in longitude and velocity aberration
    xlsa = xlsg + dpsi - _xk/rs

    #   Compute unit Sun vector
    sunvec = zeros((3,))
    sunvec[0] = cos(xlsa/_radeg)
    sunvec[1] = sin(xlsa/_radeg)*cos(eps/_radeg)
    sunvec[2] = sin(xlsa/_radeg)*sin(eps/_radeg)

    return sunvec,rs


def gha2000(t,fday,dpsi,eps):
    '''
    This subroutine computes the Greenwich hour angle in degrees for the
    input time.  It uses the model referenced in The Astronomical Almanac
    for 1984, Section S (Supplement) and documented in Exact
    closed-form geolocation algorithm for Earth survey sensors, by
    F.S. Patt and W.W. Gregg, Int. Journal of Remote Sensing, 1993.
    It includes the correction to mean sideral time for nutation
    as well as precession.

    Calling Arguments

    Name         Type    I/O     Description
  
    iyr          I*4      I      Year (four digits)
    day          R*8      I      Day (time of day as fraction)

    Returns

    gha          R*8      O      Greenwich hour angle (degrees)

    Program written by:     Frederick S. Patt
                            General Sciences Corporation
                            November 2, 1992

    t    :: floating point days since Jan 1.5, 2000
    fday :: fractional day
    dPsi :: Nutation in longitude (degrees)
    Eps  :: Obliquity of the Ecliptic (degrees)
    '''
    #  Compute Greenwich Mean Sidereal Time (degrees)
    gmst = 100.4606184 + 0.9856473663*t + 2.908E-13*t*t

    #  Include apparent time correction and time-of-day
    gha = gmst + dpsi*cos(eps/_radeg) + fday*360.0
    gha = mod(gha, 360.0)

    return gha


def ephparms(t):
    '''
    This subroutine computes ephemeris parameters used by other Mission
    Operations routines:  the solar mean longitude and mean anomaly, and
    the lunar mean longitude and mean ascending node.  It uses the model
    referenced in The Astronomical Almanac for 1984, Section S
    (Supplement) and documented and documented in Exact closed-form
    geolocation algorithm for Earth survey sensors, by F.S. Patt and
    W.W. Gregg, Int. Journal of Remote Sensing, 1993.  These parameters
    are used to compute the solar longitude and the nutation in
    longitude and obliquity.

    Calling Arguments

    Name         Type    I/O     Description
  
    t            R*8      I      Time in days since January 1, 2000 at
                                  12 hours UT

    Return:

    xls          R*8      O      Mean solar longitude (degrees)
    gs           R*8      O      Mean solar anomaly (degrees)
    xlm          R*8      O      Mean lunar longitude (degrees)
    omega        R*8      O      Ascending node of mean lunar orbit
                                  (degrees)
  
  
         Program written by:     Frederick S. Patt
                                 General Sciences Corporation
                                 November 2, 1992
    '''
    # Sun Mean Longitude
    xls = 280.46592 + 0.9856473516*t
    xls = mod(xls,360.0)

    # Sun Mean Anomaly
    gs = 357.52772 + 0.9856002831*t
    gs = mod(gs,360.0)

    # Moon Mean Longitude
    xlm = 218.31643 + 13.17639648*t
    xlm = mod(xlm,360.0)

    # Ascending Node of Moons Mean Orbit
    omega = 125.04452 - 0.0529537648*t
    omega = mod(omega,360.0)

    return xls, gs, xlm, omega


def nutate(t,xls,gs,xlm,omega):
    '''
    This subroutine computes the nutation in longitude and the obliquity
    of the ecliptic corrected for nutation.  It uses the model referenced
    in The Astronomical Almanac for 1984, Section S (Supplement) and
    documented in Exact closed-form geolocation algorithm for Earth
    survey sensors, by F.S. Patt and W.W. Gregg, Int. Journal of
    Remote Sensing, 1993.  These parameters are used to compute the
    apparent time correction to the Greenwich Hour Angle and for the
    calculation of the geocentric Sun vector.  The input ephemeris
    parameters are computed using subroutine ephparms.  Terms are
    included to 0.1 arcsecond.

    Calling Arguments

    Name         Type    I/O     Description
  
    t            R*8      I      Time in days since January 1, 2000 at
                                 12 hours UT
    xls          R*8      I      Mean solar longitude (degrees)
    gs           R*8      I      Mean solar anomaly   (degrees)
    xlm          R*8      I      Mean lunar longitude (degrees)
    Omega        R*8      I      Ascending node of mean lunar orbit
                                 (degrees)

    Returns

    dPsi         R*8      O      Nutation in longitude (degrees)
    Eps          R*8      O      Obliquity of the Ecliptic (degrees)
                                 (includes nutation in obliquity)

         Program written by:     Frederick S. Patt
                                 General Sciences Corporation
                                 October 21, 1992
    '''

    # Nutation in Longitude
    dpsi = (- 17.1996*sin(omega/_radeg)
            +  0.2062*sin(2.0*omega/_radeg)
            -  1.3187*sin(2.0*xls/_radeg)
            +  0.1426*sin(gs/_radeg)
            -  0.2274*sin(2.0*xlm/_radeg)
           )

    # Mean Obliquity of the Ecliptic
    epsm = 23.439291 - 3.560E-7*t

    # Nutation in Obliquity
    deps = 9.2025*cos(omega/_radeg) + 0.5736*cos(2.0*xls/_radeg)

    # True Obliquity of the Ecliptic
    eps = epsm + deps/3600.0

    dpsi = dpsi/3600.0

    return dpsi, eps

