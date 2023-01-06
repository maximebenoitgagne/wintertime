import numpy as np
from numpy import sin, cos, pi, arcsin, arccos
import string

digits = '0123456789abcdefghijklmnopqrstuvxyzABCDEFGHIJKLMNOPQRSTUVXYZ'

def ionum(s):
    try:
        i = int(s)
    except ValueError:
        try:
            i0 = int(s[0])
        except ValueError:
            i0 = digits[10:].index(s[0])
            i1 = digits.index(s[1])
            i = 620 + i0*62 + i1
        else:
            i1 = digits[10:].index(s[1])
            i = 100 + i0*52 + i1
    return i


def iofmt(i):
    if i < 100:
        return '{0:02d}'.format(i)
    elif i < 620:
        a,b = divmod(i-100, 52)
        return '{0}{1}'.format(a, digits[10+b])
    else:
        a,b = divmod(i-620, 62)
        return '{0}{1}'.format(digits[10+a], digits[b])


def names(np=78,nz=2):
    res = [ 'PO4', 'NO3', 'FeT', 'SiO2', 'DOP', 'DON', 'DOFe']
    for i in range(1,nz+1):
        res.extend([ s%i for s in ['Zoo%dP', 'Zoo%dN', 'Zoo%dFe', 'Zoo%dSi'] ])

    res.extend(['POP', 'PON', 'POFe', 'POSi', 'NH4', 'NO2'])
    for i in range(1,np+1):
        res.extend(['Phy%02d'%i])

    return res
#'Chl01',
#'...',
#'DIC',
#'DOC',
#'POC',
#'PIC',
#'ALK',
#'O2',
#'Zoo1C',
#'Zoo2C',

thresh0 = 1e-12   # uM P  thresh for sum_i P
thresh1 = 1e-8    # uM P  thresh for P
thresh2 = 1e-3    #       thresh for P / sum_i P
thresh3 = .999    #       biomass fraction
thresh4 = 1e-5    #       thresh for P / max_i P
threshsh = 1e-12  # uM P  just for numerics
threshrich = 1e-8 # uM P  abs.threshold for P

def calcdiversities(phy, rfac=1., rfacsh=1., rfacrich=None):
    ''' overwrites phy! '''
    if rfacrich is None: rfacrich = rfac
    nphy = phy.shape[0]
    shape = phy.shape[1:]

    phy[phy<0] = 0.
    total = phy.sum(0)
    mx = phy.max(0)

    diver1 = np.zeros(shape, int)
    diver2 = np.zeros(shape, int)
#    diver3 = np.zeros(shape, int)
    diver4 = np.zeros(shape, int)
    shannon = np.zeros(shape)
    simpson = np.zeros(shape)
    richness = np.zeros(shape, int)
#    present = np.zeros(shape, bool)

    valid = total >= thresh0*rfac
    for i in range(nphy):
#        present[...] = 0
#        for k in range(kmax):
#            present |= p[i,k,:,:] > thresh1
#        diver1[:,:] += present
        diver1[...] += phy[i] > thresh1*rfac
        diver2[...] += valid & (phy[i] > thresh2*total)
        diver4[...] += valid & (phy[i] > thresh4*mx)

    diver3 = calcdiver3(phy.copy())

    # for consistency recompute total with threshold
    phy[phy<threshsh*rfacsh] = 0
    total = phy.sum(0)
    for i in range(nphy):
        valid = phy[i] >= threshsh*rfacsh
        tmp = phy[i,valid]/total[valid]
        shannon[valid] += tmp*np.log(tmp)
        simpson[valid] += tmp*tmp
        valid = phy[i] >= threshrich*rfacrich
        richness[valid] += 1

    shannon *= -1
    valid = simpson != 0
    simpson[valid] = 1./simpson[valid]

#    biotot = np.zeros(shape)
#    for i2 in range(nphy):
#        imx = np.argmax(phy,axis=0)
#        mx = np.max(phy,axis=0)
#        diver3[biotot < thresh3*total] += 1
#        biotot += mx
#        for ind in np.ndindex(shape):
#            phy[(imx[ind],)+ind] = 0.

    return diver1, diver2, diver3, diver4, shannon, simpson, richness

divernames = ['Diver1','Diver2','Diver3','Diver4','Shannon','Simpson','Richness']

def calcdiver124(phy, rfac=1.):
    ''' overwrites phy! '''
    nphy = phy.shape[0]
    shape = phy.shape[1:]

    phy[phy<0] = 0.
    total = phy.sum(0)
    mx = phy.max(0)

    diver1 = np.zeros(shape, int)
    diver2 = np.zeros(shape, int)
    diver3 = np.zeros(shape, int)
    diver4 = np.zeros(shape, int)
#    present = np.zeros(shape, bool)

    valid = total >= thresh0*rfac
    for i in range(nphy):
#        present[...] = 0
#        for k in range(kmax):
#            present |= p[i,k,:,:] > thresh1
#        diver1[:,:] += present
        diver1[...] += phy[i] > thresh1*rfac
        diver2[...] += valid & (phy[i] > thresh2*total)
        diver4[...] += valid & (phy[i] > thresh4*mx)

    return diver1, diver2, diver4

def calcdiver3(phy):
    ''' overwrites phy! '''
    nphy = phy.shape[0]
    shape = phy.shape[1:]

    phy[phy<0] = 0.
    total = phy.sum(0)

    phy.sort(axis=0)
    physort = phy[::-1]
    diver3 = np.zeros(shape, int)
    biotot = np.zeros(shape)
    for i in range(nphy):
        diver3[biotot < thresh3*total] += 1
        biotot += physort[i]

    return diver3

def calcshannons(phy, rfacsh=1., rfacrich=1.):
    ''' overwrites phy! '''
    nphy = phy.shape[0]
    shape = phy.shape[1:]

    shannon = np.zeros(shape)
    simpson = np.zeros(shape)
    richness = np.zeros(shape, int)

    # for consistency recompute total with threshold
    phy[phy<threshsh*rfacsh] = 0
    total = phy.sum(0)
    for i in range(nphy):
        valid = phy[i] >= threshsh*rfacsh
        tmp = phy[i,valid]/total[valid]
        shannon[valid] += tmp*np.log(tmp)
        simpson[valid] += tmp*tmp
        valid = phy[i] >= threshrich*rfacrich
        richness[valid] += 1

    shannon *= -1
    valid = simpson != 0
    simpson[valid] = 1./simpson[valid]

    return shannon, simpson, richness


def decl(dayfrac):
    ''' computes the declination (solar zenith angle at noon) from
        dayfrac = day of year/length of year
    '''
    yday = dayfrac*2.0*pi
    delta = (0.006918- (0.399912*cos(yday)) #cosine zenith angle
           +(0.070257*sin(yday))            #(paltridge+platt)
           -(0.006758*cos(2.0*yday))
           +(0.000907*sin(2.0*yday))
           -(0.002697*cos(3.0*yday))
           +(0.001480*sin(3.0*yday)) )
    return delta


def cossolz(lat, dayfrac, daytime):
    ''' compute cosine of solar zenith angle at latitude lat
    for fractional day

        dayfrac = day/yeardays (January 1 -> 0)

    and fractional time

        daytime = (hours since noon)/24
    '''
    delta = decl(dayfrac)
    rlat = np.deg2rad(lat)
    hourangle = 2*pi*daytime
    cosphi = sin(rlat)*sin(delta) + cos(rlat)*cos(delta)*cos(hourangle)
    return cosphi

def cossolzutc(lon, lat, dayfrac, utctime):
    ''' compute cosine of solar zenith angle at longitude lon, latitude lat
    for fractional day

        dayfrac = day/yeardays (January 1 -> 0)

    and decimal universal time utctime (between 0 and 1 with 0 midnight UTC)
    '''
    daytime = utctime + lon/360. - .5
    return cossolz(lat, dayfrac, daytime)

_n_water = 1.341

def cossolzbelow(cosabove):
    sinabove = np.sqrt(1. - cosabove*cosabove)
    sinbelow = sinabove/_n_water
    cosbelow = np.sqrt(1. - sinbelow*sinbelow)
    return cosbelow


def solz360(tm, ylat):
    '''
    tm :: model time in seconds since Jan 1 of 360-day year
    ylat :: latitude in degrees
    '''
    dayfrac = np.mod(tm, 360.*86400.)/(360.*86400.)              
    delta = decl(dayfrac*3.1416/np.pi)
    lat = ylat/180.*3.1416
    sun1 = np.clip(-sin(delta)/cos(delta) * sin(lat)/cos(lat), -.999, .999)
    dayhrs = abs(arccos(sun1))
    cosz = ( sin(delta)*sin(lat)+              #average zenith angle
            (cos(delta)*cos(lat)*sin(dayhrs)/dayhrs) )
    cosz = cosz.clip(.005, .999)
    solz = arccos(cosz)*180./3.1416
    return solz

def radtrans_sfcrmud(solz):
    rsza = solz/180.*pi
    sinszaw = sin(rsza)/_n_water
    szaw = arcsin(sinszaw)
    rmud = 1.0/cos(szaw)   #avg cosine direct (1 over)
    return rmud.clip(0., 1.5)


def monod_phytotempfunc(T, Topt, e2, p):
    Tkel = 273.15
    TempAe = -4000.
    Tempref = 293.15
    tempnorm =  0.
    TempCoeff = 0.5882
    phytoTempCoeff = TempCoeff
    #swd -- this gives Arrenhius curve only
    phytoTempFunction = np.exp(TempAe*(1./(T+Tkel) - 1./(Tempref) ) )
    phytoTempFunction *= (np.exp(-e2*(abs(T - Topt))**p))
    phytoTempFunction -= tempnorm
    phytoTempFunction = phytoTempCoeff*np.maximum(phytoTempFunction, 1E-10)
    return phytoTempFunction

