from numpy import exp, log, log10, r_, zeros
from clouds import slingo, sl
from data import atmo

p0     = 1013.25
ozfac1 = 44.0/6370.0
ozfac2 = 1.0 + 22.0/6370.0

nlt = len(atmo)
ncld = len(sl)

def listgenerator(f):
    def g():
        return list(f())
    return g

@listgenerator
def make_ica():
    # map from light to cloud indices
    for nl in range(nlt):
        for nc in range(ncld):
            lamcld = int(round(sl.rnl2[nc]*1000.0))
            if atmo.lam[nl] < lamcld:
                yield nc
                break

ica = make_ica()


def light(sunz,cosunz,daycor,pres,ws,ozone,wvapor,relhum,
          ta,wa,asym,aml,Vil,cov,cldtau,clwp,re):
    '''
    Calls clrtrans.F to get cloud-free transmittance and slingo.F to
    get cloudy transmittance, then computes total irradiance in
    W/m2/(variable)nm weighted by the cloudiness.
  
    Tdclr is spectral clear sky direct transmittance
    Tsclr is spectral clear sky diffuse transmittance
    Tdcld is spectral cloudy direct transmittance
    Tscld is spectral cloudy diffuse transmittance
    '''
    Ed = zeros((nlt,))
    Es = zeros((nlt,))
    if (pres < 0.00 or ws < 0.0 or relhum < 0.00 or
            ozone < 0.00 or wvapor < 0.0):
        return Ed,Es

    #  Compute atmospheric path lengths (air mass); not pressure-corrected
    rtmp = (93.885-sunz)**(-1.253)
    rmu0 = cosunz+0.15*rtmp
    rm = 1.0/rmu0
    otmp = (cosunz*cosunz+ozfac1)**0.5
    rmo = ozfac2/otmp

    #  Compute pressure-corrected atmospheric path length (air mass)
    rmp = pres/p0*rm

    #  Loop to compute total irradiances at each grid point
    #   Compute direct and diffuse irradiance for a cloudy and non-cloudy
    #   atmosphere
    #   Account for gaseous absorption
    Tgas = zeros((nlt,))
    Edclr = zeros((nlt,))
    Esclr = zeros((nlt,))
    Edcld = zeros((nlt,))
    Escld = zeros((nlt,))
    for nl in range(nlt):
        # Ozone
        to = atmo.oza[nl]*ozone*0.001   # optical thickness
        oarg = -to*rmo
        # Oxygen/gases
        ag = atmo.ao[nl] + atmo.aco2[nl]
        gtmp = (1.0 + 118.3*ag*rmp)**0.45
        gtmp2 = -1.41*ag*rmp
        garg = gtmp2/gtmp
        # Water Vapor
        wtmp = (1.0+20.07*atmo.awv[nl]*wvapor*rm)**0.45
        wtmp2 = -0.2385*awv[nl]*wvapor*rm
        warg = wtmp2/wtmp

    #  Compute clear sky transmittances
    Tdclr,Tsclr = clrtrans(cosunz,rm,rmp,rmo,ws,relhum,aml,Vil,ta,wa,asym)

    for nl in range(nlt):
        Foinc = atmo.Fobar[nl]*daycor*cosunz
        # Direct irradiance 
        Edir = Foinc*Tgas[nl]*Tdclr[nl]
        # Diffuse irradiance
        Edif = Foinc*Tgas[nl]*Tsclr[nl]
        # Spectral components
        Edclr[nl] = Edir
        Esclr[nl] = Edif

    #  Compute cloudy transmittances
    #      call slingo(rmu0,cldtc(ic,jc),rlwp(ic,jc),cdre(ic,jc),
    #     *            Tdcld,Tscld)
    Tdcld,Tscld = slingo(rmu0,cldtau,clwp,re)

    for nl in range(nlt):
        Foinc = Fobar[nl]*daycor*cosunz
        # Direct irradiance 
        Edir = Foinc*Tgas[nl]*Tdcld[ica[nl]]
        # Diffuse irradiance
        Edif = Foinc*Tgas[nl]*Tscld[ica[nl]]
        # Spectral components
        Edcld[nl] = Edir
        Escld[nl] = Edif

    #  Sum clear and cloudy by percent
    ccov1 = cov*0.01  # convert from percent to fraction
    for nl in range(nlt):
        Ed[nl] = (1.0-ccov1)*Edclr[nl] + ccov1*Edcld[nl]
        Es[nl] = (1.0-ccov1)*Esclr[nl] + ccov1*Escld[nl]

    return Ed,Es


def clrtrans(cosunz,rm,rmp,rmo,ws,relhum,aml,Vil,ta,wa,asym):
    '''
    Model for atmospheric transmittance of solar irradiance through
    a cloudless maritime atmosphere.  Computes direct and diffuse 
    separately.  From Gregg and Carder (1990) Limnology and 
    Oceanography 35(8): 1657-1675.
  
    Td is spectral clear sky direct transmittance
    Ts is spectral clear sky diffuse transmittance
    '''
    nlt = len(ta)

    #  Obtain aerosol parameters; simplified Navy aerosol model
    beta,eta,wa1,afs,bfs = navaer(relhum,aml,Vil,ws)

    #  Compute spectral transmittance
    Td = zeros((nlt,))
    Ts = zeros((nlt,))
    for nl in range(nlt):
        #    Rayleigh
        rtra = exp(-atmo.thray[nl]*rmp)       # transmittance
        #   Aerosols
        if ta[nl] < 0.0:
            ta[nl] = beta*rlamu[nl]**eta
        if wa[nl] < 0.0:
            omegaa = wa1
        else:
            omegaa = wa[nl]
        if asym[nl] >= 0.0:
            alg = log(1.0-asym[nl])
            afs = alg*(1.459+alg*(.1595+alg*.4129))
            bfs = alg*(.0783+alg*(-.3824-alg*.5874))
        if ta[nl] < 0.0 or omegaa < 0.0:
            raise
        Fa = 1.0 - 0.5*exp((afs+bfs*cosunz)*cosunz)
        if Fa < 0.0:
            raise
        tarm = ta[nl]*rm
        atra = exp(-tarm)
        taa = exp(-(1.0-omegaa)*tarm)
        tas = exp(-omegaa*tarm)
        #  Direct transmittance
        Td[nl] = rtra*atra

        #   Diffuse transmittance
        dray = taa*0.5*(1.0-rtra**.95)
        daer = rtra**1.5*taa*Fa*(1.0-tas)

        #  Total diffuse
        Ts[nl] = dray + daer

    return Td,Ts


_ro   = r_[0.03, 0.24, 2.0]
_r    = r_[0.1, 1.0, 10.0]
_rlam = r_[0.55]

def navaer(relhum,am,Vi,ws):
    '''
    Computes aerosol parameters according to a simplified version
    of the Navy marine aerosol model.
    '''
    #  Relative humidity factor
    #      if (relhum .ge. 100.0)relhum = 99.9
    relhum = min(99.9,relhum)
    rnum = 2.0 - relhum/100.0
    rden = 6.0*(1.0-relhum/100.0)
    frh = (rnum/rden)**0.333

    #  Size distribution amplitude components
    a = zeros((3,))
    a[0] = 2000.0*am*am
    a[1] = 5.866*(ws-2.2)
    a[1] = max(0.5,a[1])
    a[2] = 0.01527*(ws-2.2)*0.05        # from Hughes 1987
    a[2] = max(1.4E-5,a[2])

    #  Compute size distribution at three selected radii according to
    #  Navy method
    dndr = zeros((3,))
    for n in range(3):
        for i in range(3):
            rden = frh*_ro[i]
            arg = log(_r[n]/rden)*log(_r[n]/rden)
            rval = a[i]*exp(-arg)/frh
            dndr[n] = dndr[n] + rval

    #  Least squares approximation
    sumx = 0.0
    sumy = 0.0
    sumxy = 0.0
    sumx2 = 0.0
    for n in range(3):
        rlrn = log10(_r[n])
        rldndr = log10(dndr[n])
        sumx = sumx + rlrn
        sumy = sumy + rldndr
        sumxy = sumxy + rlrn*rldndr
        sumx2 = sumx2 + rlrn*rlrn

    gama = sumxy/sumx2
    rlogc = sumy/3.0 - gama*sumx/3.0
    alpha = -(gama+3.0)
    eta = -alpha

    #  Compute beta
    cext = 3.91/Vi
    beta = cext*_rlam**alpha

    #  Compute asymmetry parameter -- a function of alpha
    if alpha > 1.2:
        asymp = 0.65
    elif alpha < 0.0:
        asymp = 0.82
    else:
        asymp = -0.14167*alpha + 0.82

    #  Forward scattering coefficients
    alg = log(1.0-asymp)
    afs = alg*(1.459+alg*(.1595+alg*.4129))
    bfs = alg*(.0783+alg*(-.3824-alg*.5874))

    #  Single scattering albedo at 550; function of RH
    wa = (-0.0032*am + 0.972)*exp(3.06E-4*relhum)

    return beta,eta,wa,afs,bfs

