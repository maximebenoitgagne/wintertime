'''
units for DIC, Alk, PO4, ... are mol/kg
'''
import numpy as np
from numpy import NaN
from numpy import log,sqrt,exp,log10
import scipy.optimize as opt

# mol/kg per mol/m3
permil = 1.0/1024.5

def rec2dict(ra):
    return dict((k,ra[k]) for k in ra.dtype.names)

_coeff_names = ['fugf',
'ff',
'k0',
'k1',
'k2',
'kB',
'k1P',
'k2P',
'k3P',
'kSi',
'kw',
'kS',
'kF',
'Bt',
'St',
'Ft',
]

def coeffs(t,s):
    #
    #     /==========================================================\
    #     | SUBROUTINE CARBON_COEFFS                                 |
    #     | determine coefficients for surface carbon chemistry      |
    #     | adapted from OCMIP2:  SUBROUTINE CO2CALC                 |
    #     | mick follows, oct 1999                                   |
    #     | minor changes to tidy, swd aug 2002                      |
    #     \==========================================================/
    # INPUT
    #       diclocal = total inorganic carbon (mol/m^3)
    #             where 1 T = 1 metric ton = 1000 kg
    #       ta  = total alkalinity (eq/m^3)
    #       pt  = inorganic phosphate (mol/^3)
    #       sit = inorganic silicate (mol/^3)
    #       t   = temperature (degrees C)
    #       s   = salinity (PSU)
    # OUTPUT
    # IMPORTANT: Some words about units - (JCO, 4/4/1999)
    #     - Models carry tracers in mol/m^3 (on a per volume basis)
    #     - Conversely, this routine, which was written by observationalists
    #       (C. Sabine and R. Key), passes input arguments in umol/kg
    #       (i.e., on a per mass basis)
    #     - I have changed things slightly so that input arguments are in mol/m^3,
    #     - Thus, all input concentrations (diclocal, ta, pt, and st) should be
    #       given in mol/m^3; output arguments "co2star" and "dco2star"
    #       are likewise be in mol/m^3.
    #
    # Apr 2011: fix vapour bug (following Bennington)
    #--------------------------------------------------------------------------

    #.....................................................................
    # OCMIP note:
    # Calculate all constants needed to convert between various measured
    # carbon species. References for each equation are noted in the code.
    # Once calculated, the constants are
    # stored and passed in the common block "const". The original version
    # of this code was based on the code by dickson in Version 2 of
    # "Handbook of Methods C for the Analysis of the Various Parameters of
    # the Carbon Dioxide System in Seawater", DOE, 1994 (SOP No. 3, p25-26).
    #....................................................................

    tk = 273.15E0 + t
    tk100 = tk/100.E0
    tk1002 = tk100*tk100
    invtk = 1.0E0/tk
    dlogtk = log(tk)
    iS = 19.924E0*s/(1000.E0-1.005E0*s)
    iS2 = iS*iS
    sqrtiS = sqrt(iS)
    s2 = s*s
    sqrts = sqrt(s)
    s15 = s**1.5E0
    scl = s/1.80655E0
    # -----------------------------------------------------------------------
    # added by Val Bennington Nov 2010
    # Fugacity Factor needed for non-ideality in ocean
    # ff used for atmospheric correction for water vapor and pressure
    # Weiss (1974) Marine Chemistry
    P1atm = 1.01325E0 # bars
    Rgas = 83.1451E0 # bar*cm3/(mol*K)
    RT = Rgas*tk
    delta = (57.7E0 - 0.118E0*tk)
    B1 = -1636.75E0 + 12.0408E0*tk - 0.0327957E0*tk*tk
    B = B1 + 3.16528E0*tk*tk*tk*(0.00001E0)
    fugf = exp( (B+2.E0*delta) * P1atm / RT)
    #------------------------------------------------------------------------
    # f = k0(1-pH2O)*correction term for non-ideality
    # Weiss & Price (1980, Mar. Chem., 8, 347-359; Eq 13 with table 6 values)
    ff = exp(-162.8301E0 + 218.2968E0/tk100  + \
          90.9241E0*log(tk100) - 1.47696E0*tk1002 + \
          s * (.025695E0 - .025225E0*tk100 + \
          0.0049867E0*tk1002))
    #------------------------------------------------------------------------
    # K0 from Weiss 1974
    k0 = exp(93.4517E0/tk100 - 60.2409E0 + \
        23.3585E0 * log(tk100) + \
        s * (0.023517E0 - 0.023656E0*tk100 + \
        0.0047036E0*tk1002))
    #------------------------------------------------------------------------
    # k1 = [H][HCO3]/[H2CO3]
    # k2 = [H][CO3]/[HCO3]
    # Millero p.664 (1995) using Mehrbach et al. data on seawater scale
    k1 = 10.**(-1.E0*(3670.7E0*invtk - \
          62.008E0 + 9.7944E0*dlogtk - \
          0.0118E0 * s + 0.000116E0*s2))
    k2 = 10.**(-1.E0*(1394.7E0*invtk+ 4.777E0- \
          0.0184E0*s + 0.000118E0*s2))
    #------------------------------------------------------------------------
    # kB = [H][BO2]/[HBO2]
    # Millero p.669 (1995) using data from dickson (1990)
    kB = exp((-8966.90E0- 2890.53E0*sqrts - \
          77.942E0*s + 1.728E0*s15 - 0.0996E0*s2)*invtk + \
          (148.0248E0 + 137.1942E0*sqrts + 1.62142E0*s) + \
          (-24.4344E0 - 25.085E0*sqrts - 0.2474E0*s) * \
          dlogtk + 0.053105E0*sqrts*tk)
    #------------------------------------------------------------------------
    # k1p = [H][H2PO4]/[H3PO4]
    # DOE(1994) eq 7.2.20 with footnote using data from Millero (1974)
    k1P = exp(-4576.752E0*invtk + 115.525E0 - \
          18.453E0*dlogtk + \
          (-106.736E0*invtk + 0.69171E0)*sqrts + \
          (-0.65643E0*invtk - 0.01844E0)*s)
    #------------------------------------------------------------------------
    # k2p = [H][HPO4]/[H2PO4]
    # DOE(1994) eq 7.2.23 with footnote using data from Millero (1974))
    k2P = exp(-8814.715E0*invtk + 172.0883E0 - \
          27.927E0*dlogtk + \
          (-160.340E0*invtk + 1.3566E0) * sqrts + \
          (0.37335E0*invtk - 0.05778E0) * s)
    #------------------------------------------------------------------------
    # k3p = [H][PO4]/[HPO4]
    # DOE(1994) eq 7.2.26 with footnote using data from Millero (1974)
    k3P = exp(-3070.75E0*invtk - 18.141E0 + \
          (17.27039E0*invtk + 2.81197E0) * \
          sqrts + (-44.99486E0*invtk - 0.09984E0) * s)
    #------------------------------------------------------------------------
    # ksi = [H][SiO(OH)3]/[Si(OH)4]
    # Millero p.671 (1995) using data from Yao and Millero (1995)
    kSi = exp(-8904.2E0*invtk + 117.385E0 - \
          19.334E0*dlogtk + \
          (-458.79E0*invtk + 3.5913E0) * sqrtiS + \
          (188.74E0*invtk - 1.5998E0) * iS + \
          (-12.1652E0*invtk + 0.07871E0) * iS2 + \
          log(1.0E0-0.001005E0*s))
    #------------------------------------------------------------------------
    # kw = [H][OH]
    # Millero p.670 (1995) using composite data
    kw = exp(-13847.26E0*invtk + 148.9652E0 - \
          23.6521E0*dlogtk + \
          (118.67E0*invtk - 5.977E0 + 1.0495E0 * dlogtk) \
          * sqrts - 0.01615E0 * s)
    #------------------------------------------------------------------------
    # ks = [H][SO4]/[HSO4]
    # dickson (1990, J. chem. Thermodynamics 22, 113)
    kS = exp(-4276.1E0*invtk + 141.328E0 - \
          23.093E0*dlogtk + \
   (-13856.E0*invtk + 324.57E0 - 47.986E0*dlogtk)*sqrtiS+ \
   (35474.E0*invtk - 771.54E0 + 114.723E0*dlogtk)*iS - \
          2698.E0*invtk*iS**1.5E0 + 1776.E0*invtk*iS2 + \
          log(1.0E0 - 0.001005E0*s))
    #------------------------------------------------------------------------
    # kf = [H][F]/[HF]
    # dickson and Riley (1979) -- change pH scale to total
    kF = exp(1590.2E0*invtk - 12.641E0 + \
   1.525E0*sqrtiS + log(1.0E0 - 0.001005E0*s) +  \
   log(1.0E0 + (0.1400E0/96.062E0)*(scl)/kS))
    #------------------------------------------------------------------------
    # Calculate concentrations for borate, sulfate, and fluoride
    # Uppstrom (1974)
    Bt = 0.000232E0 * scl/10.811E0
    # Morris & Riley (1966)
    St = 0.14E0 * scl/96.062E0
    # Riley (1965)
    Ft = 0.000067E0 * scl/18.9984E0
    #------------------------------------------------------------------------
    # add Bennington
    vs = vars()
    d = np.rec.fromarrays([vs[k] for k in _coeff_names], names=_coeff_names)
#    d = dict((k,vs[k]) for k in _coeff_names)

    return d


def phosfracs(h,ks):
    """ compute fractions of total inorg.P that are H3PO4,H2PO4,HPO4 and PO4
        given H+ conc and rate constants for the 3 reactions
    """
    k1p,k2p,k3p = ks
    h3po4 = h*h*h
    h2po4 = k1p*h*h
    hpo4  = k1p*k2p*h
    po4   = k1p*k2p*k3p
    denom = h3po4 + h2po4 + hpo4 + po4
    h3po4 /= denom
    h2po4 /= denom
    hpo4  /= denom
    po4   /= denom
    return h3po4,h2po4,hpo4,po4


def alkTphosfac(hguess,ks):
    """ returns factor relating P-contribution to Alk_T to total inorg.phosphorus
    """
    #mick - first estimate of contribution from phosphate
    #mick based on Dickson and Goyet
    h3po4g,h2po4g,hpo4g,po4g = phosfracs(hguess,ks)
    return h3po4g-hpo4g-2*po4g


def F(H,Pt,Sit,k1P=NaN,k2P=NaN,k3P=NaN,kB=NaN,kw=NaN,kSi=NaN,Bt=NaN,**kwargs):
    # Pt,Sit,ta,dic in mol/kg

    bohg = Bt*kB/(H+kB)

    siooh3g = Sit*kSi / (kSi + H)

    OH = kw/H
    F = - bohg - OH + H \
        + alkTphosfac(H,(k1P,k2P,k3P))*Pt \
        - siooh3g

    return F


def Fts(t,s,Pt,Sit,H):
    d = carbon_coeffs(t,s)
    res = F(H,Pt,Sit,*[d[k] for k in ['k1P','k2P','k3P','kB','kw','kSi','Bt']])
    return res


def ACoC(H,k1=NaN,k2=NaN,**kwargs):
    CO2oC = 1./(1. + k1/H + k1*k2/H/H)
    HCO3oC = CO2oC*k1/H
    CO3oC = HCO3oC*k2/H

    ACoC = HCO3oC + 2.*CO3oC

    return ACoC


def Alk(H,C,Pt,Sit,**kwargs):
    return ACoC(H,**kwargs)*C - F(H,Pt,Sit,**kwargs)


def solveACpF(H0,C,Pt,Sit,**d):
    f0 = ACoC(H0,**d)*C + F(H0,Pt,Sit,**d)
    def f(H):
        return C*ACoC(H,**d) + F(H,Pt,Sit,**d) - f0

    a,b = d['kw'], .99999999*H0
    if f(b) < 0:
        return NaN
    else:
        return opt.brentq(f, a, b)


def solve(H0, C, Pt, Sit, cc):
    H = np.zeros_like(H0)
    for ind in np.ndindex(H0.shape):
        if C[ind] > 0:
            d = rec2dict(cc[ind])
            H[ind] = solveACpF(H0[ind], C[ind], Pt[ind], Sit[ind], **d)

    return H


def solveuM(H0, C, Pt, Sit, cc):
    return solve(H0, C*permil, Pt*permil, Sit*permil, cc)


def maxAlk(H0,C,Pt,Sit,parm):
    """ parm must be recarray """
    keys = parm.dtype.names

    bc = np.broadcast(parm,C,Pt,Sit)
    res = np.empty(bc.shape)
    resf = res.flat
    for i,(p,C,Pt,Sit) in enumerate(bc):
        d = dict((k,p[k]) for k in keys)
        try:
            H = solveACpF(H0,C,Pt,Sit,**d)
        except ValueError:
            print i
            raise
        resf[i] = ACoC(H,**d)*C - F(H0,Pt,Sit,**d)

    return res


def solveACpFmin(H0,C,Pt,Sit,**d):
    f0 = ACoC(H0,**d)*C + F(H0,Pt,Sit,**d)
    def f(H):
        return C*ACoC(H,**d) + F(H,Pt,Sit,**d) - f0

    a,b = 1.00000001*H0,1.
    if f(a) > 0:
        return NaN
    else:
        return opt.brentq(f, a, b)


def minAlk(H0,C,Pt,Sit,parm):
    """ parm must be recarray """
    keys = parm.dtype.names

    bc = np.broadcast(parm,C,Pt,Sit)
    res = np.empty(bc.shape)
    resf = res.flat
    for i,(p,C,Pt,Sit) in enumerate(bc):
        d = dict((k,p[k]) for k in keys)
        try:
            H = solveACpFmin(H0,C,Pt,Sit,**d)
        except ValueError:
            print i
            raise
        resf[i] = ACoC(H,**d)*C - F(H0,Pt,Sit,**d)

    return res


def dFdH(H,Pt,Sit,k1P=NaN,k2P=NaN,k3P=NaN,kB=NaN,kw=NaN,kSi=NaN,Bt=NaN,**kwargs):
    # Pt,Sit,ta,dic in mol/kg

    bohg = Bt*kB/(H+kB)**2

    siooh3g = Sit*kSi / (kSi + H)**2

    Pnum = 4*H*k1P*k2P*k3P + k1P*k2P*H**2 + k1P*k3P*k2P**2 + 9*k2P*k3P*H**2 + 4*k2P*H**3 + H**4
    Pdenom = H*k1P*k2P + k1P*k2P*k3P + k1P*H**2 + H**3
    #         + Pt*k2P*(H*H+4*k3P*H+k2P*k3P)/(H*H+k2P*H+k2P*k3P)**2 \

    OH = kw/H**2
    dFdt = 1. + bohg + OH \
          + Pt*k1P*Pnum/Pdenom**2 \
         + siooh3g

    return dFdt


def dACoCdH(H,k1=NaN,k2=NaN,**kwargs):
    denom = H*H + k1*H + k1*k2
    dACoCdH = -k1*(H*H+4*k2*H+k1*k2)/denom/denom
    return dACoCdH


def dFOHdH(H,Pt,Sit,k1P=NaN,k2P=NaN,k3P=NaN,kB=NaN,kw=NaN,kSi=NaN,Bt=NaN,k1=NaN,k2=NaN,**kwargs):
    # Pt,Sit,ta,dic in mol/kg

    bohg = Bt*kB/(H+kB)**2

    siooh3g = Sit*kSi / (kSi + H)**2

    Pnum = 4*H*k1P*k2P*k3P + k1P*k2P*H**2 + k1P*k3P*k2P**2 + 9*k2P*k3P*H**2 + 4*k2P*H**3 + H**4
    Pdenom = H*k1P*k2P + k1P*k2P*k3P + k1P*H**2 + H**3
    #         + Pt*k2P*(H*H+4*k3P*H+k2P*k3P)/(H*H+k2P*H+k2P*k3P)**2 \

    OH = kw/H**2
    dFdt = 1. + bohg \
          + Pt*k1P*Pnum/Pdenom**2 \
         + siooh3g

    denom = H*H + k1*H + k1*k2
    dACoCdH = -k1*(H*H+4*k2*H+k1*k2)/denom/denom
    approx = -k2/(k2+H)**2

    return dFdt + approx - dACoCdH


def dAOHoCdH(H,k1=NaN,k2=NaN,kw=NaN,**kwargs):
    denom = H*H + k1*H + k1*k2
    dACoCdH = -k1*(H*H+4*k2*H+k1*k2)/denom/denom
    approx = -k2/(k2+H)**2
    OH = kw/H**2
    return approx - OH


def dFalldH(H,Pt,Sit,k1P=NaN,k2P=NaN,k3P=NaN,kB=NaN,kw=NaN,kSi=NaN,Bt=NaN,k1=NaN,k2=NaN,**kwargs):
    # Pt,Sit,ta,dic in mol/kg

    bohg = Bt*kB/(H+kB)**2

    siooh3g = Sit*kSi / (kSi + H)**2

    Pnum = 4*H*k1P*k2P*k3P + k1P*k2P*H**2 + k1P*k3P*k2P**2 + 9*k2P*k3P*H**2 + 4*k2P*H**3 + H**4
    Pdenom = H*k1P*k2P + k1P*k2P*k3P + k1P*H**2 + H**3
    #         + Pt*k2P*(H*H+4*k3P*H+k2P*k3P)/(H*H+k2P*H+k2P*k3P)**2 \

    OH = kw/H**2
    dFdt = bohg \
          + Pt*k1P*Pnum/Pdenom**2 \
         + siooh3g

    return dFdt


def dAalloCdH(H,k1=NaN,k2=NaN,kw=NaN,**kwargs):
    denom = H*H + k1*H + k1*k2
    dACoCdH = -k1*(H*H+4*k2*H+k1*k2)/denom/denom
    OH = kw/H**2
    return dACoCdH + 1. - OH


def iterateH(H,DIC,Pt,Sit,At,
             k1=None,k2=None,k1P=None,k2P=None,k3P=None,kw=None,
             kB=None,kSi=None,k0=None,
             fugf=None,ff=None,Bt=None,St=None,
             Ft=None,kS=None,kF=None,tol=1e-6,maxiter=1000,**kwargs):

    pH = 0.
    # will also stop on NaN
    iiter = 0
    while iiter < maxiter and np.any(abs(np.log10(H)+pH)) > tol:
        iiter += 1
        pH = -np.log10(H)
        H = iter_H(H,DIC,Pt,Sit,At, k1,k2,k1P,k2P,k3P,kw,
                   kB,kSi,k0, fugf,ff,Bt,St, Ft,kS,kF)

    return H,abs(np.log10(H)+pH)


def iter_H(H, DIC,Pt,Sit,At,
                k1,k2,
                k1P,k2P,k3P,
                kw,kB,kSi,
                k0, fugf,
                ff,Bt,St,Ft,
                kS,kF,
    ):

    bohg = Bt*kB/(H+kB)

    siooh3g = Sit*kSi / (kSi + H)

    h3po4 = H*H*H
    h2po4 = k1P*H*H
    hpo4  = k1P*k2P*H
    po4   = k1P*k2P*k3P
    denom = h3po4 + h2po4 + hpo4 + po4
    Fphos = Pt*(h3po4-hpo4-2*po4)/denom

    #mick - now estimate carbonate alkalinity
    AC = At - bohg - kw/H + H + Fphos - siooh3g

    gamma  = DIC/AC
    stuff = (1.-gamma)*(1.-gamma)*k1*k1 - 4.*k1*k2*(1.-2.*gamma)
    Hnew  = .5*( (gamma-1.)*k1 + np.sqrt(stuff) )

    return Hnew


def iterate_molm3(pH,dic,pt,sit,ta,
            k1=None,k2=None,k1P=None,k2P=None,k3P=None,kw=None,
            kB=None,kSi=None,k0=None,
            fugf=None,ff=None,Bt=None,St=None,
            Ft=None,kS=None,kF=None,
            eps=1e-6,**kwargs):
    dic = dic*permil
    pt  = pt *permil
    sit = sit*permil
    ta  = ta *permil
    return iterate1(pH,dic,pt,sit,ta,
            k1,k2,k1P,k2P,k3P,kw,
            kB,kSi,k0,
            fugf,ff,Bt,St,
            Ft,kS,kF,eps)

def iterate_molkg(pH,dic,pt,sit,ta,
            k1=None,k2=None,k1P=None,k2P=None,k3P=None,kw=None,
            kB=None,kSi=None,k0=None,
            fugf=None,ff=None,Bt=None,St=None,
            Ft=None,kS=None,kF=None,**kwargs):
    return iterate1(pH,dic,pt,sit,ta,
            k1,k2,k1P,k2P,k3P,kw,
            kB,kSi,k0,
            fugf,ff,Bt,St,
            Ft,kS,kF)

@np.vectorize
def iterate1(pH,dic,pt,sit,ta,
            k1,k2,k1P,k2P,k3P,kw,
            kB,kSi,k0,
            fugf,ff,Bt,St,
            Ft,kS,kF,eps):

    pHold = 0.
    # will also stop on NaN
    while abs(pH-pHold) > eps:
        pHold = pH
        pH,_ = calc_pco2_approx(dic,pt,sit,ta, k1,k2,k1P,k2P,k3P,kw,
                                kB,kSi,k0, fugf,ff,Bt,St, Ft,kS,kF,pH)

    return pH


def calc_pco2_approx(
                   diclocal,pt,sit,ta,
                   k1local,k2local,
                   k1plocal,k2plocal,k3plocal,
                   kwlocal,kblocal,ksilocal,
                   k0local, fugflocal,
                   fflocal,btlocal,stlocal,ftlocal,
                   kslocal,kflocal,
                   phlocal
    ):


    # ---------------------------------------------------------------------
    # Change units from the input of mol/m^3 -> mol/kg:
    # (1 mol/m^3)  x (1 m^3/1024.5 kg)
    # where the ocean's mean surface density is 1024.5 kg/m^3
    # Note: mol/kg are actually what the body of this routine uses
    # for calculations.  Units are reconverted back to mol/m^3 at the
    # end of this routine.
    # To convert input in mol/m^3 -> mol/kg
    # ---------------------------------------------------------------------
    # set first guess and brackets for [H+] solvers
    # first guess (for newton-raphson)
    phguess = phlocal
    #mick - new approx method
    #mick - make estimate of htotal (hydrogen ion conc) using
    #mick   appromate estimate of CA, carbonate alkalinity
    hguess = 10.0**(-phguess)
    #mick - first estimate borate contribution using guess for [H+]
    # B(OH)_4^-
    bohg = btlocal*kblocal/(hguess+kblocal)

    #mick - estimate contribution from silicate
    #mick based on Dickson and Goyet
    # SiO(OH)_3^-
    siooh3g = sit*ksilocal / (ksilocal + hguess)

    #mick - now estimate carbonate alkalinity
    cag = ta - bohg - (kwlocal/hguess) + hguess \
           + alkTphosfac(hguess,(k1plocal,k2plocal,k3plocal))*pt \
           - siooh3g

    #mick - now evaluate better guess of hydrogen ion conc
    #mick   htotal = [H+], hydrogen ion conc
    gamm  = diclocal/cag
    stuff = (1.E0-gamm)*(1.E0-gamm)*k1local*k1local \
          - 4.E0*k1local*k2local*(1.E0-2.E0*gamm)
    hnew  = 0.5E0*( (gamm-1.E0)*k1local + sqrt(stuff) )
    #mick - now determine [CO2*]
    co2s  = diclocal/ \
   (1.E0 + (k1local/hnew) + (k1local*k2local/(hnew*hnew)))
    #mick - return update pH to main routine
    phlocal = -log10(hnew)

    # NOW EVALUATE CO32-, carbonate ion concentration
    # used in determination of calcite compensation depth
    # Karsten Friis & Mick - Sep 2004
    #       co3local = k1local*k2local*diclocal /
    #    &         (hnew*hnew + k1local*hnew + k1local*k2local)

    # ---------------------------------------------------------------
    # surface pCO2 (following Dickson and Goyet, DOE...)
    fco2 = co2s/k0local
    pco2surfloc = fco2/fugflocal

    # ----------------------------------------------------------------
    # Reconvert from mol/kg -> mol/m^3
    return phlocal,pco2surfloc,


def windspeed(tau, rhoa=1.22, tol=1e-3):
    u10o = np.zeros_like(tau)
    cd = np.zeros_like(tau)
    cd += 1.15e-3
    u10 = np.zeros_like(tau)
    u10 += np.sqrt(tau/rhoa/cd)
    while np.any(abs(u10-u10o)>tol):
        u10o.setasflat(u10)
        cd.setasflat(np.where(u10<10.15385, 1.15e-3, 4.9e-4+6.5e-5*u10))
        u10.setasflat(np.sqrt(tau/rhoa/cd))
    return u10


if __name__ == '__main__':
    s,t = np.mgrid[32:37.1,-2:36.]
    d = carbon_coeffs(t,s)
    ra = np.rec.fromarrays(d.values(),names=d.keys())
    flds = sorted(ra.dtype.names)
    import oj.plot
    ag,ims,cbs = oj.plot.imgrid([ra[k] for k in flds],(8,2),extent=(t[0,0]-.5,t[-1,-1]+.5,s[0,0]-.5,s[-1,-1]+.5),cbar_mode='each',titles=flds,rect=[.03,.03,.9,.94],axes_pad=(1.2,.32))
    fig = ag[0].figure
    fig.savefig('carboncoeff.png',dpi=fig.dpi)

