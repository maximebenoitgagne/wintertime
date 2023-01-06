from numpy import sqrt, exp, zeros, r_
from .data import sl

ncld = len(sl)
U1 = 7.0/4.0

def slingo(rmu0,cldtau,clwp,cre):
    '''
    Slingo's (1989) Delta-Eddington approximation for the two-
    stream equations applied to clouds.  
    Inputs:
         rmu0    Kasten's approx for cosine of solar zenith angle
         cldtau  cloud optical thickness (at 0.6 um)
         clwp    liquid water path in cloud (g/m2)
         cre     cloud droplet effective radius (um)
    Outputs
         Tcd     spectral transmittance for downwelling direct irradiance
         Tcs     spectral transmittance for downwelling diffuse irradiance
    '''
    # Internal
    #      Tcu     spectral transmittance for upwelling irradiance
    # Compute re as funtion of cldtau and LWP according to eq. 1 in 
    # Slingo.
    #  tau is derived at this wavelength (0.6 um) in the ISCCP data set
    #     re = clwp*bsl(9)/(cldtau - clwp*asl(9))
    #     re = min(re,15.0)  !block high re -- produces excessive direct
    # Changes to the ISCCP-D2 data set make this relationship untenable
    # (excessive res are derived).  Instead choose a fixed re of 10 um
    # for ocean (Kiehl et al., 1998 -- J. Clim.)
    #      re = 10.0
    # Paper by Han et al., 1994 (J.Clim.) show mean ocean cloud radius
    # = 11.8 um
    #      re = 11.8
    # Mean of Kiehl and Han
    re = (10.0+11.8)/2.0

    #  Compute spectral cloud characteristics
    #   If MODIS re is available use it; otherwise use parameterized re above
    if cre >= 0.0:   # use MODIS re
        re = cre

    Tcd = zeros((ncld,))
    Tcs = zeros((ncld,))
    izero = 0
    for nc in range(ncld):
        tauc = clwp*(sl.a[nc]+sl.b[nc]/re)
        oneomega = sl.c[nc] + sl.d[nc]*re
        omega = 1.0 - oneomega
        g = sl.e[nc] + sl.f[nc]*re
        b0 = 3.0/7.0*(1.0-g)
        bmu0 = 0.5 - 0.75*rmu0*g/(1.0+g)
        f = g*g
        U2 = U1*(1.0-((1.0-omega)/(7.0*omega*b0)))
        U2 = max(U2,0.0)
        alpha1 = U1*(1.0-omega*(1.0-b0))
        alpha2 = U2*omega*b0
        alpha3 = (1.0-f)*omega*bmu0
        alpha4 = (1.0-f)*omega*(1.0-bmu0)
        sqarg = alpha1*alpha1 - alpha2*alpha2
        sqarg = max(sqarg,1.0E-17)
        eps = sqrt(sqarg)
        rM = alpha2/(alpha1+eps)
        E = exp(-eps*tauc)
        val1 = 1.0 - omega*f
        val2 = eps*eps*rmu0*rmu0
        rnum = val1*alpha3 - rmu0*(alpha1*alpha3+alpha2*alpha4)
        rden = val1*val1 - val2
        gama1 = rnum/rden
        rnum = -val1*alpha4 - rmu0*(alpha1*alpha4+alpha2*alpha3)
        gama2 = rnum/rden
        Tdb = exp(-val1*tauc/rmu0)
        val3 = 1.0 - E*E*rM*rM
        Rdif = rM*(1.0-E*E)/val3
        Tdif = E*(1.0-rM*rM)/val3
#        Rdir = -gama2*Rdif - gama1*Tdb*Tdif + gama1
        Tdir = -gama2*Tdif - gama1*Tdb*Rdif + gama2*Tdb
#        Tdir = max(Tdir,0.0)
        Tcd[nc] = Tdb
        Tcs[nc] = Tdir
        if Tcs[nc] < 0.0: izero=1
#        Tcu[nc] = Tdif

    if izero == 1:    # block negative diffuse irrads
        for nc in range(ncld):
            Tcs[nc] = 0.0

    return Tcd,Tcs

