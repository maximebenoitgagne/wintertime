#ifdef ALLOW_GUD

CBOP
C    !ROUTINE: GUD_INDICES.h
C    !INTERFACE:
C #include GUD_INDICES.h

C    !DESCRIPTION:
C Contains indices into ptracer array

C these cannot be modified for now

      INTEGER nNQuota, nPQuota, nSiQuota, nFeQuota
#ifdef GUD_ALLOW_NQUOTA
      PARAMETER(nNQuota=nplank)
#else
      PARAMETER(nNQuota=0)
#endif
#ifdef GUD_ALLOW_PQUOTA
      PARAMETER(nPQuota=nplank)
#else
      PARAMETER(nPQuota=0)
#endif
#ifdef GUD_ALLOW_SIQUOTA
      PARAMETER(nSiQuota=nplank)
#else
      PARAMETER(nSiQuota=0)
#endif
#ifdef GUD_ALLOW_FEQUOTA
      PARAMETER(nFeQuota=nplank)
#else
      PARAMETER(nFeQuota=0)
#endif

      integer iDIC
      integer iNH4
      integer iNO2
      integer iNO3
      integer iPO4
      integer iSiO2
      integer iFeT
      integer iDOC
      integer iDON
      integer iDOP
      integer iDOFe
      integer iPOC
      integer iPON
      integer iPOP
      integer iPOSi
      integer iPOFe
      integer iPIC
#ifdef GUD_ALLOW_CARBON
      integer iALK
      integer iO2
#endif
#ifdef GUD_ALLOW_CDOM
      integer iCDOM
#endif
      integer ic
      integer in
      integer ip
      integer isi
      integer ife
      integer iChl
      integer eCARBON
      integer eCDOM
      integer nGud
      PARAMETER (iDIC   =1)
      PARAMETER (iNH4   =iDIC+1)
      PARAMETER (iNO2   =iNH4 +1)
      PARAMETER (iNO3   =iNO2 +1)
      PARAMETER (iPO4   =iNO3 +1)
      PARAMETER (iSiO2  =iPO4 +1)
      PARAMETER (iFeT   =iSiO2+1)
      PARAMETER (iDOC   =iFeT +1)
      PARAMETER (iDON   =iDOC +1)
      PARAMETER (iDOP   =iDON +1)
      PARAMETER (iDOFe  =iDOP +1)
      PARAMETER (iPOC   =iDOFe+1)
      PARAMETER (iPON   =iPOC +1)
      PARAMETER (iPOP   =iPON +1)
      PARAMETER (iPOSi  =iPOP +1)
      PARAMETER (iPOFe  =iPOSi+1)
      PARAMETER (iPIC   =iPOFe+1)
#ifdef GUD_ALLOW_CARBON
      PARAMETER (iALK   =iPIC +1)
      PARAMETER (iO2    =iALK +1)
      PARAMETER (eCARBON=iO2)
#else
      PARAMETER (eCARBON=iPIC)
#endif
#ifdef GUD_ALLOW_CDOM
      PARAMETER (iCDOM  =eCARBON+1)
      PARAMETER (ic     =iCDOM+1)
#else
      PARAMETER (ic     =eCARBON+1)
#endif
      PARAMETER (in     =ic   +nplank)
      PARAMETER (ip     =in   +nNQuota)
      PARAMETER (ife    =ip   +nPQuota)
      PARAMETER (isi    =ife  +nFeQuota)
      PARAMETER (iChl   =isi  +nSiQuota)

      integer ec
      integer en
      integer ep
      integer esi
      integer efe
      integer eChl
      PARAMETER (ec     =ic  +nplank-1)
      PARAMETER (en     =in  +nNQuota-1)
      PARAMETER (ep     =ip  +nPQuota-1)
      PARAMETER (esi    =isi +nSiQuota-1)
      PARAMETER (efe    =ife +nFeQuota-1)
      PARAMETER (eChl   =iChl+nChl-1)

      PARAMETER (nGud=eChl)

CEOP
#endif /* ALLOW_GUD */
