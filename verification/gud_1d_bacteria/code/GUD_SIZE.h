#ifdef ALLOW_GUD

CBOP
C    !ROUTINE: GUD_SIZE.h
C    !INTERFACE:
C #include GUD_SIZE.h

C    !DESCRIPTION:
C Contains dimensions and index ranges for cell model.

      integer nplank, nGroup, nlam, nopt
      integer nPhoto
      integer iMinBact, iMaxBact
      integer iMinPrey, iMaxPrey
      integer iMinPred, iMaxPred
      integer nChl
      integer nPPplank
      integer nGRplank
      parameter(nlam=1)
      parameter(nopt=1)
      parameter(nplank=17)
      parameter(nGroup=12)
      parameter(nPhoto=6)
      parameter(iMinBact=7, iMaxBact=12)
      parameter(iMinPrey=1, iMaxPrey=iMaxBact)
      parameter(iMinPred=iMaxBact+1, iMaxPred=nplank)
      parameter(nChl=12)
      parameter(nPPplank=12)
      parameter(nGRplank=0)

CEOP
#endif /* ALLOW_GUD */
