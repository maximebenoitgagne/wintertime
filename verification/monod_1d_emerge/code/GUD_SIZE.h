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
      parameter(nplank=80)
      parameter(nGroup=5)
      parameter(nPhoto=78)
      parameter(iMinBact=nPhoto+1, iMaxBact=nPhoto)
      parameter(iMinPrey=1, iMaxPrey=iMaxBact)
      parameter(iMinPred=iMaxBact+1, iMaxPred=nplank)
      parameter(nChl=0)
      parameter(nPPplank=0)
      parameter(nGRplank=0)

CEOP
#endif /* ALLOW_GUD */
