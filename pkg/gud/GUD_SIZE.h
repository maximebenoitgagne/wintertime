#ifdef ALLOW_GUD

CBOP
C    !ROUTINE: GUD_SIZE.h
C    !INTERFACE:
C #include GUD_SIZE.h

C    !DESCRIPTION:
C Contains dimensions and index ranges for gud model.
C
C right now, some bits of code assume that
C
C   iMinPrey = 1
C   iMaxPred = nplank
C   nChl = nPhoto

      integer nplank, nGroup, nlam, nopt
      integer nPhoto
      integer iMinBact, iMaxBact
      integer iMinPrey, iMaxPrey
      integer iMinPred, iMaxPred
      integer nPPplank
      integer nGRplank
      integer nChl
      parameter(nplank=4)
      parameter(nGroup=5)
      parameter(nlam=1)
      parameter(nopt=1)
      parameter(nPhoto=2)
      parameter(iMinBact=nPhoto+1, iMaxBact=iMinBact)
      parameter(iMinPrey=1, iMaxPrey=iMaxBact)
      parameter(iMinPred=iMaxBact+1, iMaxPred=nplank)
      parameter(nChl=nPhoto)
      parameter(nPPplank=0)
      parameter(nGRplank=0)

CEOP
#endif /* ALLOW_GUD */
