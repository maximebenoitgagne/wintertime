#include "GUD_OPTIONS.h"

CBOP
C !ROUTINE: GUD_TEMPFUNC

C !INTERFACE: ==========================================================
      SUBROUTINE GUD_TEMPFUNC(
     I                  Temp,
     O                  photoFun, grazFun, reminFun, mortFun, mort2Fun,
     O                  uptakeFun,
     I                  myThid)

C !DESCRIPTION: ========================================================
C     Compute temperature functions for gud model.
C
C     GUD_TEMP_VERSION 1 used in Follows et al (2007) - without max of 1
C     GUD_TEMP_VERSION 2 used in Dutkiewicz et al (2009), Hickman et al, 
C                            Monteiro etal, Barton et al
C     GUD_TEMP_RANGE gives all phyto a  specific temperature range
C                if undefined, then full Eppley/Arrenhius curve used

C !USES: ===============================================================
      IMPLICIT NONE
#include "GUD_SIZE.h"
#include "GUD_INDICES.h"
#include "GUD_GENPARAMS.h"
#include "GUD_TRAITS.h"

C !INPUT PARAMETERS: ===================================================
C  myThid               :: thread number
      INTEGER myThid
      _RL Temp

C !OUTPUT PARAMETERS: ==================================================
C grazFun :: the prey entries are for quota-style grazing,
C            the predator ones for monod-style grazing
      _RL photoFun(nplank)
      _RL grazFun(nplank)
      _RL reminFun
      _RL mortFun
      _RL mort2Fun
      _RL uptakeFun
CEOP

#ifdef ALLOW_GUD

c !LOCAL VARIABLES: ====================================================
      _RL Tkel
      _RL TempAe, Tempref, TempCoeff
      INTEGER j

#ifdef GUD_NOTEMP

      photoFun = 1.0 _d 0
      grazFun  = 1.0 _d 0
      reminFun = 1.0 _d 0
      mortFun  = 1.0 _d 0
      mort2Fun = 1.0 _d 0
      uptakeFun = 1.0 _d 0

#elif GUD_TEMP_VERSION == 1

c +++++++++++++++++++ VERSION 1 +++++++++++++++++++++++++++++++++++++++
c steph's version 1 (pseudo-Eppley)
c plankton growth function (unitless)
      DO j = 1, nPhoto
cswd -- this gives Eppley curve only
        photoFun(j) = phytoTempExp1(j)**Temp
# ifdef GUD_TEMP_RANGE
cswd -- temperature range
        photoFun(j) = photoFun(j) *
     &      exp(-phytoTempExp2(j)*
     &           abs(Temp - phytoTempOptimum(j))**phytoDecayPower(j))
# endif
        photoFun(j) = photoFun(j) - tempnorm
        photoFun(j) = phytoTempCoeff(j)*
     &                  max(photoFun(j), 1. _d -10)
        photoFun(j) = min(photoFun(j),1. _d 0)
      ENDDO
      DO j = 1,nplank
c       grazFun(j) = zooTempCoeff(j)*EXP(
c    &             zooTempExp(j)*(Temp - zooTempOptimum(j)))
        grazFun(j) = 1.0 _d 0
      ENDDO
      reminFun = 1.0 _d 0
      mortFun  = 1.0 _d 0
      mort2Fun = 1.0 _d 0
      uptakeFun = 1.0 _d 0
c ++++++++++++++ END VERSION 1 +++++++++++++++++++++++++++++++++++++++

#elif GUD_TEMP_VERSION == 2

c +++++++++++++++++++ VERSION 2 +++++++++++++++++++++++++++++++++++++++
c steph's version 2 (pseudo-Arrenhius)
      Tkel      =   273.15 _d 0
!      TempAe    = -4000.0 _d 0
!      Tempref   =   293.15 _d 0
!      TempCoeff =     0.5882 _d 0
      DO j = 1, nPhoto
cswd -- this gives Arrenhius curve only
        photoFun(j) = exp(TempAeArr*(1. _d 0/(Temp+Tkel) -
     &                                  1. _d 0/(TemprefArr) ) )
# ifdef GUD_TEMP_RANGE
cswd -- temperature range
        photoFun(j) = photoFun(j) *
     &      exp(-phytoTempExp2(j)*
     &           abs(Temp - phytoTempOptimum(j))**phytoDecayPower(j))
# endif
        photoFun(j) = photoFun(j)
        photoFun(j) = TempCoeffArr*
     &               max(photoFun(j), 1. _d -10)
C
      ENDDO
      reminFun = exp(TempAeArr*(1.0 _d 0/(Temp+Tkel) -
     &                       1.0 _d 0/(TemprefArr) ) )
      reminFun = TempCoeffArr*
     &               max(reminFun, 1. _d -10)
      DO j = 1, nplank
        grazFun(j) = reminFun
c       grazFun(j) = 1. _d 0
      ENDDO
c     reminFun = 1. _d 0
      mortFun = reminFun
      mort2Fun = reminFun
c      mortFun = 1.0 _d 0
c      mort2Fun = 1.0 _d 0
      uptakeFun = 1.0 _d 0
c ++++++++++++++ END VERSION 2 +++++++++++++++++++++++++++++++++++++++

#elif GUD_TEMP_VERSION == 3

c +++++++++++++++++++ VERSION 3 +++++++++++++++++++++++++++++++++++++++
c ben's version 3 from quota model
      TempAe    =   0.05 _d 0
      Tempref   =  20.0 _d 0
      reminFun = MAX(1 _d -10, EXP(TempAe*(Temp-Tempref)))
      DO j = 1, nplank
        photoFun(j) = MAX(1 _d -10, EXP(TempAe*(Temp-Tempref)))
        grazFun(j) = reminFun
      ENDDO
      mortFun = reminFun
      mort2Fun = reminFun
      uptakeFun = reminFun
c ++++++++++++++ END VERSION 3 +++++++++++++++++++++++++++++++++++++++

#else
#error "GUD_TEMP_VERSION must be 1, 2 or 3.  Define in GUD_OPTIONS.h"
#endif

#ifdef GUD_NOZOOTEMP
      grazFun(:) = 1.0 _d 0
#endif

#endif  /* ALLOW_GUD */

      RETURN
      END SUBROUTINE
