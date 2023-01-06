#include "GUD_OPTIONS.h"

C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|
CBOP
C     !ROUTINE: GUD_SOLZ_IRRAD

C     !INTERFACE:
      SUBROUTINE GUD_SOLZ_IRRAD(
     O                       solz,
     I                       delta,
     I                       bi, bj, myThid )

C     !DESCRIPTION:
C     Compute solar zenith angle from declination
C     Cosine of zenith angle is averaged weighted by irradiance, i.e.,
C     by the cosine itself

C     !USES:
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"

C     !INPUT PARAMETERS:
C     delta :: declination in radians
      _RL delta
      integer bi, bj, myThid

C     !OUTPUT PARAMETERS:
C     solz :: solar zenith angle in degrees
      _RL solz(sNx,sNy)
CEOP

#ifdef ALLOW_GUD

C     !LOCAL VARIABLES:
      _RL lat, cosh0, h0, sinh0, ss, cc, av1, av2, cosz
      INTEGER i, j

      DO j=1,sNy
       DO i=1,sNx
        lat = YC(i,j,bi,bj)/180. _d 0*PI
        ss = SIN(delta)*SIN(lat)
        cc = COS(delta)*COS(lat)
        cosh0 = -ss/cc
        IF (cosh0.LT.-1. _d 0) cosh0 = -1. _d 0
        IF (cosh0.GT. 1. _d 0) cosh0 =  1. _d 0
        h0 = ABS(ACOS(cosh0))
        sinh0 = SIN(h0)
C       integrated cosine squared
        av2 = h0*(2*ss*ss + cc*cc) + 4*ss*cc*sinh0 + cc*cc*sinh0*cosh0
C       integrated cosine
        av1 = 2*h0*ss + 2*cc*sinh0
        IF (av1 .NE. 0 _d 0) THEN
          cosz = av2/av1
        ELSE
          cosz = 1. _d 0
        ENDIF
        IF (cosz.LT.0. _d 0) cosz = 0. _d 0
        IF (cosz.GT.1. _d 0) cosz = 1. _d 0
        solz(i,j) = ACOS(cosz)*180. _d 0/PI
       ENDDO
      ENDDO

#endif /* ALLOW_GUD */

      RETURN
      END
