#include "GUD_OPTIONS.h"

C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|
CBOP
C     !ROUTINE: GUD_SOLZ_DAYTIME

C     !INTERFACE:
      SUBROUTINE GUD_SOLZ_DAYTIME(
     O                       solz,
     I                       delta,
     I                       bi, bj, myThid )

C     !DESCRIPTION:
C     Compute solar zenith angle from declination
C     Cosine of zenith angle is averaged over day-time period

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
      _RL lat, sun1, dayhrs, cosz
      INTEGER i, j

      DO j=1,sNy
       DO i=1,sNx
          lat = YC(i,j,bi,bj)/180. _d 0*3.1416 _d 0
          sun1 = -sin(delta)/cos(delta) * sin(lat)/cos(lat)
          IF (sun1.le.-0.999 _d 0) sun1=-0.999 _d 0
          IF (sun1.ge. 0.999 _d 0) sun1= 0.999 _d 0
          dayhrs = abs(acos(sun1))
          cosz = ( sin(delta)*sin(lat)+
     &            (cos(delta)*cos(lat)*sin(dayhrs)/dayhrs) )
          IF (cosz.le.0.005 _d 0) cosz=0.005 _d 0
          IF (cosz.ge. 0.999 _d 0) cosz= 0.999 _d 0
          solz(i,j) = acos(cosz)*180. _d 0/3.1416 _d 0
       ENDDO
      ENDDO

#endif /* ALLOW_GUD */

      RETURN
      END
