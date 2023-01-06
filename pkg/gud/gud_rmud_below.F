#include "GUD_OPTIONS.h"

CBOP
C     !ROUTINE: GUD_RMUD_BELOW

C     !INTERFACE: ======================================================
      SUBROUTINE GUD_RMUD_BELOW(
     O                           rmud,
     I                           sunz, myThid )

C     !DESCRIPTION:
C     Compute average cosine for direct irradiance in the water column
C     given solar zenith angle (in degrees) above surface.
C
C     !USES: ===========================================================
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GUD_SIZE.h"
#include "GUD_GENPARAMS.h"

C     !INPUT PARAMETERS: ===============================================
C     sunz :: solar zenith angle above surface in degrees
      _RL sunz(sNx,sNy)
      INTEGER myThid

C     !OUTPUT PARAMETERS: ==============================================
C     rmud :: inverse cosine of zenith angle below surface
      _RL rmud(sNx,sNy)
CEOP

#ifdef ALLOW_GUD
#ifdef GUD_ALLOW_RADTRANS

C     !LOCAL VARIABLES: ================================================
      _RL sinbelow, solzbelow
      INTEGER i, j

      DO j=1,sNy
       DO i=1,sNx
        sinbelow = SIN(sunz(i,j)*deg2rad)/gud_refract_water
        solzbelow = ASIN(sinbelow)
        rmud(i,j) = MAX(0 _d 0, MIN(gud_rmud_max, 1./COS(solzbelow)))
       ENDDO
      ENDDO

#endif /* GUD_ALLOW_RADTRANS */
#endif /* ALLOW_GUD */

      RETURN
      END
