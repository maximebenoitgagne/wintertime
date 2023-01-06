#include "GUD_OPTIONS.h"

CBOP
C !ROUTINE: GUD_FE_CHEM

C !INTERFACE: ==========================================================
      SUBROUTINE GUD_FE_CHEM(
     U                        FeT,
     O                        freeFe,
     I                        bi, bj, myThid )

C !DESCRIPTION: ========================================================
C     Calculate L,FeL,Fe concentration
C
C     Stephanie Dutkiewicz, 2004
C        following from code by Payal Parekh

C !USES: ===============================================================
      IMPLICIT NONE
#include "SIZE.h"
#include "GRID.h"
#ifdef ALLOW_GUD
#include "GUD_SIZE.h"
#include "GUD_GENPARAMS.h"
#include "GUD_TRAITS.h"
#endif

C !INPUT PARAMETERS: ===================================================
      INTEGER bi, bj, myThid

C !INPUT/OUTPUT PARAMETERS: ============================================
      _RL FeT(1-OLx:sNx+OLx, 1-OLy:sNy+OLy, Nr)

C !OUTPUT PARAMETERS: ==================================================
      _RL freeFe(1-OLx:sNx+OLx, 1-OLy:sNy+OLy, Nr)
CEOP

#ifdef ALLOW_GUD

c !LOCAL VARIABLES: ====================================================
      INTEGER i, j, k
      _RL lig
      _RL FeL

      DO k=1,Nr
       DO j=1,sNy
        DO i=1,sNx
         IF (maskC(i, j, k, bi, bj) .GT. 0.0 _d 0) THEN
          lig = (-ligand_stab*FeT(i,j,k) +
     &            ligand_stab*ligand_tot - 
     &            1. _d 0 +
     &            ( ( ligand_stab*FeT(i,j,k) -
     &                ligand_stab*ligand_tot+1. _d 0
     &              )**2 +
     &              4. _d 0*ligand_stab*ligand_tot
     &            )**0.5 _d 0
     &          )/(2. _d 0*ligand_stab)
          
          FeL = ligand_tot - lig
          freefe(i,j,k) = FeT(i,j,k) - FeL
#ifdef GUD_MINFE
          freefe(i,j,k) = MIN(freefe(i,j,k), freefemax)
          IF (maskInC(i,j,bi,bj) .GT. 0.0) THEN
           FeT(i,j,k) = FeL + freefe(i,j,k)
          ENDIF
#endif
         ELSE
          freefe(i,j,k) = 0.0 _d 0
         ENDIF
        ENDDO
       ENDDO
      ENDDO

#endif /* ALLOW_GUD */

      RETURN
      END SUBROUTINE
