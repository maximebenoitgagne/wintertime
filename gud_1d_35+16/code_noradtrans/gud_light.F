#include "GUD_OPTIONS.h"

CBOP
C !ROUTINE: GUD_LIGHT
C !INTERFACE: ==========================================================
      SUBROUTINE GUD_LIGHT(
     O                         PAR,
#ifdef GUD_PARUICE
     O                         PAR_ice,
#endif
     I                         midTime, bi, bj,
     I                         myTime, myIter, myThid )

C !DESCRIPTION:

C !USES: ===============================================================
      IMPLICIT NONE
#include "SIZE.h"
#include "GRID.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
C#include "DYNVARS.h"
#include "FFIELDS.h"
#include "PTRACERS_SIZE.h"
#include "PTRACERS_FIELDS.h"
#ifdef ALLOW_GUD
#include "GUD_SIZE.h"
#include "GUD_INDICES.h"
#include "GUD_EXF.h"
#include "GUD_GENPARAMS.h"
#include "GUD_TRAITS.h"
#include "GUD_FIELDS.h"
#endif

C !INPUT PARAMETERS: ===================================================
C  myThid               :: thread number
      _RL midTime
      _RL myTime
      INTEGER bi, bj, myIter, myThid

C !OUTPUT PARAMETERS: ==================================================
      _RL PAR(sNx, sNy, Nr, nlam)
#ifdef GUD_PARUICE
      _RL PAR_ice(sNx, sNy, Nr, nlam)
#endif
CEOP

#ifdef ALLOW_GUD
#ifndef GUD_ALLOW_RADTRANS

C!LOCAL VARIABLES: ====================================================
C  i,j                  :: loop indices
C  k                    :: vertical level
      INTEGER i,j,k,l
      _RL PARlat(1-OLy:sNy+OLy)
      _RL Chl(sNx,sNy)
      _RL PARF(sNx,sNy)
      _RL PARF_ice(sNx,sNy)
      _RL iceF(sNx,sNy)
      _RL atten(sNx,sNy)

C ======================================================================
#ifdef GUD_READ_PAR
      PARF = surfPAR(1:sNx, 1:sNy, bi, bj)
#else
# ifdef GUD_USE_QSW
      PARF = -parfrac*parconv*Qsw(1:sNx, 1:sNy, bi, bj)*
     &       maskC(1:sNx, 1:sNy, 1, bi, bj)
# else
      CALL GUD_INSOL(midTime, PARlat, bj, .FALSE.)
      DO i=1,sNx
C       convert W/m2 to uEin/s/m2
        PARF(i,:) = PARlat(1:sNy)/.2174
      ENDDO
# endif /* GUD_USE_QSW */
#endif /* GUD_READ_PAR */
#ifdef ALLOW_DIAGNOSTICS
      IF (useDiagnostics .AND. myIter .GE.0) THEN
#ifdef GUD_READ_PARUICE
       CALL DIAGNOSTICS_FILL(surfPAR_ice(1-OLx,1-OLy,bi,bj),'sPAR_ice',
     &        1,1,2,bi,bj,myThid)
       CALL DIAGNOSTICS_FILL(surfPAR(1-OLx,1-OLy,bi,bj),'sPAR_ow ',
     &        1,1,2,bi,bj,myThid)
#else
       CALL DIAGNOSTICS_FILL(surfPAR(1-OLx,1-OLy,bi,bj),'surfPAR ',
     &      1,1,2,bi,bj,myThid)
#endif /* GUD_READ_PARUICE */
      ENDIF
#endif /* ALLOW_DIAGNOSTICS */

      IF (icefile .NE. ' ' .and. myiter .ge. 0) THEN
        iceF=iceFrac(1:sNx,1:sNy,bi,bj)
#ifdef GUD_READ_PARUICE
        PARF_ice=surfPAR_ice(1:sNx,1:sNy,bi,bj)
#else
cswd NOTE -- change this based on ice thickness etc
        PARF_ice=PARF * 0. _d 0
#endif /* GUD_READ_PARUICE */

#ifdef GUD_PARUICE
        PARF=PARF
        PARF_ice=PARF_ice
#else
        PARF=PARF * (1.0 _d 0 - iceF) + PARF_ice * iceF
#endif /* GUD_PARUICE */
      ENDIF

      DO k=1,Nr

       Chl = 0.0
#ifdef GUD_ALLOW_GEIDER
#ifdef GUD_ALLOW_CHLQUOTA
        DO j = 1, nPhoto
         Chl = Chl + MAX(0.0, 
     &        Ptracer(1:sNx, 1:sNy, k, bi, bj, iChl+j-1))
        ENDDO
#else
        Chl = ChlPrev(1:sNx, 1:sNy, k, bi, bj)
#endif
#else
       DO j = 1, nPhoto
        Chl = Chl + MAX(0.0,
     &    Ptracer(1:sNx, 1:sNy, k, bi, bj, ic+j-1)*R_ChlC(j))
       ENDDO
#endif

C TODO should include hFacC
       atten = (katten_w + katten_Chl*Chl)*DRF(k)
#ifdef GUD_PARUICE
#ifdef GUD_AVPAR
       PAR(:,:,k,1) = PARF*(1.0 - EXP(-atten))/atten
       PAR_ice(:,:,k,1) = PARF_ice*(1.0 - EXP(-atten))/atten
#else /* USE_MIDPAR */
       PAR(:,:,k,1) = PARF*EXP(-0.5*atten)
       PAR_ice(:,:,k,1) = PARF_ice*EXP(-0.5*atten)
#endif /* GUD_AVPAR */
       PAR(:,:,k,1) = PAR(:,:,k,1)*maskC(1:sNx,1:sNy,k,bi,bj)
       PAR_ice(:,:,k,1) = PAR_ice(:,:,k,1)*maskC(1:sNx,1:sNy,k,bi,bj)
#else
#ifdef GUD_AVPAR
       PAR(:,:,k,1) = PARF*(1.0 - EXP(-atten))/atten
#else
       PAR(:,:,k,1) = PARF*EXP(-0.5*atten)
#endif /* GUD_AVPAR */
       PAR(:,:,k,1) = PAR(:,:,k,1)*maskC(1:sNx,1:sNy,k,bi,bj)
#endif /* GUD_PARUICE */

#ifdef ALLOW_DIAGNOSTICS
       IF (useDiagnostics .AND. myIter .GE.0) THEN
        CALL DIAGNOSTICS_FILL(atten,'atten   ',k,1,3,bi,bj,myThid)
       ENDIF
#ifdef GUD_PARUICE 
       IF (useDiagnostics .AND. myIter .GE.0) THEN
        CALL DIAGNOSTICS_FILL(PARF_ice,'PARF_ice',k,1,3,bi,bj,myThid)
        CALL DIAGNOSTICS_FILL(PARF,'PARF_ow ',k,1,3,bi,bj,myThid)
       ENDIF
#else
       IF (useDiagnostics .AND. myIter .GE.0) THEN
        CALL DIAGNOSTICS_FILL(PARF,'PARF    ',k,1,3,bi,bj,myThid)
       ENDIF
#endif /* GUD_PARUICE */  
#endif /* ALLOW_DIAGNOSTICS */ 
       PARF = PARF*EXP(-atten)
#ifdef GUD_PARUICE
       PARF_ice=PARF_ice*EXP(-atten)
#endif

      ENDDO  ! k

#endif /* not GUD_ALLOW_RADTRANS */
#endif /* ALLOW_GUD */

      RETURN
      END
