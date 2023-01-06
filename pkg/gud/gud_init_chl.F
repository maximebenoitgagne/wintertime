#include "GUD_OPTIONS.h"

CBOP
C !ROUTINE: GUD_INIT_CHL
C !INTERFACE: ==========================================================
      SUBROUTINE GUD_INIT_CHL( dTsub,
     I                         bi, bj, myTime, myIter, myThid )

C !DESCRIPTION:
C     If gud_chlInitBalanced is true, initialize Chl from c
C     assuming balanced growth at PAR.
C     In any case, make sure Chl/C is between chl2cmin and chl2cmax.

C !USES: ===============================================================
      IMPLICIT NONE
#include "SIZE.h"
#include "GRID.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "DYNVARS.h"
#include "PTRACERS_SIZE.h"
#include "PTRACERS_FIELDS.h"
#ifdef ALLOW_GUD
#include "GUD_SIZE.h"
#include "GUD_INDICES.h"
#include "GUD_EXF.h"
#include "GUD_PARAMS.h"
#include "GUD_GENPARAMS.h"
#include "GUD_TRAITS.h"
#include "GUD_FIELDS.h"
#endif

C !INPUT PARAMETERS: ===================================================
C  myThid :: thread number
C  PAR    :: photosynthetically available radiation for balanced growth
      _RL dTsub(Nr)
      _RL myTime
      INTEGER bi, bj, myIter, myThid
CEOP

#ifdef ALLOW_GUD

C!LOCAL VARIABLES: ====================================================
      INTEGER j,l,i,k
      _RL PAR(sNx, sNy, Nr, nlam)
      _RL alpha_I(sNx,sNy,Nr)
      _RL chltmp(sNx, sNy, Nr)
      _RL lite, atten

C ======================================================================
C     compute PAR from current Chl
C     (usually initialized to 0 if gud_chlInitBalanced)
      IF (gud_chlInitBalanced) THEN
#ifdef GUD_ALLOW_RADTRANS
#ifdef GUD_CHL_INIT_LEGACY
       DO j=1,sNy
        DO i=1,sNx
         lite = 0 _d 0
         DO l=1,nlam
          lite = lite + WtouEins(l)*(OASIM_Ed(i,j,bi,bj,l)+
     &                               OASIM_Es(i,j,bi,bj,l))
         ENDDO
         atten = 0 _d 0
         DO k=1,nR
          IF (HFacC(i,j,k,bi,bj).GT.0. _d 0) THEN
           atten= atten + katten_w*5. _d -1*drF(k)
           IF (k.gt.1)THEN
            atten = atten + katten_w*5. _d -1*drF(k-1)
           ENDIF
           PAR(i,j,k,1) = lite*EXP(-atten)
          ENDIF
         ENDDO
        ENDDO
       ENDDO
#else
       j = -1
       CALL GUD_LIGHT_RADTRANS(PAR,myTime,bi,bj,myTime,j,myThid)
#endif
#else
C!!!!! do not take ice into account for backwards compatibility
       j = -1 
       CALL GUD_LIGHT(PAR, dTsub, bi, bj, myTime, j, myThid)
#endif
      ENDIF

      DO j = 1, nPhoto
#ifdef GUD_ALLOW_CHLQUOTA
       chltmp = Ptracer(1:sNx,1:sNy,:,bi,bj,iChl+j-1)
#else
       chltmp = 0.0 _d 0
#endif
       IF (gud_chlInitBalanced) THEN
#if defined(GUD_ALLOW_RADTRANS) && defined(GUD_CHL_INIT_LEGACY)
        alpha_I = alpha_mean(j) * PAR(:,:,:,1)
        alpha_I = 0 _d 0
#else
        alpha_I = 0.0
        DO l = 1, nlam
         alpha_I = alpha_I + alphachl(j,l) * PAR(:,:,:,l)
        ENDDO
#endif
        chltmp = Ptracer(1:sNx,1:sNy,:,bi,bj,ic+j-1)*
     &         chl2cmax(j)/(1+chl2cmax(j)*alpha_I/(2*PCmax(j)))
       ENDIF
       chltmp = MAX(Ptracer(1:sNx,1:sNy,:,bi,bj,ic+j-1)*chl2cmin(j),
     &              chltmp)
       chltmp = MIN(Ptracer(1:sNx,1:sNy,:,bi,bj,ic+j-1)*chl2cmax(j),
     &              chltmp)
#ifdef GUD_ALLOW_CHLQUOTA
       Ptracer(1:sNx,1:sNy,:,bi,bj,iChl+j-1) = chltmp
#else
#ifdef GUD_ALLOW_RADTRANS
       ChlPrev(1:sNx,1:sNy,:,bi,bj,j) = chltmp
#else
       ChlPrev(1:sNx,1:sNy,:,bi,bj)=ChlPrev(1:sNx,1:sNy,:,bi,bj)+chltmp
#endif
#endif
      ENDDO

#endif /* ALLOW_GUD */

      RETURN
      END
