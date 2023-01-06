#include "GUD_OPTIONS.h"

CBOP
C !ROUTINE: GUD_SINKING
C !INTERFACE: ==========================================================
      SUBROUTINE GUD_SINKING(
     I     Ptr,
     U     gTr,
     I     bi,bj,myTime,myIter,myThid)

C !DESCRIPTION:
C     compute tendencies from sinking of particulate organic matter

C !USES: ===============================================================
      IMPLICIT NONE
#include "SIZE.h"
c#include "EEPARAMS.h"
#include "GRID.h"
#include "GUD_SIZE.h"
#include "GUD_INDICES.h"
#include "GUD_GENPARAMS.h"
#include "GUD_TRAITS.h"

C !INPUT PARAMETERS: ===================================================
C  myThid :: thread number
C  Ptr    :: gud model tracers
C  dT     :: time step (for fake tendency with useEquilibriumC)
      _RL Ptr(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nGud)
      INTEGER bi,bj
      INTEGER myThid, myIter
      _RL myTime

C !INPUT/OUTPUT PARAMETERS: ============================================
C  gTr    :: computed tendencies
      _RL gTr(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nGud)
CEOP

#ifdef ALLOW_GUD

c !LOCAL VARIABLES: ====================================================
      INTEGER j,k
      _RL dzup(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL dzdn(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
      _RL flux(1-OLx:sNx+OLx,1-OLy:sNy+OLy)

      DO k=1,Nr-1
       dzup = DRF(k)*hFacC(:,:,k,bi,bj)
       dzdn = DRF(k+1)*hFacC(:,:,k+1,bi,bj)
       WHERE( dzdn.GT.0.0 )
         flux = wPIC_sink*MAX(0.0, Ptr(:,:,k,bi,bj,iPIC))
         gTr(:,:,k  ,iPIC ) = gTr(:,:,k  ,iPIC ) - flux/dzup
         gTr(:,:,k+1,iPIC ) = gTr(:,:,k+1,iPIC ) + flux/dzdn
         flux = wC_sink*MAX(0.0, Ptr(:,:,k,bi,bj,iPOC))
         gTr(:,:,k  ,iPOC ) = gTr(:,:,k  ,iPOC ) - flux/dzup
         gTr(:,:,k+1,iPOC ) = gTr(:,:,k+1,iPOC ) + flux/dzdn
         flux = wN_sink*MAX(0.0, Ptr(:,:,k,bi,bj,iPON))
         gTr(:,:,k  ,iPON ) = gTr(:,:,k  ,iPON ) - flux/dzup
         gTr(:,:,k+1,iPON ) = gTr(:,:,k+1,iPON ) + flux/dzdn
         flux = wP_sink*MAX(0.0, Ptr(:,:,k,bi,bj,iPOP))
         gTr(:,:,k  ,iPOP ) = gTr(:,:,k  ,iPOP ) - flux/dzup
         gTr(:,:,k+1,iPOP ) = gTr(:,:,k+1,iPOP ) + flux/dzdn
         flux = wSi_sink*MAX(0.0, Ptr(:,:,k,bi,bj,iPOSi))
         gTr(:,:,k  ,iPOSi) = gTr(:,:,k  ,iPOSi) - flux/dzup
         gTr(:,:,k+1,iPOSi) = gTr(:,:,k+1,iPOSi) + flux/dzdn
         flux = wFe_sink*MAX(0.0, Ptr(:,:,k,bi,bj,iPOFe))
         gTr(:,:,k  ,iPOFe) = gTr(:,:,k  ,iPOFe) - flux/dzup
         gTr(:,:,k+1,iPOFe) = gTr(:,:,k+1,iPOFe) + flux/dzdn
       ENDWHERE
       DO j = 1, nplank
         WHERE( dzdn.GT.0.0 )
      flux = wsink(j)*MAX(0.0, Ptr(:,:,k,bi,bj,ic+j-1))
      gTr(:,:,k  ,ic+j-1 )=gTr(:,:,k  ,ic+j-1 ) - flux/dzup
      gTr(:,:,k+1,ic+j-1 )=gTr(:,:,k+1,ic+j-1 ) + flux/dzdn
      flux = wswim(j)*MAX(0.0, Ptr(:,:,k+1,bi,bj,ic+j-1))
      gTr(:,:,k  ,ic+j-1 )=gTr(:,:,k  ,ic+j-1 ) + flux/dzup
      gTr(:,:,k+1,ic+j-1 )=gTr(:,:,k+1,ic+j-1 ) - flux/dzdn
         ENDWHERE
#ifdef GUD_ALLOW_NQUOTA
         WHERE( dzdn.GT.0.0 )
      flux = wsink(j)*MAX(0.0, Ptr(:,:,k,bi,bj,in+j-1))
      gTr(:,:,k  ,in+j-1 )=gTr(:,:,k  ,in+j-1 ) - flux/dzup
      gTr(:,:,k+1,in+j-1 )=gTr(:,:,k+1,in+j-1 ) + flux/dzdn
      flux = wswim(j)*MAX(0.0, Ptr(:,:,k+1,bi,bj,in+j-1))
      gTr(:,:,k  ,in+j-1 )=gTr(:,:,k  ,in+j-1 ) + flux/dzup
      gTr(:,:,k+1,in+j-1 )=gTr(:,:,k+1,in+j-1 ) - flux/dzdn
         ENDWHERE
#endif
#ifdef GUD_ALLOW_PQUOTA
         WHERE( dzdn.GT.0.0 )
      flux = wsink(j)*MAX(0.0, Ptr(:,:,k,bi,bj,ip+j-1))
      gTr(:,:,k  ,ip+j-1 )=gTr(:,:,k  ,ip+j-1 ) - flux/dzup
      gTr(:,:,k+1,ip+j-1 )=gTr(:,:,k+1,ip+j-1 ) + flux/dzdn
      flux = wswim(j)*MAX(0.0, Ptr(:,:,k+1,bi,bj,ip+j-1))
      gTr(:,:,k  ,ip+j-1 )=gTr(:,:,k  ,ip+j-1 ) + flux/dzup
      gTr(:,:,k+1,ip+j-1 )=gTr(:,:,k+1,ip+j-1 ) - flux/dzdn
         ENDWHERE
#endif
#ifdef GUD_ALLOW_SIQUOTA
         WHERE( dzdn.GT.0.0 )
      flux = wsink(j)*MAX(0.0, Ptr(:,:,k,bi,bj,isi+j-1))
      gTr(:,:,k  ,isi+j-1)=gTr(:,:,k  ,isi+j-1) - flux/dzup
      gTr(:,:,k+1,isi+j-1)=gTr(:,:,k+1,isi+j-1) + flux/dzdn
      flux = wswim(j)*MAX(0.0, Ptr(:,:,k+1,bi,bj,isi+j-1))
      gTr(:,:,k  ,isi+j-1)=gTr(:,:,k  ,isi+j-1) + flux/dzup
      gTr(:,:,k+1,isi+j-1)=gTr(:,:,k+1,isi+j-1) - flux/dzdn
         ENDWHERE
#endif
#ifdef GUD_ALLOW_FEQUOTA
         WHERE( dzdn.GT.0.0 )
      flux = wsink(j)*MAX(0.0, Ptr(:,:,k,bi,bj,ife+j-1))
      gTr(:,:,k  ,ife+j-1)=gTr(:,:,k  ,ife+j-1) - flux/dzup
      gTr(:,:,k+1,ife+j-1)=gTr(:,:,k+1,ife+j-1) + flux/dzdn
      flux = wswim(j)*MAX(0.0, Ptr(:,:,k+1,bi,bj,ife+j-1))
      gTr(:,:,k  ,ife+j-1)=gTr(:,:,k  ,ife+j-1) + flux/dzup
      gTr(:,:,k+1,ife+j-1)=gTr(:,:,k+1,ife+j-1) - flux/dzdn
         ENDWHERE
#endif
         IF (j .LE. nChl) THEN
           WHERE( dzdn.GT.0.0 )
      flux = wsink(j)*MAX(0.0, Ptr(:,:,k,bi,bj,iChl+j-1))
      gTr(:,:,k  ,iChl+j-1)=gTr(:,:,k  ,iChl+j-1)-flux/dzup
      gTr(:,:,k+1,iChl+j-1)=gTr(:,:,k+1,iChl+j-1)+flux/dzdn
      flux = wswim(j)*MAX(0.0, Ptr(:,:,k+1,bi,bj,iChl+j-1))
      gTr(:,:,k  ,iChl+j-1)=gTr(:,:,k  ,iChl+j-1)+flux/dzup
      gTr(:,:,k+1,iChl+j-1)=gTr(:,:,k+1,iChl+j-1)-flux/dzdn
           ENDWHERE
         ENDIF
       ENDDO ! j
      ENDDO ! k

#endif /* ALLOW_GUD */

      RETURN
      END

