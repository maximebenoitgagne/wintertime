C TODO
C - have to call fechem after model?

#include "GUD_OPTIONS.h"

CBOP
C !ROUTINE: GUD_FORCING
C !INTERFACE: ==========================================================
      SUBROUTINE GUD_FORCING( dTsub, iMin, iMax, jMin, jMax, bi, bj,
     &                  midTime, myTime, myIter, myThid )

C !DESCRIPTION:

C !USES: ===============================================================
      IMPLICIT NONE
#include "SIZE.h"
#include "GRID.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "DYNVARS.h"
#include "PTRACERS_SIZE.h"
#include "PTRACERS_PARAMS.h"
#include "PTRACERS_FIELDS.h"
#ifdef ALLOW_GUD
#include "GCHEM.h"
#include "GUD_SIZE.h"
#include "GUD_INDICES.h"
#include "GUD_DIAGS.h"
#include "GUD_GENPARAMS.h"
#include "GUD_TRAITS.h"
#include "GUD_FIELDS.h"
#include "GUD_EXF.h"
#endif

C !INPUT PARAMETERS: ===================================================
C  myThid               :: thread number
      _RL dTsub(Nr)
      _RL midTime
      _RL myTime
      INTEGER iMin, iMax, jMin, jMax, bi, bj, myIter, myThid
CEOP

#ifdef ALLOW_GUD

C!LOCAL VARIABLES: ====================================================
C  i,j                  :: loop indices
C  k                    :: vertical level
      LOGICAL  DIAGNOSTICS_IS_ON
      EXTERNAL DIAGNOSTICS_IS_ON
      INTEGER i,j,k,kdn,iTr
      CHARACTER*8 diagname
      _RL gPtr(1-OLx:sNx+OLx, 1-OLy:sNy+OLy, Nr, nGud)
      _RL PAR(sNx,sNy,Nr,nlam)
#ifdef GUD_PARUICE
      _RL PAR_ice(sNx,sNy,Nr,nlam)
#endif
      _RL diags(sNx,sNy,Nr,gud_nDiag)
      _RL gDIC(1-OLx:sNx+OLx, 1-OLy:sNy+OLy)
      _RL gALK(1-OLx:sNx+OLx, 1-OLy:sNy+OLy)
      _RL gO2(1-OLx:sNx+OLx, 1-OLy:sNy+OLy)
      _RL freeFe(1-OLx:sNx+OLx, 1-OLy:sNy+OLy, Nr)
      _RL scv,scav_poc
      _RL flx
      _RL ptr(nGud), gtr(nGud), PARl(nlam)
#ifdef GUD_PARUICE
      _RL PARl_ice(nlam)
      _RL iceFracl
#endif
      _RL chlout(nPhoto)
      _RL diagsl(gud_nDiag)
      _RL photoTempFunc(nplank)
      _RL grazTempFunc(nplank)
      _RL reminTempFunc
      _RL mortTempFunc
      _RL mort2TempFunc
      _RL uptakeTempFunc
#ifdef ALLOW_DIAGNOSTICS
      INTEGER l
#endif
#ifdef GUD_DIAG_PHOTO
      INTEGER IPhoto
      _RL PCmrep(sNx,sNy,Nr,nPhoto)
      _RL PCmrepout(nPhoto)
#endif

C === reset tendencies =================================================
      DO itr=1,nGud
      DO k=1,Nr
      DO j=1,sNy
      DO i=1,sNx
        gPtr(i,j,k,iTr) = 0.0 _d 0
      ENDDO
      ENDDO
      ENDDO
      ENDDO
#ifdef GUD_DIAG_PHOTO
      DO iPhoto=1,nPhoto
      DO k=1,Nr
      DO j=1,sNy
      DO i=1,sNx
        PCmrep(i,j,k,iPhoto) = 0.0 _d 0
      ENDDO
      ENDDO
      ENDDO
      ENDDO
#endif /* GUD_DIAG_PHOTO */

C === light ============================================================
C     Initialize Chl from balanced-growth Chl:C if requested
C     and check Chl:C bounds.
C     Note: myIter has already been incremented
c      IF (myIter-1 .EQ. gud_chlIter0) THEN
c        CALL GUD_INIT_CHL(bi, bj, myTime, myIter, myThid)
c      ENDIF

      CALL TIMER_START('GUD_LIGHT [GUD_FORCING]',myThid)
#ifdef GUD_ALLOW_RADTRANS
      CALL GUD_LIGHT_RADTRANS(PAR,myTime,bi,bj,myTime,myIter,myThid)
#else
      CALL GUD_LIGHT(
     O               PAR,
#ifdef GUD_PARUICE
     O               PAR_ice,
#endif
     I               midTime, bi, bj, myTime, myIter, myThid)
#endif
      CALL TIMER_STOP ('GUD_LIGHT [GUD_FORCING]',myThid)

C === dic ==============================================================
#ifdef GUD_ALLOW_CARBON
C carbon air-sea interaction
      CALL TIMER_START('GUD_SURFFORCING [GUD_FORCING]',myThid)
      CALL GUD_SURFFORCING(
     O                    gDIC, gALK, gO2,
     I                    bi,bj,imin,imax,jmin,jmax,
     I                    myIter,myTime,myThid)
      CALL TIMER_STOP ('GUD_SURFFORCING [GUD_FORCING]',myThid)
      DO j=1,sNy
      DO i=1,sNx
        gPtr(i,j,1,iDIC) = gPtr(i,j,1,iDIC) + gDIC(i,j)
        gPtr(i,j,1,iALK) = gPtr(i,j,1,iALK) + gALK(i,j)
        gPtr(i,j,1,iO2)  = gPtr(i,j,1,iO2)  + gO2(i,j)
      ENDDO
      ENDDO
#endif

C === iron =============================================================
      CALL TIMER_START('GUD_FE_CHEM [GUD_FORCING]',myThid)
      CALL GUD_FE_CHEM(
     U                 Ptracer(1-OLx,1-OLy,1,bi,bj,iFeT),
     O                 freeFe(1-OLx, 1-OLy, 1),
     I                 bi, bj, myThid)
      CALL TIMER_STOP ('GUD_FE_CHEM [GUD_FORCING]',myThid)

C     iron dust input
      CALL TIMER_START('GUD_FE_DUST [GUD_FORCING]',myThid)
      DO j=1,sNy
      DO i=1,sNx
      IF (hFacC(i,j,1,bi,bj) .NE. 0.) THEN
        gPtr(i,j,1,iFeT) =
     &  gPtr(i,j,1,iFeT) + alpfe * inputFe(i,j,bi,bj) /
     &                           drF(1) / hFacC(i,j,1,bi,bj)
      ENDIF
      ENDDO
      ENDDO
      CALL TIMER_STOP ('GUD_FE_DUST [GUD_FORCING]',myThid)

C     scavenging
      CALL TIMER_START('GUD_FE_SCAV [GUD_FORCING]',myThid)
      DO k=1,Nr
      DO j=1,sNy
      DO i=1,sNx
#ifdef GUD_PART_SCAV_POP
        scav_poc = MAX(0 _d 0, Ptracer(i,j,k,bi,bj,iPOP))/scav_R_POPPOC
        scv = scav_rat*scav_inter*(scav_poc**scav_exp)
#elif defined(GUD_PART_SCAV)
        scav_poc = MAX(0 _d 0, Ptracer(i,j,k,bi,bj,iPOC))
        scv = scav_rat*scav_inter*(scav_poc**scav_exp)
#else
        scv = scav
#endif
        gPtr(i,j,k,iFeT) = gPtr(i,j,k,iFeT) -
     &                           scv*freefe(i,j,k)
      ENDDO
      ENDDO
      ENDDO
      CALL TIMER_STOP ('GUD_FE_SCAV [GUD_FORCING]',myThid)

C     iron sediment source (in bottom grid cell above kMaxFeSed)
      CALL TIMER_START('GUD_FE_SED [GUD_FORCING]',myThid)
      DO k = kMinFeSed, kMaxFeSed
       kdn = MIN(Nr, k+1)
       DO j=1,sNy
        DO i=1,sNx
         IF (hFacC(i,j,k,bi,bj) .GT. 0. .AND.
     &       (hFacC(i,j,kdn,bi,bj) .EQ. 0. .OR. k .EQ. Nr)) THEN
#ifdef GUD_IRON_SED_SOURCE_VARIABLE
          flx = fesedflux_pcm*wp_sink*R_CP_fesed*
     &            MAX(0 _d 0, Ptracer(i,j,k-1,bi,bj,iPOP))
#else
          flx = fesedflux
#endif
          gPtr(i,j,k,iFeT) = gPtr(i,j,k,iFeT) +
     &                             flx/(drF(k)*hFacC(i,j,k,bi,bj))
         ENDIF
        ENDDO
       ENDDO
      ENDDO
      CALL TIMER_STOP ('GUD_FE_SED [GUD_FORCING]',myThid)

C === plankton =========================================================
      CALL TIMER_START('GUD_MODEL [GUD_FORCING]',myThid)
      do k=1,Nr
      do j=1,sNy
      do i=1,sNx
      if (hFacC(i,j,k,bi,bj) .GT. 0.) then
        CALL GUD_TEMPFUNC(Theta(i,j,k,bi,bj),
     &         photoTempFunc, grazTempFunc, reminTempFunc,
     &         mortTempFunc, mort2TempFunc,
     &         uptakeTempFunc, myThid)
        ptr(:) = Ptracer(i, j, k, bi, bj, 1:nGud)
        PARl(:) = PAR(i, j, k, :)
#ifdef GUD_PARUICE
        PARl_ice(:) = PAR_ice(i, j, k, :)
        iceFracl = iceFrac(i, j, bi, bj)
#endif
        gtr(:) = gPtr(i, j, k, 1:nGud)
        CALL GUD_MODEL(ptr,
     U                    gtr,
     O                    chlout, diagsl,
#ifdef GUD_DIAG_PHOTO
     O                    PCmrepout,
#endif
     I                    PARl, 
#ifdef GUD_PARUICE
     I                    PARl_ice,
     I                    iceFracl,
#endif
     I                    photoTempFunc, reminTempFunc,
     I                    uptakeTempFunc,
#ifdef GUD_DEBUG
     I                    myxgloballo+(bi-1)*sNx+i-1,
     I                    myygloballo+(bj-1)*sNy+j-1,
     I                    k, dTsub(k),
#endif
     I                    myTime, myIter, myThid )
        diags(i, j, k, :) = diagsl
#ifdef GUD_DIAG_PHOTO
        PCmrep(i,j,k,:) = PCmrepout
#endif
#ifdef GUD_ALLOW_CONS
        GUD_Nfix(i,j,k,bi,bj) = diagsl(iNfix)
        GUD_Ndenit(i,j,k,bi,bj) = diagsl(iDenitN)
#endif
#ifndef GUD_ALLOW_CHLQUOTA
#ifdef GUD_ALLOW_RADTRANS
        chlPrev(i, j, k, bi, bj, :) = chlout
#else
        chlPrev(i, j, k, bi, bj) = SUM(chlout)
#endif
#endif
#ifdef GUD_TIME_GRAZING
        gPtr(i, j, k, 1:nGud) = gtr(:)
      endif
      enddo
      enddo
      enddo
      CALL TIMER_STOP('GUD_MODEL [GUD_FORCING]',myThid)
      CALL TIMER_START('GUD_GRAZING [GUD_FORCING]',myThid)
      do k=1,Nr
      do j=1,sNy
      do i=1,sNx
      if (hFacC(i,j,k,bi,bj) .GT. 0.) then
        ptr(:) = Ptracer(i, j, k, bi, bj, 1:nGud)
        gtr(:) = gPtr(i, j, k, 1:nGud)
        CALL GUD_TEMPFUNC(Theta(i,j,k,bi,bj),
     &         photoTempFunc, grazTempFunc, reminTempFunc,
     &         mortTempFunc, mort2TempFunc,
     &         uptakeTempFunc, myThid)
#endif
        diagsl(:) = diags(i, j, k, :)
        CALL GUD_GRAZING(ptr,
     U                    gtr,
     U                    diagsl,
     I                    grazTempFunc, reminTempFunc,
     I                    mortTempFunc, mort2TempFunc,
     I                    myTime, myIter, myThid )
        diags(i, j, k, :) = diagsl
        gPtr(i, j, k, 1:nGud) = gtr(:)
      endif
      enddo
      enddo
      enddo
#ifdef GUD_TIME_GRAZING
      CALL TIMER_STOP('GUD_GRAZING [GUD_FORCING]',myThid)
#else
      CALL TIMER_STOP('GUD_MODEL [GUD_FORCING]',myThid)
#endif

C === sinking ==========================================================
      CALL TIMER_START('GUD_SINKING [GUD_FORCING]',myThid)
      CALL GUD_SINKING( Ptracer,gPtr,bi,bj,myTime,myIter,myThid )
      CALL TIMER_STOP ('GUD_SINKING [GUD_FORCING]',myThid)

C === apply tendencies to tracers ======================================
      CALL TIMER_START('GUD_STEP [GUD_FORCING]',myThid)
      DO iTr=1,nGud
      DO k=1,Nr
      DO j=1,sNy
      DO i=1,sNx
        pTracer(i,j,k,bi,bj,iTr)=pTracer(i,j,k,bi,bj,iTr)
     &                      +dTsub(k)*gPtr(i,j,k,iTr)*maskInC(i,j,bi,bj)
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      CALL TIMER_STOP ('GUD_STEP [GUD_FORCING]',myThid)

C === iron =============================================================
C     re-apply free iron limit to FeT
      CALL TIMER_START('GUD_FE_CHEM [GUD_FORCING]',myThid)
      CALL GUD_FE_CHEM(
     U                 pTracer(1-OLx,1-OLy,1,bi,bj,iFeT),
     O                 freeFe(1-OLx,1-OLy,1),
     I                 bi, bj, myThid)
      CALL TIMER_STOP ('GUD_FE_CHEM [GUD_FORCING]',myThid)

C === diagnostics ======================================================
#ifdef ALLOW_DIAGNOSTICS
#ifdef GUD_PARUICE
      IF (useDIAGNOSTICS) THEN
       CALL TIMER_START('DIAGS_FILL [GUD_FORCING]',myThid)
       IF (DIAGNOSTICS_IS_ON('PAR_ow  ', myThid)) THEN
        DO l = 2, nlam
         DO k = 1, Nr
          DO j = 1, sNy
           DO i = 1, sNx
            PAR(i,j,k,1) = PAR(i,j,k,1) + PAR(i,j,k,l)
           ENDDO
          ENDDO
         ENDDO
        ENDDO
        WRITE(diagname, '(A)') 'PAR_ow'
        CALL DIAGNOSTICS_FILL(PAR,diagname,0,Nr,3,bi,bj,myThid)
       ENDIF
       IF (DIAGNOSTICS_IS_ON('PAR_ice ', myThid)) THEN
        DO l = 2, nlam
         DO k = 1, Nr
          DO j = 1, sNy
           DO i = 1, sNx
            PAR_ice(i,j,k,1) = PAR_ice(i,j,k,1) + PAR_ice(i,j,k,l)
           ENDDO
          ENDDO
         ENDDO
        ENDDO
        WRITE(diagname, '(A)') 'PAR_ice'
        CALL DIAGNOSTICS_FILL(PAR_ice,diagname,0,Nr,3,bi,bj,myThid)
       ENDIF
#else
      IF (useDIAGNOSTICS) THEN
      CALL TIMER_START('DIAGS_FILL [GUD_FORCING]',myThid)
       DO l = 1, nlam
        WRITE(diagname, '(A,I3.3)') 'PAR', l
        CALL DIAGNOSTICS_FILL(PAR(1,1,1,l),diagname,0,Nr,3,bi,bj,myThid)
       ENDDO
       IF (DIAGNOSTICS_IS_ON('PAR     ', myThid)) THEN
        DO l = 2, nlam
         DO k = 1, Nr
          DO j = 1, sNy
           DO i = 1, sNx
            PAR(i,j,k,1) = PAR(i,j,k,1) + PAR(i,j,k,l)
           ENDDO
          ENDDO
         ENDDO
        ENDDO
        WRITE(diagname, '(A)') 'PAR'
        CALL DIAGNOSTICS_FILL(PAR,diagname,0,Nr,3,bi,bj,myThid)
       ENDIF
#endif /* GUD_PARUICE */
       CALL DIAGNOSTICS_FILL(diags(1,1,1,iPP),   'PP      ',0,Nr,3,
     &          bi,bj,myThid)
       CALL DIAGNOSTICS_FILL(diags(1,1,1,iNfix), 'Nfix    ',0,Nr,3,
     &          bi,bj,myThid)
       CALL DIAGNOSTICS_FILL(diags(1,1,1,iDenit),'Denit   ',0,Nr,3,
     &          bi,bj,myThid)

C      nutrient consumption diagnostics
       CALL DIAGNOSTICS_FILL(diags(1,1,1,iConsDIN),'C_DIN   ',0,Nr,3,
     &          bi,bj,myThid)
       CALL DIAGNOSTICS_FILL(diags(1,1,1,iConsPO4),'C_PO4   ',0,Nr,3,
     &          bi,bj,myThid)
       CALL DIAGNOSTICS_FILL(diags(1,1,1,iConsSi),'C_Si    ',0,Nr,3,
     &          bi,bj,myThid)
       CALL DIAGNOSTICS_FILL(diags(1,1,1,iConsFe),'C_Fe    ',0,Nr,3,
     &          bi,bj,myThid)

C      compute 'source' diagnostics: tendencies without consumption;
C      add full tendencies to consumption and store result back into
C      diags(iCons*) for convenience
       DO k = 1, Nr
        DO j = 1, sNy
         DO i = 1, sNx
            diags(i,j,k,iConsDIN) = diags(i,j,k,iConsDIN)
     &         + gPtr(i,j,k,iNH4) + gPtr(i,j,k,iNO2) + gPtr(i,j,k,iNO3)
            diags(i,j,k,iConsPO4) = diags(i,j,k,iConsPO4)
     &                            + gPtr(i,j,k,iPO4)
            diags(i,j,k,iConsSi) = diags(i,j,k,iConsSi)
     &                           + gPtr(i,j,k,iSiO2)
            diags(i,j,k,iConsFe) = diags(i,j,k,iConsFe)
     &                           + gPtr(i,j,k,iFeT)
         ENDDO
        ENDDO
       ENDDO
       CALL DIAGNOSTICS_FILL(diags(1,1,1,iConsDIN),'S_DIN   ',0,Nr,3,
     &          bi,bj,myThid)
       CALL DIAGNOSTICS_FILL(diags(1,1,1,iConsPO4),'S_PO4   ',0,Nr,3,
     &          bi,bj,myThid)
       CALL DIAGNOSTICS_FILL(diags(1,1,1,iConsSi),'S_Si    ',0,Nr,3,
     &          bi,bj,myThid)
       CALL DIAGNOSTICS_FILL(diags(1,1,1,iConsFe),'S_Fe    ',0,Nr,3,
     &          bi,bj,myThid)

       DO iTr=1,nGud
        diagname = 'gGUD'//PTRACERS_ioLabel(iTr)
        CALL DIAGNOSTICS_FILL(gPtr(1-OLx,1-OLy,1,iTr), diagname,
     &          0,Nr,2,bi,bj,myThid)
       ENDDO
       DO iTr=1,nPPplank
        WRITE(diagname, '(A,I4.4)') 'PP', iTr
        CALL DIAGNOSTICS_FILL(diags(1,1,1,iPPplank+iTr-1), diagname,
     &          0,Nr,3,bi,bj,myThid)
       ENDDO
       DO iTr=1,nGRplank
        WRITE(diagname, '(A,I4.4)') 'GR', iTr
        CALL DIAGNOSTICS_FILL(diags(1,1,1,iGRplank+iTr-1), diagname,
     &          0,Nr,3,bi,bj,myThid)
       ENDDO
#ifdef GUD_ALLOW_CARBON
       CALL DIAGNOSTICS_FILL(pH,     'pH      ',0,Nr,1,bi,bj,myThid)
       CALL DIAGNOSTICS_FILL(pCO2,   'pCO2    ',0,Nr,1,bi,bj,myThid)
       CALL DIAGNOSTICS_FILL(FluxCO2,'fluxCO2 ',0,1,1,bi,bj,myThid)
       CALL DIAGNOSTICS_FILL(gDIC,   'gDICsurf',0,1,2,bi,bj,myThid)
       CALL DIAGNOSTICS_FILL(gALK,   'gALKsurf',0,1,2,bi,bj,myThid)
       CALL DIAGNOSTICS_FILL(gO2,    'gO2surf ',0,1,2,bi,bj,myThid)
#endif
#ifdef GUD_DIAG_PHOTO
       CALL DIAGNOSTICS_FILL(PCmrep(1,1,1,15),'PCmrep15',
     &          0,Nr,3,bi,bj,myThid)
#endif
       CALL TIMER_STOP ('DIAGS_FILL [GUD_FORCING]',myThid)
      ENDIF
#endif

#endif /* ALLOW_GUD */

      RETURN
      END
