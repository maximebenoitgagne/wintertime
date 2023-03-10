#include "GCHEM_OPTIONS.h"
#ifdef ALLOW_DIC
# include "DIC_OPTIONS.h"
#endif
#ifdef ALLOW_BLING
# include "BLING_OPTIONS.h"
#endif
#ifdef ALLOW_DARWIN
# include "DARWIN_OPTIONS.h"
#endif
#ifdef ALLOW_GUD
#include "GUD_OPTIONS.h"
#endif

CBOP
C !ROUTINE: GCHEM_FORCING_SEP
C !INTERFACE: ==========================================================
      SUBROUTINE GCHEM_FORCING_SEP( myTime, myIter, myThid )

C !DESCRIPTION:
C     calls subroutine that will update passive tracers values
C     with a separate timestep. Since GCHEM_FORCING_SEP is now
C     called before DO_FIELDS_BLOCKING_EXCHANGES, the passive
C     tracer values in the halo regions are not up to date and
C     must not be used.

C !USES: ===============================================================
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "DYNVARS.h"
#include "PTRACERS_SIZE.h"
#include "PTRACERS_PARAMS.h"
#include "PTRACERS_FIELDS.h"
#include "GCHEM.h"
#ifdef ALLOW_DIC
# include "DIC_VARS.h"
#endif /* ALLOW_DIC */
#ifdef ALLOW_BLING
# include "BLING_VARS.h"
#endif /* ALLOW_BLING */
#ifdef ALLOW_DARWIN
# include "DARWIN_FLUX.h"
# include "DARWIN_SIZE.h"
#endif
#ifdef ALLOW_GUD
#include "GUD_SIZE.h"
#include "GUD_INDICES.h"
#include "GUD_GENPARAMS.h"
#ifdef GUD_ALLOW_CARBON
#include "GUD_FIELDS.h"
#endif
#endif

C !INPUT PARAMETERS: ===================================================
C  myThid               :: thread number
      _RL myTime
      INTEGER myIter, myThid
CEOP

#ifdef ALLOW_GCHEM
#ifdef GCHEM_SEPARATE_FORCING

C!LOCAL VARIABLES: ====================================================
C  bi,bj                :: tile indices
C  k                    :: vertical level
      INTEGER bi,bj,iMin,iMax,jMin,jMax,isub,iTracer
      INTEGER k
      _RL midTime, subTime
      PARAMETER( iMin = 1 , iMax = sNx )
      PARAMETER( jMin = 1 , jMax = sNy )
#if (defined ALLOW_OBCS) || (defined ALLOW_DIAGNOSTICS)
      INTEGER iTr
#endif
#ifdef ALLOW_DIAGNOSTICS
      CHARACTER*8 diagName
#endif /* ALLOW_DIAGNOSTICS */

#ifdef ALLOW_DEBUG
      IF (debugMode) CALL DEBUG_ENTER('GCHEM_FORCING_SEP',myThid)
#endif

#ifdef ALLOW_DIAGNOSTICS
      IF ( useDiagnostics ) THEN
       CALL TIMER_START('GCHEM_DIAGNOSTICS [GCHEM_FORCING_SEP]',myThid)
C--   fill-in tracer diagnostics before any GChem udate
       DO iTr = 1,gchem_sepFTr_num
        diagName = '        '
        WRITE(diagName,'(A5,A2)') 'GC_Tr', PTRACERS_ioLabel(iTr)
        CALL DIAGNOSTICS_FILL( pTracer(1-OLx,1-OLy,1,1,1,iTr), diagName,
     &                         0, Nr, 0, 1, 1, myThid )
       ENDDO
       CALL TIMER_STOP ('GCHEM_DIAGNOSTICS [GCHEM_FORCING_SEP]',myThid)
      ENDIF
#endif /* ALLOW_DIAGNOSTICS */

ccccccccccccccccccccccccc
c global calculations   c
ccccccccccccccccccccccccc
#ifdef ALLOW_OLD_VIRTUALFLUX
#ifdef ALLOW_DIC
# ifdef ALLOW_AUTODIFF
      IF ( .NOT.useDIC ) STOP 'ABNORMAL END: S/R GCHEM_FORCING_SEP (1)'
# else /* ALLOW_AUTODIFF */
      IF ( useDIC ) THEN
# endif /* ALLOW_AUTODIFF */
c find global surface averages
       gsm_s = 0. _d 0
       gsm_dic = 0. _d 0
       gsm_alk = 0. _d 0
       CALL GCHEM_SURFMEAN(salt,gsm_s,myThid)
       CALL GCHEM_SURFMEAN(
     &             pTracer(1-OLx,1-OLy,1,1,1,1), gsm_dic, myThid )
       print*,'mean surface dic', gsm_dic,gsm_s
       CALL GCHEM_SURFMEAN(
     &             pTracer(1-OLx,1-OLy,1,1,1,2), gsm_alk, myThid )
# ifndef ALLOW_AUTODIFF
      ENDIF
# endif /* ALLOW_AUTODIFF */
#endif /* ALLOW_DIC */
#ifdef ALLOW_DARWIN
c     IF ( useDARWIN ) THEN
c find global surface averages
       gsm_s = 0. _d 0
       gsm_dic = 0. _d 0
       gsm_alk = 0. _d 0
       CALL GCHEM_SURFMEAN(salt,gsm_s,myThid)
       CALL GCHEM_SURFMEAN(
     &             pTracer(1-OLx,1-OLy,1,1,1,iDIC), gsm_dic, myThid )
       print*,'mean surface dic', gsm_dic,gsm_s
       CALL GCHEM_SURFMEAN(
     &             pTracer(1-OLx,1-OLy,1,1,1,iALK), gsm_alk, myThid )
c     ENDIF
#endif
ccccccccccccccccccccccccccccccccccccccccccc
#endif /* ALLOW_OLD_VIRTUALFLUX */

#ifdef ALLOW_DARWIN
      IF ( useDARWIN ) THEN
        CALL DARWIN_CONS( myIter, myTime, myThid )
      ENDIF
#endif

ccccccccccccccccccccccccc
c chemical forcing      c
ccccccccccccccccccccccccc

#if defined(ALLOW_DIC) || defined(ALLOW_BLING) || defined(ALLOW_DARWIN)

C$taf loop = parallel
      DO bj=myByLo(myThid),myByHi(myThid)
C$taf loop = parallel
       DO bi=myBxLo(myThid),myBxHi(myThid)

ccccccccccccccccccccccccccc DIC cccccccccccccccccccccccccccccccc
#ifdef ALLOW_DIC
# ifdef ALLOW_AUTODIFF
        IF (.NOT.useDIC) STOP 'ABNORMAL END: S/R GCHEM_FORCING_SEP (2)'
# else /* ALLOW_AUTODIFF */
        IF ( useDIC ) THEN
# endif /* ALLOW_AUTODIFF */
#ifdef ALLOW_DEBUG
          IF (debugMode) CALL DEBUG_CALL('DIC_BIOTIC_FORCING',myThid)
#endif
#ifdef ALLOW_FE
          CALL DIC_BIOTIC_FORCING( pTracer(1-OLx,1-OLy,1,bi,bj,1),
     &                          pTracer(1-OLx,1-OLy,1,bi,bj,2),
     &                          pTracer(1-OLx,1-OLy,1,bi,bj,3),
     &                          pTracer(1-OLx,1-OLy,1,bi,bj,4),
     &                          pTracer(1-OLx,1-OLy,1,bi,bj,5),
     &                          pTracer(1-OLx,1-OLy,1,bi,bj,6),
     &                          bi, bj, iMin, iMax, jMin, jMax,
     &                          myIter, myTime, myThid )
#else
#ifdef ALLOW_O2
          CALL DIC_BIOTIC_FORCING( pTracer(1-OLx,1-OLy,1,bi,bj,1),
     &                          pTracer(1-OLx,1-OLy,1,bi,bj,2),
     &                          pTracer(1-OLx,1-OLy,1,bi,bj,3),
     &                          pTracer(1-OLx,1-OLy,1,bi,bj,4),
     &                          pTracer(1-OLx,1-OLy,1,bi,bj,5),
     &                          bi, bj, iMin, iMax, jMin, jMax,
     &                          myIter, myTime, myThid )
#else
          CALL DIC_BIOTIC_FORCING( pTracer(1-OLx,1-OLy,1,bi,bj,1),
     &                          pTracer(1-OLx,1-OLy,1,bi,bj,2),
     &                          pTracer(1-OLx,1-OLy,1,bi,bj,3),
     &                          pTracer(1-OLx,1-OLy,1,bi,bj,4),
     &                          bi, bj, iMin, iMax, jMin, jMax,
     &                          myIter, myTime, myThid )
#endif
#endif
# ifndef ALLOW_AUTODIFF
        ENDIF
# endif /* ALLOW_AUTODIFF */
#endif /* ALLOW_DIC */
cccccccccccccccccccccccccc END DIC cccccccccccccccccccccccccccccccccc

ccccccccccccccccccccccccccc BLING cccccccccccccccccccccccccccccccc
#ifdef ALLOW_BLING
        IF ( useBLING ) THEN
          CALL BLING_MAIN( pTracer(1-OLx,1-OLy,1,bi,bj,1),
     &                          pTracer(1-OLx,1-OLy,1,bi,bj,2),
     &                          pTracer(1-OLx,1-OLy,1,bi,bj,3),
     &                          pTracer(1-OLx,1-OLy,1,bi,bj,4),
     &                          pTracer(1-OLx,1-OLy,1,bi,bj,5),
     &                          pTracer(1-OLx,1-OLy,1,bi,bj,6),
     &                          pTracer(1-OLx,1-OLy,1,bi,bj,7),
     &                          pTracer(1-OLx,1-OLy,1,bi,bj,8),
# ifdef ADVECT_PHYTO
     &                          pTracer(1-OLx,1-OLy,1,bi,bj,9),
# endif
     &                          bi, bj, iMin, iMax, jMin, jMax,
     &                          myIter, myTime, myThid )
        ENDIF
#endif /* ALLOW_BLING */
cccccccccccccccccccccccccc END BLING cccccccccccccccccccccccccccccccccc

cccccccccccccccccccccccccc DARWIN ccccccccccccccccccccccccccccccccccc
#ifdef ALLOW_DARWIN
        IF ( useDARWIN ) THEN
#ifdef NUT_SUPPLY
c articficial supply of nutrients
#ifdef ALLOW_DEBUG
          IF (debugMode) CALL DEBUG_CALL('DARWIN_NUT_SUPPLY',myThid)
#endif
          CALL DARWIN_NUT_SUPPLY( pTracer(1-OLx,1-OLy,1,bi,bj,1),
     &                          bi, bj, iMin, iMax, jMin, jMax,
     &                          myIter, myTime, myThid )
          CALL DARWIN_NUT_SUPPLY( pTracer(1-OLx,1-OLy,1,bi,bj,2),
     &                          bi, bj, iMin, iMax, jMin, jMax,
     &                          myIter, myTime, myThid )
          CALL DARWIN_NUT_SUPPLY( pTracer(1-OLx,1-OLy,1,bi,bj,3),
     &                          bi, bj, iMin, iMax, jMin, jMax,
     &                          myIter, myTime, myThid )
          CALL DARWIN_NUT_SUPPLY( pTracer(1-OLx,1-OLy,1,bi,bj,4),
     &                          bi, bj, iMin, iMax, jMin, jMax,
     &                          myIter, myTime, myThid )
#endif
ccccccccccccccc
C darwin_forcing operates on bi,bj part only, but needs to get full
C array because of last (iPtr) index
#ifdef ALLOW_DEBUG
          IF (debugMode) CALL DEBUG_CALL('DARWIN_FORCING',myThid)
#endif
          CALL DARWIN_FORCING(  pTracer(1-OLx,1-OLy,1,1,1,1),
     &                          bi, bj, iMin, iMax, jMin, jMax,
     &                          myIter, myTime, myThid )
        ENDIF
#endif /* ALLOW_DARWIN */
cccccccccccccccccccccccccc END DARWIN ccccccccccccccccccccccccccccccc

#ifdef ALLOW_OBCS
C--   Apply (again) open boundary conditions for each passive tracer
C Note: could skip this 2nd call to OBCS_APPLY if all DIC/DARWIN
C       updates of ptracers were only done in the interior (i.e. with
C       tendency multiplied by maskInC)
        IF ( useOBCS .AND. .NOT.useDIC ) THEN
#ifdef ALLOW_DEBUG
          IF (debugMode) CALL DEBUG_CALL('OBCS_APPLY_PTRACER',myThid)
#endif
          DO iTr = 1,gchem_sepFTr_num
            CALL OBCS_APPLY_PTRACER(
     I                bi, bj, 0, iTr,
     U                pTracer(1-OLx,1-OLy,1,bi,bj,iTr),
     I                myThid )
          ENDDO
        ENDIF
#endif /* ALLOW_OBCS */

C--   end bi,bj loops.
       ENDDO
      ENDDO

#endif /* DIC or BLING or DARWIN */

#ifdef ALLOW_DARWIN
      IF ( useDARWIN ) THEN
         CALL DARWIN_CONS( myIter, myTime, myThid )
#ifdef ALLOW_CARBON
         CALL DIC_ATMOS( 1, myTime, myIter, myThid )
#endif
      ENDIF
#endif /* ALLOW_DARWIN */

#ifdef ALLOW_DIC
# ifdef ALLOW_AUTODIFF
      IF ( .NOT.useDIC ) STOP 'ABNORMAL END: S/R GCHEM_FORCING_SEP (3)'
# else /* ALLOW_AUTODIFF */
      IF ( useDIC ) THEN
# endif /* ALLOW_AUTODIFF */
#ifdef ALLOW_DEBUG
       IF (debugMode) CALL DEBUG_CALL('DIC_ATMOS',myThid)
#endif
       CALL DIC_ATMOS( myTime, myIter, myThid )
# ifdef COMPONENT_MODULE
       CALL DIC_STORE_FLUXCO2( myTime, myIter, myThid )
# endif
# ifdef ALLOW_COST
       CALL DIC_COST( myTime, myIter, myThid )
# endif
# ifndef ALLOW_AUTODIFF
      ENDIF
# endif /* ALLOW_AUTODIFF */
#endif /* ALLOW_DIC */


ccccccccccccccccccccccccccc GUD ccccccccccccccccccccccccccccc
#ifdef ALLOW_GUD
       IF ( useGUD ) THEN

         IF ( GUD_linFSConserve ) THEN
           CALL TIMER_START('GUD_CONS_SURF [GCHEM_FORCING_SEP]',myThid)
           CALL GUD_CONSERVE_SURFACE( myTime, myIter, myThid )
           CALL TIMER_STOP ('GUD_CONS_SURF [GCHEM_FORCING_SEP]',myThid)
         ENDIF

         CALL TIMER_START('GUD_CONS [GCHEM_FORCING_SEP]',myThid)
         CALL GUD_CONS( myTime, myIter, myThid )
         CALL TIMER_STOP ('GUD_CONS [GCHEM_FORCING_SEP]',myThid)

#ifdef GUD_ALLOW_CARBON
#ifdef ALLOW_OLD_VIRTUALFLUX
c find global surface averages
         CALL TIMER_START('GCHEM_SURFMEAN [GCHEM_FORCING_SEP]',myThid)
         CALL GCHEM_SURFMEAN(salt, gsm_s, myThid)
         CALL GCHEM_SURFMEAN(
     &             pTracer(1-OLx,1-OLy,1,1,1,iDIC), gsm_dic, myThid )
         CALL GCHEM_SURFMEAN(
     &             pTracer(1-OLx,1-OLy,1,1,1,iALK), gsm_alk, myThid )
         CALL TIMER_STOP ('GCHEM_SURFMEAN [GCHEM_FORCING_SEP]',myThid)
#endif /* ALLOW_OLD_VIRTUALFLUX */
#endif /* GUD_ALLOW_CARBON */

C$taf loop = parallel
         DO bj=myByLo(myThid),myByHi(myThid)
C$taf loop = parallel
          DO bi=myBxLo(myThid),myBxHi(myThid)

#ifdef GUD_ALLOW_NUT_SUPPLY
c articficial supply of nutrients
           CALL GUD_NUT_SUPPLY( pTracer(1-OLx,1-OLy,1,bi,bj,iPO4),
     &                          bi,bj,iMin,iMax,jMin,jMax,
     &                          myIter,myTime,myThid)
           CALL GUD_NUT_SUPPLY( pTracer(1-OLx,1-OLy,1,bi,bj,iNO3),
     &                          bi,bj,iMin,iMax,jMin,jMax,
     &                          myIter,myTime,myThid)
           CALL GUD_NUT_SUPPLY( pTracer(1-OLx,1-OLy,1,bi,bj,iFeT),
     &                          bi,bj,iMin,iMax,jMin,jMax,
     &                          myIter,myTime,myThid)
           CALL GUD_NUT_SUPPLY( pTracer(1-OLx,1-OLy,1,bi,bj,iSiO2),
     &                          bi,bj,iMin,iMax,jMin,jMax,
     &                          myIter,myTime,myThid)
#endif

           CALL TIMER_START('GUD_FORCING [GCHEM_FORCING_SEP]',myThid)
C          time at middle of sub-timestep
           midTime = myTime - deltaTclock + .5*deltaTclock/nsubtime
           subTime = myTime - deltaTclock + deltaTclock/nsubtime
           DO isub=1,nsubtime
            CALL GUD_FORCING( gchem_dTsub,iMin,iMax,jMin,jMax,bi,bj,
     &                        midTime,subTime,myIter,myThid)
            midTime = midTime + deltaTclock/nsubtime
            subTime = subTime + deltaTclock/nsubtime
           ENDDO
           CALL TIMER_STOP ('GUD_FORCING [GCHEM_FORCING_SEP]',myThid)

C--   end bi,bj loops.
          ENDDO
         ENDDO

         CALL TIMER_START('GUD_CONS [GCHEM_FORCING_SEP]',myThid)
         CALL GUD_CONS( myTime, myIter, myThid )
         CALL TIMER_STOP ('GUD_CONS [GCHEM_FORCING_SEP]',myThid)

       ENDIF
#endif /* ALLOW_GUD */
cccccccccccccccccccccccccc END GUD ccccccccccccccccccccccccccccccc
#ifdef ALLOW_DEBUG
      IF (debugMode) CALL DEBUG_LEAVE('GCHEM_FORCING_SEP',myThid)
#endif

#endif /* GCHEM_SEPARATE_FORCING */
#endif /* ALLOW_GCHEM */

      RETURN
      END
