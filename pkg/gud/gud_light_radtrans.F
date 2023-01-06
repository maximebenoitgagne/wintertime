#include "GUD_OPTIONS.h"

CBOP
C !ROUTINE: GUD_LIGHT_RADTRANS
C !INTERFACE: ==========================================================
      SUBROUTINE GUD_LIGHT_RADTRANS(
     O                         PAR,
     I                         solTime,
     I                         bi, bj,
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
#include "GUD_PARAMS.h"
#include "GUD_GENPARAMS.h"
#include "GUD_TRAITS.h"
#include "GUD_FIELDS.h"
#endif

C !INPUT PARAMETERS: ===================================================
C  myTime :: time at end of (sub)timestep
C  myThid :: thread number
      _RL solTime
      _RL myTime
      INTEGER bi, bj, myIter, myThid

C !OUTPUT PARAMETERS: ==================================================
      _RL PAR(sNx, sNy, Nr, nlam)
CEOP

#ifdef ALLOW_GUD
#ifdef GUD_ALLOW_RADTRANS

C!LOCAL VARIABLES: ====================================================
C  i,j                  :: loop indices
C  k                    :: vertical level
      LOGICAL  DIAGNOSTICS_IS_ON
      EXTERNAL DIAGNOSTICS_IS_ON
      CHARACTER*8 diagname
      INTEGER i,j,k,l,jp,klow
      INTEGER iyr,imon,iday,isec,lp,wd,mydate(4)
      _RL Edown
      _RL delta
      _RL solz(sNx,sNy)
      _RL rmud(sNx,sNy)
      _RL part
      _RL aCDOM(Nr,nlam)
      _RL phychl(nPhoto)
      _RL plankcar(nPlank)
#ifdef GUD_ALLOW_CDOM
      _RL CDOM
#else
      _RL actotref, atotref
#endif
      _RL PAR_k(Nr, nlam)
      _RL PARF_k(Nr, nlam)
      _RL PARF(sNx, sNy, Nr, nlam)
      _RL dz_k(Nr)
      _RL actot(Nr, nlam)
      _RL bctot(Nr, nlam)
      _RL bbctot(Nr, nlam)
C
      _RL a_k(Nr,nlam)
      _RL bt_k(Nr,nlam)
      _RL bb_k(Nr,nlam)
      _RL aprt_k(Nr,nlam)
      _RL btprt_k(Nr,nlam)
      _RL bbprt_k(Nr,nlam)
      _RL Edwsf(nlam),Eswsf(nlam)
      _RL Edbot(Nr,nlam),Esbot(Nr,nlam),Eubot(Nr,nlam)
      _RL Estop(Nr,nlam),Eutop(Nr,nlam)
      _RL amp1_k(Nr,nlam), amp2_k(Nr,nlam)
      _RL x_k(Nr,nlam), y_k(Nr,nlam)
      _RL r1_k(Nr,nlam), r2_k(Nr,nlam)
      _RL kappa1_k(Nr,nlam), kappa2_k(Nr,nlam)

#ifdef ALLOW_DIAGNOSTICS
      _RL Ed(sNx, sNy, Nr, nlam)
      _RL Es(sNx, sNy, Nr, nlam)
      _RL Eu(sNx, sNy, Nr, nlam)
      _RL Rirr(sNx, sNy, nlam)
#ifdef GUD_DIAG_RADTRANS_SOLUTION
      _RL Est(sNx, sNy, Nr, nlam)
      _RL Eub(sNx, sNy, Nr, nlam)
      _RL amp1(sNx, sNy, Nr, nlam)
      _RL amp2(sNx, sNy, Nr, nlam)
      _RL x3d(sNx, sNy, Nr, nlam)
      _RL y3d(sNx, sNy, Nr, nlam)
      _RL r1(sNx, sNy, Nr, nlam)
      _RL r2(sNx, sNy, Nr, nlam)
      _RL kap1(sNx, sNy, Nr, nlam)
      _RL kap2(sNx, sNy, Nr, nlam)
#endif
#ifdef GUD_DIAG_IOP
      _RL a3d(sNx, sNy, Nr, nlam)
      _RL bt3d(sNx, sNy, Nr, nlam)
      _RL bb3d(sNx, sNy, Nr, nlam)
      _RL aplk3d(sNx, sNy, Nr, nlam)
      _RL btplk3d(sNx, sNy, Nr, nlam)
      _RL bbplk3d(sNx, sNy, Nr, nlam)
      _RL aprt3d(sNx, sNy, Nr, nlam)
      _RL btprt3d(sNx, sNy, Nr, nlam)
      _RL bbprt3d(sNx, sNy, Nr, nlam)
      _RL aCDOM3d(sNx, sNy, Nr, nlam)
#endif
#endif

C ======================================================================
C--   compute solar zenith angles

      IF (gud_selectSolz.EQ.0) THEN
C       compute zenith angle at local noon

C       get current date and time of day: iyr/imon/iday+isec
        CALL CAL_GETDATE( myIter, mytime, mydate, mythid )
        CALL CAL_CONVDATE( mydate,iyr,imon,iday,isec,lp,wd,mythid )
        isec = 12*3600
        DO j=1,sNy
         DO i=1,sNx
          CALL SUN_SFCSOLZ(rad2deg, iyr, imon, iday, isec,
     I                     XC(i,j,bi,bj), YC(i,j,bi,bj),
     O                     solz(i,j))
         ENDDO
        ENDDO

      ELSEIF (gud_selectSolz.EQ.1) THEN
C       from day-time average of cosine of zenith angle

        CALL GUD_DECLINATION_SPENCER(delta, myTime, myIter, myThid)
        CALL GUD_SOLZ_DAYTIME(solz, delta, bi, bj, myThid)

      ELSEIF (gud_selectSolz.EQ.2) THEN
C       irradiance-weighted average of cosine of zenith angle

        CALL GUD_DECLINATION_SPENCER(delta, myTime, myIter, myThid)
        CALL GUD_SOLZ_IRRAD(solz, delta, bi, bj, myThid)

      ELSE

        STOP 'invalid gud_selectSolz'

      ENDIF

c     compute 1/cos(zenith) for direct light below surface
      CALL GUD_RMUD_BELOW(rmud, solz, myThid)

C ======================================================================
      DO j=1,sNy
       DO i=1,sNx
C ----------------------------------------------------------------------
        DO k=1,Nr

         dz_k(k) = drF(k)*HFacC(i,j,k,bi,bj)
         part = MAX(Ptracer(i,j,k,bi,bj,iPOP), 0. _d 0)
         DO jp=1,nPhoto
#ifdef GUD_ALLOW_CHLQUOTA
          phychl(jp)=MAX(Ptracer(i,j,k,bi,bj,iChl+jp-1),0.)
#else
          phychl(jp)=MAX(chlPrev(i,j,k,bi,bj,jp), 0.)
#endif
         ENDDO
         DO jp=1,nPlank
          plankcar(jp)=MAX(Ptracer(i,j,k,bi,bj,ic+jp-1),0.)
         ENDDO

#ifdef GUD_ALLOW_CDOM
c use cdom-like tracer
         CDOM = MAX(Ptracer(i,j,k,bi,bj,iCDOM), 0.0 _d 0)
         DO l = 1,nlam
          aCDOM(k,l) = CDOMcoeff*CDOM*exCDOM(l)
         ENDDO
#else
         actotref = 0.0 _d 0
         atotref = 0.0 _d 0
         DO jp = 1,nPhoto
c         nb. n,k swapped from WG
          actotref = actotref +
     &          phychl(jp)*aphy_chl(jp,laCDOM)
         ENDDO
         atotref = aw(laCDOM) + actotref
         DO l = 1,nlam
          aCDOM(k,l) = gud_aCDOM_fac*atotref*exCDOM(l)
         ENDDO
#endif

         DO l = 1,nlam
c         absorption by phyto
          actot(k,l) = 0.0
          bctot(k,l) = 0.0
          bbctot(k,l) = 0.0
          DO jp = 1, nPhoto
           actot(k,l)  = actot(k,l)  + phychl(jp)*aphy_chl(jp,l)
          ENDDO
          DO jp = 1, nPlank
c          convert mmol C to mg C
           bctot(k,l)  = bctot(k,l)  + plankcar(jp)*bphy_mgC(jp,l)*12
           bbctot(k,l) = bbctot(k,l) + plankcar(jp)*bbphy_mgC(jp,l)*12
          ENDDO
c         add water, CDOM and particles
          aprt_k(k,l) =  part*apart_P(l)
          btprt_k(k,l) = part*bpart_P(l)
          bbprt_k(k,l) = part*bbpart_P(l)
          a_k(k,l) = aw(l) + aCDOM(k,l) + actot(k,l)  + aprt_k(k,l)
          bt_k(k,l) = bw(l)             + bctot(k,l)  + btprt_k(k,l)
          bb_k(k,l) = gud_bbw*bw(l)     + bbctot(k,l) + bbprt_k(k,l)
          bb_k(k,l) = MAX(gud_bbmin, bb_k(k,l))
         ENDDO

C       k
        ENDDO
C ----------------------------------------------------------------------

C use read-in light
        DO l = 1,nlam
          Edwsf(l) = OASIM_Ed(i,j,bi,bj,l)
          Eswsf(l) = OASIM_Es(i,j,bi,bj,l)
        ENDDO

        IF (icefile .NE. ' ' .and. myiter .ge. 0) THEN
         DO l = 1,nlam
          Edwsf(l) = Edwsf(l)*(1.0 _d 0 - iceFrac(i,j,bi,bj))
          Eswsf(l) = Eswsf(l)*(1.0 _d 0 - iceFrac(i,j,bi,bj))
         ENDDO
        ENDIF

        klow = MIN(gud_radtrans_kmax, kLowC(i,j,bi,bj))

        CALL GUD_RADTRANS_DIRECT(
     I                dz_k,rmud(i,j),Edwsf,Eswsf,a_k,bt_k,bb_k,klow,
     O                Edbot,Esbot,Eubot,Estop,Eutop,
     O                PAR_k, PARF_k,
     O                amp1_k,amp2_k, x_k, y_k,
     O                r1_k,r2_k,kappa1_k,kappa2_k,
     I                myThid)

        DO l = 1,nlam
         DO k = 1,Nr
          PAR(i,j,k,l) = PAR_k(k,l)
         ENDDO
        ENDDO

#ifdef ALLOW_DIAGNOSTICS
        DO l = 1,nlam
         Ed(i,j,1,l) = Edwsf(l)
         Es(i,j,1,l) = Eswsf(l)
         Edown = Edwsf(l) + Eswsf(l)
         IF (Edown .GT. 0) THEN
           Rirr(i,j,l) = Eutop(1,l)/Edown
         ELSE
           Rirr(i,j,l) = 0 _d 0
         ENDIF
#ifdef GUD_DIAG_RADTRANS_SOLUTION
         Eub(i,j,1,l) = 0.0 _d 0
#endif
         DO k = 1,Nr-1
          Ed(i,j,k+1,l) = Edbot(k,l)
          Es(i,j,k+1,l) = Esbot(k,l)
#ifdef GUD_DIAG_RADTRANS_SOLUTION
          Eub(i,j,k+1,l) = Eubot(k,l)
#endif
         ENDDO
         DO k = 1,Nr
          PARF(i,j,k,l) = PARF_k(k,l)
          Eu(i,j,k,l) = Eutop(k,l)
#ifdef GUD_DIAG_IOP
          a3d(i,j,k,l) = a_k(k,l)
          bt3d(i,j,k,l) = bt_k(k,l)
          bb3d(i,j,k,l) = bb_k(k,l)
          aplk3d(i,j,k,l) = actot(k,l)
          btplk3d(i,j,k,l) = bctot(k,l)
          bbplk3d(i,j,k,l) = bbctot(k,l)
          aprt3d(i,j,k,l) = aprt_k(k,l)
          btprt3d(i,j,k,l) = btprt_k(k,l)
          bbprt3d(i,j,k,l) = bbprt_k(k,l)
          aCDOM3d(i,j,k,l) = aCDOM(k,l)
#endif
#ifdef GUD_DIAG_RADTRANS_SOLUTION
          Est(i,j,k,l) = Estop(k,l)
          amp1(i,j,k,l) = amp1_k(k,l)
          amp2(i,j,k,l) = amp2_k(k,l)
          x3d(i,j,k,l) = x_k(k,l)
          y3d(i,j,k,l) = y_k(k,l)
          r1(i,j,k,l) = r1_k(k,l)
          r2(i,j,k,l) = r2_k(k,l)
          kap1(i,j,k,l) = kappa1_k(k,l)
          kap2(i,j,k,l) = kappa2_k(k,l)
#endif
         ENDDO
        ENDDO
#endif

C      i,j
       ENDDO
      ENDDO

C ======================================================================

#ifdef ALLOW_DIAGNOSTICS
      IF (useDIAGNOSTICS .AND. myIter .GE.0) THEN
       CALL DIAGNOSTICS_FILL(rmud,'rmud    ',1,1,3,bi,bj,myThid)
      DO l = 1, nlam
       WRITE(diagname, '(A,I3.3)') 'Rirr', l
       CALL DIAGNOSTICS_FILL(Rirr(1,1,l),diagname,0,1,3,bi,bj,myThid)
       WRITE(diagname, '(A,I3.3)') 'Ed', l
       CALL DIAGNOSTICS_FILL(Ed(1,1,1,l),diagname,0,Nr,3,bi,bj,myThid)
       WRITE(diagname, '(A,I3.3)') 'Es', l
       CALL DIAGNOSTICS_FILL(Es(1,1,1,l),diagname,0,Nr,3,bi,bj,myThid)
       WRITE(diagname, '(A,I3.3)') 'Eu', l
       CALL DIAGNOSTICS_FILL(Eu(1,1,1,l),diagname,0,Nr,3,bi,bj,myThid)
       WRITE(diagname, '(A,I3.3)') 'PARF', l
       CALL DIAGNOSTICS_FILL(PARF(1,1,1,l),diagname,0,Nr,3,bi,bj,myThid)
#ifdef GUD_DIAG_RADTRANS_SOLUTION
       WRITE(diagname, '(A,I3.3)') 'Estop', l
       CALL DIAGNOSTICS_FILL(Est(1,1,1,l),diagname,0,Nr,3,bi,bj,myThid)
       WRITE(diagname, '(A,I3.3)') 'Eubot', l
       CALL DIAGNOSTICS_FILL(Eub(1,1,1,l),diagname,0,Nr,3,bi,bj,myThid)
       WRITE(diagname, '(A,I3.3)') 'amp1_', l
       CALL DIAGNOSTICS_FILL(amp1(1,1,1,l),diagname,0,Nr,3,bi,bj,myThid)
       WRITE(diagname, '(A,I3.3)') 'amp2_', l
       CALL DIAGNOSTICS_FILL(amp2(1,1,1,l),diagname,0,Nr,3,bi,bj,myThid)
       WRITE(diagname, '(A,I3.3)') 'x_', l
       CALL DIAGNOSTICS_FILL(x3d(1,1,1,l),diagname,0,Nr,3,bi,bj,myThid)
       WRITE(diagname, '(A,I3.3)') 'y_', l
       CALL DIAGNOSTICS_FILL(y3d(1,1,1,l),diagname,0,Nr,3,bi,bj,myThid)
       WRITE(diagname, '(A,I3.3)') 'r1_', l
       CALL DIAGNOSTICS_FILL(r1(1,1,1,l),diagname,0,Nr,3,bi,bj,myThid)
       WRITE(diagname, '(A,I3.3)') 'r2_', l
       CALL DIAGNOSTICS_FILL(r2(1,1,1,l),diagname,0,Nr,3,bi,bj,myThid)
       WRITE(diagname, '(A,I3.3)') 'att1_', l
       CALL DIAGNOSTICS_FILL(kap1(1,1,1,l),diagname,0,Nr,3,bi,bj,myThid)
       WRITE(diagname, '(A,I3.3)') 'att2_', l
       CALL DIAGNOSTICS_FILL(kap2(1,1,1,l),diagname,0,Nr,3,bi,bj,myThid)
#endif
#ifdef GUD_DIAG_IOP
       WRITE(diagname, '(A,I3.3)') 'a', l
       CALL DIAGNOSTICS_FILL(a3d(1,1,1,l),diagname,0,Nr,3,bi,bj,myThid)
       WRITE(diagname, '(A,I3.3)') 'bt', l
       CALL DIAGNOSTICS_FILL(bt3d(1,1,1,l),diagname,0,Nr,3,bi,bj,myThid)
       WRITE(diagname, '(A,I3.3)') 'bb', l
       CALL DIAGNOSTICS_FILL(bb3d(1,1,1,l),diagname,0,Nr,3,bi,bj,myThid)
       WRITE(diagname, '(A,I3.3)') 'aplk', l
       CALL DIAGNOSTICS_FILL(
     &                    aplk3d(1,1,1,l),diagname,0,Nr,3,bi,bj,myThid)
       WRITE(diagname, '(A,I3.3)') 'btplk', l
       CALL DIAGNOSTICS_FILL(
     &                    btplk3d(1,1,1,l),diagname,0,Nr,3,bi,bj,myThid)
       WRITE(diagname, '(A,I3.3)') 'bbplk', l
       CALL DIAGNOSTICS_FILL(
     &                    bbplk3d(1,1,1,l),diagname,0,Nr,3,bi,bj,myThid)
       WRITE(diagname, '(A,I3.3)') 'aprt', l
       CALL DIAGNOSTICS_FILL(
     &                    aprt3d(1,1,1,l),diagname,0,Nr,3,bi,bj,myThid)
       WRITE(diagname, '(A,I3.3)') 'btprt', l
       CALL DIAGNOSTICS_FILL(
     &                    btprt3d(1,1,1,l),diagname,0,Nr,3,bi,bj,myThid)
       WRITE(diagname, '(A,I3.3)') 'bbprt', l
       CALL DIAGNOSTICS_FILL(
     &                    bbprt3d(1,1,1,l),diagname,0,Nr,3,bi,bj,myThid)
       WRITE(diagname, '(A,I3.3)') 'aCDOM', l
       CALL DIAGNOSTICS_FILL(
     &                    aCDOM3d(1,1,1,l),diagname,0,Nr,3,bi,bj,myThid)
#endif
      ENDDO
       IF (DIAGNOSTICS_IS_ON('PARF    ', myThid)) THEN
        DO l = 2, nlam
         DO k = 1, Nr
          DO j = 1, sNy
           DO i = 1, sNx
            PARF(i,j,k,1) = PARF(i,j,k,1) + PARF(i,j,k,l)
           ENDDO
          ENDDO
         ENDDO
        ENDDO
        WRITE(diagname, '(A)') 'PARF'
        CALL DIAGNOSTICS_FILL(PARF,diagname,0,Nr,3,bi,bj,myThid)
       ENDIF
      ENDIF
#endif

#endif /* GUD_ALLOW_RADTRANS */
#endif /* ALLOW_GUD */

      RETURN
      END
