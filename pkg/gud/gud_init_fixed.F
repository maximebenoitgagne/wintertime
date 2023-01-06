#include "GUD_OPTIONS.h"
#include "EXF_OPTIONS.h"

C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|
CBOP
C     !ROUTINE: GUD_INIT_FIXED

C     !INTERFACE:
      SUBROUTINE GUD_INIT_FIXED( myThid )

C     !DESCRIPTION:
C     Initialize wavebands and read in absorption/scattering spectra

C     !USES:
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#ifdef ALLOW_GUD
#include "PTRACERS_SIZE.h"
#include "PTRACERS_PARAMS.h"
#include "GUD_SIZE.h"
#include "GUD_PARAMS.h"
#include "GUD_GENPARAMS.h"
#include "GUD_TRAITS.h"
#endif

C     !INPUT PARAMETERS:
      INTEGER myThid
CEOP

#ifdef ALLOW_GUD

C     !LOCAL VARIABLES:
      CHARACTER*(MAX_LEN_MBUF) msgBuf
      integer iUnit1, iUnit2, k, jp
      integer idummy
#ifdef GUD_ALLOW_RADTRANS
      character*80 title
      integer ios, i, l, ilambda
      _RL planck, c, hc, oavo, hcoavo, rlamm
      _RL lambdain, ain, apsin, bin, bbin
#endif

      CALL GUD_EXF_INIT_FIXED(myThid)

#ifdef ALLOW_DIAGNOSTICS
      IF ( useDiagnostics ) THEN
        CALL GUD_DIAGNOSTICS_INIT( myThid )
      ENDIF
#endif

      _BEGIN_MASTER(myThid)

      IF (gud_linFSConserve) PTRACERS_calcSurfCor = .TRUE.

#ifdef GUD_IRON_SED_SOURCE_VARIABLE
      kMinFeSed = 2
#else
      kMinFeSed = 1
#endif
      kMaxFeSed = 0
      DO k=1,Nr
        IF (RF(k) .GT. -depthFeSed) kMaxFeSed = k
      ENDDO


#ifdef GUD_ALLOW_RADTRANS

C band widths used to convert OASIM data to irradiation per nm
      wb_totalWidth = 0.0
      DO l=1,nlam
        wb_width(l) = gud_waveband_edges(l+1) -
     &                gud_waveband_edges(l)
        wb_totalWidth = wb_totalWidth + wb_width(l)
      ENDDO
      IF (wb_totalWidth.LE.0) then
        WRITE(msgBuf,'(2A)') 'GUD_INIT_FIXED: ',
     &  'please provide wavebabnds in gud_waveband_edges.'
        CALL PRINT_ERROR( msgBuf, myThid )
        STOP 'ABNORMAL END: S/R GUD_INIT_FIXED'
      ENDIF
      planck = 6.6256 _d -34   !Plancks constant J sec
      c = 2.998 _d 8                 !speed of light m/sec
      hc = 1.0/(planck*c)
      oavo = 1.0/6.023 _d 23   ! 1/Avogadros number
      hcoavo = hc*oavo
      DO l = 1,nlam
        IF (gud_waveband_centers(l) .GE. 0.0) THEN
          wb_center(l) = gud_waveband_centers(l)
        ELSE
          wb_center(l) = 0.5*(gud_waveband_edges(l) +
     &                        gud_waveband_edges(l+1))
        ENDIF
        rlamm = wb_center(l)*1 _d -9  !lambda in m
        WtouEins(l) = 1 _d 6*rlamm*hcoavo      !Watts to uEin/s conversion
      ENDDO
C     write summary
      WRITE(msgBuf,'(A)') 'GUD_INIT_FIXED:'
      CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                    SQUEEZE_RIGHT, 1 )
      WRITE(msgBuf,'(A)') 'GUD_INIT_FIXED: wavebands:'
      CALL PRINT_MESSAGE(msgBuf,standardMessageUnit,
     &                   SQUEEZE_RIGHT,myThid)
      WRITE(msgBuf,'(2A)') 'GUD_INIT_FIXED: ',
     &   ' idx       low   rep      high    width'
      CALL PRINT_MESSAGE(msgBuf,standardMessageUnit,
     &                   SQUEEZE_RIGHT,myThid)
      DO l=1,nlam
        WRITE(msgBuf,'(A,I4,F10.3,F6.0,F10.3,F9.3)')
     &  'GUD_INIT_FIXED: ', l, gud_waveband_edges(l),
     &  wb_center(l),gud_waveband_edges(l+1),wb_width(l)
        CALL PRINT_MESSAGE(msgBuf,standardMessageUnit,
     &                   SQUEEZE_RIGHT,myThid)
      ENDDO
      WRITE(msgBuf,'(A)') 'GUD_INIT_FIXED:'
      CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                    SQUEEZE_RIGHT, 1 )

      IF ( myProcId.EQ.0 .AND. myThid.EQ.1 ) THEN

c file for wavebands used
c        open(23,file='pwaves-check.dat',status='new')
        CALL MDSFINDUNIT( iUnit1, myThid )
        open(iUnit1,file='pwaves-check.dat',status='unknown')
        write(iUnit1,'(F6.0)')wb_center
        close(iUnit1)

      ENDIF

C read water absorption data
      DO l = 1, nlam
        aw(l) = -1.0 _d 0
        bw(l) = -1.0 _d 0
      ENDDO
      IF (gud_waterAbsorbFile .NE. ' '  ) THEN
        CALL MDSFINDUNIT( iUnit1, myThid )
        OPEN(iUnit1,FILE=gud_waterabsorbFile,
     &       STATUS='old',FORM='formatted')
C       skip header
        DO i = 1,6
         READ(iUnit1,'(A50)')title
        ENDDO
        ios = 0
        DO WHILE (ios .EQ. 0)
         READ(iUnit1,'(I5,F15.4,F10.4)',IOSTAT=ios) ilambda,ain,bin
         IF (ios .EQ. 0) THEN
          lambdain = ilambda
          DO l = 1,nlam
           IF (lambdain .GE. gud_waveband_edges(l) .AND.
     &         lambdain .LE. gud_waveband_edges(l+1)) THEN
            aw(l) = ain
            bw(l) = bin
           ENDIF
          ENDDO
         ENDIF
        ENDDO
        CLOSE(iUnit1)
      ELSE
        WRITE(msgBuf,'(A)')
     &   'GUD_INIT_FIXED: need to specify water absorption file'
        CALL PRINT_ERROR( msgBuf, myThid )
        STOP 'ABNORMAL END: S/R GUD_INIT_FIXED'
      ENDIF
C     check that all wavebands have been read
      DO l = 1, nlam
        IF (aw(l) .LT. 0.0) THEN
          WRITE(msgBuf,'(2A)') 'GUD_INIT_FIXED: ',
     &     "could not read water absorption data for band "
          CALL PRINT_ERROR( msgBuf, myThid )
          WRITE(msgBuf,'(A,I3,2F8.3)') 'GUD_INIT_FIXED: ',
     &     l,gud_waveband_edges(l),gud_waveband_edges(l+1)
          CALL PRINT_ERROR( msgBuf, myThid )
          STOP 'ABNORMAL END: S/R GUD_INIT_FIXED'
        ENDIF
      ENDDO
C     write summary
      WRITE(msgBuf,'(A)') 'GUD_INIT_FIXED: water spectra:'
      CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                    SQUEEZE_RIGHT, 1 )
      WRITE(msgBuf,'(A,A)') 'GUD_INIT_FIXED: ',
     &     ' lam aw             bw'
      CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                    SQUEEZE_RIGHT, 1 )
      DO l = 1,nlam
        WRITE(msgBuf,'(A,F4.0,F15.4,F10.4)') 'GUD_INIT_FIXED: ',
     &     wb_center(l), aw(l), bw(l)
        CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                      SQUEEZE_RIGHT, 1 )
      ENDDO
      WRITE(msgBuf,'(A)') 'GUD_INIT_FIXED:'
      CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                    SQUEEZE_RIGHT, 1 )


C read phyto absorption data
C   phyto input data files must have a column for absorption by PS pigs
C   easiest way to 'turn off' PS for growth is to put same values in both abs columns
      DO i = 1, nopt
       DO l = 1, nlam
        aphy_chl_type   (i,l) = -1.0 _d 0
        aphy_chl_ps_type(i,l) = -1.0 _d 0
        bphy_mgC_type   (i,l) = -1.0 _d 0
        bbphy_mgC_type  (i,l) = -1.0 _d 0
       ENDDO
      ENDDO
      IF (gud_phytoAbsorbFile .NE. ' '  ) THEN
        CALL MDSFINDUNIT( iUnit1, myThid )
        OPEN(iUnit1,FILE=gud_phytoAbsorbFile,
     &       STATUS='old',FORM='formatted')
C       skip global header
        DO i = 1,6
         READ(iUnit1,'(A50)')title
        ENDDO
C       phytoplanktontype header
        READ(iUnit1,'(A50)')title
        DO i = 1,nopt
         ios = 0
         IF (gud_allomSpectra) THEN
           READ(iUnit1,'(I4,3F10.0,F20.0)'),
     &           idummy, asize(i), apsize(i), bsize(i), bbsize(i)
         ENDIF
         DO WHILE (ios .EQ. 0)
          READ(iUnit1,'(I4,3F10.0,F20.0)',IOSTAT=ios)
     &                ilambda,ain,apsin,bin,bbin
C         next phyto type header will trigger error and move on to next i
          IF (ios .EQ. 0) THEN
           lambdain = ilambda
           DO l = 1,nlam
            IF (lambdain .GE. gud_waveband_edges(l) .AND.
     &          lambdain .LE. gud_waveband_edges(l+1)) THEN
             aphy_chl_type   (i,l) = ain
             aphy_chl_ps_type(i,l) = apsin
             bphy_mgC_type   (i,l) = bin
             bbphy_mgC_type  (i,l) = bbin
            ENDIF
           ENDDO
          ENDIF
         ENDDO
        ENDDO
        CLOSE(iUnit1)
      ELSE
        WRITE(msgBuf,'(A)')
     &   'GUD_INIT_FIXED: need to specify water absorption file'
        CALL PRINT_ERROR( msgBuf, myThid )
        STOP 'ABNORMAL END: S/R GUD_INIT_FIXED'
      ENDIF
C     check that all wavebands have been read
      DO i = 1, nopt
       DO l = 1, nlam
        IF (aphy_chl_type(i,l) .LT. 0.0) THEN
          WRITE(msgBuf,'(2A)') 'GUD_INIT_FIXED: ',
     &     "could not read phyto absorption data for type,band "
          CALL PRINT_ERROR( msgBuf, myThid )
          WRITE(msgBuf,'(A,2I3,2F8.3)') 'GUD_INIT_FIXED: ',
     &     i,l,gud_waveband_edges(l),gud_waveband_edges(l+1)
          CALL PRINT_ERROR( msgBuf, myThid )
          STOP 'ABNORMAL END: S/R GUD_INIT_FIXED'
        ENDIF
       ENDDO
      ENDDO
C     write summary
      WRITE(msgBuf,'(A)') 'GUD_INIT_FIXED: phyto spectra:'
      CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                    SQUEEZE_RIGHT, 1 )
      DO i = 1,nopt
        WRITE(msgBuf,'(A,I4)') 'GUD_INIT_FIXED: type ', i
        CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                    SQUEEZE_RIGHT, 1 )
        WRITE(msgBuf,'(A,A)') 'GUD_INIT_FIXED: ',
     &     ' lam ap        ap_ps     bp        bbp'
        CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                    SQUEEZE_RIGHT, 1 )
        DO l = 1,nlam
          WRITE(msgBuf,'(A,F4.0,3F10.4,F20.9)') 'GUD_INIT_FIXED: ',
     &     wb_center(l), aphy_chl_type(i,l), aphy_chl_ps_type(i,l),
     &     bphy_mgC_type(i,l), bbphy_mgC_type(i,l)
          CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                      SQUEEZE_RIGHT, 1 )
        ENDDO
        WRITE(msgBuf,'(A)') 'GUD_INIT_FIXED:'
        CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                    SQUEEZE_RIGHT, 1 )
      ENDDO


C read particle absorption data
C initialize particle absorption coefficient
      DO l = 1, nlam
        apart(l) = -1.0 _d 0
        bpart(l) = -1.0 _d 0
        bbpart(l) = -1.0 _d 0
      ENDDO
      IF (gud_particleAbsorbFile .NE. ' '  ) THEN
        CALL MDSFINDUNIT( iUnit1, myThid )
        OPEN(iUnit1,FILE=gud_particleAbsorbFile,
     &       STATUS='old',FORM='formatted')
C       skip header
        DO i = 1,6
         READ(iUnit1,'(A50)')title
        ENDDO
        ios = 0
        DO WHILE (ios .EQ. 0)
         READ(iUnit1,'(I4,3F15.0)',IOSTAT=ios) ilambda,ain,bin,bbin
         IF (ios .EQ. 0) THEN
          lambdain = ilambda
          DO l = 1,nlam
           IF (lambdain .GE. gud_waveband_edges(l) .AND.
     &         lambdain .LE. gud_waveband_edges(l+1)) THEN
            apart(l) = ain
            bpart(l) = bin
            bbpart(l) = bbin
            apart_P(l) = ain/gud_part_size_P
            bpart_P(l) = bin/gud_part_size_P
            bbpart_P(l) = bbin/gud_part_size_P
           ENDIF
          ENDDO
         ENDIF
        ENDDO
        CLOSE(iUnit1)
      ELSE
        WRITE(msgBuf,'(A)')
     &   'GUD_INIT_FIXED: need to specify particle file'
        CALL PRINT_ERROR( msgBuf, myThid )
        STOP 'ABNORMAL END: S/R GUD_INIT_FIXED'
      ENDIF
C     check that all wavebands have been read
      DO l = 1, nlam
        IF (apart(l) .LT. 0.0) THEN
          WRITE(msgBuf,'(2A)') 'GUD_INIT_FIXED: ',
     &     "could not read particle for band "
          CALL PRINT_ERROR( msgBuf, myThid )
          WRITE(msgBuf,'(A,I3,2F8.3)') 'GUD_INIT_FIXED: ',
     &     l,gud_waveband_edges(l),gud_waveband_edges(l+1)
          CALL PRINT_ERROR( msgBuf, myThid )
          STOP 'ABNORMAL END: S/R GUD_INIT_FIXED'
        ENDIF
      ENDDO
C     write summary
      WRITE(msgBuf,'(A)') 'GUD_INIT_FIXED: particulate spectra:'
      CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                    SQUEEZE_RIGHT, 1 )
      WRITE(msgBuf,'(A,A)') 'GUD_INIT_FIXED: ',
     &     ' lam apart          bpart          bbpart'
      CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                    SQUEEZE_RIGHT, 1 )
      DO l = 1,nlam
        WRITE(msgBuf,'(A,F4.0,1P3G15.6)')'GUD_INIT_FIXED: ',
     &     wb_center(l), apart(l), bpart(l), bbpart(l)
        CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                      SQUEEZE_RIGHT, 1 )
      ENDDO
      WRITE(msgBuf,'(A)') 'GUD_INIT_FIXED:'
      CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                    SQUEEZE_RIGHT, 1 )
c
      WRITE(msgBuf,'(2A)') 'GUD_INIT_FIXED: particulate spectra ',
     &                    'in phosphorus units:'
      CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                    SQUEEZE_RIGHT, 1 )
      WRITE(msgBuf,'(A,A)') 'GUD_INIT_FIXED: ',
     &     ' lam apart_P        bpart_P        bbpart_P'
      CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                    SQUEEZE_RIGHT, 1 )
      DO l = 1,nlam
        WRITE(msgBuf,'(A,F4.0,2F15.9,F15.12)') 'GUD_INIT_FIXED: ',
     &     wb_center(l), apart_P(l), bpart_P(l), bbpart_P(l)
        CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                      SQUEEZE_RIGHT, 1 )
      ENDDO
      WRITE(msgBuf,'(A)') 'GUD_INIT_FIXED:'
      CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                    SQUEEZE_RIGHT, 1 )
c


      DO l = 1,nlam
        exCDOM(l)=EXP(-gud_Sdom*(wb_center(l)-gud_lambda_aCDOM))
      ENDDO
#ifndef GUD_ALLOW_CDOM
C initialize CDOM absorption coefficient
      laCDOM = -1
      DO l = 1,nlam
        IF (gud_lambda_aCDOM .GE. gud_waveband_edges(l) .AND.
     &      gud_lambda_aCDOM .LE. gud_waveband_edges(l+1)) THEN
          laCDOM = l
        ENDIF
      ENDDO
      IF (laCDOM .LE. 0) THEN
        WRITE(msgBuf,'(2A)') 'GUD_INIT_FIXED: ',
     &   "could not read find aCDOM reference waveband with frequency"
        CALL PRINT_ERROR( msgBuf, myThid )
        WRITE(msgBuf,'(A,F8.3)') 'GUD_INIT_FIXED: ',
     &   gud_lambda_aCDOM
        CALL PRINT_ERROR( msgBuf, myThid )
        STOP 'ABNORMAL END: S/R GUD_INIT_FIXED'
      ELSE
        WRITE(msgBuf,'(A,I3)')
     &        'GUD_INIT_FIXED: laCDOM = ', laCDOM
        CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                      SQUEEZE_RIGHT, 1 )
      ENDIF
#endif

#endif /* GUD_ALLOW_RADTRANS */

C ======================================================================
      CALL GUD_RANDOM_INIT(gud_seed, myThid)

      CALL GUD_INIT_TRAITS(myThid)

#ifdef GUD_RANDOM_TRAITS
      CALL GUD_GENERATE_RANDOM(myThid)
#else
      CALL GUD_GENERATE_ALLOMETRIC(myThid)
#endif

C ======================================================================
C read (overrides generated) and write trait namelists

      WRITE(msgBuf,'(A)') ' GUD_INIT_FIXED: opening data.traits'
      CALL PRINT_MESSAGE(msgBuf, standardMessageUnit,
     &                   SQUEEZE_RIGHT , 1)

      CALL MDSFINDUNIT( iUnit1, myThid )
      CALL OPEN_COPY_DATA_FILE(
     I                   'data.traits', 'GUD_INIT_FIXED',
     O                   iUnit1,
     I                   myThid )

      IF ( myProcId.EQ.0 .AND. myThid.EQ.1 ) THEN
        CALL MDSFINDUNIT( iUnit2, mythid )
        open(iUnit2,file='gud_traits.txt',status='unknown')
      ELSE
        iUnit2 = -1
      ENDIF

      CALL GUD_READTRAITS(iUnit1, iUnit2, myThid)

      IF ( iUnit2 .GE. 0 ) THEN
        close(iUnit2)
      ENDIF
      CLOSE(iUnit1)

      WRITE(msgBuf,'(A)') ' ==================================='
      CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
     &                    SQUEEZE_RIGHT, myThid )

C ======================================================================
C deprecation checks

#ifndef GUD_ALLOW_EXUDE
      DO jp = 1, nPlank
       IF (ExportFrac(jp) .NE. UNINIT_RL) THEN
        WRITE(msgBuf,'(2A)')'ExportFrac can only be used with ',
     &     'GUD_ALLOW_EXUDE.'
        CALL PRINT_ERROR( msgBuf, myThid )
        WRITE(msgBuf,'(2A)')'Use ExportFracMort and ExportFracMort2 ',
     &     'for export due to mortality.'
        CALL PRINT_ERROR( msgBuf, myThid )
        STOP 'ABNORMAL END: S/R GUD_INIT_FIXED'
       ENDIF
      ENDDO
#endif

C ======================================================================
C write some traits to files

#ifdef GUD_ALLOW_RADTRANS

      IF ( myProcId.EQ.0 ) THEN

C set spectra for individual plankton according to optical type
C file for ap_types assigned via coin flips
      CALL MDSFINDUNIT( iUnit1, myThid )
      OPEN(iUnit1,file='p-ini-char-aptype.dat',status='unknown')
      CALL MDSFINDUNIT( iUnit2, myThid )
      OPEN(iUnit2,file='p_ini_char_aptype_nohead.dat',status='unknown')
      WRITE(iUnit1,*)'np   ap_type'
      DO jp = 1,nPhoto
        WRITE(iUnit1,'(2I5)') jp, ap_type(jp)
        WRITE(iUnit2,'(2I5)') jp, ap_type(jp)
      ENDDO
      CLOSE(iUnit1)
      CLOSE(iUnit2)

c file of total absorption spectra
c rows = pwaves, columns = jp      
      open(iUnit1,file='p-ini-char-apspec.dat',status='unknown')
      open(iUnit2,file='p_ini_char_aspec_nohead.dat',status='unknown')
      write(iUnit1,*)'Rows = pwaves. Columns = jp'
      write(iUnit1,*)'pwaves found in pwaves-check.dat'
      write(iUnit1,*)'col_1 to col_<nPhoto>'
      write(iUnit1,*)'is absorption aphy_chl (m-2 mg chla-1)'
      do l=1,nlam
       write(iUnit1,9999)(aphy_chl(jp,l),jp=1,nPhoto)
       write(iUnit2,9999)(aphy_chl(jp,l),jp=1,nPhoto)
      enddo
c make sure outfile is defined above with the correct size
      close(iUnit1)
      close(iUnit2)

c file for absorption spectra of PS's only
c rows = pwaves, columns = jp
      open(iUnit1,file='p-ini-char-apspec-psc.dat',status='unknown')
      open(iUnit2,file='p_ini_char_aspec_psc_nohead.dat',
     &     status='unknown')
      write(iUnit1,*)'Rows = pwaves. Columns = jp'
      write(iUnit1,*)'pwaves found in pwaves-check.dat'
      write(iUnit1,*)'Is absoprtion by photosynthetic'
      write(iUnit1,*)'pigments only aphy_chl_ps (m-2 mg chla-1)'
      do l=1,nlam
       write(iUnit1,9999)(aphy_chl_ps(jp,l),jp=1,nPhoto)
       write(iUnit2,9999)(aphy_chl_ps(jp,l),jp=1,nPhoto)
      enddo
      close(iUnit1)
      close(iUnit2)

c file of total scattering spectra
c rows = pwaves, columns = jp
      open(iUnit1,file='p-ini-char-btspec.dat',status='unknown')
      open(iUnit2,file='p_ini_char_btspec_nohead.dat',status='unknown')
      write(iUnit1,*)'Rows = pwaves. Columns = jp'
      write(iUnit1,*)'pwaves found in pwaves-check.dat'
      write(iUnit1,*)'col_1 to col_<nPlank>'
      write(iUnit1,*)'is total scattering bphy_mgC (m-2 mg C-1)'
      do l=1,nlam
       write(iUnit1,9999)(bphy_mgC(jp,l),jp=1,nplank)
       write(iUnit2,9999)(bphy_mgC(jp,l),jp=1,nplank)
      enddo
c make sure outfile is defined above with the correct size
      close(iUnit1)
      close(iUnit2)

c file of total scattering spectra
c rows = pwaves, columns = jp
      open(iUnit1,file='p-ini-char-bbspec.dat',status='unknown')
      open(iUnit2,file='p_ini_char_bbspec_nohead.dat',status='unknown')
      write(iUnit1,*)'Rows = pwaves. Columns = jp'
      write(iUnit1,*)'pwaves found in pwaves-check.dat'
      write(iUnit1,*)'col_1 to col_<nPlank>'
      write(iUnit1,*)'is backscattering bphy_mgC*1e4 (m-2 mg C-1)'
      do l=1,nlam
       write(iUnit1,9999)(bbphy_mgC(jp,l)*1e4,jp=1,nplank)
       write(iUnit2,9999)(bbphy_mgC(jp,l)*1e4,jp=1,nplank)
      enddo
c make sure outfile is defined above with the correct size
      close(iUnit1)
      close(iUnit2)

9999  format(9999f10.4)

C     IF ( myProcId.EQ.0 ) THEN
      ENDIF

#endif /* GUD_ALLOW_RADTRANS */

C ======================================================================

      _END_MASTER(myThid)

C Everyone else must wait for the parameters to be loaded
      _BARRIER

#endif /* ALLOW_GUD */

      RETURN
      END
