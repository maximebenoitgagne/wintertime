#include "GUD_OPTIONS.h"

CBOP
C !ROUTINE: GUD_GENERATE_ALLOMETRIC

C !INTERFACE: ==========================================================
      SUBROUTINE GUD_GENERATE_ALLOMETRIC( myThid )

C !DESCRIPTION:
C     Generate parameters for plankton types using an allometric approach

C !USES: ===============================================================
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GUD_SIZE.h"
#include "GUD_INDICES.h"
#include "GUD_PARAMS.h"
#include "GUD_GENPARAMS.h"
#include "GUD_TRAITPARAMS.h"
#include "GUD_TRAITS.h"

C !INPUT PARAMETERS: ===================================================
C  myThid               :: thread number
      INTEGER myThid
CEOP

#ifdef ALLOW_GUD

C !LOCAL VARIABLES: ====================================================
C     msgBuf   :: Informational/error meesage buffer
      CHARACTER*(MAX_LEN_MBUF) msgBuf
      INTEGER jpg(nGroup)
      INTEGER jp,jp2,jz,g,gz,l,gmin,iopt
      _RL logvol, vol, volmin
      _RL prd_pry
      _RL kappa
      _RL volmeas, dmmeas
      _RL dmint, dmac
      _RL scalefac, dmratio, slope
      _RL carpcellmeas, carpcellac
      _RL bphy_cell_type
      _RL bmean, bbmean
      _RL bbbratiomeas, bbbratioac


C ======================================================================
C compute cell volumes in micrometer^3
C
C in decreasing precedence (if bold quantity is set):
C
C   V = GRP_BIOVOL(j,g)
C   V = 10**(logvolbase+(GRP_BIOVOLIND(j,g)-1)*logvolinc)
C   V = 10**(logvolbase+(LOGVOL0IND(g)+j-2)*logvolinc)
C   V = BIOVOL0(g)*biovolfac(g)**(j-1)
C
C     if logvol0ind is set, use it to compute biovol0
      DO g=1,nGroup
        IF (logvol0ind(g) .GT. 0) THEN
          IF (biovol0(g) .NE. 0.0 _d 0) THEN
            WRITE(msgBuf,'(2A)') 'GUD_GENERATE_ALLOMETRIC: ',
     &      'cannot set both biovol0 and logvol0ind'
            CALL PRINT_ERROR( msgBuf, myThid )
            STOP 'ABNORMAL END: S/R GUD_GENERATE_ALLOMETRIC'
          ENDIF
          logvol = logvolbase + (logvol0ind(g)-1)*logvolinc
          biovol0(g) = 10 _d 0 ** logvol
          biovolfac(g) = 10 _d 0 ** logvolinc
        ENDIF
      ENDDO

      DO g=1,nGroup
       DO jp=1,grp_nplank(g)
        IF (grp_logvolind(jp,g) .GT. 0 .AND.
     &      grp_biovol(jp,g) .GT. 0.0 _d 0) THEN
          WRITE(msgBuf,'(2A)') 'GUD_GENERATE_ALLOMETRIC: ',
     &    'cannot set both grp_biovol and grp_logvolind'
          CALL PRINT_ERROR( msgBuf, myThid )
          STOP 'ABNORMAL END: S/R GUD_GENERATE_ALLOMETRIC'
        ELSEIF (grp_logvolind(jp,g) .GT. 0) THEN
C         pick from logrange
          logvol = logvolbase + (grp_logvolind(jp,g)-1)*logvolinc
          grp_biovol(jp,g) = 10 _d 0 ** logvol
        ELSEIF (grp_biovol(jp,g) .EQ. 0) THEN
          IF (biovol0(g) .GT. 0.0 _d 0) THEN
           grp_biovol(jp,g) = biovol0(g) * biovolfac(g)**(jp-1)
          ELSE
           WRITE(msgBuf,'(3A)') 'GUD_GENERATE_ALLOMETRIC: ',
     &     'Need to set one of grp_biovol, grp_logvolind, ',
     &     'biovol0, logvol0ind'
           CALL PRINT_ERROR( msgBuf, myThid )
           STOP 'ABNORMAL END: S/R GUD_GENERATE_ALLOMETRIC'
          ENDIF
        ENDIF
        biovol_bygroup(jp,g) = grp_biovol(jp,g)
       ENDDO
       DO jp=grp_nplank(g)+1,nplank
        IF (grp_biovol(jp,g) .NE. 0.0 _d 0) THEN
          WRITE(msgBuf,'(2A,I5,A,I5,A)') 'GUD_GENERATE_ALLOMETRIC: ',
     &    'index',jp,'group',g,'out of range for grp_biovol'
          CALL PRINT_ERROR( msgBuf, myThid )
          STOP 'ABNORMAL END: S/R GUD_GENERATE_ALLOMETRIC'
        ENDIF
        IF (grp_logvolind(jp,g) .NE. 0) THEN
          WRITE(msgBuf,'(2A,I5,A,I5,A)') 'GUD_GENERATE_ALLOMETRIC: ',
     &    'index',jp,'group',g,'out of range for grp_logvolind'
          CALL PRINT_ERROR( msgBuf, myThid )
          STOP 'ABNORMAL END: S/R GUD_GENERATE_ALLOMETRIC'
        ENDIF
        biovol_bygroup(jp,g) = 0.0 _d 0
       ENDDO
      ENDDO

C set up single list of plankton types from groups
C type order is either by group or, if gud_sort_biovol, by biovol.

      IF (gud_sort_biovol) THEN
c      sort by volume

       DO g=1,nGroup
        jpg(g) = 1
       ENDDO
       DO jp=1,nplank
        volmin = 1 _d 38
        gmin = 0
C       first check phototrophs
        DO g=1,ngroup
         IF (grp_photo(g).NE.0) THEN
          vol = grp_biovol(jpg(g),g)
          IF (jpg(g).LE.grp_nplank(g) .AND. vol.LT.volmin) THEN
           gmin = g
           volmin = vol
          ENDIF
         ENDIF
        ENDDO
C       then pure heterotrophs
        IF (gmin.EQ.0) THEN
         DO g=1,ngroup
          IF (grp_photo(g).EQ.0) THEN
           vol = grp_biovol(jpg(g),g)
           IF (jpg(g).LE.grp_nplank(g) .AND. vol.LT.volmin) THEN
            gmin = g
            volmin = vol
           ENDIF
          ENDIF
         ENDDO
        ENDIF
        IF (gmin.EQ.0) THEN
         STOP 'gmin'
        ENDIF
        group(jp) = gmin
        biovol(jp) = volmin
        igroup(jp) = jpg(gmin)
        jpg(gmin) = jpg(gmin) + 1
       ENDDO
       DO g=1,ngroup
        IF (jpg(g).NE.grp_nplank(g)+1) THEN
         STOP 'grp_nplank'
        ENDIF
       ENDDO

      ELSE
C      sort by group

       jp = 1
       DO g=1,ngroup
        DO jp2=1,grp_nplank(g)
         IF (jp .GT. nPlank) THEN
          WRITE(msgBuf,'(2A)') 'GUD_GENERATE_ALLOMETRIC: ',
     &     'need SUM(grp_nplank) = nPlank, nPlank too small'
          CALL PRINT_ERROR( msgBuf, myThid )
          STOP 'ABNORMAL END: S/R GUD_GENERATE_ALLOMETRIC'
         ENDIF
         group(jp)  = g
         igroup(jp) = jp2
         biovol(jp) = grp_biovol(jp2,g)
         jp = jp + 1
        ENDDO
       ENDDO
       IF (jp .NE. nPlank + 1) THEN
        WRITE(msgBuf,'(2A,2I4)') 'GUD_GENERATE_ALLOMETRIC: ',
     &   'need SUM(grp_nplank) = nPlank, not ',jp-1,nPlank
        CALL PRINT_ERROR( msgBuf, myThid )
        STOP 'ABNORMAL END: S/R GUD_GENERATE_ALLOMETRIC'
       ENDIF

C      endif sort order
      ENDIF

C ======================================================================
C compute traits from trait parameters
      DO jp=1,nplank
        g = group(jp)

C ----------------------------------------------------------------------
C non-allometric traits (same within group)

        isPhoto(jp)    = grp_photo(g)
        hasSi(jp)      = grp_hasSi(g)
        hasPIC(jp)     = grp_hasPIC(g)
        diazo(jp)      = grp_diazo(g)
        useNH4(jp)     = grp_useNH4(g)
        useNO2(jp)     = grp_useNO2(g)
        useNO3(jp)     = grp_useNO3(g)
        combNO(jp)     = grp_combNO(g)
        tempMort(jp)   = grp_tempMort(g)
        tempMort2(jp)  = grp_tempMort2(g)

        Xmin(jp)       = grp_Xmin(g)
        amminhib(jp)   = grp_amminhib(g)
        acclimtimescl(jp) = grp_acclimtimescl(g)

C mortality
        mort(jp) = grp_mort(g)
        mort2(jp) = grp_mort2(g)
C       if 0, temperature dependent, if 1, not.
        ExportFracMort(jp) = grp_ExportFracMort(g)
        ExportFracMort2(jp) = grp_ExportFracMort2(g)
        ExportFrac(jp) = grp_ExportFrac(g)

C temperature function parameters
        phytoTempCoeff(jp)   = grp_tempcoeff1(g)
        phytoTempExp1(jp)    = grp_tempcoeff3(g)
        phytoTempExp2(jp)    = grp_tempcoeff2(g)
        phytoTempOptimum(jp) = grp_tempopt(g)
        phytoDecayPower(jp)  = grp_tempdecay(g)

C plankton elemental ratios
        R_NC(jp)       = grp_R_NC(g)
        R_PC(jp)       = grp_R_PC(g)
        R_SiC(jp)      = grp_R_SiC(g)
        R_FeC(jp)      = grp_R_FeC(g)
        R_ChlC(jp)     = grp_R_ChlC(g)
        R_PICPOC(jp)   = grp_R_PICPOC(g)

C plankton sinking and swimming
        wsink(jp) = a_biosink(g) * biovol(jp)**b_biosink(g)
        wswim(jp) = a_bioswim(g) * biovol(jp)**b_bioswim(g)

C respiration rate is given in terms of carbon content
        qcarbon(jp) = a_qcarbon(g) * biovol(jp)**b_qcarbon(g)
        respiration(jp) = a_respir(g)
     &                    * (12. _d 9 * qcarbon(jp))**b_respir(g)
     &                    / qcarbon(jp)

C parameters relating to inorganic nutrients
        PCmax(jp)= a_vmax_DIC(g) * biovol(jp)**b_vmax_DIC(g)

        vmax_NH4(jp)  = a_vmax_NH4(g)  * biovol(jp)**b_vmax_NH4(g)
        vmax_NO2(jp)  = a_vmax_NO2(g)  * biovol(jp)**b_vmax_NO2(g)
        vmax_NO3(jp)  = a_vmax_NO3(g)  * biovol(jp)**b_vmax_NO3(g)
        vmax_N(jp)    = a_vmax_N(g)    * biovol(jp)**b_vmax_N(g)
        vmax_PO4(jp)  = a_vmax_PO4(g)  * biovol(jp)**b_vmax_PO4(g)
        vmax_SiO2(jp) = a_vmax_SiO2(g) * biovol(jp)**b_vmax_SiO2(g)
        vmax_FeT(jp)  = a_vmax_FeT(g)  * biovol(jp)**b_vmax_FeT(g)

        Qnmin(jp)     = a_qmin_n(g)  * biovol(jp)**b_qmin_n(g)
        Qnmax(jp)     = a_qmax_n(g)  * biovol(jp)**b_qmax_n(g)

        Qpmin(jp)     = a_qmin_p(g)  * biovol(jp)**b_qmin_p(g)
        Qpmax(jp)     = a_qmax_p(g)  * biovol(jp)**b_qmax_p(g)

        Qsimin(jp)    = a_qmin_si(g) * biovol(jp)**b_qmin_si(g)
        Qsimax(jp)    = a_qmax_si(g) * biovol(jp)**b_qmax_si(g)

        Qfemin(jp)    = a_qmin_fe(g) * biovol(jp)**b_qmin_fe(g)
        Qfemax(jp)    = a_qmax_fe(g) * biovol(jp)**b_qmax_fe(g)

        ksatNH4(jp)   = a_kn_NH4(g)  * biovol(jp)**b_kn_NH4(g)
        ksatNO2(jp)   = a_kn_NO2(g)  * biovol(jp)**b_kn_NO2(g)
        ksatNO3(jp)   = a_kn_NO3(g)  * biovol(jp)**b_kn_NO3(g)
        ksatPO4(jp)   = a_kn_PO4(g)  * biovol(jp)**b_kn_PO4(g)
        ksatSiO2(jp)  = a_kn_SiO2(g) * biovol(jp)**b_kn_SiO2(g)
        ksatFeT(jp)   = a_kn_FeT(g)  * biovol(jp)**b_kn_FeT(g)

C parameters relating to quota nutrients
c EXCRETION
        kexcC(jp)  = a_kexc_c(g)  * biovol(jp)**b_kexc_c(g)
        kexcN(jp)  = a_kexc_n(g)  * biovol(jp)**b_kexc_n(g)
        kexcP(jp)  = a_kexc_p(g)  * biovol(jp)**b_kexc_p(g)
        kexcSi(jp) = a_kexc_si(g) * biovol(jp)**b_kexc_si(g)
        kexcFe(jp) = a_kexc_fe(g) * biovol(jp)**b_kexc_fe(g)

        IF (GUD_effective_ksat) THEN
c        compute effective half sat for uptake of non-quota elements
C        we compute it for NO3 and scale for others
         IF (gud_select_kn_allom.EQ.1) THEN
C         following Ward et al.
          kappa=(ksatNO3(jp)*PCmax(jp)*Qnmin(jp)*(Qnmax(jp)-Qnmin(jp)))/
     &          (vmax_NO3(jp)*Qnmax(jp) +
     &                       PCmax(jp)*Qnmin(jp)*(Qnmax(jp)-Qnmin(jp)))
         ELSEIF (gud_select_kn_allom.EQ.2) THEN
C         following Follett et al.
          kappa = (ksatNO3(jp)*PCmax(jp)*Qnmin(jp))/vmax_NO3(jp)
         ELSE
          WRITE(msgBuf,'(2A)')'GUD_GENERATE_ALLOMETRIC: ',
     &      'illegal value for gud_select_kn_allom'
          CALL PRINT_ERROR( msgBuf, myThid )
          STOP 'ABNORMAL END: S/R GUD_GENERATE_ALLOMETRIC'
         ENDIF
#ifndef GUD_ALLOW_NQUOTA
         ksatNO3(jp) = kappa
         ksatNO2(jp) = kappa*grp_ksatNO2fac(g)
         ksatNH4(jp) = kappa*grp_ksatNH4fac(g)
#endif
#ifndef GUD_ALLOW_PQUOTA
         ksatPO4(jp) = kappa/R_NC(jp)*R_PC(jp)
#endif
#ifndef GUD_ALLOW_SIQUOTA
         ksatSiO2(jp) = kappa/R_NC(jp)*R_SiC(jp)
#endif
#ifndef GUD_ALLOW_FEQUOTA
         ksatFeT(jp) = kappa/R_NC(jp)*R_FeC(jp)
#endif
        ENDIF

C parameters for bacteria

        bactType(jp)  = grp_bacttype(g)
        isAerobic(jp) = grp_aerobic(g)
        isDenit(jp)   = grp_denit(g)

        yieldO2(jp) = 1.0
        yieldNO3(jp) = 1.0
        IF (isAerobic(jp) .NE. 0) THEN
          yield(jp) = yod
          yieldO2(jp) = yoe
        ELSEIF (isDenit(jp) .NE. 0) THEN
          yield(jp) = ynd
          yieldNO3(jp) = yne
        ENDIF

        ksatPON(jp)  = ksatPOM
        ksatDON(jp)  = ksatDOM
        ksatPOC(jp)  = ksatPON(jp)/R_NC(jp)
        ksatPOP(jp)  = ksatPON(jp)/R_NC(jp)*R_PC(jp)
        ksatPOFe(jp) = ksatPON(jp)/R_NC(jp)*R_FeC(jp)
        ksatDOC(jp)  = ksatDON(jp)/R_NC(jp)
        ksatDOP(jp)  = ksatDON(jp)/R_NC(jp)*R_PC(jp)
        ksatDOFe(jp) = ksatDON(jp)/R_NC(jp)*R_FeC(jp)

#ifdef GUD_ALLOW_GEIDER
        mQyield(jp) = grp_mQyield(g)
        chl2cmax(jp) = grp_chl2cmax(g)
        inhibcoef_geid(jp) = grp_inhibcoef_geid(g)
#else
        ksatPAR(jp) = grp_ksatPAR(g)
        kinhPAR(jp) = grp_kinhPAR(g)
#endif /* GUD_ALLOW_GEIDER */

#ifdef GUD_ALLOW_RADTRANS
        ap_type(jp) = grp_aptype(g)
        iopt = ap_type(jp)
        IF (1 .LE. iopt .AND. iopt .LE. nOpt) THEN
         IF (gud_allomSpectra) THEN

c FOR ABSORPTION
c in terms of volume
           volmeas = PI / 6. _d 0 * asize(iopt)**3
           scalefac = (biovol(jp)/volmeas)**gud_absorpSlope
c size specific absorption spectra
           DO l = 1, nlam
             aphy_chl(jp,l) = aphy_chl_type(iopt,l)*scalefac
             aphy_chl_ps(jp,l) = aphy_chl_ps_type(iopt,l)*scalefac
           ENDDO

c TOTAL SCATTER
c in terms of diameter
           volmeas = PI / 6. _d 0 * bsize(iopt)**3
           dmmeas = bsize(iopt)
           dmac = (6. _d 0 * biovol(jp)/PI)**(1. _d 0 / 3. _d 0)
           dmratio = dmac/dmmeas
           carpcellmeas = gud_aCarCell*volmeas**gud_bCarCell
           carpcellac = gud_aCarCell*biovol(jp)**gud_bCarCell
c size specific scattering spectra
           DO l = 1, nlam
c convert scatter spectrum from m2/mgC to m2/celll
             bphy_cell_type = bphy_mgC_type(iopt,l)*carpcellmeas
             dmint = 10.0 _d 0 ** gud_scatSwitchSizeLog(l)
             IF (dmmeas.GE.dmint) THEN
               slope = gud_scatSlopeLarge(l)
             ELSE
               slope = gud_scatSlopeSmall(l)
             ENDIF
             bphy_mgC(jp,l) = bphy_cell_type*dmratio**slope/carpcellac
           ENDDO

c BACK SCATTER
c calculate mean scatter
           bmean = 0. _d 0
           bbmean = 0. _d 0
           DO l = 1, nlam
             bmean = bmean + bphy_mgC_type(iopt,l)*wb_width(l)
             bbmean = bbmean + bbphy_mgC_type(iopt,l)*wb_width(l)
           ENDDO
           bmean = bmean/wb_totalWidth
           bbmean = bbmean/wb_totalWidth
c scale mean backscattering ratio
           bbbratiomeas = bbmean/bmean
           bbbratioac = bbbratiomeas*dmratio**gud_bbbSlope
           DO l = 1, nlam
             bbphy_mgC(jp,l) = bphy_mgC(jp,l)*bbbratioac
           ENDDO
cc
         ELSE
c OR use read in absorbtion spectra
          DO l = 1, nlam
           aphy_chl(jp,l) = aphy_chl_type(iopt,l)
           aphy_chl_ps(jp,l) = aphy_chl_ps_type(iopt,l)
           bphy_mgC(jp,l) = bphy_mgC_type(iopt,l)
           bbphy_mgC(jp,l) = bbphy_mgC_type(iopt,l)
          ENDDO
         ENDIF
        ELSEIF (jp .LE. nPhoto) THEN
          WRITE(msgBuf,'(A,2I4)')'invalid optical phyto type:',jp,iopt
          CALL PRINT_ERROR( msgBuf, myThid )
          STOP 'ABNORMAL END: S/R GUD_READTRAITS'
        ENDIF
#endif

C     jp
      ENDDO

C ======================================================================
C grazing

      DO jz=1,nplank
        gz = group(jz)
C       maximum grazing rate
        grazemax(jz) = a_graz(gz) * biovol(jz)**b_graz(gz)
C       grazing half-saturation
        kgrazesat(jz) = a_kg(gz) * biovol(jz)**b_kg(gz)
        DO jp=1,nplank
          g = group(jp)
          IF (grp_pred(gz).GT.0 .AND. grp_prey(g).GT.0) THEN
#ifdef GUD_ALLOMETRIC_PALAT
C assign grazing preference according to predator/prey radius ratio
C           grazing size preference ratio
            pp_opt(jz) = a_prdpry(gz) * biovol(jz)**b_prdpry(gz)
C           standard deviation of size preference
            pp_sig(jz) = grp_pp_sig(gz)
            prd_pry = biovol(jz) / biovol(jp)
            palat(jp,jz) =
     &        EXP(-(LOG(prd_pry/pp_opt(jz))**2) / (2*pp_sig(jz)**2))
     &        / pp_sig(jz)/2. _d 0
            IF (palat(jp,jz).LT.palat_min) THEN
              palat(jp,jz) = 0. _d 0
            ENDIF
#else
            palat(jp,jz) = 0.0 _d 0
#endif
            asseff(jp,jz) = grp_ass_eff(g,gz)
            ExportFracPreyPred(jp,jz) = grp_ExportFracPreyPred(g,gz)
          ELSE
            palat(jp,jz) = 0. _d 0
            asseff(jp,jz) = 0. _d 0
            ExportFracPreyPred(jp,jz) = 0. _d 0
          ENDIF
        ENDDO
      ENDDO

#endif  /*ALLOW_GUD*/

      RETURN
      END
