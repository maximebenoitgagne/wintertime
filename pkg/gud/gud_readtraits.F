C This file contains cog templates.
C Code between template and end marker is autogenerated.
C Add new parameters to traits.py
C
CCOG[[[cog import cog; from traits import *]]]
CCOG[[[end]]] (checksum: d41d8cd98f00b204e9800998ecf8427e)

#include "GUD_OPTIONS.h"

      SUBROUTINE GUD_READTRAITS(iUnit,oUnit,myThid)
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

      CHARACTER*(MAX_LEN_MBUF) msgBuf
      INTEGER iUnit,oUnit,myThid

#ifdef ALLOW_GUD

      INTEGER jp
      INTEGER jp2
      INTEGER jz
      INTEGER l
      INTEGER iopt

CCOG[[[cog
CCOGfor name,(nml,conds,conde) in nmls.items():
CCOG    cog.out(conds)
CCOG    cog.out('''
CCOG      NAMELIST/{name}/
CCOG'''.format(name=name)[1:])
CCOG    cog.out(',\n'.join('''
CCOG{pre}     &    {param}
CCOG{post}'''.format(param=param,pre=pre,post=post)[1:-1]
CCOG        for tp,dims,param,pre,post in nml))
CCOG    cog.out('\n')
CCOG    cog.out(conde)
CCOG    cog.out('\n')
CCOG]]]
      NAMELIST/GUD_TRAITS/
     &    isPhoto,
     &    bactType,
     &    isAerobic,
     &    isDenit,
     &    hasSi,
     &    hasPIC,
     &    diazo,
     &    useNH4,
     &    useNO2,
     &    useNO3,
     &    combNO,
     &    Xmin,
     &    amminhib,
     &    acclimtimescl,
     &    mort,
     &    mort2,
     &    tempMort,
     &    tempMort2,
     &    ExportFracMort,
     &    ExportFracMort2,
     &    ExportFrac,
     &    phytoTempCoeff,
     &    phytoTempExp1,
     &    phytoTempExp2,
     &    phytoTempOptimum,
     &    phytoDecayPower,
     &    R_NC,
     &    R_PC,
     &    R_SiC,
     &    R_FeC,
     &    R_ChlC,
     &    R_PICPOC,
     &    wsink,
     &    wswim,
     &    respiration,
     &    PCmax,
     &    Qnmax,
     &    Qnmin,
     &    Qpmax,
     &    Qpmin,
     &    Qsimax,
     &    Qsimin,
     &    Qfemax,
     &    Qfemin,
     &    Vmax_NH4,
     &    Vmax_NO2,
     &    Vmax_NO3,
     &    Vmax_N,
     &    Vmax_PO4,
     &    Vmax_SiO2,
     &    Vmax_FeT,
     &    ksatNH4,
     &    ksatNO2,
     &    ksatNO3,
     &    ksatPO4,
     &    ksatSiO2,
     &    ksatFeT,
     &    kexcC,
     &    kexcN,
     &    kexcP,
     &    kexcSi,
     &    kexcFe,
#ifdef GUD_ALLOW_GEIDER
     &    inhibcoef_geid,
#else
     &    ksatPAR,
     &    kinhPAR,
#endif
     &    mQyield,
     &    chl2cmax,
     &    grazemax,
     &    kgrazesat,
     &    palat,
     &    asseff,
     &    ExportFracPreyPred,
     &    yield,
     &    yieldO2,
     &    yieldNO3,
     &    ksatPON,
     &    ksatPOC,
     &    ksatPOP,
     &    ksatPOFe,
     &    ksatDON,
     &    ksatDOC,
     &    ksatDOP,
     &    ksatDOFe

#ifdef GUD_ALLOW_RADTRANS
      NAMELIST/GUD_RADTRANS_TRAITS/
     &    aphy_chl,
     &    aphy_chl_ps,
     &    bphy_mgC,
     &    bbphy_mgC
#endif

      NAMELIST/GUD_DEPENDENT_TRAITS/
#ifndef GUD_ALLOW_GEIDER
     &    normI,
#endif
#ifdef GUD_ALLOW_RADTRANS
     &    ap_type,
#endif
     &    biovol,
     &    group,
     &    igroup,
     &    qcarbon,
     &    pp_opt,
     &    pp_sig,
     &    biovol_bygroup,
     &    alphachl,
     &    alpha_mean,
     &    chl2cmin,
     &    mortTempFuncMin,
     &    mort2TempFuncMin

CCOG[[[end]]] (checksum: 8c81ab14b02efa21d5af86e6fdaa2eb5)

CCOG[[[cog
CCOGfor name,(lines,conds,conde) in code.items():
CCOG    cog.out('''
CCOGCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCOGC     {name}
CCOGCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCOG
CCOG'''.format(name=name)[1:])
CCOG    cog.out(conds)
CCOG    cog.out('\n'.join(lines))
CCOG    if name.lower() not in noread:
CCOG        cog.out('''
CCOG      READ(UNIT=iUnit, NML={name})
CCOG        '''.format(name=name)[1:-8])
CCOG    cog.out(conde)
CCOG    cog.out('\n')
CCOG]]]
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     GUD_TRAITS
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      READ(UNIT=iUnit, NML=GUD_TRAITS)

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     GUD_RADTRANS_TRAITS
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

#ifdef GUD_ALLOW_RADTRANS
      READ(UNIT=iUnit, NML=GUD_RADTRANS_TRAITS)
#endif

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     GUD_DEPENDENT_TRAITS
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC


CCOG[[[end]]] (checksum: 056ae7f9cc78f111910c1b44e7fb7910)

CCOG[[[cog
CCOGfor param in denomparams:
CCOG    cog.out('''
CCOG      {param} = {param}/{param}_denom
CCOG'''.format(param=param)[1:])
CCOG]]]
CCOG[[[end]]] (checksum: d41d8cd98f00b204e9800998ecf8427e)

      DO jp = 1,nPlank
        vmax_NH4(jp) = vmax_NH4(jp) * useNH4(jp)
        vmax_NO2(jp) = vmax_NO2(jp) * useNO2(jp)
        vmax_NO3(jp) = vmax_NO3(jp) * useNO3(jp)
#ifndef GUD_ALLOW_NQUOTA
        IF (diazo(jp).NE.0) THEN
          useNH4(jp) = 0
          useNO2(jp) = 0
          useNO3(jp) = 0
        ENDIF
#endif
        IF (useNO3(jp).EQ.0 .OR. useNO2(jp).EQ.0) THEN
          combNO(jp)=0
        ENDIF
      ! Silicate parameters to zero for non-diatoms
        IF (hasSi(jp) .EQ. 0) THEN
          vmax_SiO2(jp) = 0.0 _d 0
          ksatSiO2(jp) = 0.0 _d 0
          R_SiC(jp) = 0.0 _d 0
        ENDIF
      ! only Coccolithophores have PIC
        IF (hasPIC(jp) .EQ. 0) THEN
          R_PICPOC(jp) = 0.0 _d 0
        ENDIF
      ENDDO
      
      
      mortTempFuncMin(:)  = MIN(1.0, 1.0 - tempMort(:))
      mort2TempFuncMin(:) = MIN(1.0, 1.0 - tempMort2(:))
      

#ifdef GUD_ALLOW_RADTRANS
      
      DO jp = 1, nPhoto
       alpha_mean(jp) = 0.0 _d 0
       DO l = 1, nlam
        alphachl(jp,l) = mQyield(jp) * aphy_chl_ps(jp,l)
        alpha_mean(jp) = alpha_mean(jp) + wb_width(l)*alphachl(jp,l)
       ENDDO
       alpha_mean(jp) = alpha_mean(jp)/wb_totalWidth
      ENDDO
#else
      DO jp = 1, nPhoto
       alphachl(jp,1) = mQyield(jp) * aphy_chl_ave
       alpha_mean(jp) = alphachl(jp,1)
      ENDDO
#endif /* GUD_ALLOW_RADTRANS */

#ifdef GUD_ALLOW_RADTRANS
      DO jp = 1, nplank
       chl2cmin(jp)=chl2cmax(jp)/
     &   (1+(chl2cmax(jp)* alpha_mean(jp) *2000. _d 0)/(2*pcmax(jp)))
      ENDDO
#else
      DO jp = 1, nPhoto
       chl2cmin(jp) = 0. _d 0
      ENDDO
#endif

#ifndef GUD_ALLOW_GEIDER
      DO jp = 1, nplank
       IF (ksatPAR(jp)*kinhPAR(jp) .GT. 0.0) THEN
        normI(jp) = 1.0/(ksatPAR(jp)/(ksatPAR(jp)+kinhPAR(jp))*
     &    EXP(kinhPAR(jp)/ksatPAR(jp)*
     &        LOG(kinhPAR(jp)/(ksatPAR(jp)+kinhPAR(jp)))))
       ELSE
        normI(jp) = 1.0
       ENDIF
      ENDDO
#endif

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! checks

      ! check types are within type ranges
      DO jp=1,nplank
       IF (isPhoto(jp) .NE. 0) THEN
        IF (jp.GT.nPhoto) THEN
          WRITE(msgBuf,'(2A,I4)') 'GUD_READTRAITS: ERROR: ',
     &    'isPhoto set for type outside of photo range: ', jp
          CALL PRINT_ERROR( msgBuf , 1)
          STOP 'ABNORMAL END: S/R GUD_READTRAITS'
        ENDIF
#ifdef GUD_ALLOW_CHLQUOTA
        IF (jp.GT.nChl) THEN
          WRITE(msgBuf,'(2A,I4)') 'GUD_READTRAITS: ERROR: ',
     &    'isPhoto set for type outside of Chl range: ', jp
          CALL PRINT_ERROR( msgBuf , 1)
          STOP 'ABNORMAL END: S/R GUD_READTRAITS'
        ENDIF
#endif
       ELSE
        PCmax(jp) = 0 _d 0
        Vmax_PO4(jp) = 0 _d 0
        Vmax_NO3(jp) = 0 _d 0
        Vmax_NO2(jp) = 0 _d 0
        Vmax_NH4(jp) = 0 _d 0
        Vmax_FeT(jp) = 0 _d 0
        Vmax_SiO2(jp) = 0 _d 0
       ENDIF

       IF (bactType(jp) .NE. 0) THEN
        IF (jp.LT.iMinBact .OR. jp.GT.iMaxBact) THEN
          WRITE(msgBuf,'(2A,I4)') 'GUD_READTRAITS: ERROR: ',
     &    'bactType set for type outside of bacteria range: ', jp
          CALL PRINT_ERROR( msgBuf , 1)
          STOP 'ABNORMAL END: S/R GUD_READTRAITS'
        ENDIF
        IF (isAerobic(jp).NE.0 .AND. isDenit(jp).NE.0) THEN
          WRITE(msgBuf,'(2A,I4)') 'GUD_READTRAITS: ERROR: ',
     &    'isAerobic and isDenit both set: ', jp
          CALL PRINT_ERROR( msgBuf , 1)
          STOP 'ABNORMAL END: S/R GUD_READTRAITS'
        ENDIF
       ELSE
        IF (isAerobic(jp).NE.0) THEN
          WRITE(msgBuf,'(2A,I4)') 'GUD_READTRAITS: ERROR: ',
     &    'isAerobic set for non-bacteria type: ', jp
          CALL PRINT_ERROR( msgBuf , 1)
          STOP 'ABNORMAL END: S/R GUD_READTRAITS'
        ENDIF
        IF (isDenit(jp).NE.0) THEN
          WRITE(msgBuf,'(2A,I4)') 'GUD_READTRAITS: ERROR: ',
     &    'isDenit set for non-bacteria type: ', jp
          CALL PRINT_ERROR( msgBuf , 1)
          STOP 'ABNORMAL END: S/R GUD_READTRAITS'
        ENDIF
       ENDIF

       DO jz=1,nplank
        IF (palat(jp,jz) .NE. 0.) THEN
         IF (jp.LT.iMinPrey .OR. jp.GT.iMaxPrey) THEN
          WRITE(msgBuf,'(2A,I4)') 'GUD_READTRAITS: WARNING: ',
     &    'palat set for type outside of prey range: ', jp
          CALL PRINT_ERROR( msgBuf , 1)
         ENDIF
         IF (jz.LT.iMinPred .OR. jz.GT.iMaxPred) THEN
          WRITE(msgBuf,'(2A,I4)') 'GUD_READTRAITS: WARNING: ',
     &    'palat set for type outside of predator range: ', jz
          CALL PRINT_ERROR( msgBuf , 1)
         ENDIF
        ENDIF
       ENDDO
      ENDDO

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! zero out unused trait entries, so they do not mess up the traits file
      DO jp=1,nplank
#ifndef GUD_ALLOW_NQUOTA
       Vmax_NH4(jp) = GUD_UNUSED
       Vmax_NO2(jp) = GUD_UNUSED
       Vmax_NO3(jp) = GUD_UNUSED
       Vmax_N(jp) = GUD_UNUSED
       Qnmax(jp) = GUD_UNUSED
       Qnmin(jp) = GUD_UNUSED
#endif
#ifndef GUD_ALLOW_PQUOTA
       Vmax_PO4(jp) = GUD_UNUSED
       Qpmax(jp) = GUD_UNUSED
       Qpmin(jp) = GUD_UNUSED
#endif
#ifndef GUD_ALLOW_SIQUOTA
       Vmax_SiO2(jp) = GUD_UNUSED
       Qsimax(jp) = GUD_UNUSED
       Qsimin(jp) = GUD_UNUSED
#endif
#ifndef GUD_ALLOW_FEQUOTA
       Vmax_FeT(jp) = GUD_UNUSED
       Qfemax(jp) = GUD_UNUSED
       Qfemin(jp) = GUD_UNUSED
#endif
#ifndef GUD_ALLOW_EXUDE
       kexcC(jp) = GUD_UNUSED
       kexcN(jp) = GUD_UNUSED
       kexcP(jp) = GUD_UNUSED
       kexcSi(jp) = GUD_UNUSED
       kexcFe(jp) = GUD_UNUSED
#endif
       IF (jp.GT.nPhoto) THEN
        diazo(jp) = GUD_UNUSED
        useNH4(jp) = GUD_UNUSED
        useNO2(jp) = GUD_UNUSED
        useNO3(jp) = GUD_UNUSED
        combNO(jp) = GUD_UNUSED
        amminhib(jp) = GUD_UNUSED
        acclimtimescl(jp) = GUD_UNUSED
        PCmax(jp) = GUD_UNUSED

        phytoTempCoeff(jp) = GUD_UNUSED
        phytoTempExp1(jp) = GUD_UNUSED
        phytoTempExp2(jp) = GUD_UNUSED
        phytoTempOptimum(jp) = GUD_UNUSED
        phytoDecayPower(jp) = GUD_UNUSED

        Vmax_NH4(jp) = GUD_UNUSED
        Vmax_NO2(jp) = GUD_UNUSED
        Vmax_NO3(jp) = GUD_UNUSED
        Vmax_PO4(jp) = GUD_UNUSED
        Vmax_SiO2(jp) = GUD_UNUSED
        Vmax_FeT(jp) = GUD_UNUSED

        ksatNH4(jp) = GUD_UNUSED
        ksatNO2(jp) = GUD_UNUSED
        ksatNO3(jp) = GUD_UNUSED
        ksatPO4(jp) = GUD_UNUSED
        ksatSiO2(jp) = GUD_UNUSED
        ksatFeT(jp) = GUD_UNUSED

#ifdef GUD_ALLOW_GEIDER
        inhibcoef_geid(jp) = GUD_UNUSED
#else
        ksatPAR(jp) = GUD_UNUSED
        kinhPAR(jp) = GUD_UNUSED
        normI(jp) = GUD_UNUSED
#endif
        mQyield(jp) = GUD_UNUSED
        chl2cmax(jp) = GUD_UNUSED
        chl2cmin(jp) = GUD_UNUSED
        DO l=1,nlam
         alphachl(jp,l) = GUD_UNUSED
        ENDDO
        alpha_mean(jp) = GUD_UNUSED
       ENDIF
       IF (jp.LT.iMinPred .OR. jp.GT.iMaxPred) THEN
        grazemax(jp) = GUD_UNUSED
        kgrazesat(jp) = GUD_UNUSED
       ENDIF
       DO jp2=1,nplank
        IF (jp .LT.iMinPred .OR. jp .GT.iMaxPred .OR.
     &      jp2.LT.iMinPrey .OR. jp2.GT.iMaxPrey) THEN
          palat(jp2,jp) = GUD_UNUSED
          asseff(jp2,jp) = GUD_UNUSED
          ExportFracPreyPred(jp2,jp) = GUD_UNUSED
        ENDIF
       ENDDO
      ENDDO

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     log all parameters to a namelist file

      IF ( oUnit .GE. 0 ) THEN
CCOG[[[cog
CCOGfor name,(nml,conds,conde) in coms.items():
CCOG    cog.out(conds)
CCOG    cog.out('''
CCOG       WRITE(UNIT=oUnit, NML={name})
CCOG    '''.format(name=name)[1:-4])
CCOG    cog.out(conde)
CCOG]]]
       WRITE(UNIT=oUnit, NML=GUD_TRAITS)
#ifdef GUD_ALLOW_RADTRANS
       WRITE(UNIT=oUnit, NML=GUD_RADTRANS_TRAITS)
#endif
       WRITE(UNIT=oUnit, NML=GUD_DEPENDENT_TRAITS)
CCOG[[[end]]] (checksum: d5dcd83f6d5fd5f1dd20ce3514543e85)
      ENDIF

#endif /* ALLOW_GUD */

      RETURN
      END
