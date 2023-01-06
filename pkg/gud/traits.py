'''
&GUD_TRAITS

! per-plankton traits (generated, maybe overwritten by data.traits)

INTEGER(nplank)  isPhoto
INTEGER(nplank)  bactType
INTEGER(nplank)  isAerobic
INTEGER(nplank)  isDenit
INTEGER(nplank)  hasSi
INTEGER(nplank)  hasPIC
INTEGER(nplank)  diazo
INTEGER(nplank)  useNH4
INTEGER(nplank)  useNO2
INTEGER(nplank)  useNO3
INTEGER(nplank)  combNO

_RL(nplank)      Xmin              ! mmol C m^-3
_RL(nplank)      amminhib          ! (mmol N m^-3)^-1
_RL(nplank)      acclimtimescl     ! s^-1

_RL(nplank)      mort              ! s^-1                   ! linear mortality rate
_RL(nplank)      mort2             ! (mmol C m^-3)^-1 s^-1  ! quadratic mortality coefficient
INTEGER(nplank)  tempMort          ! 1  ! 1: mort temperature dependent 0: not
INTEGER(nplank)  tempMort2         ! 1  ! 1: mort2 temperature dependent 0: not
_RL(nplank)      ExportFracMort    ! 1  ! fraction of linear mortality to POM
_RL(nplank)      ExportFracMort2   ! 1  ! fraction of quadratic mortality to POM
_RL(nplank)      ExportFrac        ! 1  ! fraction of exudation to POM

_RL(nplank)      phytoTempCoeff    ! 1
_RL(nplank)      phytoTempExp1     ! ln(degree)
_RL(nplank)      phytoTempExp2     ! (degree C)^-phytoDecayPower
_RL(nplank)      phytoTempOptimum  ! degree C
_RL(nplank)      phytoDecayPower   ! 1

_RL(nplank)      R_NC              ! mmol N (mmol C)^-1
_RL(nplank)      R_PC              ! mmol P (mmol C)^-1
_RL(nplank)      R_SiC             ! mmol Si (mmol C)^-1
_RL(nplank)      R_FeC             ! mmol Fe (mmol C)^-1
_RL(nplank)      R_ChlC            ! mg Chl (mmol C)^-1
_RL(nplank)      R_PICPOC          ! mmol PIC (mmol POC)^-1
			     
_RL(nplank)      wsink             ! m s^-1
_RL(nplank)      wswim             ! m s^-1

_RL(nplank)      respiration       ! s^-1
_RL(nplank)      PCmax             ! s^-1
_RL(nplank)      Qnmax             ! mmol N (mmol C)^-1
_RL(nplank)      Qnmin             ! mmol N (mmol C)^-1
_RL(nplank)      Qpmax             ! mmol P (mmol C)^-1
_RL(nplank)      Qpmin             ! mmol P (mmol C)^-1
_RL(nplank)      Qsimax            ! mmol Si (mmol C)^-1
_RL(nplank)      Qsimin            ! mmol Si (mmol C)^-1
_RL(nplank)      Qfemax            ! mmol Fe (mmol C)^-1
_RL(nplank)      Qfemin            ! mmol Fe (mmol C)^-1
_RL(nplank)      Vmax_NH4          ! mmol N (mmol C)^-1 s^-1
_RL(nplank)      Vmax_NO2          ! mmol N (mmol C)^-1 s^-1
_RL(nplank)      Vmax_NO3          ! mmol N (mmol C)^-1 s^-1
_RL(nplank)      Vmax_N            ! mmol N (mmol C)^-1 s^-1
_RL(nplank)      Vmax_PO4          ! mmol P (mmol C)^-1 s^-1
_RL(nplank)      Vmax_SiO2         ! mmol Si (mmol C)^-1 s^-1
_RL(nplank)      Vmax_FeT          ! mmol Fe (mmol C)^-1 s^-1
	            	     
_RL(nplank)      ksatNH4           ! mmol N m^-3
_RL(nplank)      ksatNO2           ! mmol N m^-3
_RL(nplank)      ksatNO3           ! mmol N m^-3
_RL(nplank)      ksatPO4           ! mmol P m^-3
_RL(nplank)      ksatSiO2          ! mmol Si m^-3
_RL(nplank)      ksatFeT           ! mmol Fe m^-3
                 
_RL(nplank)      kexcC             ! s^-1
_RL(nplank)      kexcN             ! s^-1
_RL(nplank)      kexcP             ! s^-1
_RL(nplank)      kexcSi            ! s^-1
_RL(nplank)      kexcFe            ! s^-1

#ifdef GUD_ALLOW_GEIDER
_RL(nplank)      inhibcoef_geid    ! 1
#else
_RL(nplank)      ksatPAR           ! (uEin m^-2 s^-1)^-1
_RL(nplank)      kinhPAR           ! (uEin m^-2 s^-1)^-1
#endif

_RL(nplank)      mQyield           ! mmol C (uEin)^-1
_RL(nplank)      chl2cmax          ! mg Chl (mmol C)^-1
			     

_RL(nplank)      grazemax          ! s^-1
_RL(nplank)      kgrazesat         ! mmol C m^-3


_RL(nplank,nplank)  palat               ! 1
_RL(nplank,nplank)  asseff              ! 1
_RL(nplank,nplank)  ExportFracPreyPred  ! 1

! bacteria

_RL(nplank)  yield     ! bacterial growth yield for all organic matter
_RL(nplank)  yieldO2   ! bacterial growth yield for oxygen
_RL(nplank)  yieldNO3  ! bacterial growth yield for nitrate
_RL(nplank)  ksatPON   ! mmol N m^-3   ! half-saturation of PON for bacterial growth
_RL(nplank)  ksatPOC   ! mmol C m^-3   ! half-saturation of POC for bacterial growth
_RL(nplank)  ksatPOP   ! mmol P m^-3   ! half-saturation of POP for bacterial growth
_RL(nplank)  ksatPOFe  ! mmol Fe m^-3  ! half-saturation of POFe for bacterial growth
_RL(nplank)  ksatDON   ! mmol N m^-3   ! half-saturation of DON for bacterial growth
_RL(nplank)  ksatDOC   ! mmol C m^-3   ! half-saturation of DOC for bacterial growth
_RL(nplank)  ksatDOP   ! mmol P m^-3   ! half-saturation of DOP for bacterial growth
_RL(nplank)  ksatDOFe  ! mmol Fe m^-3  ! half-saturation of DOFe for bacterial growth



#ifdef GUD_ALLOW_RADTRANS

&GUD_RADTRANS_TRAITS

_RL(nplank,nlam)    aphy_chl       ! m^-1 (mg Chl m^-3)^-1
_RL(nplank,nlam)    aphy_chl_ps    ! m^-1 (mg Chl m^-3)^-1
_RL(nplank,nlam)    bphy_mgC       ! m^-1 (mg C m^-3)^-1
_RL(nplank,nlam)    bbphy_mgC      ! m^-1 (mg C m^-3)^-1

#endif



&GUD_DEPENDENT_TRAITS-

! dependent and constant (not read-in) parameters

#ifndef GUD_ALLOW_GEIDER
_RL(nplank)         normI
#endif

#ifdef GUD_ALLOW_RADTRANS
INTEGER(nplank)     ap_type
#endif

_RL(nplank)         biovol
INTEGER(nplank)     group
INTEGER(nplank)     igroup
_RL(nplank)         qcarbon
_RL(nplank)         pp_opt
_RL(nplank)         pp_sig
                    
_RL(nplank,ngroup)  biovol_bygroup
                    
_RL(nplank,nlam)    alphachl            ! mmol C s-1 (uEin m^-2 s^-1)^-1 (mg Chl)^-1
                                        
_RL(nplank)         alpha_mean          ! mmol C s-1 (uEin m^-2 s^-1)^-1 (mg Chl)^-1
_RL(nplank)         chl2cmin            ! mg Chl (mmol C)^-1
                    
_RL(nplank)         mortTempFuncMin
_RL(nplank)         mort2TempFuncMin
'''

import parser
globals().update(parser.parse(__doc__))

