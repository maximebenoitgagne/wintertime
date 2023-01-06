'''
&locals
_RL UNINIT_RL = -999999999 _d 0
_RL pday = 86400.0 _d 0


&GUD_RANDOM_PARAMS
                                             ! set growth days ...small or big organism?
_RL Smallgrow      = .7 _d 0
_RL Biggrow        = .4 _d 0
_RL Smallgrowrange = 0. _d 0
_RL Biggrowrange   = 0. _d 0
_RL diaz_growfac   = 2. _d 0
_RL cocco_growfac  = 1.3 _d 0
_RL diatom_growfac = 0.95 _d 0
                                             ! set mort days ...small or big organism?
_RL Smallmort      = 10. _d 0
_RL Bigmort        = 10. _d 0
_RL Smallmortrange = 0. _d 0
_RL Bigmortrange   = 0. _d 0
                                             ! set export fraction ...small or big organism?
_RL Smallexport = 0.2 _d 0
_RL Bigexport   = 0.5 _d 0
                                             ! set temperature function
_RL tempcoeff1       = 1. _d 0/3. _d 0
_RL tempcoeff2_small = 0.001 _d 0
_RL tempcoeff2_big   = 0.0003 _d 0
_RL tempcoeff3       = 1.04 _d 0
_RL tempmax          = 30. _d 0              ! 32. _d 0
_RL temprange        = 32. _d 0              ! 30. _d 0
_RL tempdecay        = 4. _d 0

                                             ! set nutrient ratios for phyto
_RL val_R_NC         = 16.0 _d 0/120.0 _d 0
_RL val_R_NC_diaz    = 40.0 _d 0/120.0 _d 0
_RL val_R_PC         =  1.0 _d 0/120.0 _d 0
_RL val_R_SiC_diatom = 16.0 _d 0/120.0 _d 0  ! 32 for Fanny's runs
_RL val_R_FeC        = 1.0 _d -3/120.0 _d 0
_RL val_R_FeC_diaz   = 30.0 _d 0 * val_R_FeC
_RL val_R_PICPOC     = 0.8 _d 0
_RL val_R_ChlC       = 16.0 _d 0/120 _d 0    ! for atten_chl   = atten_p * 16

_RL val_R_NC_zoo     = 16.0 _d 0/120.0 _d 0
_RL val_R_PC_zoo     = 1 _d 0/120.0 _d 0
_RL val_R_SiC_zoo    = 0.0 _d 0
_RL val_R_FeC_zoo    = 1.0 _d -3/120.0 _d 0
_RL val_R_PICPOC_zoo = 0.0 _d 0
_RL val_R_ChlC_zoo   = 0 _d 0/120 _d 0       ! for atten_chl = atten_p * 16

                                             ! set sinking rates (m/s)... small or big organism?
_RL SmallSink = 0.0 _d 0/pday
_RL BigSink   = 0.5 _d 0/pday                ! 0.5 _d 0/pday

                               ! set phosphate half stauration constants .. small or big organism
_RL SmallPsat      = 0.015 _d 0
_RL BigPsat        = 0.035 _d 0
_RL ProcPsat       = 0.01 _d 0
_RL UniDzPsat      = 0.012 _d 0
_RL CoccoPsat      = 0.035 _d 0              ! by default same as big
_RL SmallPsatrange = 0.02 _d 0
_RL BigPsatrange   = 0.02 _d 0
_RL ProcPsatrange  = 0.005 _d 0
_RL UniDzPsatrange = 0.02 _d 0
_RL CoccoPsatrange = 0.02 _d 0
                               ! set NH4/NO2 frac, so that NH4/NO2 can be preferred nitrogen source
_RL ksatNH4fac = .50 _d 0
_RL ksatNO2fac = 1.0 _d 0

_RL val_amminhib = 4.6 _d 0    ! coefficient for NH4 inhibition of NO uptake ((mmol N/m3)-1)

                               ! set si half sat
_RL val_ksatsio2 = 1. _d 0

_RL smallksatpar    = 0.12 _d -1             ! 0.8 _d 0
_RL smallksatparstd = 0.20 _d -1             ! 0.3 _d 0
_RL smallkinhpar    = 6.0 _d -3              ! 2.0 _d 0
_RL smallkinhparstd = 0.10 _d -3             ! 0.5 _d 0
_RL Bigksatpar      = 0.12 _d -1             ! 0.35 _d 0
_RL Bigksatparstd   = 0.06 _d -1             ! 0.1 _d 0
_RL Bigkinhpar      = 1.0 _d -3              ! 0.5 _d 0
_RL Bigkinhparstd   = 0.05 _d -3             ! 0.1 _d 0
_RL LLProkinhpar    = 6.0 _d -3
_RL Coccokinhpar    = 0.5 _d -3

                                             ! inhib for Prochl?
! inhibcoef_geid_val   = 1.2 _d 0            ! DUMMY VAL
_RL inhibcoef_geid_val = 0 _d 0              ! DUMMY VAL

! ANNA_Q units for alpha are same as expected: mmol C (mg chla)-1 (uEin)-1 (m)2
! smallalphachl      = 1. _d -6                ! mmol C (uEin/m-2)-1 (mg Chl)-1
! smallalphachlrange = 1. _d -6                ! mmol C (uEin/m-2)-1 (mg Chl)-1
! Bigalphachl        = 6. _d -7                ! mmol C (uEin/m-2)-1 (mg Chl)-1
! Bigalphachlrange   = 4. _d -7                ! mmol C (uEin/m-2)-1 (mg Chl)-1
                         ! ANNA mQyield vals are from alphachl / aphy_chl which for now is 0.02
                         ! ANNA ranges for mQyield are same as alphachl but reduced by factor 100
_RL smallmQyield      = 5. _d -5             ! mmol C (uEin)-1
_RL smallmQyieldrange = 1. _d -4             ! mmol C (uEin)-1
_RL BigmQyield        = 3. _d -5             ! mmol C (uEin)-1
_RL BigmQyieldrange   = 4. _d -5             ! mmol C (uEin)-1

_RL smallchl2cmax      = 0.2 _d 0            ! mg Chl (mmol C)
_RL smallchl2cmaxrange = 0.3 _d 0            ! mg Chl (mmol C)
_RL Bigchl2cmax        = 0.5 _d 0            ! mg Chl (mmol C)
_RL Bigchl2cmaxrange   = 0.3 _d 0            ! mg Chl (mmol C)

                       ! ANNA value of aphy_chl_ave = 0.02 - its the mean of all spectras used as input data
_RL aphy_chl_ave = 0.02 _d 0                 ! m2 (mg chla)-1 (ie. x chla gives absorption m-1)

_RL val_acclimtimescl=1./(60. _d 0*60. _d 0*24. _d 0*20. _d 0)  ! inverse time scale for Chl acclimation

! GRAZERS

LOGICAL oldTwoGrazers = .FALSE.              ! old defaults for 2 grazers
                                             ! set grazing rates .. small or big organism?
_RL GrazeFast = 1.0 _d 0/(2.0 _d 0*pday)     ! was 1/5 days
_RL GrazeSlow = 1.0 _d 0/(7.0 _d 0*pday)     ! was 1/30 days
                                             ! set zoo exportfrac
_RL ZooexfacSmall = 0.2 _d 0
_RL ZooexfacBig   = 0.7 _d 0
                                             ! set zoo mortality
_RL ZoomortSmall  = 1.0 _d 0/(30.0 _d 0*pday)
_RL ZoomortBig    = 1.0 _d 0/(30.0 _d 0*pday)
_RL ZoomortSmall2 = 0. _d 0
_RL ZoomortBig2   = 0. _d 0
                                             ! set faction graz to POM
_RL ExGrazfracbig   = 0.8 _d 0
_RL ExGrazfracsmall = 0.8 _d 0
                                             ! set palatibility
_RL palathi = 1.0 _d 0
_RL palatlo = 0.2 _d 0
                                             ! set palatibilty diatom factor
_RL diatomgraz = 0.7 _d 0
_RL coccograz  = 0.6 _d 0
_RL olargegraz = 1.0 _d 0
                                             ! set grazing effeciency
_RL GrazeEfflow = 0.2 _d 0
_RL GrazeEffmod = 0.5 _d 0
_RL GrazeEffhi  = 0.7 _d 0


! new defaults for many grazers

_RL GrazeRate  = 1.0 _d 0/(2.0 _d 0*pday)    ! grazing rate
_RL ExGrazfrac = 0.8 _d 0                    ! fraction of sloppy feeding that goes to particulate

_RL val_palat = 0.0 _d 0                     ! need to set in data.traits

_RL val_ass_eff = 0.70 _d 0                  ! grazing efficiency

_RL kgrazesat_val = 12 _d 0                  ! = 0.1 mmol P m-3

_RL Zoomort  = 1.0 _d 0/(30.0 _d 0*pday)     ! zoo linear mortality rate (s-1)
_RL Zoomort2 = 0. _d 0                       ! zoo quadratic mortality ((mmol C)-1 s-1)
_RL Zooexfac = 0.7 _d 0                      ! fraction of dead zoo that goes to particulate

_RL ZooDM = 100 _d 0                         ! diameter (not used so far)



&GUD_TRAIT_PARAMS

! used in gud_generate_allometric

LOGICAL gud_sort_biovol = .FALSE.

LOGICAL GUD_effective_ksat = .FALSE.         ! compute effective half-saturation for non-quota elements
INTEGER gud_select_kn_allom = 2              ! 1: use Ward et al formulation, 2: use Follett et al


_RL logvolbase = 0.0 _d 0                    ! draw volumes from list 10**logvolbase, 10**(logvolbase+logvolinc), ...
_RL logvolinc  = 0.0 _d 0
_RL(nGroup)        biovol0       = 0.0 _d 0  ! set grp_biovol to biovol0, biovol0*biovolfac, ...
_RL(nGroup)        biovolfac     = 1.0 _d 0
INTEGER(nGroup)    logvol0ind    = 0         ! or use consecutive values starting here for each group
_RL(nPlank,nGroup) grp_logvolind = 0 _d 0    ! or set individual indices into logvol list
_RL(nPlank,nGroup) grp_biovol    = 0 _d 0    ! or set volumes of types in each group directly (um^3)

CHARACTER*80(nGroup) grp_names   = ''
INTEGER(nGroup)    grp_nplank    = 0
INTEGER(nGroup)    grp_photo     = 1
INTEGER(nGroup)    grp_bacttype  = 0         ! 1: particle-associated, 2: free-living
INTEGER(nGroup)    grp_aerobic   = 0
INTEGER(nGroup)    grp_denit     = 0
INTEGER(nGroup)    grp_pred      = 0
INTEGER(nGroup)    grp_prey      = 1
INTEGER(nGroup)    grp_hasSi     = 0
INTEGER(nGroup)    grp_hasPIC    = 0
INTEGER(nGroup)    grp_diazo     = 0
INTEGER(nGroup)    grp_useNH4    = 1
INTEGER(nGroup)    grp_useNO2    = 1
INTEGER(nGroup)    grp_useNO3    = 1
INTEGER(nGroup)    grp_combNO    = 1
INTEGER(nGroup)    grp_aptype    = 0
INTEGER(nGroup)    grp_tempMort  = 1
INTEGER(nGroup)    grp_tempMort2 = 1
_RL(nGroup)        grp_Xmin            = 0 _d 0                 ! used to be 120D-20 for phyto
_RL(nGroup)        grp_R_NC            = 16.0 _d 0/120.0 _d 0
_RL(nGroup)        grp_R_PC            = 1 _d 0/120.0 _d 0
_RL(nGroup)        grp_R_SiC           = 0 _d 0
_RL(nGroup)        grp_R_FeC           = 1.0 _d -3/120.0 _d 0
_RL(nGroup)        grp_R_ChlC          = 16 _d 0/120 _d 0
_RL(nGroup)        grp_R_PICPOC        = 0.8 _d 0
_RL(nGroup)        grp_ExportFracMort  = 0.5 _d 0
_RL(nGroup)        grp_ExportFracMort2 = 0.5 _d 0
_RL(nGroup)        grp_ExportFrac      = UNINIT_RL
_RL(nGroup)        grp_mort            = 0.02 _d 0 / pday
_RL(nGroup)        grp_mort2           = 0.0 _d 0
_RL(nGroup)        grp_tempcoeff1      = 1. _d 0/3. _d 0
_RL(nGroup)        grp_tempcoeff2      = 0.001 _d 0
_RL(nGroup)        grp_tempcoeff3      = 1.04 _d 0
_RL(nGroup)        grp_tempopt         = 2. _d 0
_RL(nGroup)        grp_tempdecay       = 4. _d 0
_RL(nGroup)        grp_pp_sig          = 1.0 _d 0
#ifdef GUD_ALLOW_GEIDER
_RL(nGroup)        grp_mQyield         = 7.5 _d -5
_RL(nGroup)        grp_chl2cmax        = .3 _d 0
_RL(nGroup)        grp_inhibcoef_geid  = 0 _d 0
#else
_RL(nGroup)        grp_ksatPAR         = 0.012 _d 0
_RL(nGroup)        grp_kinhPAR         = 6.0 _d -3  ! big: 1 _d -3
#endif

_RL(nGroup)        grp_ksatNH4fac = 0.5 _d 0  ! these are only used for gud_effective_ksat
_RL(nGroup)        grp_ksatNO2fac = 1.0 _d 0

_RL(nGroup)        grp_amminhib            = 4.6 _d 0          ! coefficient for NH4 inhibition of NO uptake ((mmol N/m3)-1)
_RL(nGroup)        grp_aphy_chl            = 0.02 _d 0         ! mean of all spectra (m2 (mg chla)-1)
_RL(nGroup)        grp_acclimtimescl       = 1 _d 0/(20*pday)  ! inverse time scale for Chl acclimation
_RL(nGroup)        grp_acclimtimescl_denom = 1 _d 0


! parameter = aV^b
! if errors are relative (*/) then state as "log(error)"
! if errors are absolute (+-) then state as "error"
                                             !ccccccccccccccccccccccccccccccccccccccccccccccccccccc
                                             ! maximum specific grazing rate
_RL(nGroup) a_graz       = 21.9 _d 0 / pday
_RL(nGroup) a_graz_denom = 1. _d 0
_RL(nGroup) b_graz       = -0.16 _d 0
                                             ! half saturation grazing prey carbon concentration
_RL(nGroup) a_kg         = 1.00 _d 0         ! mmol C m^-3  ! was 22.4
_RL(nGroup) b_kg         = 0.00 _d 0
                                             !ccccccccccccccccccccccccccccccccccccccccccccccccccccc
                                             ! sinking (enter as positive downwards)
_RL(nGroup) a_biosink        =  0.28 _d -1 / pday  ! m s^-1
_RL(nGroup) a_biosink_denom  =  1. _d 0
_RL(nGroup) b_biosink        =  0.39 _d 0
                                             ! swimming velocity (enter as positive upwards - converted in quota_generate_phyto.F)
_RL(nGroup) a_bioswim        =  0.00 _d 0 / pday   ! m s^-1
_RL(nGroup) a_bioswim_denom  =  1. _d 0
_RL(nGroup) b_bioswim        =  0.18 _d 0
                                             !ccccccccccccccccccccccccccccccccccccccccccccccccccccc
                                             ! mortality
!a_mort           =  0.02 _d 0 / pday
!a_mort_denom     =  1. _d 0
!b_mort           =  0.00 _d 0
                                             ! predator prey preference distribution parameters
_RL(nGroup) a_prdpry         =  1024. _d 0         ! dimensionless
_RL(nGroup) b_prdpry         =  0.00 _d 0
                                             !ccccccccccccccccccccccccccccccccccccccccccccccccccccc
                                             ! carbon
                                             ! max photosynthetic rate (modified in quota_generate_plankton.F)
_RL(nGroup) a_vmax_DIC       =  1.00 _d 0 / pday
_RL(nGroup) a_vmax_DIC_denom =  1. _d 0
_RL(nGroup) b_vmax_DIC       = -0.15 _d 0
                                             ! cellular carbon content
_RL(nGroup) a_qcarbon        =  1.80 _d -11        ! mmol C cell^-1
_RL(nGroup) b_qcarbon        =  0.94 _d 0
                                             ! respiration (Note function of cellular C --> aC^b!)
_RL(nGroup) a_respir         =  0.00 _d 0    ! mmol C cell^-1 s^-1 ! was 3.21D-11 / pday
_RL(nGroup) a_respir_denom   =  1. _d 0
_RL(nGroup) b_respir         =  0.93 _d 0
                                             ! carbon excretion
_RL(nGroup) a_kexc_c         =  0.00 _d 0    ! was 0.32 _d -1 / pday
_RL(nGroup) b_kexc_c         = -0.33 _d 0
                                             ! fraction grazing to DOC
!a_beta_graz_c    =  1.00 _d 0
!b_beta_graz_c    = -0.40 _d 0
                                             ! fraction mortality to DOC
!a_beta_mort_c    =  1.00 _d 0
!b_beta_mort_c    = -0.40 _d 0
                                             ! nitrogen & nitrate
                                             ! maximum NO3 uptake rate
_RL(nGroup) a_vmax_NO3       =  0.51 _d 0 / pday   ! mmol N (mmol C)^-1 s^-1
_RL(nGroup) a_vmax_NO3_denom =  1. _d 0
_RL(nGroup) b_vmax_NO3       = -0.27 _d 0
                                             ! NO3 half-saturation
_RL(nGroup) a_kn_NO3         =  0.17 _d 0          ! mmol N m^-3
_RL(nGroup) b_kn_NO3         =  0.27 _d 0
                                             ! N minimum quota
_RL(nGroup) a_qmin_n         =  0.07 _d 0          ! mmol N (mmol C)^-1
_RL(nGroup) b_qmin_n         = -0.17 _d 0
                                             ! N maximum quota
_RL(nGroup) a_qmax_n         =  0.25 _d 0          ! mmol N (mmol C)^-1
_RL(nGroup) b_qmax_n         = -0.13 _d 0
                                             ! nitrogen excretion
_RL(nGroup) a_kexc_n         =  0.00 _d 0    ! was 0.24 _d -1 / pday
_RL(nGroup) b_kexc_n         = -0.33 _d 0
                                             ! fraction grazing to DON
!a_beta_graz_n    =  1.00 _d 0
!b_beta_graz_n    = -0.40 _d 0
                                             ! fraction mortality to DON
!a_beta_mort_n    =  1.00 _d 0
!b_beta_mort_n    = -0.40 _d 0
                                             ! nitrite
                                             ! maximum NO2 uptake rate
_RL(nGroup) a_vmax_NO2       =  0.51 _d 0 / pday   ! mmol N (mmol C)^-1 s^-1
_RL(nGroup) a_vmax_NO2_denom =  1.0 _d 0
_RL(nGroup) b_vmax_NO2       = -0.27 _d 0
                                             ! NO2 half-saturation
_RL(nGroup) a_kn_NO2         =  0.17 _d 0          ! mmol N m^-3
_RL(nGroup) b_kn_NO2         =  0.27 _d 0
                                             ! ammonium
                                             ! maximum NH4 uptake rate
_RL(nGroup) a_vmax_NH4       =  0.26 _d 0 / pday   ! mmol N (mmol C)^-1 s^-1
_RL(nGroup) a_vmax_NH4_denom =  1.0 _d 0
_RL(nGroup) b_vmax_NH4       = -0.27 _d 0
                                             ! NH4 half-saturation
_RL(nGroup) a_kn_NH4         =  0.85 _d -1         ! mmol N m^-3
_RL(nGroup) b_kn_NH4         =  0.27 _d 0

_RL(nGroup) a_vmax_N         =  1.28 _d 0 / pday   ! vmax_n has to be >= vmax_no3+vmax_no2+vmax_nh4 for all sizes!
_RL(nGroup) a_vmax_N_denom   =  1. _d 0
_RL(nGroup) b_vmax_N         = -0.27 _d 0

                                             ! phosphate
                                             ! maximum PO4 uptake rate
_RL(nGroup) a_vmax_PO4       =  0.77 _d -1 / pday  ! mmol P (mmol C)^-1 s^-1
_RL(nGroup) a_vmax_PO4_denom =  1.0 _d 0
_RL(nGroup) b_vmax_PO4       = -0.27 _d 0
                                             ! PO4 half-saturation
_RL(nGroup) a_kn_PO4         =  0.26 _d -1         ! mmol P m^-3
_RL(nGroup) b_kn_PO4         =  0.27 _d 0
                                             ! minimum P quota
_RL(nGroup) a_qmin_p         =  2.00 _d -3         ! mmol P (mmol C)^-1
_RL(nGroup) b_qmin_p         =  0.00 _d 0
                                             ! maximum P quota
_RL(nGroup) a_qmax_p         =  0.01 _d 0          ! mmol P (mmol C)^-1
_RL(nGroup) b_qmax_p         =  0.00 _d 0
                                             ! P excretion
_RL(nGroup) a_kexc_p         =  0.24 _d -1 / pday  ! s^-1
_RL(nGroup) b_kexc_p         = -0.33 _d 0
                                             ! fraction grazing to DOP
_RL(nGroup) a_beta_graz_p    =  1.00 _d 0
_RL(nGroup) b_beta_graz_p    = -0.40 _d 0
                                             ! fraction mortality to DOP
_RL(nGroup) a_beta_mort_p    =  1.00 _d 0
_RL(nGroup) b_beta_mort_p    = -0.40 _d 0
                                             ! silicate
                                             ! maximum Si uptake rate
_RL(nGroup) a_vmax_SiO2      =  0.77 _d -1 / pday  ! mmol Si (mmol C)^-1 s^-1
_RL(nGroup) a_vmax_SiO2_denom =  1.0 _d 0
_RL(nGroup) b_vmax_SiO2      = -0.27 _d 0
                                             ! Si half-saturation
_RL(nGroup) a_kn_SiO2        =  0.24 _d -1         ! mmol Si m^-3
_RL(nGroup) b_kn_SiO2        =  0.27 _d 0
                                             ! minimum Si quota
_RL(nGroup) a_qmin_si        =  2.00 _d -3   ! mmol Si (mmol C)^-1  ! was 0.84 _d -1
_RL(nGroup) b_qmin_si        =  0.00 _d 0    ! was -0.17
                                             ! maximum Si quota
_RL(nGroup) a_qmax_si        =  4.00 _d -3   ! mmol Si (mmol C)^-1  ! was 0.30 _d 0
_RL(nGroup) b_qmax_si        =  0.00 _d 0    ! was -0.13
                                             ! Si excretion
_RL(nGroup) a_kexc_si        =  0.00 _d 0  / pday  ! s^-1
_RL(nGroup) b_kexc_si        =  0.00 _d 0
                                             ! fraction grazing to DOSi
_RL(nGroup) a_beta_graz_si   =  0.00 _d 0
_RL(nGroup) b_beta_graz_si   =  0.00 _d 0
                                             ! fraction mortality to DOSi
_RL(nGroup) a_beta_mort_si   =  0.00 _d 0
_RL(nGroup) b_beta_mort_si   =  0.00 _d 0
                                             ! iron
                                             ! maximum Fe uptake rate
_RL(nGroup) a_vmax_FeT       =  14.0 _d -6 / pday  ! mmol Fe (mmol C)^-1 s^-1  ! was 96.2 _d -6 / pday
_RL(nGroup) a_vmax_FeT_denom =  1.0 _d 0
_RL(nGroup) b_vmax_FeT       = -0.27 _d 0
                                             ! Fe half-saturation
_RL(nGroup) a_kn_feT         =  80.0 _d -6   ! mmol Fe m^-3  ! was 32.1 _d -6
_RL(nGroup) b_kn_FeT         =  0.27 _d 0
                                             ! minimum Fe quota
_RL(nGroup) a_qmin_fe        =  1.50 _d -6   ! mmol Fe (mmol C)^-1 - Mongin (2006)
!_RL(nGroup) a_qmin_fe        =  5.00 _d -6   ! mmol Fe (mmol C)^-1
_RL(nGroup) b_qmin_fe        =  0.00 _d 0
                                             ! maximum Fe quota
_RL(nGroup) a_qmax_fe        =  80.0 _d -6   ! mmol Fe (mmol C)^-1 - Mongin (2006)
!_RL(nGroup) a_qmax_fe        =  15.0 _d -6   ! mmol Fe (mmol C)^-1
_RL(nGroup) b_qmax_fe        =  0.00 _d 0
                                             ! Fe excretion
_RL(nGroup) a_kexc_fe        =  0.00 _d 0  / pday  ! s^-1
_RL(nGroup) b_kexc_fe        =  0.00 _d 0
                                             ! fraction grazing to DOFe
_RL(nGroup) a_beta_graz_fe   =  1.00 _d 0
_RL(nGroup) b_beta_graz_fe   = -0.40 _d 0
                                             ! fraction mortality to DOFe
_RL(nGroup) a_beta_mort_fe   =  1.00 _d 0
_RL(nGroup) b_beta_mort_fe   = -0.40 _d 0

_RL(nGroup,nGroup) grp_ExportFracPreyPred = 0.5 _d 0
_RL(nGroup,nGroup) grp_ass_eff            = 0.70 _d 0
'''

import parser
globals().update(parser.parse(__doc__))

