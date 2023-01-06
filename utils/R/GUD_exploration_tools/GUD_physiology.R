#'@title Exploration of physiological relationship from MITGCM/GUD
#'@description
#'Read input files to get parameters' values, and physiological functions.
#'Plot physiological relationships.
#'
#'@param None yet.
#'@return list(s) of parameters and plots.
#'@author
#'F. Maps 2020
#'@export

outfile='../../../output/run_20200710_0001_for_GUD_physiology5/GUD_physiology.network.png'

### Loading modules & functions

library("dplyr")

library('ggplot2')

library('reshape2')

source('data_options.R')

source('data_default.R')

source('data_nml.R')

### Get parameters values from namelists in data.gud

# Get option keys for the GUD package
opt      <- read.options()
attach(opt)

# Get default parameters values
pars     <- read.default( opt )

# Read run-specific values from namelists
pars_nml <- read.nml( pars )
attach(pars_nml)

# WARNING: for convenience, greate the ngroup variable here
ngroup <- nGroup

# Groups names !!! CHECK
species    <- rep( '', nplank )
cum_nplank <- cumsum( grp_nplank )

species[1:cum_nplank[1]]                     <- "Prochlorococcus"
species[ (cum_nplank[1] +1):cum_nplank[2]  ] <- "Synechococcus"
species[ (cum_nplank[2] +1):cum_nplank[3]  ] <- "Small eukaryots"
species[ (cum_nplank[3] +1):cum_nplank[4]  ] <- "Coccolithophore"
species[ (cum_nplank[4] +1):cum_nplank[5]  ] <- "Diazotrophs"
species[ (cum_nplank[5] +1):cum_nplank[6]  ] <- "Trichodesmium"
species[ (cum_nplank[6] +1):cum_nplank[7]  ] <- "Diatoms"
species[ (cum_nplank[7] +1):cum_nplank[8]  ] <- "Dinoflagellates"
species[ (cum_nplank[8] +1):cum_nplank[9]  ] <- "Microzooplankton"
# species[ (cum_nplank[9] +1):cum_nplank[10] ] <- "Metazoo - active"
# species[ (cum_nplank[10]+1):cum_nplank[11] ] <- "Metazoo - dormant"


### Compute biovolumes and ESD (gud_generate_allometric.F)

# compute cell volumes in micrometer^3
#
# in decreasing precedence (if bold quantity is set):
#
#   V = GRP_BIOVOL(j,g)
#   V = 10**(logvolbase+(GRP_BIOVOLIND(j,g)-1)*logvolinc)
#   V = 10**(logvolbase+(LOGVOL0IND(g)+j-2)*logvolinc)
#   V = BIOVOL0(g)*biovolfac(g)**(j-1)

# if logvol0ind is set, use it to compute biovol0
if( sum( logvol0ind ) > 0 ) {
  if( sum( biovol0 ) > 0 ) { warning( 'GUD_GENERATE_ALLOMETRIC: cannot set both biovol0 and logvol0ind' ) }

  logvol    <- logvolbase + ( logvol0ind - 1 ) * logvolinc
  biovol0   <- 10 ** logvol
  biovolfac <- 10 ** logvolinc

}

# Matrix of plankton types indices, by groups
jp <- matrix( NA, ncol = ngroup, nrow = nplank )
for( g in 1:ngroup ) {
  if( grp_nplank[g] > 0 ) {
    jp[1:grp_nplank[g],g] <- seq( 1, grp_nplank[g] )
  }
}

# Biovolumes, by groups
grp_biovol     <- t( apply( jp, 1, function(x) biovol0 * biovolfac ** (x-1) ) )
biovol_bygroup <- grp_biovol

# ESD for plotting

esd <- 2 * ( 3 * grp_biovol / (4*pi) ) ^ (1/3)

# Single list of plankton types, ordered by groups

j  <- as.vector( jp )
jx <- is.finite( j  )

group  <- rep( 1:ngroup, grp_nplank )
igroup <- jp[jx]
biovol <- grp_biovol[jx]

#################################################
### Plot size structure

theme_set( theme_classic() )

plot_biovol <- data.frame( esd=esd[jx], biovol = biovol, ptr = 1:nplank, species = species )

  print( ggplot( plot_biovol, aes( x = as.factor(ptr), y = esd ) ) +

         geom_point( aes( colour = species ), size = 4 ) +

         geom_hline( yintercept = 20, lty = 2 ) +

         scale_y_log10() +

         labs( title = "Planktonic types generated in GUD", x = "Type ID", y = "ESD in μm")
       )


#################################################
### Compute traits from trait parameters
### ( gud_generate_allometric.F )

#--- Non-allometric traits (same WITHIN groups)

# Key group properties

isPhoto       <- grp_photo[group]
hasSi         <- grp_hasSi[group]
hasPIC        <- grp_hasPIC[group]
diazo         <- grp_diazo[group]
useNH4        <- grp_useNH4[group]
useNO2        <- grp_useNO2[group]
useNO3        <- grp_useNO3[group]
combNO        <- grp_combNO[group]

Xmin          <- grp_Xmin[group]
amminhib      <- grp_amminhib[group]
acclimtimescl <- grp_acclimtimescl[group]

# Mortality

mort            <- grp_mort[group]
mort2           <- grp_mort2[group]
tempMort        <- grp_tempMort[group]
tempMort2       <- grp_tempMort2[group]
ExportFracMort  <- grp_ExportFracMort[group]
ExportFracMort2 <- grp_ExportFracMort2[group]
ExportFrac      <- grp_ExportFrac[group]

# Temperature function parameters

phytoTempCoeff   <- grp_tempcoeff1[group]
phytoTempExp1    <- grp_tempcoeff3[group]
phytoTempExp2    <- grp_tempcoeff2[group]
phytoTempOptimum <- grp_tempopt[group]
phytoDecayPower  <- grp_tempdecay[group]

# Plankton elemental ratios

R_NC     <- grp_R_NC[group]
R_PC     <- grp_R_PC[group]
R_SiC    <- grp_R_SiC[group]
R_FeC    <- grp_R_FeC[group]
R_ChlC   <- grp_R_ChlC[group]
R_PICPOC <- grp_R_PICPOC[group]

#--- Allometric traits (change BETWEEN types)

# Plankton sinking and swimming

wsink <- a_biosink[group] * biovol ** b_biosink[group]
wswim <- a_bioswim[group] * biovol ** b_bioswim[group]

# Respiration rate is given in terms of carbon content

qcarbon     <- a_qcarbon[group] * biovol ** b_qcarbon[group]
respiration <- a_respir[group] *
              (12.9 * qcarbon ) ** b_respir[group] / qcarbon

# Parameters relating to inorganic nutrients

PCmax     <- a_vmax_DIC[group]  * biovol ** b_vmax_DIC[group]

vmax_NH4  <- a_vmax_NH4[group]  * biovol ** b_vmax_NH4[group]
vmax_NO2  <- a_vmax_NO2[group]  * biovol ** b_vmax_NO2[group]
vmax_NO3  <- a_vmax_NO3[group]  * biovol ** b_vmax_NO3[group]
vmax_N    <- a_vmax_N[group]    * biovol ** b_vmax_N[group]
vmax_PO4  <- a_vmax_PO4[group]  * biovol ** b_vmax_PO4[group]
vmax_SiO2 <- a_vmax_SiO2[group] * biovol ** b_vmax_SiO2[group]
vmax_FeT  <- a_vmax_FeT[group]  * biovol ** b_vmax_FeT[group]

Qnmin     <- a_qmin_n[group]    * biovol ** b_qmin_n[group]
Qnmax     <- a_qmax_n[group]    * biovol ** b_qmax_n[group]
Qpmin     <- a_qmin_p[group]    * biovol ** b_qmin_p[group]
Qpmax     <- a_qmax_p[group]    * biovol ** b_qmax_p[group]
Qsimin    <- a_qmin_si[group]   * biovol ** b_qmin_si[group]
Qsimax    <- a_qmax_si[group]   * biovol ** b_qmax_si[group]
Qfemin    <- a_qmin_fe[group]   * biovol ** b_qmin_fe[group]
Qfemax    <- a_qmax_fe[group]   * biovol ** b_qmax_fe[group]

ksatNH4   <- a_kn_NH4[group]    * biovol ** b_kn_NH4[group]
ksatNO2   <- a_kn_NO2[group]    * biovol ** b_kn_NO2[group]
ksatNO3   <- a_kn_NO3[group]    * biovol ** b_kn_NO3[group]
ksatPO4   <- a_kn_PO4[group]    * biovol ** b_kn_PO4[group]
ksatSiO2  <- a_kn_SiO2[group]   * biovol ** b_kn_SiO2[group]
ksatFeT   <- a_kn_feT[group]    * biovol ** b_kn_FeT[group]    # NOT defined in namelist

# Parameters relating to quota nutrients

# Excretion

kexcC  <- a_kexc_c[group]  * biovol ** b_kexc_c[group]
kexcN  <- a_kexc_n[group]  * biovol ** b_kexc_n[group]
kexcP  <- a_kexc_p[group]  * biovol ** b_kexc_p[group]
kexcSi <- a_kexc_si[group] * biovol ** b_kexc_si[group]
kexcFe <- a_kexc_fe[group] * biovol ** b_kexc_fe[group]

if( GUD_effective_ksat ) {
  # compute effective half sat for uptake of non-quota elements
  # we compute it for NO3 and scale for other

  if( gud_select_kn_allom == 1) {
    # Following Ward et al.
    kappa <- ( ksatNO3 *
               PCmax * Qnmin * ( Qnmax - Qnmin ) ) /
             ( vmax_NO3 * Qnmax +
               PCmax * Qnmin * ( Qnmax - Qnmin ) )
  } else if( gud_select_kn_allom == 2 ) {
    # Following Follet et al.
    kappa <- ksatNO3 * PCmax * Qnmin / vmax_NO3
  } else {
    warning( 'GUD_GENERATE_ALLOMETRIC: illegal value for gud_select_kn_allom' )
    stop(    'ABNORMAL END: S/R GUD_GENERATE_ALLOMETRIC' )
  }

  if( !GUD_ALLOW_NQUOTA ) {
   ksatNO3 <- kappa
   ksatNO2 <- kappa * grp_ksatNO2fac[group]
   ksatNH4 <- kappa * grp_ksatNH4fac[group]
  }
  if( !GUD_ALLOW_PQUOTA ) {
    ksatPO4 <- kappa / R_NC * R_PC
  }
  if( !GUD_ALLOW_SIQUOTA ) {
    ksatSiO2 <- kappa / R_NC * R_SiC
  }
  if( !GUD_ALLOW_FEQUOTA ) {
    ksatFeT <- kappa / R_NC * R_FeC
  }
}

#################################################
### Plot half saturation coefficients
if(0==1) {
theme_set( theme_classic() )

plot_ksat <- data.frame( esd=esd[jx], ksat = ksatNO3, ptr = 1:nplank, species = species )

print( ggplot( plot_ksat, aes( x = as.factor(ptr), y = ksat ) ) +

       geom_point( aes( colour = species ), size = 4 ) +

       scale_y_log10() +

       labs( title = "Half saturation for growth on NO3", x = "Planktonic type ID", y = "Ksat NO3")
     )
}
#################################################

# Parameters for bacteria

bactType   <- grp_bacttype[group] # NOT defined in namelist; default in gud_readgenparams.F
isAerobic  <- grp_aerobic[group]  # NOT defined in namelist; default in gud_readgenparams.F
isDenit    <- grp_denit[group]    # NOT defined in namelist; default in gud_readgenparams.F

yieldo2    <- 1.0
yieldno3   <- 1.0

yield  [isAerobic != 0] <- yod
yieldo2[isAerobic != 0] <- yoe

yield   [isDenit != 0]  <- ynd
yieldno3[isDenit != 0]  <- yne

ksatPOC    <- ksatPON / R_NC
ksatPOP    <- ksatPON / R_NC * R_PC
ksatPOFe   <- ksatPON / R_NC * R_FeC
ksatDOC    <- ksatDON / R_NC
ksatDOP    <- ksatDON / R_NC * R_PC
ksatDOFe   <- ksatDON / R_NC * R_FeC

# Parameters relating to light

if( GUD_ALLOW_GEIDER ) {
  mQyield        <- grp_mQyield[group]
  chl2cmax       <- grp_chl2cmax[group]
  inhibcoef_geid <- grp_inhibcoef_geid[group]
} else {
  ksatPAR        <- grp_ksatpar[group]
  kinhPAR        <- grp_kinhpar[group]
}

# Grazing by zooplankton

# maximum grazing rate (s^-1)
grazemax <- a_graz[group] * biovol ** b_graz[group]

# grazing half-saturation (mmol C m^-3)
kgrazesat <- a_kg[group] * biovol ** b_kg[group]

#################################################
### Plot maximum grazing rate
theme_set( theme_classic() )

plot_grazx <- data.frame( esd=esd[jx], grazx = grazemax * 86400, ptr = 1:nplank, species = species )

print( ggplot( plot_grazx, aes( x = as.factor(ptr), y = grazx ) ) +

       geom_point( aes( colour = species ), size = 4 ) +

       scale_y_log10() +

       labs( title = "Maximum grazing rate", x = "Planktonic type ID", y = "d^-1")
)
#################################################

if( GUD_ALLOMETRIC_PALAT ) {
# Assign grazing preferences according to predator/prey radius ratio

  # grazing size preference ratio
  pp_opt  <- as.vector( a_prdpry[group] * biovol ** b_prdpry[group] )

  # standard deviation of size preference
  pp_sig  <- as.vector( grp_pp_sig[group] )

  # predator / prey ratio
  # 0 if the type is NOT tagged as a potential prey
  # transpose matrix for future calculations in R

  #!!! grp_prey[group] NEEDS TO BE DEFINED AS A VECTOR TO BE * WITH A matrix(...)
  prd_pry <- as.vector(grp_prey[group]) * matrix(biovol, nrow = nplank, ncol = nplank, byrow=T) / biovol
  prd_pry <- t( as.vector(grp_pred[group]) * t(prd_pry) )

  # prey palatability ~ Normal distribution PDF
  palat   <- t( exp( -( log( t(prd_pry) / pp_opt ) ** 2 ) / ( 2 * pp_sig ** 2 ) ) / pp_sig / 2 )

              palat_min  <- 1e-2
  palat[palat<palat_min] <- 0

} else if( !exists( "palat" ) ) {
  palat   <- 0
}

ExportFracPreyPred <- grp_ExportFracPreyPred


#################################################
### Plot trophic network

require(qgraph)

palati <- palat > 0

          netwk   <- cbind( which( palati, arr.ind = TRUE ), palat[palati] )
colnames( netwk ) <- c( "from", "to", "weight" )

# Circle
#qc <- qgraph( as.matrix( netwk ), layout = "circle" )

# Hubs
png(outfile)
qh <- qgraph( as.matrix( netwk ) )
dev.off()
# All the info needed is inside the returned object, including the layout of the graph.

# ### Plot prey:predator preference size window
# 
# theme_set( theme_classic() )
# 
# # Here I extract the predator:prey preferences for the largest microzoo [48] & the smalest metazoo [49]
# #plot_prdpry <- data.frame( p2p = log10( prd_pry[,48:49]^(1/3) ), pal = palat[,48:49], Preys = species )
# plot_prdpry <- data.frame( p2p = log10( prd_pry[,48:49]^(1/3) ), pal = palat[,48:49], Preys = species )
# 
# print( ggplot( plot_prdpry ) + 
#          
#          geom_line( aes( x = p2p.1, y = pal.1, colour = Preys ), size = 1.5 ) +
#          geom_line( aes( x = p2p.2, y = pal.2, colour = Preys ), size = 1.5 ) +
# 
#          geom_vline( xintercept = log10(20), lty = 2 ) +
#          
#          scale_x_reverse( labels = c("1:1","10:1","100:1","1000:1") ) +
#          
#          labs( title = "Prey:Predator ESD; preference for micro vs metazoo in GUD", x = "Prey:Pred Ratio", y = "Preference")
# )
# 
# 
# #################################################
# ### Compute light-related functions
# ### ( gud_light.F )
# 
# ### !!! NOT WORKING !!! MAXIME CHECK HOW TO READ INPUT FILES HERE !!!
# 
# if( FALSE) {
# #--- Compute light fields
# 
# if( GUD_READ_PAR ) {
#   PARF <- 100 #surfPAR
# } else if( GUD_USE_QSW ) {
#   PARF <- -parfrac * parconv * Qsw * maskC
# } else {
#   GUD_INSOL(midTime, PARlat, bj, .FALSE.)
#   # convert W/m2 to uEin/s/m2
#   PARF <- PARlat / 0.2174
# }
# 
# # Take into account the impact of ice on surface irradiance
# if( GUD_PARUICE ) {
#   PARF_ice <- PARF * 0.
# } else {
#   PARF     <- PARF * ( 1 - iceFrac )
# }
# 
# # Compute differential PAR profiles under open water | sea ice
# for( k in 1:Nr ) {
# 
#   Chl <- 0
#   
#   if( GUD_ALLOW_GEIDER ) {
#     if( GUD_ALLOW_CHLQUOTA ) {
# 
#       for( j in 1:nPhoto ) {
#         Chl <- Chl + max( 0, Ptracer[k, iChl+j-1] )
#       }
#     } else {
#       Chl <- ChlPrev[k]
#     }
#   } else {
#     for( j in 1:nPhoto ) {
#       Chl <- Chl + max( 0, Ptracer[ k, ic+j-1] * R_ChlC[j] )
#     }
#   }
# 
#   # TODO should include hFacC
#   atten <- ( katten_w + katten_Chl * Chl ) * DRF[k]
# 
#   if( GUD_PARUICE ) {
#     if( GUD_AVPAR ) {
#       PAR    [k,1] <- PARF     * ( 1 - exp( -atten ) ) / atten
#       PAR_ice[k,1] <- PARF_ice * ( 1 - exp( -atten ) ) / atten
#     } else { # USE_MIDPAR
#       PAR    [k,1] <- PARF     * exp( -0.5 * atten )
#       PAR_ice[k,1] <- PARF_ice * exp( -0.5 * atten )
#     }
#   } else {
#     if( GUD_AVPAR ) {
#       PAR[k,1] <- PARF * ( 1 - exp( -atten ) ) / atten
#     } else { # USE_MIDPAR
#       PAR[k,1] <- PARF * exp( -0.5 * atten )
#     }
#   }
# 
#   PARF <- PARF * exp( -atten )
#   if( GUD_PARUICE ) {
#     PARF_ice <- PARF_ice * exp( -atten )
#   }
#   
# } # k loop
# }
# 
# #################################################
# ### ( gud_tempfunc.F )
# ### Compute temperature functions for gud model
# #   GUD_TEMP_VERSION 1 used in Follows et al (2007) - without max of 1
# #   GUD_TEMP_VERSION 2 used in Dutkiewicz et al (2009), Hickman et al, 
# #                              Monteiro et al, Barton et al
# #   GUD_TEMP_RANGE gives all phyto a specific temperature range
# #   if undefined, then full Eppley/Arrenhius curve used
# #
# # !USES: ===============================================================
# # IMPLICIT NONE
# #include "GUD_SIZE.h"
# #include "GUD_INDICES.h"
# #include "GUD_GENPARAMS.h"
# #include "GUD_TRAITS.h"
# #
# # !INPUT PARAMETERS: ===================================================
# # INTEGER myThid :: thread number
# # _RL Temp
# #
# # !OUTPUT PARAMETERS: ==================================================
# # grazFun :: the prey entries are for quota-style grazing,
# #            the predator ones for monod-style grazing
# # _RL photoFun(nplank)
# # _RL grazFun(nplank)
# # _RL reminFun
# # _RL mortFun
# # _RL mort2Fun
# # _RL uptakeFun
# #
# # !LOCAL VARIABLES: ====================================================
# # _RL Tkel
# # _RL TempAe, Tempref, TempCoeff
# # INTEGER j
# 
# gud_tempfunc <- function( GUD_NOTEMP = FALSE, GUD_TEMP_RANGE = FALSE, GUD_NOZOOTEMP = FALSE,
#                           GUD_TEMP_VERSION = 2, temp = 0 ) {
# 
#   photoFun  <- NULL
#   grazFun   <- NULL
#   reminFun  <- NULL
#   mortFun   <- NULL
#   mort2Fun  <- NULL
#   uptakeFun <- NULL
#   
#   if( GUD_NOTEMP ) {
#   
#     photoFun[1:nplank] <- 1.
#     grazFun [1:nplank] <- 1.
#     reminFun           <- 1.
#     mortFun            <- 1.
#     mort2Fun           <- 1.
#     uptakeFun          <- 1.
#   
#   } else if( GUD_TEMP_VERSION == 1 ) {
#   # +++++++++++++++++++ VERSION 1 +++++++++++++++++++++++++++++++++++++++
#   # Steph's version 1 (pseudo-Eppley)
#    
#     # plankton growth function (unitless)
#   
#     #swd -- this gives Eppley curve only
#     photoFun[1:nphoto] <- phytoTempExp1[1:nphoto] ** temp
#   
#     if( GUD_TEMP_RANGE ) {
#       #swd -- temperature range
#       photoFun[1:nphoto] <- photoFun[1:nphoto] *
#                             exp( -phytoTempExp2[1:nphoto] *
#                                   abs( temp - 
#                                        phytoTempOptimum[1:nphoto] ** phytoDecayPower[1:nphoto] 
#                                      ) 
#                                 )
#     }
#           
#     photoFun[1:nphoto] <-  photoFun[1:nphoto] - tempnorm
#     photoFun[1:nphoto] <-  pmax(  photoFun[1:nphoto], 1e-10 ) * phytoTempCoeff[1:nphoto]
#     photoFun[1:nphoto] <-  pmin(  photoFun[1:nphoto], 1 )
#   
#     #grazFun[1:nplank]  <- zooTempCoeff[1:nplank] *
#     #                      exp( zooTempExp[1:nplank] * 
#     #                      ( Temp - zooTempOptimum[1:nplank] ) )
#     grazFun[1:nplank]  <- 1.
#     reminFun           <- 1.
#     mortFun            <- 1.
#     mort2Fun           <- 1.
#     uptakeFun          <- 1.
#   # ++++++++++++++ END VERSION 1 +++++++++++++++++++++++++++++++++++++++
#   
#   } else if( GUD_TEMP_VERSION == 2 ) {
#   # +++++++++++++++++++ VERSION 2 +++++++++++++++++++++++++++++++++++++++
#   # Steph's version 2 (pseudo-Arrenhius)
#   
#     Tkel         <-   273.15
#     tempaearr    <- -4000.0
#     temprefarr   <-   293.15
#     tempcoeffarr <-     0.5882
#   
#     #swd -- this gives Arrenhius curve only
#     photoFun[1:nphoto] <- exp(  tempaearr *
#                               ( 1. / ( temp + Tkel ) - 1. / temprefarr ) 
#                              )
#       
#     if( GUD_TEMP_RANGE ) {
#       #swd -- temperature range
#       photoFun[1:nphoto] <- photoFun[1:nphoto] *
#                             exp( -phytoTempExp2[1:nphoto] *
#                                   abs( temp -
#                                        phytoTempOptimum[1:nphoto] ** phytoDecayPower[1:nphoto]
#                                      ) 
#                                 )
#     }
#   
#     photoFun[1:nphoto] <-  tempcoeffarr * pmax( photoFun[1:nphoto], 1e-10 )
#   
#     reminFun           <- exp(  tempaearr *
#                               ( 1. / ( temp + Tkel ) - 1. / temprefarr )
#                              )
#     reminFun           <-  tempcoeffarr * pmax( reminFun, 1e-10 )
#   
#     grazFun[1:nplank]  <- reminFun
#     mortFun            <- reminFun
#     mort2Fun           <- reminFun
#   
#     uptakeFun          <- 1.
#   # ++++++++++++++ END VERSION 2 ++++++++++++++++++++++++++++++++++++++++
#     
#   } else if( GUD_TEMP_VERSION == 3 ) {
#   # +++++++++++++++++++ VERSION 3 +++++++++++++++++++++++++++++++++++++++
#   # Ben's version 3 from quota model
#   
#     TempAe  <-   0.05
#     Tempref <-  20.0
#       
#     reminFun           <- pmax( 1e-10, exp( TempAe * ( temp - Tempref) ) )
#   
#     photoFun[1:nplank] <- reminFun
#     grazFun [1:nplank] <- reminFun
#   
#     mortFun            <- reminFun
#     mort2Fun           <- reminFun
#     uptakeFun          <- reminFun
#   # ++++++++++++++ END VERSION 3 +++++++++++++++++++++++++++++++++++++++
#   
#   } else {
#     print("GUD_TEMP_VERSION must be 1, 2 or 3. Define in GUD_OPTIONS.h")
#   }
#   
#   if( GUD_NOZOOTEMP ) {
#     grazFun[1:nplank] <- 1.
#   }
# 
#   # return list of temperature functions
#   temp_func <- list( photoFun  = photoFun,
#                      grazFun   = grazFun,
#                      reminFun  = reminFun,
#                      mortFun   = mortFun,
#                      mort2Fun  = mort2Fun,
#                      uptakeFun = uptakeFun )
#   return( temp_func )
# 
# } # END gud_tempfunc()
# 
# 
# #################################################
# ### Temperature function
# 
# Temp <- seq( -1, 28 )
# 
# photoFun  <- array( 1., dim = c( nplank, length(Temp) ) )
# grazFun   <- array( 1., dim = c( nplank, length(Temp) ) )
# reminFun  <- 1.
# mortFun   <- 1.
# mort2Fun  <- 1.
# uptakeFun <- 1.
# 
# for( i in 1:length(Temp) ) {
#   
#   out <- gud_tempfunc( GUD_NOTEMP, GUD_TEMP_RANGE, GUD_NOZOOTEMP, GUD_TEMP_VERSION, Temp[i] )
#   
#   photoFun [1:length(out$photoFun),i] <- out$photoFun
#   grazFun  [1:length(out$grazFun), i] <- out$grazFun
#   reminFun [i]                        <- out$reminFun
#   mortFun  [i]                        <- out$mortFun
#   mort2Fun [i]                        <- out$mort2Fun
#   uptakeFun[i]                        <- out$uptakeFun
# }
# 
# # plot
# if(0==1) {
# tempplot           <- t( photoFun )
# colnames(tempplot) <- paste( species, igroup )
# 
# plot_tempfunc <- melt( data          = as.data.frame( cbind( Temp, tempplot ) ), 
#                        id.vars       = "Temp", 
#                        variable.name = "Plankton.type", 
#                        value.name    = "Temperature.function" )
# 
# plot_species  <- rep( species, each = length(Temp) )
# 
# theme_set( theme_classic() )
# 
# print( ggplot( plot_tempfunc, aes( x = Temp, y = Temperature.function, group = Plankton.type ) ) + 
#   
#        geom_line( aes( colour = plot_species ), size = 2 ) +
#   
#        ylim( c( 0, 1 ) ) +
# 
#        labs( title = "Planktonic types response", x = "Temperature (°C)", y = "photoFun")
#      )
# }
# 
# 
# #################################################
# ### ( gud_grazing.F )
# 
# #include "GUD_OPTIONS.h"
# 
# #CBOP
# #C !ROUTINE: GUD_MODEL
# #C !INTERFACE: ==========================================================
# #  SUBROUTINE GUD_GRAZING(
# #    I     Ptr,
# #    U     gTr,
# #    U     diags,
# #    I     grazTempFunc, reminTempFunc, mortTempFunc, mort2TempFunc,
# #    I     myTime,myIter,myThid)
# 
# #C !DESCRIPTION:
# #  C     add quota-style grazing tendencies to gPtr
# 
# #C !USES: ===============================================================
# #  IMPLICIT NONE
# #include "GUD_SIZE.h"
# #include "GUD_INDICES.h"
# #include "GUD_DIAGS.h"
# #include "GUD_GENPARAMS.h"
# #include "GUD_TRAITS.h"
# 
# #C !INPUT PARAMETERS: ===================================================
# #  C  Ptr    :: gud model tracers
# #C  Temp   :: temperature field (degrees C)
# #C  myTime :: current time
# #C  myIter :: current iteration number
# #C  myThid :: thread number
# #_RL Ptr(nGud)
# #_RL grazTempFunc(nplank)
# #_RL mortTempFunc
# #_RL mort2TempFunc
# #_RL reminTempFunc
# #INTEGER myThid, myIter
# #_RL myTime
# 
# #C !INPUT/OUTPUT PARAMETERS: ============================================
# #  C  gTr    :: accumulates computed tendencies
# #C  diags  :: accumulates diagnostics
# #_RL gTr(nGud)
# #_RL diags(gud_nDiag)
# #CEOP
# 
# #ifdef ALLOW_GUD
# 
# #c !LOCAL VARIABLES: ====================================================
# #  INTEGER jz, jp
# 
# #_RL Qc  (nplank)
# #_RL Qn  (nplank)
# #_RL Qp  (nplank)
# #_RL Qsi (nplank)
# #_RL Qfe (nplank)
# #_RL QChl(nChl)
# #_RL X   (nplank)
# #_RL Xi  (nplank)
# 
# #_RL regQc, regQn, regQp, regQfe
# #_RL sumprey, sumpref, grazphy
# 
# #_RL preygraz   (nplank)
# #_RL predgrazc  (nplank)
# #ifdef GUD_ALLOW_NQUOTA
# #_RL predgrazn  (nplank)
# #endif
# #ifdef GUD_ALLOW_PQUOTA
# #_RL predgrazp  (nplank)
# #endif
# #ifdef GUD_ALLOW_FEQUOTA
# #_RL predgrazfe (nplank)
# #endif
# 
# #_RL totkillc, totkilln, totkillp, totkillsi, totkillfe
# #ifdef GUD_ALLOW_CARBON
# #_RL totkillPIC
# #endif
# #_RL totkillexpc, totkillexpn, totkillexpp, totkillexpfe
# #_RL predexpc, predexpn, predexpp, predexpfe
# #_RL graz2OC, graz2ON, graz2OP, graz2OFe
# #_RL graz2POC, graz2PON, graz2POP, graz2POSi, graz2POFe
# #_RL graz2PIC
# 
# #_RL tmp, expfrac
# 
# #_RL Xe
# #_RL mortX
# #_RL mortX2
# 
# #_RL exude_DOC
# #_RL exude_DON
# #_RL exude_DOP
# #_RL exude_DOFe
# 
# #_RL exude_PIC
# #_RL exude_POC
# #_RL exude_PON
# #_RL exude_POP
# #_RL exude_POSi
# #_RL exude_POFe
# 
# #_RL mort_c(nplank)
# 
# #_RL respir
# #_RL respir_c
# 
# #ifdef GUD_ALLOW_CDOM
# #_RL graz2CDOM, exude_CDOM
# #endif
# 
# #_RL EPS
# #PARAMETER (EPS=1D-38)
# 
# #==== make all bio fields non-negative and compute quotas ==============
# 
# # fixed carbon quota, for now 1.0 (may change later)
# Qc <- 1
# X  <- pmax( 0, Ptr[ic:ec] ) / Qc
# 
# # other elements: get quota from corresponding ptracer or set to fixed
# # ratio if not variable.
# Xi <- 1 / pmax( EPS, X )
# 
# if( GUD_ALLOW_NQUOTA ) {
#   Qn <- pmax( 0, Ptr[iN:en] ) * Xi
# } else {
#   Qn <- R_NC
# }
# 
# if( GUD_ALLOW_PQUOTA ) {
#   Qp <- pmax( 0, Ptr[ip:ep] ) * Xi
# } else {
#   Qp <- R_PC
# }
# 
# if( GUD_ALLOW_SIQUOTA ) {
#   Qsi <- pmax( 0, Ptr[isi:esi] ) * Xi
# } else {
#   Qsi <- R_SiC
# }
# 
# if( GUD_ALLOW_FEQUOTA ) {
#   Qfe <- pmax( 0, Ptr[ife:efe] ) * Xi
# } else {
#   Qfe <- R_FeC
# }
# 
# if( GUD_ALLOW_CHLQUOTA ) {
#   QChl <- pmax( 0, Ptr[ichl:echl] ) * Xi[1:nChl]
# }
# 
# preygraz   <- 0
# predgrazc  <- 0
# 
# if( GUD_ALLOW_NQUOTA ) {
#   predgrazn  <- 0
# }
# if( GUD_ALLOW_PQUOTA ) {
#   predgrazp  <- 0
# }
# if( GUD_ALLOW_FEQUOTA ) {
#   predgrazfe <- 0
# }
# 
# graz2POC  <- 0
# graz2PON  <- 0
# graz2POP  <- 0
# graz2POSI <- 0
# graz2POFE <- 0
# graz2OC   <- 0
# graz2ON   <- 0
# graz2OP   <- 0
# graz2OFE  <- 0
# graz2PIC  <- 0
# 
# regQn  <- 1.0
# regQp  <- 1.0
# regQfe <- 1.0
# regQc  <- 1.0
# 
# #=======================================================================
# for( jz in iMinPred:iMaxPred ) {
#   
#   # regulate grazing near full quota
#   regQc <- 1
#   if( GUD_ALLOW_NQUOTA ) {
#     regQn <- pmax( 0, pmin( 1, ( Qnmax[jz] - Qn[jz] ) / ( Qnmax[jz] - Qnmin[jz] ) ) )
#     regQc <- pmin( regQc, 1 - regQn )
#     regQn <- regQn ** hillnum
#   }
#   if( GUD_ALLOW_PQUOTA ) {
#     regQp <- pmax( 0, pmin( 1, ( Qpmax[jz] - Qp[jz] ) / ( Qpmax[jz] - Qpmin[jz] ) ) )
#     regQc <- pmin( regQc, 1 - regQp )
#     regQp <- regQp ** hillnum
#   }
#   if( GUD_ALLOW_FEQUOTA ) {
#     regQfe <- pmax( 0, pmin( 1, ( Qfemax[jz] - Qfe[jz] ) / ( Qfemax[jz] - Qfemin[jz] ) ) )
#     regQc  <- pmin( regQc, 1 - regQfe )
#     regQfe <- regQfe ** hillnum
#   }
#   regQc <- regQc ** hillnum
#   
#   sumprey <- 0.0
#   sumpref <- 0.0
#   
#   for( jp in iMinPrey:iMaxPrey ) {
#     sumprey <- sumprey + palat[jp,jz] * X[jp]
#     if( GUD_GRAZING_SWITCH ) {
#       sumpref <- sumpref + palat[jp,jz] * palat[jp,jz] * X[jp] * X[jp]
#     } else {
#       sumpref <- sumpref + palat[jp,jz] * X[jp]
#     }
#   }
#   
#   sumprey <- pmax( 0, sumprey - phygrazmin )
#   sumpref <- pmax( phygrazmin, sumpref )
#   tmp     <-  grazemax[jz] * grazTempFunc[jz] * X[jz] *
#     ( sumprey ** hollexp / ( sumprey ** hollexp + kgrazesat[jz] ** hollexp ) ) *
#     ( 1 - exp( -inhib_graz * sumprey ) ) ** inhib_graz_exp
#   
#   totkillc  <- 0
#   totkilln  <- 0
#   totkillp  <- 0
#   totkillsi <- 0
#   totkillfe <- 0
#   if( GUD_ALLOW_CARBON ) {
#     totkillPIC <- 0
#   }
#   totkillexpc  <- 0
#   totkillexpn  <- 0
#   totkillexpp  <- 0
#   totkillexpfe <- 0
#   
#   predexpc  <- 0
#   predexpn  <- 0
#   predexpp  <- 0
#   predexpfe <- 0
#   
#   for( jp in iMinPrey:iMaxPrey ) {
#     if( GUD_GRAZING_SWITCH ) {
#       grazphy <- tmp * palat[jp,jz] * palat[jp,jz] * X[jp] * X[jp] / sumpref
#     } else {
#       grazphy <- tmp * palat[jp,jz] * X[jp] / sumpref
#     }
#     
#     preygraz[jp] <- preygraz[jp] + grazphy
#     
#     totkillc  <- totkillc  + grazphy
#     totkilln  <- totkilln  + grazphy * Qn [jp]
#     totkillp  <- totkillp  + grazphy * Qp [jp]
#     totkillsi <- totkillsi + grazphy * Qsi[jp]
#     totkillfe <- totkillfe + grazphy * Qfe[jp]
#     if( GUD_ALLOW_CARBON ) {
#       totkillPIC <- totkillPIC + grazphy * R_PICPOC[jp]
#     }
#     
#     expFrac <- ExportFracPreyPred[jp,jz]
#     
#     totkillexpc  <- totkillexpc  + expFrac * grazphy
#     totkillexpn  <- totkillexpn  + expFrac * grazphy * Qn [jp]
#     totkillexpp  <- totkillexpp  + expFrac * grazphy * Qp [jp]
#     totkillexpfe <- totkillexpfe + expFrac * grazphy * Qfe[jp]
#     
#     predgrazc[jz] <- predgrazc[jz]      + grazphy * asseff[jp,jz] * regQc
#     predexpc      <- predexpc + expFrac * grazphy * asseff[jp,jz] * regQc
#     if( GUD_ALLOW_NQUOTA ) {
#       predgrazn[jz] <- predgrazn[jz]       + grazphy * asseff[jp,jz] * regQn * Qn[jp]
#       predexpn      <- predexpn + expFrac  * grazphy * asseff[jp,jz] * regQn * Qn[jp]
#     }
#     if( GUD_ALLOW_PQUOTA ) {
#       predgrazp[jz] <- predgrazp[jz]      + grazphy * asseff[jp,jz] * regQp * Qp[jp]
#       predexpp      <- predexpp + expFrac * grazphy * asseff[jp,jz] * regQp * Qp[jp]
#     }
#     if( GUD_ALLOW_FEQUOTA ) {
#       predgrazfe[jz] <- predgrazfe[jz]      + grazphy * asseff[jp,jz] * regQfe * Qfe[jp]
#       predexpfe      <- predexpfe + expFrac * grazphy * asseff[jp,jz] * regQfe * Qfe[jp]
#     }
#   }
#   
#   graz2OC   <- graz2OC   + totkillc    - predgrazc[jz]
#   graz2POC  <- graz2POC  + totkillexpc - predexpc
#   
#   if( GUD_ALLOW_NQUOTA ) {
#     graz2ON   <- graz2ON   + totkilln    - predgrazn[jz]
#     graz2PON  <- graz2PON  + totkillexpn - predexpn
#   } else {
#     graz2ON   <- graz2ON   + totkilln    - predgrazc[jz] * Qn[jz]
#     graz2PON  <- graz2PON  + totkillexpn - predexpc      * Qn[jz]
#   }
#   if( GUD_ALLOW_PQUOTA ) {
#     graz2OP   <- graz2OP   + totkillp    - predgrazp[jz]
#     graz2POP  <- graz2POP  + totkillexpp - predexpp
#   } else {
#     graz2OP   <- graz2OP   + totkillp    - predgrazc[jz] * Qp[jz]
#     graz2POP  <- graz2POP  + totkillexpp - predexpc      * Qp[jz]
#   }
#   if( GUD_ALLOW_FEQUOTA ) {
#     graz2OFe  <- graz2OFe   + totkillfe    - predgrazfe[jz]
#     graz2POFe <- graz2POFe  + totkillexpfe - predexpfe
#   } else {
#     graz2OFe  <- graz2OFe   + totkillfe    - predgrazc[jz] * Qfe[jz]
#     graz2POFe <- graz2POFe  + totkillexpfe - predexpc      * Qfe[jz]
#   }
#   
#   graz2POSi <- graz2POSi + totkillsi
#   
#   if( GUD_ALLOW_CARBON ) {
#     graz2PIC <- graz2PIC + totkillPIC
#   }
#   
#   # end predator loop
# }
# 
# #==== tendencies =======================================================
# 
# gTr[iDOC ] <- gTr[iDOC ] + graz2OC  - graz2POC
# gTr[iDON ] <- gTr[iDON ] + graz2ON  - graz2PON
# gTr[iDOP ] <- gTr[iDOP ] + graz2OP  - graz2POP
# gTr[iDOFe] <- gTr[iDOFe] + graz2OFe - graz2POFe
# gTr[iPOC ] <- gTr[iPOC ] + graz2POC
# gTr[iPON ] <- gTr[iPON ] + graz2PON
# gTr[iPOP ] <- gTr[iPOP ] + graz2POP
# gTr[iPOSi] <- gTr[iPOSi] + graz2POSi
# gTr[iPOFe] <- gTr[iPOFe] + graz2POFe
# if( GUD_ALLOW_CARBON ) {
#   gTr[iPIC ] <- gTr[iPIC ] + graz2PIC
# }
# if( GUD_ALLOW_CDOM ) {
#   graz2CDOM  <- fracCDOM * ( graz2OP - graz2POP )
#   gTr[iCDOM] <- gTr[iCDOM] + graz2CDOM
#   gTr[iDOC ] <- gTr[iDOC ]             - R_CP_CDOM  * graz2CDOM
#   gTr[iDON ] <- gTr[iDON ]             - R_NP_CDOM  * graz2CDOM
#   gTr[iDOP ] <- gTr[iDOP ] - graz2CDOM
#   gTr[iDOFe] <- gTr[iDOFe]             - R_FeP_CDOM * graz2CDOM
# }
# for( jp in iMinPrey:iMaxPrey ) {
#   gTr[ic+jp-1] <- gTr[ic+jp-1] - preygraz[jp]
# }
# for( jz in iMinPred:iMaxPred ) {
#   gTr[ic+jz-1] <- gTr[ic+jz-1] + predgrazc[jz]
# }
# if( GUD_ALLOW_NQUOTA ) {
#   gTr[iN:en] <- gTr[iN:en] + predgrazn - preygraz * Qn
# }
# if( GUD_ALLOW_PQUOTA ) {
#   gTr[ip:ep] <- gTr[ip:ep] + predgrazp - preygraz * Qp
# }
# if( GUD_ALLOW_SIQUOTA ) {
#   gTr[isi:esi] <- gTr[isi:esi] - preygraz * Qsi
# }
# if( GUD_ALLOW_FEQUOTA ) {
#   gTr[ife:efe] <- gTr[ife:efe] + predgrazfe - preygraz * Qfe
# }
# if( GUD_ALLOW_CHLQUOTA ) {
#   gTr[iChl:eChl] <- gTr[iChl:eChl] - preygraz[1:nChl] * QChl
# }
# 
# for( jp in 1:nGRplank ) {
#   diags[iGRplank+jp-1] <- preygraz[jp]
# }
# 
# #==== mortality ========================================================
# exude_DOC  <- 0
# exude_POC  <- 0
# exude_DON  <- 0
# exude_PON  <- 0
# exude_DOFe <- 0
# exude_POFe <- 0
# exude_DOP  <- 0
# exude_POP  <- 0
# exude_POSi <- 0
# exude_PIC  <- 0
# 
# respir     <- 0
# 
# for( jp in 1:nplank ) {
#   
#   Xe <- pmax( 0, X[jp] - Xmin[jp] )
#   mortX  <- mort[jp]  * Xe      * pmax( mortTempFuncMin[jp],  mortTempFunc  )
#   mortX2 <- mort2[jp] * Xe * Xe * pmax( mort2TempFuncMin[jp], mort2TempFunc )
#   
#   mort_c[jp] <- mortX + mortX2
#   
#   exude_DOC <- exude_DOC + ( 1 - ExportFracMort[jp]  ) * mortX +
#     ( 1 - ExportFracMort2[jp] ) * mortX2
#   exude_POC <- exude_POC +       ExportFracMort[jp]    * mortX +
#     ExportFracMort2[jp]   * mortX2
#   
#   exude_DON <- exude_DON + ( 1 - ExportFracMort[jp]  ) * mortX  * Qn[jp] +
#     ( 1 - ExportFracMort2[jp] ) * mortX2 * Qn[jp]
#   exude_PON <- exude_PON +       ExportFracMort[jp]    * mortX  * Qn[jp] +
#     ExportFracMort2[jp]   * mortX2 * Qn[jp]
#   
#   exude_DOP <- exude_DOP + ( 1 - ExportFracMort[jp]  ) * mortX  * Qp[jp] +
#     ( 1 - ExportFracMort2[jp] ) * mortX2 * Qp[jp]
#   exude_POP <- exude_POP +       ExportFracMort[jp]    * mortX  * Qp[jp] +
#     ExportFracMort2[jp]   * mortX2 * Qp[jp]
#   
#   exude_DOFe<- exude_DOFe+ ( 1 - ExportFracMort[jp]  ) * mortX  * Qfe[jp] +
#     ( 1 - ExportFracMort2[jp] ) * mortX2 * Qfe[jp]
#   exude_POFe<- exude_POFe+       ExportFracMort[jp]    * mortX  * Qfe[jp] +
#     ExportFracMort2[jp]   * mortX2 * Qfe[jp]
#   
#   exude_POSi<- exude_POSi + mort_c[jp] * Qsi[jp]
#   
#   exude_PIC <- exude_PIC  + mort_c[jp] * R_PICPOC[jp]
#   
#   respir_c  <- respiration[jp] * Xe * reminTempFunc
#   respir    <- respir + respir_c
#   
#   gTr[ic+jp-1]    <- gTr[ic+jp-1]  - mort_c[jp] - respir_c
#   if( GUD_ALLOW_NQUOTA ) {
#     gTr[iN+jp-1]  <- gTr[iN+jp-1]  - mort_c[jp] * Qn[jp]
#   }
#   if( GUD_ALLOW_PQUOTA ) {
#     gTr[ip+jp-1]  <- gTr[ip+jp-1]  - mort_c[jp] * Qp[jp]
#   }
#   if( GUD_ALLOW_SIQUOTA ) {
#     gTr[isi+jp-1] <- gTr[isi+jp-1] - mort_c[jp] * Qsi[jp]
#   }
#   if( GUD_ALLOW_FEQUOTA ) {
#     gTr[ife+jp-1] <- gTr[ife+jp-1] - mort_c[jp] * Qfe[jp]
#   }
#   
#   if( GUD_ALLOW_EXUDE ) {
#     exude_DOC  <- exude_DOC  + ( 1 - ExportFrac[jp] ) * kexcC[jp]  * Xe
#     exude_POC  <- exude_POC  +       ExportFrac[jp]   * kexcC[jp]  * Xe
#     exude_DON  <- exude_DON  + ( 1 - ExportFrac[jp] ) * kexcN[jp]  * Xe * Qn[jp]
#     exude_PON  <- exude_PON  +       ExportFrac[jp]   * kexcN[jp]  * Xe * Qn[jp]
#     exude_DOP  <- exude_DOP  + ( 1 - ExportFrac[jp] ) * kexcP[jp]  * Xe * Qp[jp]
#     exude_POP  <- exude_POP  +       ExportFrac[jp]   * kexcP[jp]  * Xe * Qp[jp]
#     exude_DOFe <- exude_DOFe + ( 1 - ExportFrac[jp] ) * kexcFe[jp] * Xe * Qfe[jp]
#     exude_POFe <- exude_POFe +       ExportFrac[jp]   * kexcFe[jp] * Xe * Qfe[jp]
#     exude_POSi <- exude_POSi +                          kexcSi[jp] * Xe * Qsi[jp]
#     
#     gTr[ic+jp-1]   <- gTr[ic+jp-1]   - kexcC[jp]  * Xe
#     if( GUD_ALLOW_NQUOTA ) {
#       gTr[iN+jp-1] <- gTr[iN+jp-1]   - kexcN[jp]  * Xe * Qn[jp]
#     }
#     if( GUD_ALLOW_PQUOTA ) {
#       gTr[ip+jp-1] <- gTr[ip+jp-1]   - kexcP[jp]  * Xe * Qp[jp]
#     }
#     if( GUD_ALLOW_SIQUOTA ) {
#       gTr[isi+jp-1] <- gTr[isi+jp-1] - kexcSi[jp] * Xe * Qsi[jp]
#     }
#     if( GUD_ALLOW_FEQUOTA ) {
#       gTr[ife+jp-1] <- gTr[ife+jp-1] - kexcFe[jp] * Xe * Qfe[jp]
#     }
#   }
# }
# 
# if( GUD_ALLOW_CHLQUOTA ) {
#   for( jp in 1:nChl ) {
#     gTr[iChl+jp-1] <- gTr[iChl+jp-1] - mort_c[jp] * QChl[jp]
#   }
# }
# 
# gTr[iDIC ] <- gTr[iDIC ] + respir
# gTr[iDOC ] <- gTr[iDOC ] + exude_DOC
# gTr[iDON ] <- gTr[iDON ] + exude_DON
# gTr[iDOP ] <- gTr[iDOP ] + exude_DOP
# gTr[iDOFe] <- gTr[iDOFe] + exude_DOFe
# 
# gTr[iPIC ] <- gTr[iPIC ] + exude_PIC
# gTr[iPOC ] <- gTr[iPOC ] + exude_POC
# gTr[iPON ] <- gTr[iPON ] + exude_PON
# gTr[iPOP ] <- gTr[iPOP ] + exude_POP
# gTr[iPOSi] <- gTr[iPOSi] + exude_POSi
# gTr[iPOFe] <- gTr[iPOFe] + exude_POFe
# 
# if( GUD_ALLOW_CDOM ) {
#   exude_CDOM <-  fracCDOM  * exude_DOP
#   gTr[iCDOM] <- gTr[iCDOM] + exude_CDOM
#   gTr[iDOC ] <- gTr[iDOC ]              - R_CP_CDOM  * exude_CDOM
#   gTr[iDON ] <- gTr[iDON ]              - R_NP_CDOM  * exude_CDOM
#   gTr[iDOP ] <- gTr[iDOP ] - exude_CDOM
#   gTr[iDOFe] <- gTr[iDOFe]              - R_FeP_CDOM * exude_CDOM
# }
# 
# #RETURN
# #END SUBROUTINE
# 
# 
# #################################################
# ### ( gud_model.F ) main routine
# 
# #==== phytoplankton ====================================================
#   
# # fixed carbon quota, for now 1.0 (may change later)
# # other elements: get quota from corresponding ptracer or set to fixed ratio if not variable.
# 
# #X  <-  pmax( 0., Ptr[ic+j-1] ) !!! Ptr is biomass of plankton type
# X  <- isPhoto * 0 + 1 # !!! FOR TESTING ONLY
# Qc <- 1.
# 
# #==== uptake and nutrient limitation ===================================
# # for quota elements, growth is limiteed by available quota,
# # for non-quota elements, ...            by available nutrients in medium.
# 
# # to not use PO4, ..., set ksatPO4=0 and Vmax_PO4=0 (if GUD_ALLOW_PQUOTA)
# # or R_PC=0 (if not)
# # the result will be limitp = 1, uptakePO4 = 0
# 
# uptakeTempFunc <- uptakeFun[2] # !!! FOR TESTING ONLY; uptakeFun at 0 deg C
# EPS <- 1e-12 # !!! FOR TESTING ONLY
# 
# #--- phosphorus
# PO4 <- 1e-6 # !!! FOR TESTING ONLY
# 
# limitp <- PO4 / ( PO4 + ksatPO4[1:nphoto] )
# 
# if( GUD_ALLOW_PQUOTA ) {
#   Qp        <- pmax( EPS * R_PC[1:nphoto], Ptr[ip+j-1] ) / pmax(EPS, X) # !!! DEFINE [ip+j-1] / DEFINE EPS
#   regQ      <- pmax( 0., 
#                      pmin( 1., ( Qpmax[1:nphoto] - Qp ) / ( Qpmax[1:nphoto] - Qpmin[1:nphoto] ) )
#                    )
# 
#   uptakePO4 <- Vmax_PO4[1:nphoto] * limitp * regQ * uptakeTempFunc * X
# 
#   # normalized Droop limitation
#   limitp   <- pmax( 0., 
#                     pmin( 1., ( 1. - Qpmin[1:nphoto] / Qp ) / ( 1. - Qpmin[1:nphoto] / Qpmax[1:nphoto] ) )
#                   )
# }
# 
# #--- silica
# SiO2 <- 1e-6 # !!! FOR TESTING ONLY
# 
# limitsi <- SiO2 / ( SiO2 + ksatSiO2[1:nphoto] )
# 
# if( GUD_ALLOW_SIQUOTA ) {
#   Qsi        <- pmax( EPS * R_SiC[1:nphoto], Ptr[isi+j-1] ) / pmax(EPS, X) # !!! DEFINE [isi+j-1] / DEFINE EPS
#   regQ       <- pmax( 0., 
#                       pmin( 1., ( Qsimax[1:nphoto] - Qsi ) / ( Qsimax[1:nphoto] - Qsimin[1:nphoto] ) )
#                     )
#   uptakeSiO2 <- Vmax_SiO2[1:nphoto] * limitsi * regQ * uptakeTempFunc * X
# 
#   # linear limitation
#   limitsi    <- pmax( 0., 
#                       pmin( 1., ( Qsi - Qsimin[1:nphoto] ) / ( Qsimax[1:nphoto] - Qsimin[1:nphoto] ) )
#                     )
# }
# 
# #--- iron
# FeT <- 1e-6 # !!! FOR TESTING ONLY
# 
# limitfe <-  FeT / (FeT + ksatFeT[1:nphoto] )
# 
# if( GUD_ALLOW_FEQUOTA ) {
#   Qfe       <- pmax( EPS * R_FeC[1:nphoto], Ptr[ife+j-1] ) / pmax( EPS, X ) # !!! DEFINE [ife+j-1] / DEFINE EPS
#   regQ      <- pmax( 0., 
#                      pmin( 1., ( Qfemax[1:nphoto] - Qfe ) / ( Qfemax[1:nphoto] - Qfemin[1:nphoto] ) ) 
#                    )
#   uptakeFeT <- Vmax_FeT[1:nphoto] * limitfe * regQ * uptakeTempFunc * X
# 
#   # normalized Droop limitation
#   limitfe   <- pmax( 0., 
#                      pmin( 1., ( 1. - Qfemin[1:nphoto] / Qfe ) / ( 1. - Qfemin[1:nphoto] / Qfemax[1:nphoto] ) )
#                    )
# }
# 
# #--- nitrogen
# NH4 <- NO2 <- NO3 <- 1e-6 # !!! FOR TESTING ONLY
# 
# if( GUD_ALLOW_NQUOTA ) {
#   # have nitrogen quota
#   inhibNH4  <- exp( -amminhib[1:nphoto] * NH4 )
#   limitNH4  <- NH4 / ( NH4 + ksatNH4[1:nphoto] )
#   limitNO2  <- NO2 / ( NO2 + ksatNO2[1:nphoto] ) * inhibNH4
#   limitNO3  <- NO3 / ( NO3 + ksatNO3[1:nphoto] ) * inhibNH4
# 
#   Qn        <- pmax( EPS * R_NC[1:nphoto], Ptr[iN+j-1] ) / pmax( EPS, X )
#   regQ      <- pmax( 0., 
#                      pmin( 1., ( Qnmax[1:nphoto] - Qn ) / ( Qnmax[1:nphoto] - Qnmin[1:nphoto] ) )
#                    )
# 
#   uptakeNH4 <- Vmax_NH4[1:nphoto] * limitNH4 * regQ * uptakeTempFunc * X
#   uptakeNO2 <- Vmax_NO2[1:nphoto] * limitNO2 * regQ * uptakeTempFunc * X
#   uptakeNO3 <- Vmax_NO3[1:nphoto] * limitNO3 * regQ * uptakeTempFunc * X
# 
#   if( GUD_ALLOW_FEQUOTA ) {
#     uptakeNO3 <- uptakeNO3 * limitfe
#   }
# 
#   uptakeN   <- pmax( uptakeNH4 + uptakeNO2 + uptakeNO3,
#                      Vmax_N[1:nphoto] * regQ * uptakeTempFunc * X * diazo[1:nphoto] 
#                    )
# 
#   # linear limitation
#   limitn    <- pmax( 0.,
#                      pmin( 1., ( Qn - Qnmin[1:nphoto] ) / ( Qnmax[1:nphoto] - Qnmin[1:nphoto] ) )
#                    )
# }else{ # /* not GUD_ALLOW_NQUOTA */
# 
#   Qn       <-  R_NC[1:nphoto]
#   inhibNH4 <-  exp( -amminhib[1:nphoto] * NH4 )
#   limitNH4 <-  useNH4[1:nphoto] * NH4 / ( NH4 + ksatNH4[1:nphoto] )
#   limitNO2 <-  useNO2[1:nphoto] * NO2 /
#              ( NO2 + combNO[1:nphoto] * ( NO3 + ksatNO3[1:nphoto] - ksatNO2[1:nphoto] ) + ksatNO2[1:nphoto] ) * 
#                inhibNH4
#   limitNO3 <-  useNO3[1:nphoto] * NO3 / ( combNO[1:nphoto] * NO2 + NO3 + ksatNO3[1:nphoto] ) * inhibNH4
#   limitn   <-  limitNH4 + limitNO2 + limitNO3
# 
#   # normalize to sum (approx) 1
#   fracNH4  <-  limitNH4 / ( limitn + EPS )
#   fracNO2  <-  limitNO2 / ( limitn + EPS )
#   fracNO3  <-  limitNO3 / ( limitn + EPS )
# 
#   # if diazo, all fracN* == 0 but want no N limitation
#   limitn   <- pmin( 1., limitn + diazo[1:nphoto] )
# } # /* GUD_ALLOW_NQUOTA */
# 
# #if( limitn > 0 ) {
# #  ngrow <- ( ( 10 * 4 + 2 ) / ( 10 * 4 + 2 * limitNH4 / limitn +
# #                                         8 * limitNO2 / limitn + 
# #                                        10 * limitNO3 / limitn )
# #           )
# #} else {
#   ngrow <- 1
# #}
# 
# limitnut <- pmin( limitn, limitp, limitsi )
# 
# if( !GUD_ALLOW_FEQUOTA ) {
#   limitnut <- pmin( limitnut, limitfe )
# }
# 
# limitpCO2 <- 1
# 
# #==== growth ===========================================================
# 
# if( GUD_ALLOW_GEIDER ) {
#   
#   alpha_I <- 0
#   for( l in 1:nlam ) {
#     alpha_I <- alpha_I + alphachl[1:nPhoto,l] * PAR[l]
#   }
#   # NB: for quota, PCmax(j) = Vmax_c(j)
#   PCm <- PCmax[1:nPhoto] * limitnut * photoTempFunc[1:nPhoto] * limitpCO2
# 
#   if( PCm > 0 ) {
#     acclim <- pmax( chl2cmin[1:nPhoto], 
#                     pmin( chl2cmax[1:nPhoto],
#                           chl2cmax[1:nPhoto] / ( 1 + chl2cmax[1:nPhoto] * alpha_I / ( 2 * PCm ) ) 
#                         )
#                   )
#   } else {
#     acclim <- chl2cmin[1:nPhoto]
#   }
# 
#   if( GUD_ALLOW_CHLQUOTA ) {
#     QChl  <- pmax( EPS * R_ChlC[1:nPhoto], Ptr(ichl+j-1) ) / pmax( EPS, X )
# #   quotas are already relative to carbon
#     chl2c <- QChl
#   } else {
#     chl2c <- acclim
#   }
# 
#   alpha_I_growth <- alpha_I
# # a la quota
#   if( GUD_ALLOW_FEQUOTA ) {
#     alpha_I_growth <- alpha_I_growth * limitfe
#   }
# 
# # carbon-specific growth rate
# # PC = PCm*(1-EXP(-alpha_I_growth*chl2c/MAX(EPS, PCm)))
#   if( PCm > 0 && PARtot > PARmin ) {
#     PC <- PCm * ( 1 - exp( -alpha_I_growth * chl2c / PCm ) )
#   } else {
#     PC <- 0
#   }
# 
#   if( inhibcoef_geid[1:nPhoto] > 0 ) {
# #   "total" PAR:
#     tmp     <- alpha_I /           alpha_mean[1:nPhoto]
#     Ek      <- PCm     / ( chl2c * alpha_mean[1:nPhoto] )
#     EkoverE <- Ek / tmp
#     if( tmp > Ek ) {
#       PC <- PC * EkoverE * inhibcoef_geid[1:nPhoto]
#     }
#   }
# 
# } else { #/* not GUD_ALLOW_GEIDER */
# 
#   if( PARtot > PARmin ) {
# #   only 1 waveband without GUD_ALLOW_GEIDER
#     limitI <- ( 1 - exp( -PARtot * ksatPAR[1:nPhoto] ) ) *
#                     exp( -PARtot * kinhPAR[1:nPhoto] )   * normI[1:nPhoto]
#     PC <- PCmax[1:nPhoto] * limitnut * limitI * photoTempFunc[1:nPhoto] * limitpCO2
#   } else {
#     PC <- 0
#   }
# 
# } #/* GUD_ALLOW_GEIDER */
# 
# if( GUD_PARUICE ) {
#   if( GUD_ALLOW_GEIDER ) {
# 
#     alpha_I_ice <- 0
#     for( l in 1:nlam ) {
#       alpha_I_ice <- alpha_I_ice + alphachl[1:nPhoto,l] * PAR_ice[l]
#     }
# #   NB: for quota, PCmax[1:nPhoto] = Vmax_c[1:nPhoto]
#     PCm <- PCmax[1:nPhoto] * limitnut * photoTempFunc[1:nPhoto] * limitpCO2
# 
#     if( PCm > 0 ) {
#       acclim_ice <- pmax( chl2cmin[1:nPhoto], 
#                           pmin( chl2cmax[1:nPhoto],
#                                 chl2cmax[1:nPhoto] / ( 1 + chl2cmax[1:nPhoto] * alpha_I_ice / ( 2 * PCm ) )
#                               )
#                         )
#     } else {
#       acclim_ice <- chl2cmin[1:nPhoto]
#     }
# 
#     if( GUD_ALLOW_CHLQUOTA ) {
#       QChl <- pmax( EPS * R_ChlC[1:nPhoto], Ptr(ichl+j-1) ) / pmax( EPS, X )
# #     quotas are already relative to carbon
#       chl2c_ice <- QChl
#     } else {
#      chl2c_ice <- acclim_ice
#     }
# 
#     alpha_I_growth <- alpha_I_ice
# #   a la quota
#     if( GUD_ALLOW_FEQUOTA ) {
#       alpha_I_growth <- alpha_I_growth * limitfe
#     }
# 
# #       carbon-specific growth rate
# #       PC = PCm*(1-EXP(-alpha_I_growth*chl2c/MAX(EPS, PCm)))
#     if( PCm > 0 && PARtot_ice > PARmin ) {
#       PC_ice <- PCm * ( 1 - exp( -alpha_I_growth * chl2c_ice / PCm ) )
#     } else {
#       PC_ice <- 0
#     }
# 
#     if( inhibcoef_geid[1:nPhoto] > 0 ) {
# #     "total" PAR_ice:
#       tmp     <- alpha_I_ice / alpha_mean[1:nPhoto]
#       Ek      <- PCm / ( chl2c_ice * alpha_mean[1:nPhoto] )
#       EkoverE <- Ek / tmp
#       if( tmp > Ek ) {
#         PC_ice <- PC_ice * EkoverE * inhibcoef_geid[1:nPhoto]
#       }
#     }
# 
#   } else { #/* not GUD_ALLOW_GEIDER */
# 
#       if( PARtot_ice > PARmin ) {
# #       only 1 waveband without GUD_ALLOW_GEIDER
#         limitI <- ( 1 - exp( -PARtot_ice * ksatPAR[1:nPhoto] ) ) *
#                         exp( -PARtot_ice * kinhPAR[1:nPhoto] )   * normI[1:nPhoto]
#         PC_ice <- PCmax[1:nPhoto] * limitnut * limitI * photoTempFunc[1:nPhoto] * limitpCO2
#       } else {
#         PC_ice <- 0
#       }
# 
#   } #/* GUD_ALLOW_GEIDER */
# 
#   PC      <- PC      * ( 1 - iceFrac ) + PC_ice      * iceFrac
#   alpha_i <- alpha_i * ( 1 - iceFrac ) + alpha_i_ice * iceFrac
#   chl2c   <- chl2c   * ( 1 - iceFrac ) + chl2c_ice   * iceFrac
#   acclim  <- acclim  * (1 - iceFrac ) + acclim_ice   * iceFrac 
# 
# } #/* GUD_PARUICE */
# 
# growth    <- PC * ngrow * X
# 
# uptakeDIC <- growth
# 
# # non-quota elements are taken up with fixed stoichiometry
# if( !GUD_ALLOW_NQUOTA ) {
#   uptakeN    <- growth  * R_NC[1:nPhoto]
#   uptakeNH4  <- uptakeN * fracNH4
#   uptakeNO2  <- uptakeN * fracNO2
#   uptakeNO3  <- uptakeN * fracNO3
# }
# if( !GUD_ALLOW_PQUOTA ) {
#   uptakePO4  <- growth * R_PC[1:nPhoto]
# }
# if( !GUD_ALLOW_SIQUOTA ) {
#   uptakeSiO2 <- growth * R_SiC[1:nPhoto]
# }
# if( !GUD_ALLOW_FEQUOTA ) {
#   uptakeFeT <- growth * R_FeC[1:nPhoto]
# }
# 
# #==== chlorophyll ======================================================
# if( GUD_ALLOW_GEIDER ) {
#   if( GUD_ALLOW_CHLQUOTA ) {
#     if( GUD_ALLOW_NQUOTA ) {
# #       Geider 1998
#       if( alpha_I * chl2c > 0 ) {
# #       rhochl = Chl2Nmax/(alpha_I*chl2c)*ngrow ???
#         rhochl <- Chl2Nmax * PC * ngrow / ( alpha_I * chl2c )
#       } else {
#         rhochl <- Chl2Nmax
#       }
#       uptakeDIC <-  uptakeDIC - synthcost * uptakeN
#       synthChl  <- rhochl * uptakeN
#     } else {
#       if( GUD_GEIDER_RHO_SYNTH ) {
#         if( alpha_I > 0 && acclim > 0 ) {
#           rhochl <- Chl2Cmax[1:nPhoto] * PC * ngrow / ( alpha_I * acclim )
#         } else {
#           rhochl <- 0 # should be Chl2Cmax(j) ?????
#         }
#         synthChl <- rhochl * growth + acclimtimescl[1:nPhoto] * ( acclim - chl2c ) * X
#       } else {
#         synthChl <- acclim * growth + acclimtimescl[1:nPhoto] * ( acclim - chl2c ) * X
#       }
#     } #/* GUD_ALLOW_NQUOTA */
#   } else { #/* GUD_ALLOW_CHLQUOTA */
#       chlout[1:nPhoto] <- X * Qc * chl2c
#       synthChl <- 0
#   } #/* GUD_ALLOW_CHLQUOTA */
# } else {
#   synthChl <- 0
# } #/* GUD_ALLOW_GEIDER */
# #=======================================================================
