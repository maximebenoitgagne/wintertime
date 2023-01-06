#ifndef GUD_OPTIONS_H
#define GUD_OPTIONS_H
#include "PACKAGES_CONFIG.h"
#ifdef ALLOW_GUD

#include "CPP_OPTIONS.h"

CBOP
C    !ROUTINE: GUD_OPTIONS.h
C    !INTERFACE:

C    !DESCRIPTION:
C options for gud package
CEOP

C tracer selection

#undef  GUD_ALLOW_NQUOTA
#undef  GUD_ALLOW_PQUOTA
#undef  GUD_ALLOW_FEQUOTA
#undef  GUD_ALLOW_SIQUOTA
#undef  GUD_ALLOW_CHLQUOTA
#undef  GUD_ALLOW_CDOM
#undef  GUD_ALLOW_CARBON

C optional bits

#undef  GUD_ALLOW_DENIT
#undef  GUD_ALLOW_EXUDE
#undef  ALLOW_OLD_VIRTUALFLUX

C light

#define GUD_READ_PAR
#undef  GUD_USE_QSW
#undef  GUD_AVPAR
#undef  GUD_ALLOW_GEIDER
#undef  GUD_ALLOW_RADTRANS

C initialize chl with radtrans as in darwin2
#undef  GUD_CHL_INIT_LEGACY

#undef  GUD_GEIDER_RHO_SYNTH

#undef  GUD_DIAG_IOP
#undef  GUD_DIAG_RADTRANS_SOLUTION

C grazing

C for quadratic grazing a la quota
#undef  GUD_GRAZING_SWITCH

C compute palat from size ratios
#undef  GUD_ALLOMETRIC_PALAT

C turn off grazing temperature dependence
#undef  GUD_NOZOOTEMP

#undef  GUD_TIME_GRAZING

C temperature

#undef  GUD_NOTEMP
#define GUD_TEMP_VERSION 1
#undef  GUD_TEMP_RANGE

C iron

#define GUD_MINFE
#undef  GUD_PART_SCAV
#undef  GUD_IRON_SED_SOURCE_VARIABLE

C debugging

#undef GUD_DEBUG

#define GUD_ALLOW_CONS

#define GUD_UNUSED 0

C deprecated

C base particle scavenging on POP as in darwin2
#undef  GUD_PART_SCAV_POP

C these are for gud_generate_random
#undef  GUD_RANDOM_TRAITS
#undef  GUD_TWO_SPECIES_SETUP
#undef  GUD_NINE_SPECIES_SETUP
#undef  GUD_ALLOW_DIAZ

#endif /* ALLOW_GUD */
#endif /* GUD_OPTIONS_H */

