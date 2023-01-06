GENPARAMS_TGT=GUD_GENPARAMS.h gud_readgenparams.F
TRAITPARAMS_TGT=GUD_TRAITPARAMS.h gud_readtraitparams.F
TRAITS_TGT=GUD_TRAITS.h gud_init_traits.F gud_readtraits.F
EXF_TGT=GUD_EXF.h gud_exf_init_fixed.F gud_exf_init_varia.F gud_exf_load.F gud_readparms.F gud_monitor.F
COG_TGT=$(GENPARAMS_TGT) $(TRAITPARAMS_TGT) $(TRAITS_TGT) $(EXF_TGT) gud_check.F

all: cog

cog: FORCE
	../../tools/python/cog -c -r $(COG_TGT)

FORCE:

clean_old:

