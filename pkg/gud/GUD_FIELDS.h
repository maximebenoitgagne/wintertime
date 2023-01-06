#ifdef ALLOW_GUD

CBOP
C     !ROUTINE: GUD_FIELDS.h
C     !INTERFACE:
C #include GUD_FIELDS.h

C     !DESCRIPTION:
C Contains fields for cell package
C
C Requires: SIZE.h
C

#ifndef GUD_ALLOW_CHLQUOTA
      COMMON /GUD_CHL_STORE/ chlPrev
#ifdef GUD_ALLOW_RADTRANS
      _RL chlPrev(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy,nPhoto)
#else
      _RL chlPrev(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy)
#endif
#endif


#ifdef GUD_ALLOW_CONS
      COMMON /GUD_CHECK_CONS/
     &      GUD_cons_C_unit,
     &      GUD_cons_P_unit,
     &      GUD_cons_N_unit,
     &      GUD_cons_Fe_unit,
     &      GUD_cons_Si_unit,
     &      GUD_cons_A_unit,
     &      GUD_cons_O_unit
      INTEGER GUD_cons_C_unit
      INTEGER GUD_cons_P_unit
      INTEGER GUD_cons_N_unit
      INTEGER GUD_cons_Fe_unit
      INTEGER GUD_cons_Si_unit
      INTEGER GUD_cons_A_unit
      INTEGER GUD_cons_O_unit

      COMMON /GUD_CONS_3D/ GUD_Nfix, GUD_Ndenit
      _RL GUD_Nfix(sNx,sNy,Nr,nSx,nSy)
      _RL GUD_Ndenit(sNx,sNy,Nr,nSx,nSy)
#endif

C Carbon Variables

       COMMON /CARBON_NEEDS/
     &              pH, pCO2, Atmosp, FluxCO2, FluxO2
      _RL  pH(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy)
      _RL  pCO2(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy)
      _RL  AtmosP(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL  FluxCO2(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL  FluxO2(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)

       COMMON /CARBON_CHEM/
     &                     ak0,ak1,ak2,akw,akb,aks,akf,
     &                     ak1p,ak2p,ak3p,aksi, fugf, 
     &                     ff,ft,st,bt, Ksp_TP_Calc
      _RL  ak0(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL  ak1(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL  ak2(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL  akw(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL  akb(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL  aks(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL  akf(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL  ak1p(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL  ak2p(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL  ak3p(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL  aksi(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL  ff(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
C Fugacity Factor added by Val Bennington Nov. 2010
      _RL  fugf(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL  ft(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL  st(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL  bt(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL  Ksp_TP_Calc(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)

C m3perkg : is conversion factor for mol/m3 to mol/kg
C          assumes uniform (surface) density
C Pa2Atm : for conversion of atmospheric pressure
C          when coming from atmospheric model
       COMMON /GLOBAL_SURF_MEAN/
     &                          gsm_alk,gsm_s,gsm_t,gsm_dic,
     &                          gsm_c14
      _RL  gsm_alk
      _RL  gsm_s
      _RL  gsm_t
      _RL  gsm_DIC
      _RL  gsm_C14

CEOP
#endif /* ALLOW_GUD */
