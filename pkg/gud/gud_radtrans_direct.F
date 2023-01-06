#include "GUD_OPTIONS.h"

CBOP
C !ROUTINE: GUD_RADTRANS_DIRECT

C !INTERFACE: ==========================================================
      subroutine GUD_RADTRANS_DIRECT(
     I                   H,rmud,Edsf,Essf,a_k,bt_k,bb_k,kbot,
     O                   Edbot,Esbot,Eubot,Estop,Eutop,
     O                   PAR, PARF,
     O                   amp1, amp2, x, y,
     O                   r1, r2, kappa1, kappa2,
     I                   myThid)

C !DESCRIPTION:
c
c  Model of irradiance in the water column.  Accounts for three
c  irradiance streams [Ackleson, Balch, Holligan, JGR, 1994],
c
c  Edbot = direct downwelling irradiance in W/m2 per waveband
c  Esbot = diffuse downwelling irradiance in W/m2 per waveband
c  Eubot = diffuse upwelling irradiance in W/m2 per waveband
c
c  all defined at the bottom of each layer.  Also computed are Estop,
c  Eutop at the top of each layer which should be very close to Esbot,
c  Eubot of the layer above.
c
c  The Ed equation is integrated exactly, Es and Eu are computed by
c  solving a set of linear equation for the amplitudes in the exact
c  solution [see, e.g., Kylling, Stamnes, Tsay, JAC, 1995].
c  The boundary condition in the deepest wet layer is
c  downward-decreasing modes only (i.e., zero irradiance at infinite
c  depth, assuming the optical properties of the last layer).
c
c  Also computed are scalar radiance and PAR at the grid cell center
c  (both in uEin/m2/s).
c
C !USES: ===============================================================
      IMPLICIT NONE
#include "SIZE.h"                 /* Nr */
#include "EEPARAMS.h"
#include "GUD_SIZE.h"          /* nlam */
#ifdef GUD_ALLOW_RADTRANS
#include "GUD_PARAMS.h"        /* WtouEin */
#include "GUD_GENPARAMS.h"  /* gud_PAR_ilamLo/Hi
                                     gud_radmodThresh
                                     gud_rmus gud_rmuu */
#endif

C !INPUT PARAMETERS: ===================================================
C     H     :: layer thickness (including hFacC!)
C     rmud  :: inv.cosine of direct (underwater solar) zenith angle
C     Edsf  :: direct downwelling irradiance below surface per waveband
C     Essf  :: diffuse downwelling irradiance below surface per waveband
C     a_k   :: absorption coefficient per level and waveband (1/m)
C     bt_k  :: total scattering coefficient per level and waveband (1/m)
C              = forward + back scattering coefficient
C     bb_k  :: backscattering coefficient per level and waveband (1/m)
C     kmax  :: maximum number of layers to compute
      _RL H(Nr)
      _RL rmud
      _RL Edsf(nlam), Essf(nlam)
      _RL a_k(Nr,nlam), bt_k(Nr,nlam), bb_k(Nr,nlam)
      INTEGER kbot
      INTEGER myThid

C !OUTPUT PARAMETERS: ==================================================
C     Edbot  :: direct downwelling irradiance at bottom of layer
C     Esbot  :: diffuse downwelling irradiance at bottom of layer
C     Eubot  :: diffuse upwelling irradiance at bottom of layer
C     Estop  :: diffuse downwelling irradiance at top of layer
C     Eutop  :: diffuse upwelling irradiance at top of layer
C     PAR    :: PAR at cell center per waveband
C     PARF   :: PAR at top of cell per waveband
C     amp1   :: amplitude of downward increasing mode
C     amp2   :: amplitude of downward decreasing mode
      _RL Edbot(Nr,nlam),Esbot(Nr,nlam),Eubot(Nr,nlam)
      _RL Estop(Nr,nlam),Eutop(Nr,nlam)
      _RL PAR(Nr,nlam)
      _RL PARF(Nr,nlam)
      _RL amp1(Nr,nlam), amp2(Nr,nlam)
      _RL x(Nr,nlam), y(Nr,nlam)
      _RL kappa1(Nr,nlam),kappa2(Nr,nlam)
      _RL r2(Nr,nlam),r1(Nr,nlam)
CEOP

#ifdef GUD_ALLOW_RADTRANS

C !LOCAL VARIABLES: ====================================================
      INTEGER k, l
      _RL Edtop(Nr,nlam)
      _RL Etopwq, Ebotwq
      _RL zd
      _RL rmus,rmuu
      _RL cd,au,Bu,Cu
      _RL as,Bs,Cs,Bd,Fd
      _RL bquad,D
      _RL denom
      _RL c1,c2
      _RL ed(Nr),e2(Nr),e1(Nr)
      _RL a3d(2*Nr), b3d(2*Nr), c3d(2*Nr), y3d(2*Nr)
      _RL rd, ru
      data rd /1.5 _d 0/   !these are taken from Ackleson, et al. 1994 (JGR)
      data ru /3.0 _d 0/

      rmus = gud_rmus
      rmuu = gud_rmuu

      DO l = 1,nlam
       DO k=1,Nr
        Edtop(k,l) = 0.0
        Estop(k,l) = 0.0
        Eutop(k,l) = 0.0
        Edbot(k,l) = 0.0
        Esbot(k,l) = 0.0
        Eubot(k,l) = 0.0
        amp1(k,l) = 0.0
        amp2(k,l) = 0.0
        kappa1(k,l) = 0.0
        kappa2(k,l) = 0.0
        r1(k,l) = 0.0
        r1(k,l) = 0.0
        x(k,l) = 0.0
        y(k,l) = 0.0
       ENDDO
      ENDDO
      IF (kbot.GT.0) THEN
       DO l = 1,nlam
        IF (Edsf(l) .GE. gud_radmodThresh .OR.
     &      Essf(l) .GE. gud_radmodThresh) THEN
         DO k=1,kbot
          zd = H(k)
          cd = (a_k(k,l)+bt_k(k,l))*rmud
          au = a_k(k,l)*rmuu
          Bu = ru*bb_k(k,l)*rmuu
          Cu = au+Bu
          as = a_k(k,l)*rmus
          Bs = rd*bb_k(k,l)*rmus
          Cs = as+Bs
          Bd = bb_k(k,l)*rmud
          Fd = (bt_k(k,l)-bb_k(k,l))*rmud
          bquad = Cs + Cu
          D = 0.5*(bquad + SQRT(bquad*bquad - 4.0*Bs*Bu))
          kappa1(k,l) = D - Cs
          kappa2(k,l) = Cs - Bs*Bu/D  ! == D - Cu
          r1(k,l) = Bu/D
          r2(k,l) = Bs/D
          denom = (cd-Cs)*(cd+Cu) + Bs*Bu
          x(k,l) = -((cd+Cu)*Fd+Bu*Bd)/denom
          y(k,l) = (-Bs*Fd+(cd-Cs)*Bd)/denom
          ed(k) = EXP(-cd*zd)
          e1(k) = EXP(-kappa1(k,l)*zd)
          e2(k) = EXP(-kappa2(k,l)*zd)
         ENDDO

C integrate Ed equation first
         Edtop(1,l) = Edsf(l)
         DO k=1,kbot-1
          Edbot(k,l) = Edtop(k,l)*ed(k)
          Edtop(k+1,l) = Edbot(k,l)
         ENDDO
         Edbot(kbot,l) = Edtop(kbot,l)*ed(kbot)

C setup tridiagonal matrix of continuity/boundary conditions
C variables: c2(1), c1(1), c2(2), ..., c1(kbot)
C a3d,b3d,c3d: lower, main and upper diagonal
C y3d: right-hand side
C
C top b.c.: c2(1) + e1(1)*r1(1)*c1(1) = Essf - x(1)*Edsf
         a3d(1) = 0. _d 0  ! not used
         b3d(1) = 1.           ! A(1,1)*c2(1)
         c3d(1) = e1(1)*r1(1,l)  ! A(1,2)*c1(1)
         y3d(1) = Essf(l) - x(1,l)*Edsf(l)
C continuity at layer boundaries
         DO k=1, kbot-1
           a3d(2*k) = (1. - r2(k,l)*r1(k+1,l))*e2(k)  !   A(2k,2k-1)*c2(k)
           b3d(2*k) = r1(k,l) - r1(k+1,l)             ! + A(2k,2k  )*c1(k)
           c3d(2*k) = -1. + r2(k+1,l)*r1(k+1,l)       ! + A(2k,2k+1)*c2(k+1)
           y3d(2*k)=(x(k+1,l) - x(k,l) - r1(k+1,l)*(y(k+1,l)-y(k,l)))
     &              *Edbot(k,l)
           a3d(2*k+1) = 1 - r1(k,l)*r2(k,l)                !   A(2k+1,2k  )*c1(k)
           b3d(2*k+1) = r2(k,l) - r2(k+1,l)                ! + A(2k+1,2k+1)*c2(k+1)
           c3d(2*k+1) = (-1. + r1(k+1,l)*r2(k,l))*e1(k+1)  ! + A(2k+1,2k+2)*c1(k+1)
           y3d(2*k+1)=(y(k+1,l) - y(k,l) - r2(k,l)*(x(k+1,l)-x(k,l)))
     &                *Edbot(k,l)
         ENDDO
c bottom boundary condition: c1 = 0
         a3d(2*kbot) = 0. _d 0  !   A(2*kbot,2*kbot-1)*c2(kbot)
         b3d(2*kbot) = 1. _d 0  ! + A(2*kbot,2*kbot  )*c1(kbot)
         c3d(2*kbot) = 0. _d 0  ! not used
         y3d(2*kbot) = 0. _d 0  ! = 0

         CALL GUD_SOLVE_TRIDIAG(a3d,b3d,c3d,y3d,2*kbot,myThid)

C compute irradiances
         DO k=1,kbot
          c2 = y3d(2*k-1)
          c1 = y3d(2*k)
          Estop(k,l) = c2 + r1(k,l)*e1(k)*c1 + x(k,l)*Edtop(k,l)
          Esbot(k,l) = e2(k)*c2 + r1(k,l)*c1 + x(k,l)*Edbot(k,l)
          Eutop(k,l) = r2(k,l)*c2 + e1(k)*c1 + y(k,l)*Edtop(k,l)
          Eubot(k,l) = r2(k,l)*e2(k)*c2 + c1 + y(k,l)*Edbot(k,l)
          amp1(k,l) = c1
          amp2(k,l) = c2
         ENDDO
         IF (kbot .LT. Nr) THEN
          Estop(kbot+1,l) = Esbot(kbot,l)
          Eutop(kbot+1,l) = Eubot(kbot,l)
         ENDIF

C       endif thresh
        ENDIF

        DO k = 1,Nr
C convert to scalar irradiance in quanta
         Etopwq = (rmud*Edtop(k,l)+rmus*Estop(k,l)+rmuu*Eutop(k,l))
     &            *WtouEins(l)
         Ebotwq = (rmud*Edbot(k,l)+rmus*Esbot(k,l)+rmuu*Eubot(k,l))
     &            *WtouEins(l)
C        and interpolate
         PAR(k,l) = SQRT(Etopwq*Ebotwq)
         PARF(k,l) = Etopwq
        ENDDO

C      enddo l
       ENDDO
C     endif kbot.gt.0
      ENDIF

#endif /* GUD_ALLOW_RADTRANS */

      return
      end

