#ifdef ALLOW_SUN

C     nutime :: time of precomputed dpsi,eps
C     dpsi   :: Nutation in longitude (degrees)
C     eps    :: Obliquity of the Ecliptic (degrees)
C               (includes nutation in obliquity)
C
      COMMON /SUN_NUTATE_VARS/dpsi,eps,nutime

      INTEGER nutime
      _RL dpsi, eps

#endif /* ALLOW_SUN */

