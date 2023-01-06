#ifdef ALLOW_GUD

#ifdef GUD_ALLOW_RADTRANS

      COMMON /radtrans_params_r/
     &  wb_totalWidth, wb_width, wb_center, WtouEins,
     &  aw, bw,
     &  aphy_chl_type, aphy_chl_ps_type, bphy_mgC_type, bbphy_mgC_type,
     &  asize, apsize, bsize, bbsize,
     &  apart, bpart, bbpart, apart_P, bpart_P, bbpart_P
      _RL wb_totalWidth
      _RL wb_width(nlam)
      _RL wb_center(nlam)
      _RL WtouEins(nlam)
      _RL aw(nlam)
      _RL bw(nlam)
      _RL aphy_chl_type   (nopt,nlam)
      _RL aphy_chl_ps_type(nopt,nlam)
      _RL bphy_mgC_type   (nopt,nlam)
      _RL bbphy_mgC_type  (nopt,nlam)
      _RL asize(nopt)
      _RL apsize(nopt)
      _RL bsize(nopt)
      _RL bbsize(nopt)
      _RL apart(nlam)
      _RL bpart(nlam)
      _RL bbpart(nlam)
      _RL apart_P(nlam)
      _RL bpart_P(nlam)
      _RL bbpart_P(nlam)

      COMMON /gud_acdom_params_r/ exCDOM
      _RL exCDOM(nlam)

#ifndef GUD_ALLOW_CDOM
      COMMON /gud_acdom_params_i/ laCDOM
      INTEGER laCDOM
#endif

#endif /* GUD_ALLOW_RADTRANS */

#endif /* ALLOW_GUD */
