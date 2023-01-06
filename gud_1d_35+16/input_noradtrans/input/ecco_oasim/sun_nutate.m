function [dpsi, eps]=sun_nutate(radeg, t, xls, gs, xlm, omega)
%  This subroutine computes the nutation in longitude and the obliquity
%  of the ecliptic corrected for nutation.  It uses the model referenced
%  in The Astronomical Almanac for 1984, Section S (Supplement) and 
%  documented in Exact closed-form geolocation algorithm for Earth 
%  survey sensors, by F.S. Patt and W.W. Gregg, Int. Journal of 
%  Remote Sensing, 1993.  These parameters are used to compute the 
%  apparent time correction to the Greenwich Hour Angle and for the 
%  calculation of the geocentric Sun vector.  The input ephemeris 
%  parameters are computed using subroutine ephparms.  Terms are 
%  included to 0.1 arcsecond.
%
%  Calling Arguments
%
%  Name		Type 	I/O	Description
%
%  t		R*8	 I	Time in days since January 1, 2000 at 
%				 12 hours UT
%  xls		R*8	 I	Mean solar longitude (degrees)
%  gs		R*8	 I	Mean solar anomaly   (degrees)
%  xlm		R*8	 I	Mean lunar longitude (degrees)
%  Omega	R*8	 I	Ascending node of mean lunar orbit 
%  				 (degrees)
%  dPsi		R*8	 O	Nutation in longitude (degrees)
%  Eps		R*8	 O	Obliquity of the Ecliptic (degrees)
% 				 (includes nutation in obliquity)
%
%
%	Program written by:	Frederick S. Patt
%				General Sciences Corporation
%				October 21, 1992
%
% [dpsi, eps]=sun_nutate(radeg, t, xls, gs, xlm, omega)
%
% ref: https://gud.mit.edu/MITgcm/source/pkg/sun/sun_nutate.F?v=gud

% Nutation in Longitude
dpsi = - 17.1996*sin(omega/radeg) ...
       + 0.2062*sin(2.0D0*omega/radeg) ...
       - 1.3187*sin(2.0*xls/radeg) ...
       + 0.1426*sin(gs/radeg) ...
       - 0.2274*sin(2.0*xlm/radeg);

% Mean Obliquity of the Ecliptic
epsm = 23.439291 - (3.560e-7)*t;

% Nutation in Obliquity
deps = 9.2025*cos(omega/radeg) + 0.5736*cos(2.0*xls/radeg);

% True Obliquity of the Ecliptic
eps = epsm + deps/3600.0;

dpsi = dpsi/3600.0;