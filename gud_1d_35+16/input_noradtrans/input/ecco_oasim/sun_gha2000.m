function gha=sun_gha2000(radeg, iyr, imon, day)
%  This subroutine computes the Greenwich hour angle in degrees for the
%  input time.  It uses the model referenced in The Astronomical Almanac
%  for 1984, Section S (Supplement) and documented in Exact 
%  closed-form geolocation algorithm for Earth survey sensors, by 
%  F.S. Patt and W.W. Gregg, Int. Journal of Remote Sensing, 1993.
%  It includes the correction to mean sideral time for nutation
%  as well as precession.
%
%  Calling Arguments
%
%  Name		Type 	I/O	Description
%
%  iyr		I*4	 I	Year (four digits)
%  day		R*8	 I	Day (time of day as fraction)
%  gha		R*8	 O	Greenwich hour angle (degrees)
%
%
%	Subprograms referenced:
%
%	JD		Computes Julian day from calendar date
%	EPHPARMS	Computes mean solar longitude and anomaly and
%			 mean lunar lontitude and ascending node
%	NUTATE		Compute nutation corrections to lontitude and 
%			 obliquity
% 	
%
%	Program written by:	Frederick S. Patt
%				General Sciences Corporation
%				November 2, 1992

%  Compute days since J2000
iday = fix(day);
fday = day - iday;
jday = sun_jd(iyr,imon,iday);
t = jday - 2451545.5 + fday;

% Compute Greenwich Mean Sidereal Time	(degrees)
gmst = 100.4606184 + 0.9856473663*t + 2.908e-13*t*t;

% Check if need to compute nutation corrections for this day
nutime=-99999;
nt = fix(t);
if (nt ~= nutime)
    nutime = nt;
    [xls, gs, xlm, omega]=sun_ephparms(t);
    [dpsi, eps]=sun_nutate(radeg, t, xls, gs, xlm, omega);
end

% Include apparent time correction and time-of-day
gha = gmst + dpsi*cos(eps/radeg) + fday*360.0;
gha = mod(gha,360.0);
if (gha < 0.0)
    gha = gha + 360.0;
end
