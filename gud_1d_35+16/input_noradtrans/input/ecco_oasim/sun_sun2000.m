function [sunvec, rs]=sun_sun2000(radeg, iyr, imon, iday, sec)
%  This subroutine computes the Sun vector in geocentric inertial 
%  (equatorial) coodinates.  It uses the model referenced in The 
%  Astronomical Almanac for 1984, Section S (Supplement) and documented
%  in Exact closed-form geolocation algorithm for Earth survey
%  sensors, by F.S. Patt and W.W. Gregg, Int. Journal of Remote
%  Sensing, 1993.  The accuracy of the Sun vector is approximately 0.1 
%  arcminute.
%
%	Arguments:
%
%	Name	Type	I/O	Description
%	--------------------------------------------------------
%	IYR	I*4	 I	Year, four digits (i.e, 1993)
%	IDAY	I*4	 I	Day of month (1-31) % correction in the comment by
%                                         Maxime Benoit-Gagne
%	SEC	R*8	 I	Seconds of day 
%	SUN(3)	R*8	 O	Unit Sun vector in geocentric inertial 
%				 coordinates of date
%	RS	R*8	 O	Magnitude of the Sun vector (AU)
%
%	Subprograms referenced:
%
%	JD		Computes Julian day from calendar date
%	EPHPARMS	Computes mean solar longitude and anomaly and
%			 mean lunar lontitude and ascending node
%	NUTATE		Compute nutation corrections to lontitude and 
%			 obliquity
%
%	Coded by:  Frederick S. Patt, GSC, November 2, 1992
%	Modified to include Earth constants subroutine by W. Gregg,
%		May 11, 1993.
%
% [sunvec, rs]=(radeg, iyr, imon, iday, sec);
%
% ref: https://gud.mit.edu/MITgcm/source/pkg/sun/sun_sun2000.F?v=gud

sunvec=[NaN,NaN,NaN];
xk=0.0056932; %Constant of aberration

% Compute floating point days since Jan 1.5, 2000
% Note that the Julian day starts at noon on the specified date
rjd = sun_jd(iyr,imon,iday);
t = rjd - 2451545.0 + (sec-43200.0)/86400.0;

% Compute solar ephemeris parameters
[xls, gs, xlm, omega]=sun_ephparms(t);

% Check if need to compute nutation corrections for this day
nutime=-99999;
nt = fix(t);
if (nt ~= nutime)
    nutime = nt;
    [dpsi, eps]=sun_nutate(radeg, t, xls, gs, xlm, omega);
end

% Compute planet mean anomalies
 % Venus Mean Anomaly
g2 = 50.40828 + 1.60213022*t;
g2 = mod(g2,360.0);

 % Mars Mean Anomaly
g4 = 19.38816 + 0.52402078*t;
g4 = mod(g4,360.0);

 % Jupiter Mean Anomaly
g5 = 20.35116 + 0.08309121*t;
g5 = mod(g5,360.0);

 % Compute solar distance (AU)
rs = 1.00014 - 0.01671*cos(gs/radeg)- 0.00014*cos(2.0*gs/radeg);

 % Compute Geometric Solar Longitude
dls = (6893.0 - 4.6543463e-4*t)*sin(gs/radeg) ...
     + 72.0*sin(2.0*gs/radeg) ...
     - 7.0*cos((gs - g5)/radeg) ...
     + 6.0*sin((xlm - xls)/radeg) ...
     + 5.0*sin((4.0*gs - 8.0*g4 + 3.0*g5)/radeg) ...
     - 5.0*cos((2.0*gs - 2.0*g2)/radeg) ...
     - 4.0*sin((gs - g2)/radeg) ...
     + 4.0*cos((4.0*gs - 8.0*g4 + 3.0*g5)/radeg) ...
     + 3.0*sin((2.0*gs - 2.0*g2)/radeg) ...
     - 3.0*sin(g5/radeg) ...
     - 3.0*sin((2.0*gs - 2.0*g5)/radeg);  %arcseconds

 xlsg = xls + dls/3600.0;
 
 % Compute Apparent Solar Longitude; includes corrections for nutation
  % in longitude and velocity aberration
 xlsa = xlsg + dpsi - xk/rs;
 
 % Compute unit Sun vector
 sunvec(1) = cos(xlsa/radeg);
 sunvec(2) = sin(xlsa/radeg)*cos(eps/radeg);
 sunvec(3) = sin(xlsa/radeg)*sin(eps/radeg);
