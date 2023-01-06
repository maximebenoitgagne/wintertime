function [sunz,rs]=sun_sunmod(rad,iday,imon,iyr,gmt,up,no,ea)
%  Given year, day of year, time in hours (GMT) and latitude and 
%  longitude, returns an accurate solar zenith and azimuth angle.  
%  Based on IAU 1976 Earth ellipsoid.  Method for computing solar 
%  vector and local vertical from Patt and Gregg, 1994, Int. J. 
%  Remote Sensing.  Only outputs solar zenith angle.  This version
%  utilizes a pre-calculation of the local up, north, and east 
%  vectors, since the locations where the solar zenith angle are 
%  calculated in the model are fixed.
%
%  Subroutines required: sun2000
%                        gha2000
%                        jd
%
% [sunz,rs]=sun_sunmod(rad,iday,imon,iyr,gmt,up,no,ea);
%
% ref: https://gud.mit.edu/MITgcm/source/pkg/sun/sun_sunmod.F?v=gud

suni=[NaN,NaN,NaN];
sung=[NaN,NaN,NaN];

radeg = rad;
% Compute sun vector
% Compute unit sun vector in geocentric inertial coordinates
sec = gmt*3600.0;
[suni, rs]=sun_sun2000(radeg, iyr, imon, iday, sec);

% Get Greenwich mean sidereal angle
day = iday;
day = day + sec/86400.0;
gha = sun_gha2000 (radeg, iyr, imon, day);
ghar = gha/radeg;

% Transform Sun vector into geocentric rotating frame
sung(1) = suni(1)*cos(ghar) + suni(2)*sin(ghar);
sung(2) = suni(2)*cos(ghar) - suni(1)*sin(ghar);
sung(3) = suni(3);

% Compute components of spacecraft and sun vector in the
% vertical (up), North (no), and East (ea) vectors frame
sunv = 0.0;
sunn = 0.0;
sune = 0.0;
for j = 1:3
    sunv = sunv + sung(j)*up(j);
    sunn = sunn + sung(j)*no(j);
    sune = sune + sung(j)*ea(j);
end

% Compute the solar zenith
sunz = radeg*atan2(sqrt(sunn*sunn+sune*sune),sunv);