function [xls, gs, xlm, omega]=sun_ephparms(t)
%  This subroutine computes ephemeris parameters used by other Mission
%  Operations routines:  the solar mean longitude and mean anomaly, and
%  the lunar mean longitude and mean ascending node.  It uses the model
%  referenced in The Astronomical Almanac for 1984, Section S 
%  (Supplement) and documented and documented in Exact closed-form 
%  geolocation algorithm for Earth survey sensors, by F.S. Patt and 
%  W.W. Gregg, Int. Journal of Remote Sensing, 1993.  These parameters 
%  are used to compute the solar longitude and the nutation in 
%  longitude and obliquity.
%
%	Program written by:	Frederick S. Patt
%				General Sciences Corporation
%				November 2, 1992
%
% [xls, gs, xlm, omega]=sun_ephparms(t)
%
% ref: https://gud.mit.edu/MITgcm/source/pkg/sun/sun_ephparms.F?v=gud
%
% Parameter : t      :: Time in days since January 1, 2000 at 12 hours UT
%
% Return    : xls	 :: Mean solar longitude (degrees)
%             gs	 :: Mean solar anomaly (degrees)
%             xlm	 :: Mean lunar longitude (degrees)
%             omega :: Ascending node of mean lunar orbit (degrees)

% Sun Mean Longitude
xls = 280.46592 + 0.9856473516*t;
xls = mod(xls,360.0);

% Sun Mean Anomaly
gs = 357.52772 + 0.9856002831*t;
gs = mod(gs,360.0);

% Moon Mean Longitude
xlm = 218.31643 + 13.17639648*t;
xlm = mod(xlm,360.0);

% Ascending Node of Moons Mean Orbit
omega = 125.04452 - 0.0529537648*t;
omega = mod(omega,360.0);
