function [up,no,ea]=sun_localvec(rad,xlon,ylat)
% [up,no,ea]--Create arrays of up, north, and east vectors for fixed 
%             locations corresponding to these nwater indicators.
%
% [up,no,ea]=sun_localvec(rad,xlon,xlat);
%
% ref: https://gud.mit.edu/MITgcm/source/pkg/sun/sun_localvec.F?v=gud

% Compute local east, north, and vertical vectors 
up=[0,0,0];
no=[0,0,0];
ea=[0,0,0];
    
% Convert geodetic lat/lon to Earth-centered, earth-fixed (ECEF)
% vector (geodetic unit vector)
rlon = xlon/rad;
cosx = cos(rlon);
sinx = sin(rlon);
rlat = ylat/rad;
cosy = cos(rlat);
siny = sin(rlat);

% Compute the local up, East and North unit vectors
up(1) = cosy*cosx;
up(2) = cosy*sinx;
up(3) = siny;
upxy = sqrt(up(1)*up(1)+up(2)*up(2));
ea(1) = -up(2)/upxy;
ea(2) = up(1)/upxy;
no(1) = up(2)*ea(3) - up(3)*ea(2);  %cross product
no(2) = up(3)*ea(1) - up(1)*ea(3);
no(3) = up(1)*ea(2) - up(2)*ea(1);
end