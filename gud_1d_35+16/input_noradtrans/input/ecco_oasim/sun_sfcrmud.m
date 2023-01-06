function rmud=sun_sfcrmud(rad,sunz)
% Compute average cosine for direct irradiance in the water
% column given solar zenith angle (in degrees) at surface.
%
% ref: https://gud.mit.edu/MITgcm/source/pkg/sun/sun_sfcrmud.F?v=gud

rn=1.341;
rsza = sunz/rad;
sinszaw = sin(rsza)/rn;
szaw = asin(sinszaw);
rmudl = 1.0/cos(szaw);   %avg cosine direct (1 over)
rmud = min(rmudl,1.5);
rmud = max(rmud,0.0);
