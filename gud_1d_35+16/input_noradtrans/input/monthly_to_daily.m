function vec_iday_var = monthly_to_daily(vec_imonth_var)
% vec_iday_var = monthly_to_daily(vec_imonth_var);
%
% Interpolates monthly values to 365 daily values.
% The values from day 350 to day 365 and from day 1 to day 15 are
% interpolated between the values of December and January.
%
% e.g.
% >> vec_imonth_var=2*(1:12);
% >> vec_iday_var = monthly_to_daily(vec_imonth_var);
% >> plot(vec_iday_var);
vec_imonth_iday=[16 45.5 75 105.5 136 166.5 197 228 258.5 289 319.5 ...
    350 380];
vec_imonth_var=vec_imonth_var([1:length(vec_imonth_var) 1]);
vec_iday_iday=16:380;
vec_iday_var=interp1(vec_imonth_iday,vec_imonth_var,vec_iday_iday);
vec_iday_var=vec_iday_var([(366-15):(380-15) (16-15):(365-15)]);
end