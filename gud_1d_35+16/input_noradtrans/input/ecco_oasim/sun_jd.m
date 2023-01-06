function sun_jd=sun_jd(i,j,k)
%    This function converts a calendar date to the corresponding Julian
%    day starting at noon on the calendar date.  The algorithm used is
%    from Van Flandern and Pulkkinen, Ap. J. Supplement Series 41, 
%    November 1979, p. 400.
%
%
%	Arguments
%     
%     	Name    Type 	I/O 	Description
%     	----	---- 	--- 	-----------
%     	i	I*4  	 I 	Year - e.g. 1970
%     	j       I*4  	 I  	Month - (1-12)
%     	k       I*4  	 I  	Day  - (1-31)
%     	jd      I*4  	 O  	Julian day
%
%     external references
%     -------------------
%      none
%
%
%     Written by Frederick S. Patt, GSC, November 4, 1992
%
% sun_jd=sun_jd(i,j,k)
%
% ref: https://gud.mit.edu/MITgcm/source/pkg/sun/sun_jd.F?v=gud
sun_jd = 367*i - 7*(i+(j+9)/12)/4 + 275*j/9 + k + 1721014;
%  This additional calculation is needed only for dates outside of the 
%   period March 1, 1900 to February 28, 2100
%     	sun_jd = sun_jd + 15 - 3*((i+(j-9)/7)/100+1)/4