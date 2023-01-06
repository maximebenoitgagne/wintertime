% Preprocess direct and diffuse downward planar irradiance from 
% monthly climatology of OASIM (W m^-2 (25 nm)^-1) in 32 bits to get total
% downward scalar irradiance (W m^-2 (25 nm)^-1) and then 
% Photosynthetically Available Radiation (PAR)
% (micromol photons m^-2 s^-1) in 64 bits.

clear all

% Direct irradiance monthly forcing
% Amundsen Gulf (71N, -125E)
% ilon = 360 - 125
% ilat = 80 + 71
ilon=235;
ilat=151;
array1d_ilambda_lambda = [400,425,450,475,500,525,550,575,600, ...
    625,650,675,700];
months = {'January', 'February', 'March', 'April', 'May', 'June', ...
	'July', 'August', 'September', 'October', 'November', 'December'};
array2d_ilambda_imonth_edp = zeros(13,12);
for ilambda=1:13
    fid=fopen(['ecco_oasim_edp',padstr0(ilambda,2), ...
        '_below.bin'],'r','b');
    field=fread(fid,'float32');
    field_r=reshape(field,360,160,12);
    edp = squeeze(field_r(ilon,ilat,:));
    array2d_ilambda_imonth_edp(ilambda,:) = edp;
    fclose(fid);
end
for imonth=1:12
    subplot(4,3,imonth)
    array1d_ilambda_edp = array2d_ilambda_imonth_edp(:,imonth);
    plot(array1d_ilambda_lambda, array1d_ilambda_edp)
    title(months{imonth})
    xlabel('Wavelength (nm)')
    ylabel({'Direct downwelling plane'; ...
        'irradiance (W m^{-2} (25 nm){^-1})'})
    ylim([0,10])
end
figure;

% Diffuse irradiance monthly forcing
array2d_ilambda_imonth_esp = zeros(13,12);
for ilambda=1:13
    fid=fopen(['ecco_oasim_esp',padstr0(ilambda,2), ...
        '_below.bin'],'r','b');
    field=fread(fid,'float32');
    field_r=reshape(field,360,160,12);
    esp = squeeze(field_r(ilon,ilat,:));
    array2d_ilambda_imonth_esp(ilambda,:) = esp;
    fclose(fid);
end
for imonth=1:12
    subplot(4,3,imonth)
    array1d_ilambda_esp = array2d_ilambda_imonth_esp(:,imonth);
    plot(array1d_ilambda_lambda, array1d_ilambda_esp)
    title(months{imonth})
    xlabel('Wavelength (nm)')
    ylabel({'Diffuse downwelling plane'; 'irradiance (W m^{-2} (25 nm)^{-1})'})
    ylim([0,10])
end
figure;

% Total irradiance monthly forcing
array2d_ilambda_imonth_etp ...
    = array2d_ilambda_imonth_edp + array2d_ilambda_imonth_esp;

for imonth=1:12
    subplot(4,3,imonth)
    array1d_ilambda_etp = array2d_ilambda_imonth_etp(:,imonth);
    plot(array1d_ilambda_lambda, array1d_ilambda_etp)
    title(months{imonth})
    xlabel('Wavelength (nm)')
    ylabel({'Total downwelling plane'; 'irradiance (W m^{-2} (25 nm)^{-1})'})
    ylim([0,10])
end
figure;

% Location of Amundsen Gulf
xlon = -125; % degrees East (-180 to 180)
ylat =   71; % degrees North

rad = 180/pi; % degrees/rad = radians

% arrays of up, north, and east vectors for fixed
% locations corresponding to these nwater indicators.
[up,no,ea] = sun_localvec(rad,xlon,ylat);

% For the 15th day of each month at noon (local time)
iyr=1995; % mean of the time interval 1979 to 2011 for irradiance data
          % from OASIM
iday=15;
gmt=12-xlon/15;
array1d_imonth_month = 1:12;
array1d_imonth_sunz = zeros(1,12);
array1d_imonth_sune = zeros(1,12); % solar elevation angle
for imonth=array1d_imonth_month
    % solar zenith angle and azimuth angle.
    [sunz,rs]=sun_sunmod(rad,iday,imonth,iyr,gmt,up,no,ea);
    array1d_imonth_sunz(imonth) = sunz;
    array1d_imonth_sune(imonth) = 90. - sunz;
end

% Solar elevation angle for each month from sun_sunmod.m
p1=plot(array1d_imonth_month,array1d_imonth_sune);
title('Solar elevation angle')
xlim([1,12])
xlabel('Time')
ax=gca;
ax.XTickLabel = ({'Jan 15', 'Feb 15', 'Mar 15', 'Apr 15', 'May 15', ...
    'Jun 15', 'Jul 15', 'Aug 15', 'Sep 15', 'Oct 15', 'Nov 15', 'Dec 15', });
ylabel('Solar elevation angle (degree)')

% Solar elevation angle for each month from the NOAA Solar Calculator
% https://www.esrl.noaa.gov/gmd/grad/solcalc/
filename='sune_NOAA.csv';
col1_imonth_col2_sunz=csvread(filename,4,0);
p2=line(col1_imonth_col2_sunz(:,1),col1_imonth_col2_sunz(:,2),'Color',...
    'red','LineStyle','-.');
p3=line([1,12],[0,0],'Color','black','LineStyle','--');
legend([p1 p2],'sun\_sunmod.m','NOAA Solar Calculator')
figure;

% Solar zenith angle for each month from sun_sunmod.m
plot(array1d_imonth_month,array1d_imonth_sunz)
title('Solar zenith angle')
xlim([1,12])
xlabel('Time')
ax=gca;
ax.XTickLabel = ({'Jan 15', 'Feb 15', 'Mar 15', 'Apr 15', 'May 15', 'Jun 15', 'Jul 15', 'Aug 15', 'Sep 15', 'Oct 15', 'Nov 15', 'Dec 15', });
ylabel('Solar zenith angle (degree)')
y = 90;
line([1,12],[y,y],'Color','black','LineStyle','--')
figure;

% Average cosine of direct planar downward irradiance for each month
array1d_imonth_rmud = zeros(1,12);
array1d_imonth_mud  = zeros(1,12);
for imonth=array1d_imonth_month
    sunz=array1d_imonth_sunz(imonth);
    rmud=sun_sfcrmud(rad,sunz);
    array1d_imonth_mud(imonth) = 1/rmud;
    array1d_imonth_rmud(imonth)=rmud;
end

plot(array1d_imonth_month,array1d_imonth_mud)
title('Downwelling average cosine')
xlim([1,12])
xlabel('Time')
ax=gca;
ax.XTickLabel = ({'Jan 15', 'Feb 15', 'Mar 15', 'Apr 15', 'May 15', 'Jun 15', 'Jul 15', 'Aug 15', 'Sep 15', 'Oct 15', 'Nov 15', 'Dec 15', });
ylim([0,1])
ylabel('Downwelling average cosine (no units)')
figure;

plot(array1d_imonth_month,array1d_imonth_rmud)
title('Inverse of downwelling average cosine')
xlim([1,12])
xlabel('Time')
ax=gca;
ax.XTickLabel = ({'Jan 15', 'Feb 15', 'Mar 15', 'Apr 15', 'May 15', 'Jun 15', 'Jul 15', 'Aug 15', 'Sep 15', 'Oct 15', 'Nov 15', 'Dec 15', });
ylabel('Inverse of downwelling average cosine (no units)')
figure;

% Photosynthetically Available Radiation
[array2d_ilambda_imonth_eds,array2d_ilambda_imonth_ess, ...
    array2d_ilambda_imonth_ets,array1d_imonth_PAR] ...
    =spectral_planar_2_PAR(array1d_imonth_rmud, ...
    array2d_ilambda_imonth_edp,array2d_ilambda_imonth_esp);

for ilambda=1:13
    for imonth=1:12
        subplot(4,3,imonth)
        array1d_ilambda_eds = array2d_ilambda_imonth_eds(:,imonth);
        plot(array1d_ilambda_lambda, array1d_ilambda_eds)
        title(months{imonth})
        xlabel('Wavelength (nm)')
        xlim([400,700]);
        ylabel({'Direct downwelling scalar'; ...
            'irradiance (W m^{-2} (25 nm)^{-1})'})
        ylim([0,10])
    end
end
figure;

for ilambda=1:13
    for imonth=1:12
        subplot(4,3,imonth)
        array1d_ilambda_ess = array2d_ilambda_imonth_ess(:,imonth);
        plot(array1d_ilambda_lambda, array1d_ilambda_ess)
        title(months{imonth})
        xlabel('Wavelength (nm)')
        xlim([400,700]);
        ylabel({'Diffuse downwelling scalar'; ...
            'irradiance (W m^{-2} (25 nm)^{-1})'})
        ylim([0,10])
    end
end
figure;

for ilambda=1:13
    for imonth=1:12
        subplot(4,3,imonth)
        array1d_ilambda_ets = array2d_ilambda_imonth_ets(:,imonth);
        plot(array1d_ilambda_lambda, array1d_ilambda_ets)
        title(months{imonth})
        xlabel('Wavelength (nm)')
        xlim([400,700]);
        ylabel({'Total downwelling scalar'; ...
            'irradiance (W m^{-2} (25 nm)^{-1})'})
        ylim([0,10])
    end
end
figure;

p1=plot(array1d_imonth_month,array1d_imonth_PAR);
title('Photosynthetically available radiation (PAR)')
xlim([1,12])
xlabel('Time')
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('Photosynthetically available radiation (PAR) (\mumol photons m^{-2} s^{-1})')

% PAR(0-) for Beaufort Sea from Belanger et al., 2013
% doi:10.5194/bg-10-4087-2013
array1d_imonth_PAR_Belanger13=zeros(1,12);
for imonth=1:4
    array1d_imonth_PAR_Belanger13(imonth)=NaN;
end
for imonth=10:12
    array1d_imonth_PAR_Belanger13(imonth)=NaN;
end
array1d_imonth_PAR_Belanger13(5)=4.7; % mol photons m^-2 d^-1
array1d_imonth_PAR_Belanger13(6)=14.3;
array1d_imonth_PAR_Belanger13(7)=21.2;
array1d_imonth_PAR_Belanger13(8)=17.5;
array1d_imonth_PAR_Belanger13(9)=11.2;
% from mol photons m^-2 d^-1 to umol photons m^-2 s^-1
scale_fac=1e6/(24*3600);
array1d_imonth_PAR_Belanger13=array1d_imonth_PAR_Belanger13*scale_fac;
p2=line(array1d_imonth_month,array1d_imonth_PAR_Belanger13,'Color', ...
    'red','LineStyle','-.');
legend([p1 p2],'Computed from OASIM (1979-2011)', ...
    'Belanger et al., 2013 (1998-2009)', 'Location', 'northwest')
% figure;

% Write PAR file in 32 bits
outfile='AG_ecco_oasim_par_below_ueinm2s.32bits.bin';
outfileID = fopen(outfile, 'w');
fwrite(outfileID, array1d_imonth_PAR, 'float32', 0, 'ieee-be');
fclose(outfileID);