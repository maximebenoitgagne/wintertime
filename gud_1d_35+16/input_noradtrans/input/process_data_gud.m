% process forcings from data.gud configuration file

clear all;
clf;

ndays=365;

%%%%%%%%%% ice processing %%%%%%%%%%

% 2013 to 2020

% icefile
% variable ice_concentration at indices (2,2) meaning center of grid point
% was used.
% Ice_d.nc is the output of NEMO+LIM3+PISCES simulation by Gaetan
% Olivier (UBO).
icefile='Ice_d.nc';
% ncdisp(icefile);
icefull=ncread(icefile,'ice concentration',[2 2 1], [1 1 1825]);
icefull=reshape(icefull,[],1);

outfile='ice_processing.png';
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 15], ...
        'PaperUnits', 'Inches', 'PaperSize', [20, 15]);
subplot(3,1,1);
plot(1:ndays*5, icefull);
title('ice concentration 2013 to 2018');
xlim([1,ndays*5]);
xlabel('Time');
xticks(1:365:1826);
ax=gca;
ax.XTickLabel = (2013:1:2018);
ylabel('ice concentration (unitless)');

% 2016
ice=icefull(365*3+1:365*4);

subplot(3,1,2);
plot(1:ndays, ice);
title('ice concentration 2016');
xlim([1,ndays]);
xlabel('Time');
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('ice concentration (unitless)');

% siarea.nemo.2016.365.32bits.bin

icefile2016='siarea.nemo.2016.365.32bits.bin';
outfileID = fopen(icefile2016, 'w');
fwrite(outfileID, ice, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(icefile2016, 'r', 'ieee-be');
ice=fread(fileID, 'float32');
fclose(fileID);

subplot(3,1,3);
plot(1:ndays, ice);
title('siarea.nemo.2016.365.32bits.bin');
xlim([1,ndays]);
xlabel('Time');
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('ice concentration (unitless)');

saveas(gcf, outfile);

%%%%%%%%%% iron processing %%%%%%%%%%

% iron file
% is file Hamiltonetlal2019_MIMIv1.0_Iron_ConcDep_MonMean.nc
% It was downloaded at 
% http://www.geo.cornell.edu/eas/PeoplePlaces/Faculty/mahowald/dust/Hamiltonetal2019/
% reference: Hamilton et al. 2019 in Geoscientific Model Development
% doi: ?10.5194/gmd-12-3835-2019
% The variable SFEDEP is the monhtly mean soluble-Fe deposition
% (in kg Fe m^-2 s^-1).
iron_source_file='Hamiltonetlal2019_MIMIv1.0_Iron_ConcDep_MonMean.nc';
% ncdisp(iron_source_file)
% Be x,y the coordinates (1-based) on the equirectangular grid
% location of the Green Edge ice camp
lat= 67.4797; % degree N
lon=-63.7895; % degree E
x=round((lon+360)*144/360+1);
y=round((lat+90)*(96-1)/180+1);
iron_monthly=ncread(iron_source_file,'SFEDEP',[x y 1],[1 1 12]);
kg2g=1000;
atomic_mass_iron=55.84; % g Fe / mol Fe
% Conversion factor to transform atmospheric Fe deposition into all the
% exterior sources of Fe (atmospheric iron deposition on open water,
% atmospheric iron deposition on sea ice followed by sea ice melting,
% river runoff and melting of glaciers from the Penny Ice Cap).
% This conversion factor was empirical. There was no references to
% support it.
atm2extsrc=10;
iron_monthly=iron_monthly*kg2g/atomic_mass_iron*atm2extsrc;
iron_file_monthly='GE_hamilton2019_SFEDEP_times10.monthly.32bits.bin';
outfileID = fopen(iron_file_monthly, 'w');
fwrite(outfileID, iron_monthly, 'float32', 0, 'ieee-be');
fclose(outfileID);

%%%%%%%%%% PAR processing from GDPS %%%%%%%%%%

% qswfile
% is file 1D_GDPS_qsw_y2016.nc
% variable qsw at indices (2,2) meaning center of grid point was used.
% file 1D_GDPS_qsw_y2016.nc is the output of CMC GDPS reforecasts (CGRF).
% CMC: Canadian Meteorological Centre's.
% GDPS: global deterministic prediction system.
% The data was provided by Gregory Smith (Environment Canada).
% Gaetan Olivier (UBO) preprocessed it to get the data at the Green Edge
% 2016 ice camp (67.48N, -63.79E).

qswfile='1D_GDPS_qsw_y2016.nc';
% ncdisp(qswfile)
qsw=ncread(qswfile,'qsw',[2 2 1], [1 1 8784]);
qsw=reshape(qsw,[],1);

outfile='PAR_processing_GDPS.m.png';
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 15], ...
        'PaperUnits', 'Inches', 'PaperSize', [20, 15]);
subplot(4,2,1);
plot(1:(ndays+1)*24, qsw);
title('hourly shortwave radiation just above water surface in 2016');
xlim([1,(ndays+1)*24]);
ylim([-200,800]);
xlabel('Time');
xticks([1, 745, 1417, 2161, 2881, 3625, 4345, 5089, 5833, 6553, 7297, 8017]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('irradiance (W m^{-2})');

% daily shortwave radiation just above water surface in 2016
array1d_idoy_qsw=zeros(ndays,1);
for doy=1:ndays
    time_counter_start=(doy-1)*24+1;
    time_counter_end=time_counter_start+23;
    qswdaily=0;
    for time_counter=time_counter_start:time_counter_end
        qswdaily=qswdaily+qsw(time_counter);
    end % for time_counter
    qswdaily=qswdaily/24;
    array1d_idoy_qsw(doy)=qswdaily;
end % for doy

subplot(4,2,2);
plot(1:ndays, array1d_idoy_qsw);
title('daily shortwave radiation just above water surface in 2016');
xlim([1,ndays]);
ylim([-200,800]);
xlabel('Time');
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('irradiance (W m^{-2})');

% daily shortwave radiation just below water surface in 2016
albedo=0.066; % reference: PISCES
array1d_idoy_qswbelow=array1d_idoy_qsw*(1-albedo);

subplot(4,2,3);
plot(1:ndays, array1d_idoy_qswbelow);
title('daily shortwave radiation just below water surface in 2016');
xlim([1,ndays]);
ylim([-200,800]);
xlabel('Time');
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('irradiance (W m^{-2})');

% daily PAR just below water surface in 2016
Wm2_to_PAR=0.43; % TODO: find a reference
array1d_idoy_PARbelow=array1d_idoy_qswbelow*Wm2_to_PAR;

subplot(4,2,4);
plot(1:ndays, array1d_idoy_PARbelow);
title('daily PAR just below water surface in 2016');
xlim([1,ndays]);
ylim([-200,800]);
xlabel('Time');
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('PAR (W m^{-2})');

% daily irradiance of each 3 RGB bands just below water surface in 2016
% W m^-2
onecolor=1/3;
array1d_idoy_blue=array1d_idoy_PARbelow*onecolor;
array1d_idoy_green=array1d_idoy_PARbelow*onecolor;
array1d_idoy_red=array1d_idoy_PARbelow*onecolor;

subplot(4,2,5);
plot(1:ndays, array1d_idoy_blue, ':b');
hold on;
plot(1:ndays, array1d_idoy_green, '--g');
plot(1:ndays, array1d_idoy_red, '-.r');
mytitle='daily irradiance of each 3 RGB bands just below water surface in 2016';
title(mytitle);
xlim([1,ndays]);
ylim([-200,800]);
xlabel('Time');
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('PAR (W m^{-2})');
h=zeros(3, 1);
h(1)=plot(NaN, NaN, '--b');
h(2)=plot(NaN, NaN, '-.g');
h(3)=plot(NaN, NaN, ':r');
legend(h, 'blue', 'green', 'red');

% daily irradiance of each 3 RGB bands just below water surface in 2016
% umol photons m^-2 s^-1
% ref: https://gud.mit.edu/MITgcm/source/pkg/gud/gud_radtrans_direct.F?v=gud
rmus=1/0.83;
planck = 6.6256e-34;            %Plancks constant J sec
c = 2.998e8;                    %speed of light m/sec
hc = 1.0/(planck*c);
oavo = 1.0/6.023e23;            % 1/Avogadros number
hcoavo = hc*oavo;
wb_center=[450,550,650];        % [blue, green, red]
WtouEins=zeros(3,1);
for l=1:3
   rlamm = wb_center(l)*1e-9;      %lambda in m
   WtouEins(l) = 1e6*rlamm*hcoavo; %Watts to uEin/s conversion
end
array1d_idoy_blue =array1d_idoy_blue *WtouEins(1);
array1d_idoy_green=array1d_idoy_green*WtouEins(2);
array1d_idoy_red  =array1d_idoy_red  *WtouEins(3);

subplot(4,2,6);
plot(1:ndays, array1d_idoy_blue, ':b');
hold on;
plot(1:ndays, array1d_idoy_green, '--g');
plot(1:ndays, array1d_idoy_red, '-.r');
mytitle='daily irradiance of each 3 RGB bands just below water surface in 2016';
title(mytitle);
xlim([1,ndays]);
ylim([0,700]);
xlabel('Time');
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('PAR (\mumol photons m^{-2} s^{-1})');
h=zeros(3, 1);
h(1)=plot(NaN, NaN, '--b');
h(2)=plot(NaN, NaN, '-.g');
h(3)=plot(NaN, NaN, ':r');
legend(h, 'blue', 'green', 'red');

% daily PAR just below water surface in 2016
array1d_idoy_PARuEin=array1d_idoy_blue+array1d_idoy_green+array1d_idoy_red;

subplot(4,2,7);
plot(1:ndays, array1d_idoy_PARuEin);
title('daily PAR just below water surface in 2016');
xlim([1,ndays]);
ylim([0,700]);
xlabel('Time');
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('PAR (\mumol photons m^{-2} s^{-1})');

% 1D_GDPS_PAR_y2016.365.32bits.m.bin
PARfile='1D_GDPS_PAR_y2016.365.32bits.m.bin';
outfileID = fopen(PARfile, 'w');
fwrite(outfileID, array1d_idoy_PARuEin, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(PARfile, 'r', 'ieee-be');
array1d_idoy_PARuEin=fread(fileID, 'float32');
fclose(fileID);

subplot(4,2,8);
plot(1:ndays, array1d_idoy_PARuEin);
title('1D\_GDPS\_PAR\_y2016.365.32bits.bin');
xlim([1,ndays]);
ylim([0,700]);
xlabel('Time');
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('PAR (\mumol photons m^{-2} s^{-1})');

saveas(gcf, outfile);

%%%%%%%%%% PAR processing from NEMO %%%%%%%%%%
% standard_name: surface_downwelling_shortwave_in water.
% long_name: solar_heat_flux_under_ice_for_100:100_ice_cover.
% file: GE_mod_var.nc.
% Variable solar_heat_flux_under_ice_for_100:100_ice_cover is the surface
% downwelling shortwave in water (below ice for 100% ice cover).
% It is at indices (2,2) meaning center of grid point was used.
% File GE_mod_var.nc is the output of NEMO+LIM3 model configured by Gaetan
% Olivier (UBO) for the Green Edge ice camp 2016 (67.48N, -63.79E).
clf;
outfile='PARice_processing_NEMO.m.png';

% daily shortwave radiation just below sea ice from 2013 to 2017
qswice_file='GE_mod_var.nc';
array1d_iT_qswicefull=ncread(qswice_file, ...
    'solar heat flux under ice for 100:100 ice cover',[2 2 1],[1 1 1825]);
array1d_iT_qswicefull=reshape(array1d_iT_qswicefull,[],1);

subplot(4,2,1);
plot(1:ndays*5, array1d_iT_qswicefull, 'LineWidth', 2);
title('daily shortwave radiation just below sea ice from 2013 to 2017');
xlim([1,ndays*5]);
xticks(1:365:1826);
ylim([0,40]);
ax=gca;
ax.XTickLabel = (2013:1:2018);
ylabel('irradiance (W m^{-2})');
grid on;

% daily shortwave radiation just below sea ice from 2016
array1d_idoy_qswice=array1d_iT_qswicefull(365*3+1:365*4);

subplot(4,2,2);
plot(1:ndays, array1d_idoy_qswice, 'LineWidth', 2);
title('daily shortwave radiation just below sea ice in 2016');
xlim([1,ndays]);
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ylim([0,40]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('irradiance (W m^{-2})');
grid on;

% daily PAR just below sea ice in 2016
Wm2_to_PAR=0.85; % ask a reference to Marion Lebrun (ULaval)
array1d_idoy_PARice=array1d_idoy_qswice*Wm2_to_PAR;

subplot(4,2,3);
plot(1:ndays, array1d_idoy_PARice, 'LineWidth', 2);
title('daily PAR just below sea ice in 2016');
xlim([1,ndays]);
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ylim([0,40]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('PAR (W m^{-2})');
grid on;

% daily irradiance of each 3 RGB bands just below sea ice in 2016
% W m^-2

% reference:
% Lebrun et al. (submitted). Values for the Green Edge ice camp 2016.
blue_fraction=0.43;
green_fraction=0.46;
red_fraction=0.11;

array1d_idoy_blue=array1d_idoy_PARice*blue_fraction;
array1d_idoy_green=array1d_idoy_PARice*green_fraction;
array1d_idoy_red=array1d_idoy_PARice*red_fraction;

subplot(4,2,4);
plot(1:ndays, array1d_idoy_blue, ':b', 'LineWidth', 2);
hold on;
plot(1:ndays, array1d_idoy_green, '--g', 'LineWidth', 2);
plot(1:ndays, array1d_idoy_red, '-.r', 'LineWidth', 2);
mytitle='daily irradiance of each 3 RGB bands just below sea ice in 2016';
title(mytitle);
xlim([1,ndays]);
ylim([0,40]);
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('PAR (W m^{-2})');
h=zeros(3, 1);
h(1)=plot(NaN, NaN, '--b', 'LineWidth', 2);
h(2)=plot(NaN, NaN, '-.g', 'LineWidth', 2);
h(3)=plot(NaN, NaN, ':r', 'LineWidth', 2);
legend(h, 'blue', 'green', 'red');
grid on;

% daily irradiance of each 3 RGB bands just below water surface in 2016
% umol photons m^-2 s^-1
% ref: https://gud.mit.edu/MITgcm/source/pkg/gud/gud_radtrans_direct.F?v=gud
rmus=1/0.83;
planck = 6.6256e-34;            %Plancks constant J sec
c = 2.998e8;                    %speed of light m/sec
hc = 1.0/(planck*c);
oavo = 1.0/6.023e23;            % 1/Avogadros number
hcoavo = hc*oavo;
wb_center=[450,550,650];        % [blue, green, red]
WtouEins=zeros(3,1);
for l=1:3
   rlamm = wb_center(l)*1e-9;      %lambda in m
   WtouEins(l) = 1e6*rlamm*hcoavo; %Watts to uEin/s conversion
end
array1d_idoy_blue =array1d_idoy_blue *WtouEins(1);
array1d_idoy_green=array1d_idoy_green*WtouEins(2);
array1d_idoy_red  =array1d_idoy_red  *WtouEins(3);

subplot(4,2,5);
plot(1:ndays, array1d_idoy_blue, ':b', 'LineWidth', 2);
hold on;
plot(1:ndays, array1d_idoy_green, '--g', 'LineWidth', 2);
plot(1:ndays, array1d_idoy_red, '-.r', 'LineWidth', 2);
mytitle='daily irradiance of each 3 RGB bands just below sea ice in 2016';
title(mytitle);
xlim([1,ndays]);
ylim([0,120]);
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('PAR (\mumol photons m^{-2} s^{-1})');
h=zeros(3, 1);
h(1)=plot(NaN, NaN, '--b', 'LineWidth', 2);
h(2)=plot(NaN, NaN, '-.g', 'LineWidth', 2);
h(3)=plot(NaN, NaN, ':r', 'LineWidth', 2);
legend(h, 'blue', 'green', 'red');
grid on;

% daily PAR just below sea ice in 2016
array1d_idoy_PARiceuEin=array1d_idoy_blue+array1d_idoy_green+ ...
    array1d_idoy_red;

subplot(4,2,6);
plot(1:ndays, array1d_idoy_PARiceuEin, 'LineWidth', 2);
title('daily PAR just below sea ice in 2016');
xlim([1,ndays]);
ylim([0,120]);
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('PAR (\mumol photons m^{-2} s^{-1})');
grid on;

% 1D_NEMO_PARice_y2016.365.32bits.m.bin
array1d_idoy_PARiceuEin(isnan(array1d_idoy_PARiceuEin))=0;
PARicefile='1D_NEMO_PARice_y2016.365.32bits.m.bin';
outfileID = fopen(PARicefile, 'w');
fwrite(outfileID, array1d_idoy_PARiceuEin, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(PARicefile, 'r', 'ieee-be');
array1d_idoy_PARiceuEin=fread(fileID, 'float32');
fclose(fileID);

subplot(4,2,7);
plot(1:ndays, array1d_idoy_PARiceuEin, 'LineWidth', 2);
title('1D\_NEMO\_PARice\_y2016.365.32bits.bin');
xlim([1,ndays]);
ylim([0,120]);
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('PAR (\mumol photons m^{-2} s^{-1})');
grid on;
    
saveas(gcf, outfile);

%%%%%%%%% source %%%%%%%%%%

clf;

outfile='data.gud.source.png';

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 15], ...
        'PaperUnits', 'Inches', 'PaperSize', [20, 15]);
    
% icefile
icefile='siarea.nemo.2016.365.32bits.bin';
fileID = fopen(icefile, 'r', 'ieee-be');
ice=fread(fileID, 'float32');
fclose(fileID);
subplot(3,2,1);
plot(1:ndays, ice, 'LineWidth', 2);
title('icefile');
xlim([1,ndays]);
xlabel('Time');
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('ice area');

fileID=fopen(iron_file_monthly, 'r', 'ieee-be');
iron_monthly_temp=fread(fileID, 'float32');
fclose(fileID);
subplot(3,2,2);
plot(1.5:12.5, iron_monthly_temp, 'LineWidth', 2);
title('ironfile');
xlim([1,13]);
xlabel('Time');
xticks(1:12);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
     'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('iron (mol Fe m^{-2} s^{-1})');

% PARfile
fileID = fopen(PARfile, 'r', 'ieee-be');
array1d_idoy_PARuEin=fread(fileID, 'float32');
fclose(fileID);
subplot(3,2,3);
plot(1:ndays, array1d_idoy_PARuEin, 'LineWidth', 2);
title('PARFile');
xlim([1,ndays]);
ylim([0,700]);
xlabel('Time');
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('PAR (\mumol photons m^{-2} s^{-1})');

% PARicefile
fileID = fopen(PARicefile, 'r', 'ieee-be');
array1d_idoy_PARiceuEin=fread(fileID, 'float32');
fclose(fileID);
subplot(3,2,4);
plot(1:ndays, array1d_idoy_PARiceuEin, 'LineWidth', 2);
title('PARiceFile');
xlim([1,ndays]);
ylim([0,700]);
xlabel('Time');
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('PAR (\mumol photons m^{-2} s^{-1})');

saveas(gcf, outfile);

%%%%%%%%%% 32 bits %%%%%%%%%%
clf;

outfile32='data.gud.32bits.png';

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 15], ...
        'PaperUnits', 'Inches', 'PaperSize', [20, 15]);
    
% icefile
icefile='siarea.nemo.2016.365.32bits.bin';
fileID = fopen(icefile, 'r', 'ieee-be');
ice=fread(fileID, 'float32');
fclose(fileID);
subplot(3,2,1);
plot(1:ndays, ice, 'LineWidth', 2);
title('icefile');
xlim([1,ndays]);
xlabel('Time');
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('ice area');

%ironfile
fileID=fopen(iron_file_monthly, 'r', 'ieee-be');
iron_monthly=fread(fileID, 'float32');
fclose(fileID);
subplot(3,2,2);
plot(1.5:12.5, iron_monthly, 'LineWidth', 2);
title('ironfile');
xlim([1,13]);
xlabel('Time');
xticks(1:12);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
     'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('iron (mol Fe m^{-2} s^{-1})');

% PARFile
fileID = fopen(PARfile, 'r', 'ieee-be');
array1d_idoy_PARuEin=fread(fileID, 'float32');
fclose(fileID);
subplot(3,2,3);
plot(1:ndays, array1d_idoy_PARuEin, 'LineWidth', 2);
title('PARFile');
xlim([1,ndays]);
ylim([0,700]);
xlabel('Time');
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('PAR (\mumol photons m^{-2} s^{-1})');

% PARicefile
fileID = fopen(PARicefile, 'r', 'ieee-be');
array1d_idoy_PARiceuEin=fread(fileID, 'float32');
fclose(fileID);
subplot(3,2,4);
plot(1:ndays, array1d_idoy_PARiceuEin, 'LineWidth', 2);
title('PARiceFile');
xlim([1,ndays]);
ylim([0,700]);
xlabel('Time');
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('PAR (\mumol photons m^{-2} s^{-1})');

saveas(gcf, outfile32);

%%%%%%%%%% daily %%%%%%%%%%
clf;

outfile_daily='data.gud.32bits.daily.png';

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 15], ...
        'PaperUnits', 'Inches', 'PaperSize', [20, 15]);
    
% icefile
icefile='siarea.nemo.2016.365.32bits.bin';
fileID = fopen(icefile, 'r', 'ieee-be');
ice=fread(fileID, 'float32');
fclose(fileID);
subplot(3,2,1);
plot(1:ndays, ice, 'LineWidth', 2);
title('icefile');
xlim([1,ndays]);
xlabel('Time');
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('ice area');

% ironfile
iron_daily=monthly_to_daily(iron_monthly);
iron_file_daily='GE_hamilton2019_SFEDEP_times10.daily.32bits.bin';
outfileID = fopen(iron_file_daily, 'w');
fwrite(outfileID, iron_daily, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(iron_file_daily, 'r', 'ieee-be');
iron_daily_temp=fread(fileID, 'float32');
fclose(fileID);
subplot(3,2,2);
plot(1:ndays, iron_daily_temp, 'LineWidth', 2);
title('ironFile');
xlim([1,ndays]);
% ylim([0,20E-13]);
xlabel('Time');
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('iron (mol Fe m^{-2} s^{-1})');

% PARFile
fileID = fopen(PARfile, 'r', 'ieee-be');
array1d_idoy_PARuEin=fread(fileID, 'float32');
fclose(fileID);
subplot(3,2,3);
plot(1:ndays, array1d_idoy_PARuEin, 'LineWidth', 2);
title('PARFile');
xlim([1,ndays]);
ylim([0,700]);
xlabel('Time');
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('PAR (\mumol photons m^{-2} s^{-1})');

% PARicefile
fileID = fopen(PARicefile, 'r', 'ieee-be');
array1d_idoy_PARiceuEin=fread(fileID, 'float32');
fclose(fileID);
subplot(3,2,4);
plot(1:ndays, array1d_idoy_PARiceuEin, 'LineWidth', 2);
title('PARiceFile');
xlim([1,ndays]);
ylim([0,700]);
xlabel('Time');
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('PAR (\mumol photons m^{-2} s^{-1})');

% windFile
% generating wind file
% wind file is a constant value of 12 N m^-2
% it is an arbitrary value since the goal is to have a system
% flooded with iron such that it is never iron limited
wind_daily=12+zeros(1,365);
windFiledaily='constant_wind.daily.32bits.bin';
outfileID = fopen(windFiledaily, 'w');
fwrite(outfileID, wind_daily, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(windFiledaily, 'r', 'ieee-be');
wind_daily=fread(fileID, 'float32');
fclose(fileID);
subplot(3,2,5);
plot(1:ndays, wind_daily, 'LineWidth', 2);
title('windFile');
xlim([1,ndays]);
ylim([0,20]);
xlabel('Time');
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ax=gca;
ax.XTickLabel = ({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', ...
    'Aug', 'Sep', 'Oct', 'Nov', 'Dec' });
ylabel('Wind (N/m^2 ?)');

saveas(gcf, outfile_daily);
