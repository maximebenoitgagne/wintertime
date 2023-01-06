% make sure solar_heat_flux_under_ice =
% solar_heat_flux_under_ice_for_100:100_ice_cover * ice_concentration

clear all;
clf;

ndays=365;
outfile='explore_GE_mod_var.png';

% standard_name: surface_downwelling_shortwave_in water.
% long_name: solar_heat_flux_under_ice_for_100:100_ice_cover.
% file: GE_mod_var.nc.
% Variable solar_heat_flux_under_ice_for_100:100_ice_cover is the surface
% downwelling shortwave in water (below ice for 100% ice cover).
% It is at indices (2,2) meaning center of grid point was used.
% File GE_mod_var.nc is the output of NEMO+LIM3 model configured by Gaetan
% Olivier (UBO) for the Green Edge ice camp 2016 (67.48N, -63.79E).

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 15], ...
    'PaperUnits', 'Inches', 'PaperSize', [20, 15]);
subplot(3,2,1);
qswice_file='GE_mod_var.nc';
array1d_iT_qswicefull=ncread(qswice_file, ...
    'solar heat flux under ice for 100:100 ice cover',[2 2 1], [1 1 1825]);
array1d_iT_qswicefull=reshape(array1d_iT_qswicefull,[],1);
plot(1:ndays*5, array1d_iT_qswicefull, 'LineWidth', 2);
title({'solar\_heat\_flux\_under\_ice\_for\_100:100\_ice\_cover',''});
xlim([1,ndays*5]);
ylim([0,40]);
xlabel('Time');
xticks(1:365:1826);
ax=gca;
ax.XTickLabel = (2013:1:2018);
ylabel('irradiance (W m^{-2})');
set(gca,'fontsize', 24);
grid on;

% standard_name: sea_ice_area_fraction.
% long_name: ice_concentration.
% file: Ice_d.nc.
% Variable ice_concentration is at indices (2,2) meaning center of grid 
% point was used.
% File Ice_d.nc is the output of NEMO+LIM3 model configured by Gaetan
% Olivier (UBO) for the Green Edge ice camp 2016 (67.48N, -63.79E).
icefile='Ice_d.nc';
icefull=ncread(icefile,'ice concentration',[2 2 1], [1 1 1825]);
icefull=reshape(icefull,[],1);
subplot(3,2,3);
plot(1:ndays*5, icefull, 'LineWidth', 2);
title({'ice concentration',''});
xlim([1,ndays*5]);
xlabel('Time');
xticks(1:365:1826);
ax=gca;
ax.XTickLabel = (2013:1:2018);
ylabel('ice concentration (%)');
set(gca,'fontsize', 24);
grid on;

% solar_heat_flux_under_ice_for_100:100_ice_cover * ice concentration
ndaysfull=length(array1d_iT_qswicefull);
if ndaysfull~=length(icefull)
    error('The shortwave file (%s) and the ice file (%s) dont have the same number of days', ...
        qswice_file, icefile);
end % ndaysfull~=length(icefull)
array1d_iT_qswice_times_icefull_calculated=NaN(ndaysfull,1);
for iT=1:ndaysfull
    ice=icefull(iT);
    qswice=array1d_iT_qswicefull(iT);
    if ice>0
        array1d_iT_qswice_times_icefull_calculated(iT)=qswice*ice;
    end % sic > 0
end % for iT
subplot(3,2,5);
plot(1:ndays*5, array1d_iT_qswice_times_icefull_calculated, ...
    'LineWidth', 2);
title({'solar\_heat\_flux\_under\_ice\_for\_100:100\_ice\_cover', ...
    '* ice\_concentration'});
xlim([1,ndays*5]);
ylim([0,40]);
xlabel('Time');
xticks(1:365:1826);
ax=gca;
ax.XTickLabel = (2013:1:2018);
ylabel('irradiance (W m^{-2})');
set(gca,'fontsize', 24);
grid on;

% standard_name: surface_downwelling_shortwave_in water.
% long_name: solar_heat_flux_under_ice.
% file: GE_mod_var.nc.
% Variable solar_heat_flux_under_ice is the surface
% downwelling shortwave in water (below ice for 100% ice cover) times ice
% concentration.
% It is at indices (2,2) meaning center of grid point was used.
% File GE_mod_var.nc is the output of NEMO+LIM3 model configured by Gaetan
% Olivier (UBO) for the Green Edge ice camp 2016 (67.48N, -63.79E).
subplot(3,2,6);
array1d_iT_qswice_times_icefull=ncread(qswice_file, ...
    'solar heat flux under ice', ...
    [2 2 1], [1 1 1825]);
array1d_iT_qswice_times_icefull= ...
    reshape(array1d_iT_qswice_times_icefull,[],1);
plot(1:ndays*5, array1d_iT_qswice_times_icefull, 'LineWidth', 2);
title({'solar\_heat\_flux\_under\_ice',''});
xlim([1,ndays*5]);
ylim([0,40]);
xlabel('Time');
xticks(1:365:1826);
ax=gca;
ax.XTickLabel = (2013:1:2018);
ylabel('irradiance (W m^{-2})');
set(gca,'fontsize', 24);
grid on;

saveas(gcf, outfile);