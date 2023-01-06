% plot forcing fields of surface: PAR, sea ice concentration, wind and 
% iron

clear all;

% The only use of the foo block of code below is that 
% PAR.ice.2016.365.png is not the first figure produced.
% The first figure produced has different dimensions than the other
% figures for an unknown reason.
% TODO: Find a solution.
outfile='tmp.png';
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 15], ...
        'PaperUnits', 'Inches', 'PaperSize', [20, 15]);
plot(1,1);
saveas(gcf,outfile);
delete 'tmp.png';

outfile='PAR.ice.2016.365.png';

ndays=365;

interval_days=[1 32 60 91 121 152 182 213 244 274 305 335];

% days
vec_iday_day=1:ndays;

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 15], ...
        'PaperUnits', 'Inches', 'PaperSize', [20, 15]);

% PAR
infile='1D_GDPS_PAR_y2016.365.32bits.bin';
fileID = fopen(infile, 'r', 'ieee-be');
array1d_iday_val = fread(fileID, 'float32');
fclose(fileID);
subplot(2,1,1);
p=plot(vec_iday_day,array1d_iday_val);
grid on;
p.LineWidth=10;
xlim([1,ndays]);
ylim([0,800]);
ax=gca;
ax.FontSize=24;
ax.XTick=interval_days;
ax.XTickLabel=({'Jan-1','Feb-1','Mar-1','Apr-1','May-1','Jun-1', ...
    'Jul-1','Aug-1','Sep-1','Oct-1','Nov-1','Dec-1'});
title('PAR');
ax.TitleFontSizeMultiplier = 2;
ax.TickDir='both';
ax.Layer='top';
ylabel('PAR (\mumol photons m^{-2} s^{-1})')

% sea ice concentration
infile='siarea.nemo.2016.365.32bits.bin';
fileID = fopen(infile, 'r', 'ieee-be');
array1d_iday_val = fread(fileID, 'float32');
fclose(fileID);
subplot(2,1,2);
p=plot(vec_iday_day,array1d_iday_val);
grid on;
p.LineWidth=10;
xlim([1,ndays]);
ax=gca;
ax.FontSize=24;
ax.XTick=interval_days;
ax.XTickLabel=({'Jan-1','Feb-1','Mar-1','Apr-1','May-1','Jun-1', ...
    'Jul-1','Aug-1','Sep-1','Oct-1','Nov-1','Dec-1'});
title('Sea ice concentration');
ax.TitleFontSizeMultiplier = 2;
ax.TickDir='both';
ax.Layer='top';
ylabel('Sea ice concentration (no unit)')

saveas(gcf,outfile);

% wind
clf;
outfile='wind.iron.365.png';
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 15], ...
        'PaperUnits', 'Inches', 'PaperSize', [20, 15]);
infile='loc1_tren_speed_daily-2d.32bits.bin';
fileID = fopen(infile, 'r', 'ieee-be');
array1d_iday_val = fread(fileID, 'float32');
fclose(fileID);
subplot(2,1,1);
p=plot(vec_iday_day,array1d_iday_val);
grid on;
p.LineWidth=10;
xlim([1,ndays]);
ax=gca;
ax.FontSize=24;
ax.XTick=interval_days;
ax.XTickLabel=({'Jan-1','Feb-1','Mar-1','Apr-1','May-1','Jun-1', ...
    'Jul-1','Aug-1','Sep-1','Oct-1','Nov-1','Dec-1'});
title('Wind');
ax.TitleFontSizeMultiplier = 2;
ax.TickDir='both';
ax.Layer='top';
ylabel('Wind (N m^{-2} ?)');

% iron
infile='loc1_mahowald2009_solubile_current_smooth_oce_daily-2d.32bits.bin';
fileID = fopen(infile, 'r', 'ieee-be');
array1d_iday_val = fread(fileID, 'float32');
fclose(fileID);
subplot(2,1,2);
p=plot(vec_iday_day,array1d_iday_val);
grid on;
p.LineWidth=10;
xlim([1,ndays]);
ax=gca;
ax.FontSize=24;
ax.XTick=interval_days;
ax.XTickLabel=({'Jan-1','Feb-1','Mar-1','Apr-1','May-1','Jun-1', ...
    'Jul-1','Aug-1','Sep-1','Oct-1','Nov-1','Dec-1'});
title('Iron');
ax.TitleFontSizeMultiplier = 2;
ax.TickDir='both';
ax.Layer='top';
ylabel('Iron (units ?)');
    
saveas(gcf,outfile);