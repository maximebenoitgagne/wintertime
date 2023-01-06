% plot sea ice area for one location over one year

clear all

ndays=365;
YYYY=2016;
vec_iday_day=1:ndays;

infile='siarea.nemo.2016.365.32bits.bin';
outfile='siarea.nemo.2016.365.32bits.png';
fileID = fopen(infile, 'r', 'ieee-be');
array1d_iday_val = fread(fileID, 'float32');
fclose(fileID);

p=plot(vec_iday_day, array1d_iday_val);
p.LineWidth=2;
title("Sea ice concentration in "+YYYY);
ylabel('Sea ice (no unit)');
x0=10;
y0=10;
width=400;
height=150;
set(gcf,'position',[x0,y0,width,height])
xlim([1,ndays]);
xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]);
ax=gca;
ax.FontSize = 12;
ax.XTickLabel = ({'Jan', '', 'Mar', '', 'May', '', 'Jul', '', 'Sep', ...
    '', 'Nov', ''});

saveas(gcf,outfile);