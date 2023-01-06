% process initial conditions and forcings from data configuration file

clear all;
clf;

%%%%%%%%%% source %%%%%%%%%%

outfile='data.png';

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 15], ...
        'PaperUnits', 'Inches', 'PaperSize', [20, 15]);


% bathyFile
% from Green Edge data accessed by Gaetan Olivier
% (UBO).
% -> bathyneg.32bits.bin

bathyFile32='bathyneg.32bits.bin';
fileID = fopen(bathyFile32, 'r', 'ieee-be');
bathyneg=fread(fileID, 'float32');
fclose(fileID);
bar(bathyneg);
title('bathyFile');
ylabel('bathymetry (m)');

saveas(gcf, outfile);