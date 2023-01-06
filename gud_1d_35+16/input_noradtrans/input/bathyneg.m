% write the negative bathymetry in bathyneg.bin.
% The bathymetry comes from preprint of Massicotte et al.
% https://doi.org/10.5194/essd-2019-160

clear all;

bathyn=-360;

outfile='bathyneg.32bits.bin';
outfileID = fopen(outfile, 'w');
fwrite(outfileID, bathyn, 'float32', 0, 'ieee-be');
fclose(outfileID);

infile='bathyneg.32bits.bin';
fileID = fopen(infile, 'r', 'ieee-be');
bathyn_32bits=fread(fileID, 'float32');
fclose(fileID);