% write the relax mask file for rbcs package.
% See https://mitgcm.readthedocs.io/en/latest/phys_pkgs/rbcs.html
% This relax mask file contains 75 times 1 in 32-bits big-endian.

clear all;

mask=ones(75,1);

outfile_matlab='rbcs_mask.32bits.matlab.bin';
outfileID_matlab = fopen(outfile_matlab, 'w');
fwrite(outfileID_matlab, mask, 'float32', 0, 'ieee-be');
fclose(outfileID_matlab);

infile_matlab='rbcs_mask.32bits.matlab.bin';
fileID_matlab = fopen(infile_matlab, 'r', 'ieee-be');
mask_32bits_matlab=fread(fileID_matlab, 'float32');
fclose(fileID_matlab);

infile_python='rbcs_mask.32bits.bin';
fileID_python = fopen(infile_python, 'r', 'ieee-be');
mask_32bits_python=fread(fileID_python, 'float32');
fclose(fileID_python);