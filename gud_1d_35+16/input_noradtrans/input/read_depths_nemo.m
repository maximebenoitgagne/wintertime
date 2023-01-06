clear all;

% Read the depths from 1D_BB_NO3_GE_spring.delZ1015mm.nc.
% The depths are the vertical cell center spacing 1D array.
% It is also known as the parameter delRc in MITgcm.
% See the MITgcm manual at section 3.8.1.2
% (https://mitgcm.readthedocs.io/en/latest/getting_started/getting_started.html#grid)

depthfile='1D_BB_NO3_GE_spring.delZ1015mm.nc';
depth=ncread(depthfile,'depth');
outfile='depth.txt';
fileID=fopen(outfile,'w');
fprintf(fileID,'%011.6f,',-depth);
fclose(fileID);

% Read the depths from GE_mod_var.nc.
% It is the depths used as outputs in the simulation of NEMO+PISCES by
% Gaetan Olivier (UBO) at the Green Edge ice camp (67.48N; 63.79W) for 
% 2016.
% The depths are the vertical cell center spacing 1D array.
% It is also known as the parameter delRc in MITgcm.
% See the MITgcm manual at section 3.8.1.2
% (https://mitgcm.readthedocs.io/en/latest/getting_started/getting_started.html#grid)

depthfile='GE_mod_var.nc';
depth=ncread(depthfile,'deptht');
outfile='deptht.txt';
fileID=fopen(outfile,'w');
fprintf(fileID,'%011.6f,',-depth);
fclose(fileID);

% Read the vertical grid spacing from GE_mod_var.nc.
% It is the vertical grid spacing used as outputs in the 
% simulation of NEMO+PISCES by Gaetan Olivier (UBO) at the Green Edge ice 
% camp (67.48N; 63.79W) for 2016.
% The vertical grid spacing are the vertical differences between the 
% upper and lower surfaces of the grid cells.
% It is also known as the parameter delZ or delR in MITgcm.
% See the MITgcm manual at section 3.8.1.2
% (https://mitgcm.readthedocs.io/en/latest/getting_started/getting_started.html#grid)

depthfile='GE_mod_var.nc';
outfile='delZ.txt';
ndepths=75;
array1d_idepth_upper=(ncread(depthfile,'deptht_bounds',[1 1], ...
    [1 ndepths]) )';
array1d_idepth_lower=(ncread(depthfile,'deptht_bounds',[2 1], ...
    [1 ndepths]) )';
array1d_idepth_delZ=array1d_idepth_lower-array1d_idepth_upper;
fileID=fopen(outfile,'w');
fprintf(fileID,'%08.6f,',array1d_idepth_delZ);
fclose(fileID);
