% process initial conditions for data.ptracers configuration file

clear all;

% The only use of the foo block of code below is that 
% data.ptracers.source.png is not the first figure produced.
% The first figure produced has different dimensions than the other
% figures for an unknown reason.
% TODO: Find a solution.
outfile='tmp.png';
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 15], ...
        'PaperUnits', 'Inches', 'PaperSize', [20, 15]);
plot(1,1);
saveas(gcf,outfile);
delete 'tmp.png';

clf;

%%%%%%%%%% source %%%%%%%%%%

outfile='data.ptracers.nutrients.source.png';
tenthyearjan1=365*9+1; % January 1 of 10th year
% previous_run_file='car.0000000000.t001.1D_BB_GE_run_20211022_0000.nc';
% this file is available on https://bit.ly/31mOGaE
% previous_run_file='car.0000000000.t001.1D_BB_GE_run_20220504_0000.nc';
% this file is available on https://bit.ly/3wm80k3
previous_run_file='car.0000000000.t001.1D_BB_GE_run_20220511_0000.nc';
% this file is available on https://bit.ly/3lrCEnj
ncid=netcdf.open(previous_run_file,'NC_NOWRITE');

% previous_run_file is an absolute file name
[pathstr, ~, ~] = fileparts(previous_run_file);
% previous_run_file is a relative file name
if strcmp(pathstr,'')
    pathstr=pwd;
end % if strcmp(pathstr,'')
gridfile=strcat(pathstr,'/grid.t001.nc');
if exist(gridfile,'file') == 2
    Z=ncread(gridfile,'Z');
else
    error('file does not exist:\n%s\ngrid.t001.nc is required because it contains the values of the depths.', gridfile);
end % if exist
Z=Z(Z>-400);
ndepths=length(Z);
   
delZ1015mm=[-000.502836,-001.528090,-002.586202,-003.687317, ...
    -004.844414,-006.073893,-007.396130,-008.835898,-010.422464, ...
    -012.189137,-014.172002,-016.407684,-018.930130,-021.766829, ...
    -024.935165,-028.439867,-032.272266,-036.411629,-040.828140, ...
    -045.486675,-050.350471,-055.384087,-060.555321,-065.836227, ...
    -071.203377,-076.637634,-082.123665,-087.649414,-093.205460, ...
    -098.784546,-104.381088,-109.990845,-115.610580,-121.237839, ...
    -126.870773,-132.507965,-138.148392,-143.791214,-149.435883, ...
    -155.081894,-160.728943,-166.376770,-172.025177,-177.674011, ...
    -183.323181,-188.972610,-194.622208,-200.271942,-205.921799, ...
    -211.571716,-217.221710,-222.871735,-228.521790,-234.171890, ...
    -239.821991,-245.472107,-251.122238,-256.772369,-262.422516, ...
    -268.072662,-273.722839,-279.372986,-285.023132,-290.673309, ...
    -296.323456,-301.973633,-307.623779,-313.273956,-318.924103, ...
    -324.574280,-330.224426,-335.874603,-341.524750,-347.174927, ...
    -352.825073];

delZ1016mm=[-000.502938,-001.529108,-002.589326,-003.694104, ...
    -004.856893,-006.094686,-007.428598,-008.884300,-010.492119, ...
    -012.286569,-014.305046,-016.585499,-019.163105,-022.066351, ...
    -025.313251,-028.908703,-032.843723,-037.096832,-041.637131, ...
    -046.428223,-051.432011,-056.611771,-061.934135,-067.370148, ...
    -072.895531,-078.490440,-084.139015,-089.828743,-095.549858, ...
    -101.294853,-107.057938,-112.834717,-118.621834,-124.416748, ...
    -130.217529,-136.022751,-141.831299,-147.642349,-153.455292, ...
    -159.269653,-165.085068,-170.901291,-176.718124,-182.535400, ...
    -188.353012,-194.170883,-199.988953,-205.807159,-211.625488, ...
    -217.443878,-223.262344,-229.080856,-234.899399,-240.717972, ...
    -246.536560,-252.355164,-258.173767,-263.992401,-269.811035, ...
    -275.629669,-281.448303,-287.266937,-293.085571,-298.904205, ...
    -304.722870,-310.541504,-316.360138,-322.178802,-327.997437, ...
    -333.816101,-339.634735,-345.453369,-351.272034,-357.090668, ...
    -362.909332];

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 15], ...
        'PaperUnits', 'Inches', 'PaperSize', [20, 15]);

% DIC
% from 1D_\ Dic.nc of Gaetan Olivier (UBO).
% renamed 1D_BB_DIC_GE_spring.delZ1015mm.nc.
% 1D_\ Dic.nc is a file containing in situ data of DIC at the Green Edge
% ice camp (67.48N; 63.79W) from mid-May to mid-June 2016.
% ASSUMPTION TO VERIFY: units are mmol C/m^-3).
DICfile='1D_BB_DIC_GE_spring.delZ1015mm.nc';
DIC=ncread(DICfile,'DIC',[2 2 1 1], [1 1 75 1]);
DIC=reshape(DIC,[],1);
subplot(5,4,1);
plot(DIC, delZ1015mm, 'green');
title('DIC');
xlabel('DIC (mmol C/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);
hold on;
h=zeros(3,1);
h(1)=plot(NaN,NaN,'-green');
h(2)=plot(NaN,NaN,'-blue');
h(3)=plot(NaN,NaN,'-black');
legend(h, 'Green Edge', 'previous run', ...
    'World Ocean Atlas', 'Location','southwest');
legend boxoff;

% NH4
var=ncread(previous_run_file,'TRAC02');
var=squeeze(var);
NH4=var(1:ndepths,tenthyearjan1);
subplot(5,4,2);
plot(NH4, Z, 'blue');
title('NH4');
xlabel('NH4 (mmol N/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% NO2
var=ncread(previous_run_file,'TRAC03');
var=squeeze(var);
NO2=var(1:ndepths,tenthyearjan1);
subplot(5,4,3);
plot(NO2, Z, 'blue');
title('NO2');
xlabel('NO2 (mmol N/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% NO3
% from 1D_NO3.nc of Gaetan Olivier (UBO).
% rename 1D_BB_NO3_GE_spring.delZ1015mm.nc
% 1D_NO3.nc is a file containing in situ data of NO3 at the Green Edge
% ice camp (67.48N; 63.79W) from mid-May to mid-June 2016.
% ASSUMPTION TO VERIFY: units are mmol N/m^-3).
NO3file='1D_BB_NO3_GE_spring.delZ1015mm.nc';
NO3=ncread(NO3file,'NO3',[2 2 1 1], [1 1 75 1]);
NO3=reshape(NO3,[],1);
subplot(5,4,4);
plot(NO3, delZ1015mm, 'green');
title('NO3');
xlabel('NO3 (mmol N/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% PO4
% from 1D_PO4.nc of Gaetan Olivier (UBO).
% rename 1D_BB_PO4_GE_spring.delZ1015mm.nc
% 1D_PO4.nc is a file containing in situ data of PO4 at the Green Edge
% ice camp (67.48N; 63.79W) from mid-May to mid-June 2016.
% ASSUMPTION TO VERIFY: units are mmol P/m^-3).
PO4file='1D_BB_PO4_GE_spring.delZ1015mm.nc';
PO4=ncread(PO4file,'PO4',[2 2 1 1], [1 1 75 1]);
PO4=reshape(PO4,[],1);
subplot(5,4,5);
plot(PO4, delZ1015mm, 'green');
title('PO4');
xlabel('PO4 (mmol P/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% SiO2
% from 1D_SiOH4.nc of Gaetan Olivier (UBO).
% rename 1D_BB_Si_GE_spring.delZ1015mm.nc
% 1D_SiOH4.nc is a file containing in situ data of SiO2 at the Green Edge
% ice camp (67.48N; 63.79W) from mid-May to mid-June 2016.
% ASSUMPTION TO VERIFY: units are mmol Si/m^-3).
SiO2file='1D_BB_Si_GE_spring.delZ1015mm.nc';
SiO2=ncread(SiO2file,'SiOH4',[2 2 1 1], [1 1 75 1]);
SiO2=reshape(SiO2,[],1);
subplot(5,4,6);
plot(SiO2, delZ1015mm, 'green');
title('SiO2');
xlabel('SiO2 (mmol Si/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% FeT
var=ncread(previous_run_file,'TRAC07');
var=squeeze(var);
FeT=var(1:ndepths,tenthyearjan1);
subplot(5,4,7);
plot(FeT, Z, 'blue');
title('FeT');
xlabel('FeT (mmol Fe/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% DOC
var=ncread(previous_run_file,'TRAC08');
var=squeeze(var);
DOC=var(1:ndepths,tenthyearjan1);
subplot(5,4,8);
plot(DOC, Z, 'blue');
title('DOC');
xlabel('DOC (mmol C/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% DON
var=ncread(previous_run_file,'TRAC09');
var=squeeze(var);
DON=var(1:ndepths,tenthyearjan1);
subplot(5,4,9);
plot(DON, Z, 'blue');
title('DON');
xlabel('DON (mmol N/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% DOP
var=ncread(previous_run_file,'TRAC10');
var=squeeze(var);
DOP=var(1:ndepths,tenthyearjan1);
subplot(5,4,10);
plot(DOP, Z, 'blue');
title('DOP');
xlabel('DOP (mmol P/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% DOFe
var=ncread(previous_run_file,'TRAC11');
var=squeeze(var);
DOFe=var(1:ndepths,tenthyearjan1);
subplot(5,4,11);
plot(DOFe, Z, 'blue');
title('DOFe');
xlabel('DOFe (mmol Fe/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% POC
var=ncread(previous_run_file,'TRAC12');
var=squeeze(var);
POC=var(1:ndepths,tenthyearjan1);
subplot(5,4,12);
plot(POC, Z, 'blue');
title('POC');
xlabel('POC (mmol C/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% PON
var=ncread(previous_run_file,'TRAC13');
var=squeeze(var);
PON=var(1:ndepths,tenthyearjan1);
subplot(5,4,13);
plot(PON, Z, 'blue');
title('PON');
xlabel('PON (mmol N/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% POP
var=ncread(previous_run_file,'TRAC14');
var=squeeze(var);
POP=var(1:ndepths,tenthyearjan1);
subplot(5,4,14);
plot(POP, Z, 'blue');
title('POP');
xlabel('POP (mmol P/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% POSi
var=ncread(previous_run_file,'TRAC15');
var=squeeze(var);
POSi=var(1:ndepths,tenthyearjan1);
subplot(5,4,15);
plot(POSi, Z, 'blue');
title('POSi');
xlabel('POSi (mmol Si/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% POFe
var=ncread(previous_run_file,'TRAC16');
var=squeeze(var);
POFe=var(1:ndepths,tenthyearjan1);
subplot(5,4,16);
plot(POFe, Z, 'blue');
title('POFe');
xlabel('POFe (mmol Fe/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% PIC
var=ncread(previous_run_file,'TRAC17');
var=squeeze(var);
PIC=var(1:ndepths,tenthyearjan1);
subplot(5,4,17);
plot(PIC, Z, 'blue');
title('PIC');
xlabel('PIC (mmol C/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% ALK
% from 1D_Total_Alkalinity.nc of Gaetan Olivier (UBO).
% rename 1D_BB_ALK_GE_spring.delZ1015mm.nc
% 1D_Total_Alkalinity.nc is a file containing in situ data of alkalinity 
% at the Green Edge ice camp (67.48N; 63.79W) from mid-May to mid-June 2016.
% ASSUMPTION TO VERIFY: units are mmol eq/m^3).
ALKfile='1D_BB_ALK_GE_spring.delZ1015mm.nc';
ALK=ncread(ALKfile,'Total_Alkalinity',[2 2 1 1], [1 1 75 1]);
ALK=reshape(ALK,[],1);
subplot(5,4,18);
plot(ALK, delZ1015mm, 'green');
title('ALK');
xlabel('ALK (mmol eq/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% O2
% from 1D_O2.nc of Gaetan Olivier (UBO).
% rename 1D_BB_O2_GE_spring.delZ1015mm.nc
% 1D_O2.nc is a file containing Levitus data of O2 at the Green Edge
% ice camp (67.48N; 63.79W)
% when?
% ASSUMPTION TO VERIFY: units are mmol O/m^3).
O2file='1D_BB_O2_GE_spring.delZ1015mm.nc';
O2=ncread(O2file,'O2',[2 2 1 1], [1 1 75 1]);
O2=reshape(O2,[],1);
maO=15.999; % atomic mass of O u
rmaO=1/maO;
O2=O2*rmaO*1e3; % mg O L^-1 to mmol O m^-3
subplot(5,4,19);
plot(O2, delZ1015mm, 'black');
title('O2');
xlabel('O2 (mmol O/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% CDOM
var=ncread(previous_run_file,'TRAC20');
var=squeeze(var);
CDOM=var(1:ndepths,tenthyearjan1);
subplot(5,4,20);
plot(CDOM, Z, 'blue');
title('CDOM');
xlabel('CDOM (mmol P/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

saveas(gcf, outfile);

% % phyto biomass
% % from https://github.com/maximebenoitgagne/gud/blob/gud/gud_1d_35%2B16/input_so/loc1_biomass_x120_run83_janprof.bin
% % rename loc1_biomass_x120_run83_janprof.delZ14t10.32bits.bin
% phytofile='loc1_biomass_x120_run83_janprof.delZ14t10.32bits.bin';
% fileID = fopen(phytofile, 'r', 'ieee-be');
% phyto=fread(fileID, 'float32');
% fclose(fileID);
% subplot(5,4,20);
% plot(phyto, delZ14t10, 'blue');
% title('phyto biomass for each species');
% xlabel('phyto biomass (mmol C/m^3)');
% ylabel('Depth (m)');
% ylim([-400,0]);

clf;
outfile='data.ptracers.prok.small.other.diazo.source.png';
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 15], ...
        'PaperUnits', 'Inches', 'PaperSize', [20, 15]);
biomassphyto_total=zeros(ndepths,1);
for itracer=21:34
    varname=num2str(itracer,'TRAC%02d');
    var=ncread(previous_run_file,varname);
    var=squeeze(var);
    biomassphyto_onetype=var(1:ndepths,tenthyearjan1);
    subplot(5,4,itracer-20);
    plot(biomassphyto_onetype, Z, 'blue');
    longname=get_longname(itracer);
    title(longname);
    xlabel('biomass (mmol C/m^3)');
    ylabel('Depth (m)');
    xlim([0,0.08]);
    ylim([-400,0]);
    biomassphyto_total=biomassphyto_total+biomassphyto_onetype;
end % for itracer
saveas(gcf, outfile);

clf;
outfile='data.ptracers.diatoms.dino.source.png';
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 15], ...
        'PaperUnits', 'Inches', 'PaperSize', [20, 15]);
for itracer=35:53
    varname=num2str(itracer,'TRAC%02d');
    var=ncread(previous_run_file,varname);
    var=squeeze(var);
    biomassphyto_onetype=var(1:ndepths,tenthyearjan1);
    subplot(5,4,itracer-34);
    plot(biomassphyto_onetype, Z, 'blue');
    longname=get_longname(itracer);
    title(longname);
    xlabel('biomass (mmol C/m^3)');
    ylabel('Depth (m)');
    xlim([0,0.08]);
    ylim([-400,0]);
    biomassphyto_total=biomassphyto_total+biomassphyto_onetype;
end % for itracer
saveas(gcf, outfile);

clf;
outfile='data.ptracers.zoo.source.png';
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 15], ...
        'PaperUnits', 'Inches', 'PaperSize', [20, 15]);
biomasszoo_total=zeros(ndepths,1);
for itracer=54:69
    varname=num2str(itracer,'TRAC%02d');
    var=ncread(previous_run_file,varname);
    var=squeeze(var);
    biomasszoo_onetype=var(1:ndepths,tenthyearjan1);
    subplot(5,4,itracer-53);
    plot(biomasszoo_onetype, Z, 'blue');
    longname=get_longname(itracer);
    title(longname);
    xlabel('biomass (mmol C/m^3)');
    ylabel('Depth (m)');
    xlim([0,0.08]);
    ylim([-400,0]);
    biomasszoo_total=biomasszoo_total+biomasszoo_onetype;
end % for itracer
saveas(gcf, outfile);

netcdf.close(ncid);

%%%%%%%%%% delZ1016mm %%%%%%%%%%

clf;

outfiledelZ1016mm='data.ptracers.nutrients.delZ1016mm.png';

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 15], ...
        'PaperUnits', 'Inches', 'PaperSize', [20, 15]);

% DIC
DIC=interp1(delZ1015mm,DIC,delZ1016mm,'linear',DIC(end));

subplot(5,4,1);
plot(DIC, delZ1016mm, 'green');
title('DIC');
xlabel('DIC (mmol C/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);
hold on;
h=zeros(3,1);
h(1)=plot(NaN,NaN,'-green');
h(2)=plot(NaN,NaN,'-blue');
h(3)=plot(NaN,NaN,'-black');
legend(h, 'Green Edge', 'previous run', ...
    'World Ocean Atlas', 'Location','southwest');
legend boxoff;

% NH4
subplot(5,4,2);
plot(NH4, delZ1016mm, 'blue');
title('NH4');
xlabel('NH4 (mmol N/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% NO2
subplot(5,4,3);
plot(NO2, delZ1016mm, 'blue');
title('NO2');
xlabel('NO2 (mmol N/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% NO3
NO3=interp1(delZ1015mm,NO3,delZ1016mm,'linear',NO3(end));

subplot(5,4,4);
plot(NO3, delZ1016mm, 'green');
title('NO3');
xlabel('NO3 (mmol N/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% PO4
PO4=interp1(delZ1015mm,PO4,delZ1016mm,'linear',PO4(end));

subplot(5,4,5);
plot(PO4, delZ1016mm, 'green');
title('PO4');
xlabel('PO4 (mmol P/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% SiO2
SiO2=interp1(delZ1015mm,SiO2,delZ1016mm,'linear',SiO2(end));

subplot(5,4,6);
plot(SiO2, delZ1016mm, 'green');
title('SiO2');
xlabel('SiO2 (mmol Si/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% FeT
subplot(5,4,7);
plot(FeT, delZ1016mm, 'blue');
title('FeT');
xlabel('FeT (mmol Fe/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% DOC
subplot(5,4,8);
plot(DOC, delZ1016mm, 'blue');
title('DOC');
xlabel('DOC (mmol C/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% DON
subplot(5,4,9);
plot(DON, delZ1016mm, 'blue');
title('DON');
xlabel('DON (mmol N/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% DOP
subplot(5,4,10);
plot(DOP, delZ1016mm, 'blue');
title('DOP');
xlabel('DOP (mmol N/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% DOFe
subplot(5,4,11);
plot(DOFe, delZ1016mm, 'blue');
title('DOFe');
xlabel('DOFe (mmol Fe/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% POC
subplot(5,4,12);
plot(POC, delZ1016mm, 'blue');
title('POC');
xlabel('POC (mmol C/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% PON
subplot(5,4,13);
plot(PON, delZ1016mm, 'blue');
title('PON');
xlabel('PON (mmol N/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% POP
subplot(5,4,14);
plot(POP, delZ1016mm, 'blue');
title('POP');
xlabel('POP (mmol P/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% POSi
subplot(5,4,15);
plot(POSi, delZ1016mm, 'blue');
title('POSi');
xlabel('POSi (mmol Si/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% POFe
subplot(5,4,16);
plot(POFe, delZ1016mm, 'blue');
title('POFe');
xlabel('POFe (mmol Fe/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% PIC
subplot(5,4,17);
plot(PIC, delZ1016mm, 'blue');
title('PIC');
xlabel('PIC (mmol C/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% ALK
ALK=interp1(delZ1015mm,ALK,delZ1016mm,'linear',ALK(end));

subplot(5,4,18);
plot(ALK, delZ1016mm, 'green');
title('ALK');
xlabel('ALK (mmol eq/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% O2
O2=interp1(delZ1015mm,O2,delZ1016mm,'linear',O2(end));

subplot(5,4,19);
plot(O2, delZ1016mm, 'black');
title('O2');
xlabel('O2 (mmol O/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% CDOM
subplot(5,4,20);
plot(CDOM, delZ1016mm, 'blue');
title('CDOM');
xlabel('CDOM (mmol P/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

saveas(gcf, outfiledelZ1016mm);

%%%%%%%%%% 32 bits %%%%%%%%%%

clf;
outfile32='data.ptracers.nutrients.delZ1016mm.32bits.png';
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 15], ...
        'PaperUnits', 'Inches', 'PaperSize', [20, 15]);

% DIC
DICfile='1D_BB_DIC_GE_spring.delZ1016mm.32bits.bin';
outfileID = fopen(DICfile, 'w');
fwrite(outfileID, DIC, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(DICfile, 'r', 'ieee-be');
DIC=fread(fileID, 'float32');
fclose(fileID);

subplot(5,4,1);
plot(DIC, delZ1016mm, 'green');
title('DIC');
xlabel('DIC (mmol C/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);
hold on;
h=zeros(3,1);
h(1)=plot(NaN,NaN,'-green');
h(2)=plot(NaN,NaN,'-blue');
h(3)=plot(NaN,NaN,'-black');
legend(h, 'Green Edge', 'previous run', ...
    'World Ocean Atlas', 'Location','southwest');
legend boxoff;

% NH4
NH4file=...
'1D_BB_NH4_GE_run_20220511_0000_10thyear_jan01prof.delZ1016mm.32bits.bin';
outfileID = fopen(NH4file, 'w');
fwrite(outfileID, NH4, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(NH4file, 'r', 'ieee-be');
NH4=fread(fileID, 'float32');
fclose(fileID);

subplot(5,4,2);
plot(NH4, Z, 'blue');
title('NH4');
xlabel('NH4 (mmol N/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% NO2
NO2file=...
'1D_BB_NO2_GE_run_20220511_0000_10thyear_jan01prof.delZ1016mm.32bits.bin';
outfileID = fopen(NO2file, 'w');
fwrite(outfileID, NO2, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(NO2file, 'r', 'ieee-be');
NO2=fread(fileID, 'float32');
fclose(fileID);

subplot(5,4,3);
plot(NO2, Z, 'blue');
title('NO2');
xlabel('NO2 (mmol N/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% NO3
NO3file='1D_BB_NO3_GE_spring.delZ1016mm.32bits.bin';
outfileID = fopen(NO3file, 'w');
fwrite(outfileID, NO3, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(NO3file, 'r', 'ieee-be');
NO3=fread(fileID, 'float32');
fclose(fileID);

subplot(5,4,4);
plot(NO3, delZ1016mm, 'green');
title('NO3');
xlabel('NO3 (mmol N/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% PO4
PO4file='1D_BB_PO4_GE_spring.delZ1016mm.32bits.bin';
outfileID = fopen(PO4file, 'w');
fwrite(outfileID, PO4, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(PO4file, 'r', 'ieee-be');
PO4=fread(fileID, 'float32');
fclose(fileID);

subplot(5,4,5);
plot(PO4, delZ1016mm, 'green');
title('PO4');
xlabel('PO4 (mmol P/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% SiO2
SiO2file='1D_BB_Si_GE_spring.delZ1016mm.32bits.bin';
outfileID = fopen(SiO2file, 'w');
fwrite(outfileID, SiO2, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(SiO2file, 'r', 'ieee-be');
SiO2=fread(fileID, 'float32');
fclose(fileID);

subplot(5,4,6);
plot(SiO2, delZ1016mm, 'green');
title('SiO2');
xlabel('SiO2 (mmol Si/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% FeT
FeTfile=...
'1D_BB_FeT_GE_run_20220511_0000_10thyear_jan01prof.delZ1016mm.32bits.bin';
outfileID = fopen(FeTfile, 'w');
fwrite(outfileID, FeT, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(FeTfile, 'r', 'ieee-be');
FeT=fread(fileID, 'float32');
fclose(fileID);

subplot(5,4,7);
plot(FeT, Z, 'blue');
title('FeT');
xlabel('FeT (mmol Fe/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% DOC
DOCfile=...
'1D_BB_DOC_GE_run_20220511_0000_10thyear_jan01prof.delZ1016mm.32bits.bin';
outfileID = fopen(DOCfile, 'w');
fwrite(outfileID, DOC, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(DOCfile, 'r', 'ieee-be');
DOC=fread(fileID, 'float32');
fclose(fileID);

subplot(5,4,8);
plot(DOC, Z, 'blue');
title('DOC');
xlabel('DOC (mmol C/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% DON
DONfile=...
'1D_BB_DON_GE_run_20220511_0000_10thyear_jan01prof.delZ1016mm.32bits.bin';
outfileID = fopen(DONfile, 'w');
fwrite(outfileID, DON, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(DONfile, 'r', 'ieee-be');
DON=fread(fileID, 'float32');
fclose(fileID);

subplot(5,4,9);
plot(DON, Z, 'blue');
title('DON');
xlabel('DON (mmol N/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% DOP
DOPfile=...
'1D_BB_DOP_GE_run_20220511_0000_10thyear_jan01prof.delZ1016mm.32bits.bin';
outfileID = fopen(DOPfile, 'w');
fwrite(outfileID, DOP, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(DOPfile, 'r', 'ieee-be');
DOP=fread(fileID, 'float32');
fclose(fileID);

subplot(5,4,10);
plot(DOP, Z, 'blue');
title('DOP');
xlabel('DOP (mmol P/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% DOFe
DOFefile=...
'1D_BB_DOFe_GE_run_20220511_0000_10thyear_jan01prof.delZ1016mm.32bits.bin';
outfileID = fopen(DOFefile, 'w');
fwrite(outfileID, DOFe, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(DOFefile, 'r', 'ieee-be');
DOFe=fread(fileID, 'float32');
fclose(fileID);

subplot(5,4,11);
plot(DOFe, Z, 'blue');
title('DOFe');
xlabel('DOFe (mmol Fe/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% POC
POCfile=...
'1D_BB_POC_GE_run_20220511_0000_10thyear_jan01prof.delZ1016mm.32bits.bin';
outfileID = fopen(POCfile, 'w');
fwrite(outfileID, POC, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(POCfile, 'r', 'ieee-be');
POC=fread(fileID, 'float32');
fclose(fileID);

subplot(5,4,12);
plot(POC, Z, 'blue');
title('POC');
xlabel('POC (mmol C/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% PON
PONfile=...
'1D_BB_PON_GE_run_20220511_0000_10thyear_jan01prof.delZ1016mm.32bits.bin';
outfileID = fopen(PONfile, 'w');
fwrite(outfileID, PON, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(PONfile, 'r', 'ieee-be');
PON=fread(fileID, 'float32');
fclose(fileID);

subplot(5,4,13);
plot(PON, Z, 'blue');
title('PON');
xlabel('PON (mmol N/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% POP
POPfile=...
'1D_BB_POP_GE_run_20220511_0000_10thyear_jan01prof.delZ1016mm.32bits.bin';
outfileID = fopen(POPfile, 'w');
fwrite(outfileID, POP, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(POPfile, 'r', 'ieee-be');
POP=fread(fileID, 'float32');
fclose(fileID);

subplot(5,4,14);
plot(POP, Z, 'blue');
title('POP');
xlabel('POP (mmol P/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% POSi
POSifile=...
'1D_BB_POSi_GE_run_20220511_0000_10thyear_jan01prof.delZ1016mm.32bits.bin';
outfileID = fopen(POSifile, 'w');
fwrite(outfileID, POSi, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(POSifile, 'r', 'ieee-be');
POSi=fread(fileID, 'float32');
fclose(fileID);

subplot(5,4,15);
plot(POSi, Z, 'blue');
title('POSi');
xlabel('POSi (mmol Si/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% POFe
POFefile=...
'1D_BB_POFe_GE_run_20220511_0000_10thyear_jan01prof.delZ1016mm.32bits.bin';
outfileID = fopen(POFefile, 'w');
fwrite(outfileID, POFe, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(POFefile, 'r', 'ieee-be');
POFe=fread(fileID, 'float32');
fclose(fileID);

subplot(5,4,16);
plot(POFe, Z, 'blue');
title('POFe');
xlabel('POFe (mmol Fe/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% PIC
PICfile=...
'1D_BB_PIC_GE_run_20220511_0000_10thyear_jan01prof.delZ1016mm.32bits.bin';
outfileID = fopen(PICfile, 'w');
fwrite(outfileID, PIC, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(PICfile, 'r', 'ieee-be');
PIC=fread(fileID, 'float32');
fclose(fileID);

subplot(5,4,17);
plot(PIC, Z, 'blue');
title('PIC');
xlabel('PIC (mmol C/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% ALK
ALKfile='1D_BB_ALK_GE_spring.delZ1016mm.32bits.bin';
outfileID = fopen(ALKfile, 'w');
fwrite(outfileID, ALK, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(ALKfile, 'r', 'ieee-be');
ALK=fread(fileID, 'float32');
fclose(fileID);

subplot(5,4,18);
plot(ALK, delZ1016mm, 'green');
title('ALK');
xlabel('ALK (mmol eq/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% O2
O2file='1D_BB_O2_GE_spring.delZ1016mm.32bits.bin';
outfileID = fopen(O2file, 'w');
fwrite(outfileID, O2, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(O2file, 'r', 'ieee-be');
O2=fread(fileID, 'float32');
fclose(fileID);

subplot(5,4,19);
plot(O2, delZ1016mm, 'black');
title('O2');
xlabel('O2 (mmol O/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

% CDOM
CDOMfile=...
'1D_BB_CDOM_GE_run_20220511_0000_10thyear_jan01prof.delZ1016mm.32bits.bin';
outfileID = fopen(CDOMfile, 'w');
fwrite(outfileID, CDOM, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(CDOMfile, 'r', 'ieee-be');
CDOM=fread(fileID, 'float32');
fclose(fileID);

subplot(5,4,20);
plot(CDOM, Z, 'blue');
title('CDOM');
xlabel('CDOM (mmol P/m^3)');
ylabel('Depth (m)');
ylim([-400,0]);

saveas(gcf, outfile32);

clf;
outfile32='data.ptracers.prok.small.other.diazo.32bits.png';
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 15], ...
        'PaperUnits', 'Inches', 'PaperSize', [20, 15]);

biomassprokfile='75zeros.32bits.bin';
zeros75=zeros(75,1);

outfileID = fopen(biomassprokfile, 'w');
fwrite(outfileID, zeros75, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(biomassprokfile, 'r', 'ieee-be');
zeros75=fread(fileID, 'float32');
fclose(fileID);

biomassphytofile=...
'1D_BB_biomassphyto_GE_run_20220511_0000_10thyear_jan01prof.delZ1016mm.32bits.bin';
biomassphyto=biomassphyto_total/26;

outfileID = fopen(biomassphytofile, 'w');
fwrite(outfileID, biomassphyto, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(biomassphytofile, 'r', 'ieee-be');
biomassphyto=fread(fileID, 'float32');
fclose(fileID);

for itracer=21:34
    subplot(5,4,itracer-20);
    if ( (itracer==21) || (itracer==22) || (30<=itracer && itracer <=34) )
        plot(zeros75, Z, '--b');
        longname=get_longname(itracer);
        title(longname);
        xlabel('biomass (mmol C/m^3)');
        ylabel('Depth (m)');
        xlim([0,0.08]);
        ylim([-400,0]);
        if itracer==21
            hold on;
            h=zeros(2,1);
            h(1)=plot(NaN,NaN,'-blue');
            h(2)=plot(NaN,NaN,'--blue');
            legend(h, 'previous run', ...
                'set to zero', 'Location','southwest');
            legend boxoff;
        end % if itracer==21
    else
        plot(biomassphyto, Z, 'blue');
        longname=get_longname(itracer);
        title(longname);
        xlabel('biomass (mmol C/m^3)');
        ylabel('Depth (m)');
        xlim([0,0.08]);
        ylim([-400,0]);
    end % if ( (itracer==21) || (itracer==22) ||
        %      (30<=itracer && itracer <=34) )
end % for itracer
saveas(gcf, outfile32);

clf;
outfile32='data.ptracers.diatoms.dino.32bits.png';
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 15], ...
        'PaperUnits', 'Inches', 'PaperSize', [20, 15]);
for itracer=35:53
    subplot(5,4,itracer-34);
    plot(biomassphyto, Z, 'blue');
    longname=get_longname(itracer);
    title(longname);
    xlabel('biomass (mmol C/m^3)');
    ylabel('Depth (m)');
    xlim([0,0.08]);
    ylim([-400,0]);
end % for itracer
saveas(gcf, outfile32);

clf;
outfile32='data.ptracers.zoo.32bits.png';
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 20, 15], ...
        'PaperUnits', 'Inches', 'PaperSize', [20, 15]);
  
biomasszoofile=...
'1D_BB_biomasszoo_GE_run_20220511_0000_10thyear_jan01prof.delZ1016mm.32bits.bin';
biomasszoo=biomasszoo_total/16;

outfileID = fopen(biomasszoofile, 'w');
fwrite(outfileID, biomasszoo, 'float32', 0, 'ieee-be');
fclose(outfileID);

fileID = fopen(biomasszoofile, 'r', 'ieee-be');
biomasszoo=fread(fileID, 'float32');
fclose(fileID);

for itracer=54:69
    subplot(5,4,itracer-53);
    plot(biomasszoo, Z, 'blue');
    longname=get_longname(itracer);
    title(longname);
    xlabel('biomass (mmol C/m^3)');
    ylabel('Depth (m)');
    xlim([0,0.08]);
    ylim([-400,0]);
end % for itracer
saveas(gcf, outfile32);


function longname=get_longname(itracer)
% Input:
%   itracer Index of the living tracer in data.ptracers.
%           The name of the corresponding variable in the NetCDF file
%           generated by MITgcm is TRAC+itracer.
%           For example, TRAC21 is Prochlorococcus.
%
% Output:
%   longname: The long name of the group plus its equivalent spherical
%             diameter (ESD; um).
%             For example, 'Prochlorococcus 1 um' is Prochlorococcus with
%             an ESD of 1 um.
%             If tracer is less than 21 or greater than 69, it is not a
%             living tracer and the long name will be the empty word.

    if itracer==21
        longname='Prochlorococcus 1 um';
    elseif itracer==22
        longname='Synechococcus 1 um';
    elseif itracer==23
        longname='Small Eukaryotes 1 um';
    elseif itracer==24
        longname='Small Eukaryotes 2 um';
    elseif itracer==25
        longname='Other Eukaryotes 3 um';
    elseif itracer==26
        longname='Other Eukaryotes 4 um';
    elseif itracer==27
        longname='Other Eukaryotes 7 um';
    elseif itracer==28
        longname='Other Eukaryotes 10 um';
    elseif itracer==29
        longname='Other Eukaryotes 15 um';
    elseif itracer==30
        longname='Diazotrophs 3 um';
    elseif itracer==31
        longname='Diazotrophs 4 um';
    elseif itracer==32
        longname='Diazotrophs 7 um';
    elseif itracer==33
        longname='Diazotrophs 10 um';
    elseif itracer==34
        longname='Trichodesmium 15 um';
    elseif itracer==35
        longname='Diatoms 7 um';
    elseif itracer==36
        longname='Diatoms 10 um';
    elseif itracer==37
        longname='Diatoms 15 um';
    elseif itracer==38
        longname='Diatoms 22 um';
    elseif itracer==39
        longname='Diatoms 32 um';
    elseif itracer==40
        longname='Diatoms 47 um';
    elseif itracer==41
        longname='Diatoms 70 um';
    elseif itracer==42
        longname='Diatoms 104 um';
    elseif itracer==43
        longname='Diatoms 154 um';
    elseif itracer==44
        longname='Dinoflagellates 7 um';
    elseif itracer==45
        longname='Dinoflagellates 10 um';
    elseif itracer==46
        longname='Dinoflagellates 15 um';
    elseif itracer==47
        longname='Dinoflagellates 22 um';
    elseif itracer==48
        longname='Dinoflagellates 32 um';
    elseif itracer==49
        longname='Dinoflagellates 47 um';
    elseif itracer==50
        longname='Dinoflagellates 70 um';
    elseif itracer==51
        longname='Dinoflagellates 104 um';
    elseif itracer==52
        longname='Dinoflagellates 154 um';
    elseif itracer==53
        longname='Dinoflagellates 228 um';
    elseif itracer==54
        longname='Zooplankton 7 um';
    elseif itracer==55
        longname='Zooplankton 10 um';
    elseif itracer==56
        longname='Zooplankton 15 um';
    elseif itracer==57
        longname='Zooplankton 22 um';
    elseif itracer==58
        longname='Zooplankton 32 um';
    elseif itracer==59
        longname='Zooplankton 47 um';
    elseif itracer==60
        longname='Zooplankton 70 um';
    elseif itracer==61
        longname='Zooplankton 104 um';
    elseif itracer==62
        longname='Zooplankton 154 um';
    elseif itracer==63
        longname='Zooplankton 228 um';
    elseif itracer==64
        longname='Zooplankton 339 um';
    elseif itracer==65
        longname='Zooplankton 502 um';
    elseif itracer==66
        longname='Zooplankton 744 um';
    elseif itracer==67
        longname='Zooplankton 1103 um';
    elseif itracer==68
        longname='Zooplankton 1636 um';
    elseif itracer==69
        longname='Zooplankton 2425 um';
    else 
        longname='';
    end % if tracer
end % function longname
