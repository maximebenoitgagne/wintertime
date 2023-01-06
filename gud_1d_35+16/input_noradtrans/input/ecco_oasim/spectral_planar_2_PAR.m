function [array2d_ilambda_imonth_eds,array2d_ilambda_imonth_ess, ...
     array2d_ilambda_imonth_ets,array1d_imonth_PAR] ...
     =spectral_planar_2_PAR(array1d_imonth_rmud, ...
     array2d_ilambda_imonth_edp,array2d_ilambda_imonth_esp)
% Compute scalar irradiances and PAR (both in umol photons/m2/s).
%
% ref: https://gud.mit.edu/MITgcm/source/pkg/gud/gud_radtrans_direct.F?v=gud
rmus=1/0.83;
planck = 6.6256e-34;            %Plancks constant J sec
c = 2.998e8;                    %speed of light m/sec
hc = 1.0/(planck*c);
oavo = 1.0/6.023e23;            % 1/Avogadros number
hcoavo = hc*oavo;
wb_center=[400,425,450,475,500,525,550,575,600,625,650,675,700];
WtouEins=zeros(13,1);
for l=1:13
    rlamm = wb_center(l)*1e-9;      %lambda in m
    WtouEins(l) = 1e6*rlamm*hcoavo; %Watts to uEin/s conversion
end
array2d_ilambda_imonth_eds=zeros(13,12);
array2d_ilambda_imonth_ess=zeros(13,12);
array2d_ilambda_imonth_ets=zeros(13,12);
array1d_imonth_PAR=zeros(1,12);

% plot(wb_center,WtouEins)
% title('Scale factor to convert W m^{-2} into \mumol photons m^{-2} s^{-1}')
% xlabel('Wavelength (nm)')
% ylabel('Scale factor (\mumol photons J^{-1})')
% y = 4.57; % ref: https://www.controlledenvironments.org/wp-content/uploads/sites/6/2017/06/Ch01.pdf
% line([400,700],[y,y],'Color','red','LineStyle','--')
% txt='4.57';
% text(385,4.57,txt,'Color','red')
% figure;

for imonth=1:12
    for ilambda=1:13
        edp=array2d_ilambda_imonth_edp(ilambda,imonth);
        rmud=array1d_imonth_rmud(imonth);
        eds=rmud*edp;
        array2d_ilambda_imonth_eds(ilambda,imonth)=eds;
        
        esp=array2d_ilambda_imonth_esp(ilambda,imonth);
        ess=rmus*esp;
        array2d_ilambda_imonth_ess(ilambda,imonth)=ess;
        
        ets=eds+ess;
        array2d_ilambda_imonth_ets(ilambda,imonth)=ets;
        
        array1d_imonth_PAR(imonth) ...
            =array1d_imonth_PAR(imonth)+ets*WtouEins(ilambda);
    end
end