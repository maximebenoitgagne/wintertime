istart=1;
close all


if (istart==1),
 clear all

% setup for 35+16 sizes and function
 nphyt=35; nzoo=16; nplank=nphyt+nzoo;
 tsize=[1 2 3:4  5:9  5:9 5:15 7:16];
 ptype=[1 2 3 3 4 4 4 4 4 5 5  5 5 5 6 6 6 6 6  6 6 6 6 6 6 7 7 7  7 7 7 7 7 7 7];

 logvolexp=[-0.6900:.513:-0.6900+17*.513];
 logvol=(10*ones(size(logvolexp))).^logvolexp;
 logdm=2*((3/4*logvol/pi).^(1/3));

% directory pathway for diagnostic results
% dirstr on server
%  dirstr=['../run_noradtrans/diags_20181009_0001'];
% dirstr on Maxime's MacBook Pro
 dirstr=['../../../../../output/gud/run_noradtrans_20181019_0001_first_run/diags_20181019_0001'];
 tstr='0000000000.t001';
% averaging period
 avetim=259200;

% read in the model output
 for idiag=1:4,
   if (idiag==1), ncid1=netcdf.open([dirstr,'/car.',tstr,'.nc']); end
   if (idiag==2), ncid1=netcdf.open([dirstr,'/chl.',tstr,'.nc']); end
   if (idiag==3), ncid1=netcdf.open([dirstr,'/rates.',tstr,'.nc']); end
   if (idiag==3), ncid1=netcdf.open([dirstr,'/grid.t001.nc']); end
     [ndim,nvar,natt,unlim]=netcdf.inq(ncid1);

     for i=0:ndim-1,
      [dimname, dimlength]=netcdf.inqDim(ncid1,i);
      diml(i+1)=dimlength;
     end % for i

     for i=0:nvar-1;

      [varname, xtype,dimid,natt]=netcdf.inqVar(ncid1,i);
      if (length(dimid)==1),
        eval([varname,'=netcdf.getVar(ncid1,i,0,diml(dimid+1));']);
      end % if
      if (length(dimid)==2),
        eval([varname,'=netcdf.getVar(ncid1,i,[0 0],[diml(dimid(1)+1) diml(dimid(2)+1)] );']);
      end % if
      if (length(dimid)==3),
        eval([varname,'=netcdf.getVar(ncid1,i,[0 0 0],[diml(dimid(1)+1) diml(dimid(2)+1) diml(dimid(3)+1)] );']);
      end % if
      if (length(dimid)==4),
         eval([varname,'=netcdf.getVar(ncid1,i,[0 0 0 0],[diml(dimid(1)+1) diml(dimid(2)+1) diml(dimid(3)+1)  diml(dimid(4)+1)] );']);
       end % if
        eval([varname,'=double(',varname,');']);
     end % for i
  end % for idiag

      clear mstr
       mstr=textread(['diag_list_rad_car.data'],'%c','delimiter','\n','whitespace','');
       for ip=1:nplank
         str=mstr(6*(ip-1)+1:6*ip)';
         eval(['car(',num2str(ip),',:,:)=',str,';']);
       end % for ip
% 
       clear mstr
         mstr=textread(['diag_list_rad_chl_noquota.data'],'%c','delimiter','\n','whitespace','');
         for ip=1:nphyt
          str=mstr(6*(ip-1)+1:6*ip)';
          eval(['chl(',num2str(ip),',:,:)=',str,';']);
         end % for ip
  tim=squeeze(T/avetim);
end % istart


%%%% PLOT RESULTS %%%%%%%%%%
kn1=1; kn2=24;
cax1=0; cax2=1;
for igr=1:5,
 if (igr==1), npl=[1 2 3 4]; tstr='pico'; end
 if (igr==2), npl=[5 6 7 8 9]; tstr='cocco'; end
 if (igr==3), npl=[10 11 12 13 14]; tstr='diaz'; end
 if (igr==4), npl=[15 16 17 18 19 20]; tstr='diatom'; end
 if (igr==5), npl=[25 26 27 28 29 30]; tstr='dino'; end
 ip1=0;
 figure
 colormap(fake_parula())
 for ip=npl, %nplank
   ip1=ip1+1;
   clear tmp, tmp=squeeze(car(ip,kn1:kn2,:));
   subplot(6,1,ip1), 
   pcolor(tim,squeeze(Z(kn1:kn2)), tmp); shading flat, 
   caxis([cax1 cax2]); colorbar
   if (ip1==1), title(tstr); end
 end % for ip
end % for igr
