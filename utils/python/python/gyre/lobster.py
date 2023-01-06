import os
try:
    from gluemnc.pupynere import netcdf_file
except:
    from scipy.io.netcdf import netcdf_file

root = os.path.join(os.environ['HOME'], 'gyre')
ncdir = os.path.join(root, 'data', 'lobster', 'nc')
B9gridTfile = netcdf_file(ncdir+'/grid_T.nc')
# time_counter time
# nav_lon      longitude
# nav_lat      latitude
# deptht       model_level_number
#
# sobowlin     Bowl Index
# sohefldo     Net Downward Heat Flux
# sohefldp     Surface Heat Flux: Damping
# soicecov     Ice fraction
# somixhgt     Turbocline Depth
# somxl010     Mixed Layer Depth 0.01
# sosafldp     Surface salt flux: damping
# sosalflx     Surface Salt Flux
# sosaline     Sea Surface Salinity
# soshfldo     Shortwave Radiation
# sossheig     Sea Surface Height
# sosstsst     Sea Surface temperature
# sowaflcd     concentration/dilution water flux
# sowafldp     Surface Water Flux: Damping
# sowaflup     Net Upward Water Flux
# vosaline     Salinity
# votemper     Temperature

B9 = dict(B9gridTfile.variables)
B9['nav_lat_t'] = B9['nav_lat']
B9['nav_lon_t'] = B9['nav_lon']

B9gridUfile = netcdf_file(ncdir+'/grid_U.nc')
# sozotaux     Wind Stress along i-axis
# vozocrtx     Zonal Current
B9.update(B9gridUfile.variables)
B9['nav_lat_u'] = B9['nav_lat']
B9['nav_lon_u'] = B9['nav_lon']

B9gridVfile = netcdf_file(ncdir+'/grid_V.nc')
# sometauy     Wind Stress along j-axis
# vomecrty     Meridional Current
B9.update(B9gridVfile.variables)
B9['nav_lat_v'] = B9['nav_lat']
B9['nav_lon_v'] = B9['nav_lon']

B9gridWfile = netcdf_file(ncdir+'/grid_W.nc')
# vobn2avt     Vertical Eddy Diffusivity times BV frequency
# vologavt     Log of Vertical Eddy Diffusivity
# votkeavt     Vertical Eddy Diffusivity
# votkeevd     Enhanced Vertical Diffusivity
# vovecrtz     Vertical Velocity
B9.update(B9gridWfile.variables)
B9['nav_lat_w'] = B9['nav_lat']
B9['nav_lon_w'] = B9['nav_lon']

del B9['nav_lat']
del B9['nav_lon']

B9names = dict((k,v.standard_name) for k,v in B9.items())

