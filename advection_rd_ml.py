# ----------------------------------------------------------
# Calculate horizontal advection for mixed layer heat budget
# Shay Magahey, 03/26/2024, UCSB
# ML heat budget equation based on Graham et al 2014
# Calculates ubar*delT', ekman, and horizontal eddy terms (terms 2-4 in the Graham eqn)
# -----------------------------------------------------------

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import dask
import numpy.ma as ma
import datetime
import cftime

def advection_rd_ml(sal,lat,lon,time,Uh,Vh,st,end):
    # probably need to isolate Uh, Vh over ML before inputting
    # 1. Calculate horizontal tracer gradient
    RE=6378000
    latarr=lat*np.pi/180.;
    lonarr=lon*np.pi/180.;

    dSdx=sal.differentiate('lon',edge_order=1)*np.cos(lonarr)/lonarr.differentiate("lon",edge_order=1)
    dSdy=sal.differentiate("lat",edge_order=1)/latarr.differentiate("lat",edge_order=1)

    # Convert units by dividing by radius of Earth
    dSdx=dSdx/RE
    dSdy=dSdy/RE

    #------------------------------------------------------------
    # 2. Get climatological mean salinity and currents
    #------------------------------------------------------------
    # from advection_ml_rd.m
    # monthly grouped means of currents and salt

    Sm=sal.groupby('time.month').mean()
    Um=Uh.groupby('time.month').mean()
    Vm=Vh.groupby('time.month').mean()

    #------------------------------------------------------
    # 3. Gradient of climatological mean salinity
    #------------------------------------------------------
    dS=Sm.differentiate('lon')
    dStmp=dS*np.cos(latarr.isel(lat=slice(0,-1)))/(lonarr.differentiate('lon'))
    dSmdx=0.5*(dStmp.isel(lon=slice(0,-1))+dStmp.isel(lon=slice(1,None)))/RE

    dStmp=(Sm.differentiate('lat'))/(latarr.differentiate('lat'))
    dSmdy=0.5*(dStmp.isel(lat=slice(0,-1))+dStmp.isel(lat=slice(1,None)))/RE

    arrU=Um*dSmdx
    arrV=Vm*dSmdy
    mn_adv_clim_u=xr.DataArray(np.tile(arrU, (1, 1, 1)),coords=arrU.coords, dims=arrU.dims) # all nans yuck 
    mn_adv_clim_v=xr.DataArray(np.tile(arrV, (1, 1, 1)),coords=arrV.coords, dims=arrV.dims)
    # I actually don't know where this is used lol

    #-------------------------------------------------------
    # 4. Anomalous advection of climatological mean gradient
    #-------------------------------------------------------
    # from advection_ml_rd.m
    anomadv_U=(Uh.groupby('time.month')-Um)*dSmdx
    anomadv_V=(Vh.groupby('time.month')-Vm)*dSmdy 
    # u' * dTbar/dx + v'*dTbar/dy
    ekman=anomadv_U + anomadv_V  # dims are fine

    #-------------------------------------------------------
    # 5 Climatological mean advection of anomalous gradient
    #-------------------------------------------------------
    # from advection_ml_rd.m

    climadv_U=Um*(dSdx.groupby('time.month') - dSmdx)
    climadv_V=Vm*(dSdy.groupby('time.month') - dSmdy)

    T1= climadv_U + climadv_V
    dSdx=dSdx.convert_calendar("proleptic_gregorian",use_cftime=False)
    dSdy=dSdy.convert_calendar("proleptic_gregorian",use_cftime=False)

    #-------------------------------------------------------
    # 6 Anomalous advection of anomalous gradient
    #-------------------------------------------------------
    

    anomadv_anomU=(Uh.groupby('time.month')-Um)*(dSdx.groupby('time.month') - dSmdx)
    anomadv_anomV=(Vh.groupby('time.month')-Vm)*(dSdy.groupby('time.month') - dSmdy)
    # overall value minus mean gives anomaly

    eddy1=anomadv_anomU + anomadv_anomV
    mneddy=eddy1.mean('time')
    h_eddy=eddy1+mneddy

    #-------------------------------------------------------
    # 7. Convert times to seconds to make python not angry about time derivative
    #-------------------------------------------------------
    
    ti = time.values  # Extract the cftime.datetime object

    # Define the reference time
    reference_time = cftime.datetime(1850, 1, 1, calendar="noleap")

    # Calculate the seconds since the reference time for each time point
    seconds_since_epoch = np.array([(t - reference_time).total_seconds() for t in ti])

    # Convert seconds to float32
    time_32 = xr.DataArray(np.float32(seconds_since_epoch))
    time32 = time_32.rename({'dim_0':'time'})
    time32=time32[st:end]

    #-------------------------------------------------------
    # 8. Time Derivative
    #-------------------------------------------------------

    num=sal.differentiate('time')
    den=time32.differentiate('time')
    dStmp=num/den
    dSpdt=0.5*(dStmp.isel(time=slice(0,-1))+dStmp.isel(time=slice(1,None)))

    # nowhere for averaging over lev? Need?
    return mn_adv_clim_u, mn_adv_clim_v, ekman, T1, h_eddy, dSpdt


