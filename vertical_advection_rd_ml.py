#----------------------------------------------------------
# Calculate vertical advection for mixed layer heat budget
# Shay Magahey, 03/26/2024, UCSB
# ML heat budget equation based on Graham et al 2014
# Calculates upwelling, thermocline, and vertical eddy terms (terms 5-7 in the Graham eqn)
#-----------------------------------------------------------
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import dask
import numpy.ma as ma
import datetime
import cftime

def vertical_advection_rd_ml(MLD,lat,lon,int_T,int_U,int_V,sub_int_W,sub_int_T):
    #INputs:
    # MLD: make sure right size
    # lat, lon
    # int_U, int_V: make sure right size
    # sub_int_W
    # int_T, sub_int_T
    # 
    #---------------------------------
    # 1. Compute entrainment velocity
    #----------------------------------
    
    RE=6378000
    latarr=lat*np.pi/180.;
    lonarr=lon*np.pi/180.;

    dHtmp=MLD.differentiate('time')
    changeH_changetime=0.5*(dHtmp.isel(time=slice(0,-1))+dHtmp.isel(time=slice(1,None)))


    #---------------------------------
    # 2. Time, horizontal derivatives of MLD
    #----------------------------------

    dH=MLD.differentiate('time')
    dHtmp=(dH*np.cos(latarr.isel(lat=slice(0,None))))/(lonarr.differentiate('lon'))
    dHdx=0.5*(dHtmp.isel(lon=slice(0,None))+dHtmp.isel(lon=slice(1,None)))/RE

    dHtmp=MLD.differentiate('lat')/(latarr.differentiate('lat'))
    dHdy=0.5*(dHtmp.isel(lat=slice(0,None))+dHtmp.isel(lat=slice(1,None)))/RE


    #---------------------------------
    # 3. Entrainment velocity
    #----------------------------------

    dHdX=(dHdx.convert_calendar("proleptic_gregorian",use_cftime=False)).transpose('member','time','lat','lon')
    dHdY=(dHdy.convert_calendar("proleptic_gregorian",use_cftime=False)).transpose('member','time','lat','lon')
    changeH_change_time=(changeH_changetime.convert_calendar("proleptic_gregorian",use_cftime=False)).transpose('member','time','lat','lon')
    w_entr=changeH_change_time/86400 + int_U*dHdX+ int_V*dHdY + sub_int_W
    # do you need to divide changeH_change_time by 86400 like is done in the matlab code?

    #---------------------------------
    # 4. Apply reynolds decomp to entrainment velocity,cross-MLD temperature difference
    #----------------------------------

    MLD=MLD.convert_calendar("proleptic_gregorian",use_cftime=False)
    dTdz=(int_T-sub_int_T)/MLD; # Tmld, Tsub inputs


    #---------------------------------
    # 5. Compute climatological means
    #----------------------------------

    # Climatological mean vertical temperature gradient
    Tm_mld=int_T.groupby('time.month').mean()
    Tm_sub=sub_int_T.groupby('time.month').mean()

    # climatological mean entrainmenrt velocity
    wm=w_entr.groupby('time.month').mean()

    # climatological mean temperature difference
    dTm=Tm_mld-Tm_sub
    




    #---------------------------------
    # 6. Heaviside function
    #----------------------------------

    wsgn=wm
    wsgn = wsgn.where(wsgn > 0, 0) # wsgn > 0 is true, wsgn < 0 is false, replace false w 0
    wsgn = wsgn.where(wsgn == 0, 1) # wsgn=0 is true, not zero is false, replace false with 1
    wsgn= wsgn.where(wsgn !=0, np.nan) # wsgn != is true, so 0 is false, replace false with nan
                     
    #wsgn[wsgn==0]=np.nan

    #---------------------------------
    # 7. Calculate various advection terms
    #----------------------------------

    mn_w_dTdz=(wsgn*wm*dTm)/MLD
    wtmp= w_entr-wm # value seems fine
    # anomaly of vertical velocity = wtmp
    
    #print("dTm=",dTm) #also fine
    #print("dTm max=",dTm.max())
    #print("dTm min=",dTm.min())
    #print("max wm=",wm.max())
    #print(wsgn)
    print("tm_mld shape=",Tm_mld.shape)
    print("wtmp shape=", wtmp.shape)
    
    upwelling=wsgn*(wtmp*(Tm_mld-Tm_sub))/MLD # upwelling # wtmp is anomaly? of w?
    # upwelling: anomaly w times mean MLD temp minus mean sub-MLD temp divided by MLD
    tmp1=int_T-Tm_mld # T' MLD
    tmp2=sub_int_T-Tm_sub # T' sub MLD
    # integrated temp anomalies in ML minus temp anomalies below mixed layer: tmp1-tmp2

    thermo=(wsgn*wm*(tmp1-tmp2))/MLD # thermocline

    eddy=(wsgn*wtmp*(tmp1-tmp2))/MLD # eddy

    updTpbar=eddy.groupby('time.month').mean()

    
    return upwelling,thermo,eddy,updTpbar





