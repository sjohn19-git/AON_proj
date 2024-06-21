#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 13:09:10 2023

@author: sebinjohn
"""
from obspy import UTCDateTime
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import os
import glob
import pyproj
import xarray as xr
import pygmt
from global_land_mask import globe
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature
from tqdm import tqdm
import obspy.signal.cross_correlation as cr
import pandas as pd
import matplotlib.dates as mdates

##########

import pickle
from os.path import isfile, join 
os.chdir("/Users/sebinjohn/AON_PROJECT/Data/codes")
from medfilt import medfilt

#stations 
stati=["A21K","C16K","O14K","H21K"]
#stati=["O14K"]

#load and process the spectra of each station
for sta in stati:
    os.chdir("/Users/sebinjohn/AON_PROJECT/Data/Video Making")
    with open("metadta.pkl","rb") as f:
       long,lat,stationo ,env = pickle.load(f) 
    
    datapath="/Users/sebinjohn/AON_PROJECT/Data/*/"
    st=UTCDateTime(2018,1,1)
    et=UTCDateTime(2021,1,1)
    windw=1
    tim_len=int((et-st)/(3600*windw))
    time_frame=[]
    for i in range (int((et-st)/(3600*windw))):
        time_frame.append(st)
        st=st+(3600*windw)
    st=UTCDateTime(2018,1,1)
    et=UTCDateTime(2021,1,1)
    
    freq=[]
    name= pd.read_xml("pdf0.xml", xpath="/PsdRoot/Psds[1]/Psd[1]/value[@freq]")
    for i in range (95):
        freq.append(name.iloc[i]['freq'])
    freq.append(19.740300000000000)
    
    #loading spectra
    os.chdir(join("/Users/sebinjohn/AON_PROJECT/Data",sta))
    with open((str(sta)+".pkl"),"rb") as f:
       sta, starttimeta, endtimeta,starttimeak,endtimeak,sta,cha,loc = pickle.load(f) 
       j=int((st-starttimeta)/(3600))
       m=int((et-st)/3600)+j
       if j<0:
           print(i)
           pass
       else:
           with open((glob.glob(join(datapath+'final_'+sta+'.npy'))[0]), 'rb') as g:
               final=np.load(g)
           res=final[53:61,j:m]
           mean=np.mean(res,axis=0)
           out_mean_med=medfilt(mean,7)
           interp_inde=np.array([])
           interp_x=out_mean_med.copy()
           datagap=np.where(interp_x==0)[0]
           data_x=np.where(interp_x!=0)[0]
           for i in range(len(data_x)-1):
               if data_x[i+1]-data_x[i]<12 and data_x[i+1]-data_x[i]>1:
                   interp_inde=np.append(interp_inde,np.array( [i for i in range(int(data_x[i])+1,int(data_x[i+1]))]))
               else:
                   continue
           if len(interp_inde)>1:
               interp=np.interp(interp_inde, data_x.reshape(np.shape(data_x)[0]), interp_x[data_x].reshape(np.shape(data_x)[0]))
               interp_inde=(interp_inde+1).astype("int32")
               interp_x[interp_inde]=interp
           else:
               pass
    
    sum_tot=interp_x
    #setting outliers
    sum_tot[sum_tot<-150]=np.nan
    sum_tot[sum_tot>-90]=np.nan
    plt.plot(sum_tot)
    
    
    wave_all=xr.open_dataset("/Users/sebinjohn/AON_PROJECT/Data/wave/wave_2018-2021_fulltime_highlat.nc")
    
    plot_time=[]
    for ele in time_frame:
        plot_time.append(ele.matplotlib_date) 
        
    plot_time=np.array(plot_time)
    
    #cross correlation
    numval=np.array([])
    corre_grid=np.zeros([np.shape(wave_all.wave)[0],np.shape(wave_all.wave)[1]])
    corre_inde=np.zeros(np.shape(corre_grid))
    for i in tqdm(range(np.shape(wave_all.wave)[0])):
        for j in range(np.shape(wave_all.wave)[1]):
            w=wave_all.wave[i,j,:len(time_frame)].copy()
            nanind=np.logical_and(~np.isnan(w),~np.isnan(sum_tot))
            #numval=np.append(numval,np.sum(nanind))
            w=w[nanind]
            s=sum_tot[nanind]
            if len(s)>5000:
                k=cr.correlate(w,s,10)
                corre_grid[i,j]=k[10]
                corre_inde[i,j]=cr.xcorr_max(k)[0]
                if k[10]>6:
                    plt.figure()
                    plt.plot(plot_time[nanind],w)
                    plt.plot(plot_time[nanind],s)
                    plt.show()
            else:
                corre_grid[i,j]=np.nan
            
    
    
    grid_corre_wave = xr.DataArray(
        corre_grid, dims=["lat", "lon"], coords={"lat": wave_all.lat, "lon": wave_all.lon}
        )
    
    #############plotting
    df1=pd.read_csv("/Users/sebinjohn/AON_PROJECT/Data/station Meta Data.csv")
    stat=sta
    lat_sta=df1[df1["Station\xa0"]==stat]["Latitude\xa0"]
    lon_sta=df1[df1["Station\xa0"]==stat]["Longitude\xa0"]
    
    xmw=wave_all.lon
    ymw=wave_all.lat
    xf,yf=np.meshgrid(xmw,ymw)
    xf=xf.flatten()
    yf=yf.flatten()
    ww=grid_corre_wave.data.flatten()
    
    os.chdir("/Users/sebinjohn/AON_PROJECT/Final_codes_paper/")
    grid_corre_wave.data[grid_corre_wave.data==0]=np.nan
    #cmp=pygmt.makecpt(cmap="rainbow",output="hot1.cpt", series=[0,1,0.01,],reverse=True)
    cmp=pygmt.makecpt(cmap="hot",output="hot4.cpt", series=[0.5,0.9,0.01,],reverse=True)
    grid1 = '/Users/sebinjohn/AON_PROJECT/Final_codes_paper/grid2.nc'
    pr="L-159/35/33/45/10c"
    fig=pygmt.Figure()
    pygmt.config(MAP_FRAME_TYPE="plain",MAP_FRAME_PEN="1p,black",FONT_LABEL="10p")
    reg="181/48/245/75r"
    #reg=[148,230,40, 85]
    fig.grdimage(grid=grid_corre_wave,region=reg,projection=pr, cmap="hot5.cpt",nan_transparent=True)
    #fig.colorbar(cmap="hot1.cpt",frame='af+l"correlation coefficient"',projection=pr) 
    #fig.colorbar(cmap="hot4.cpt",frame='af+l"correlation coefficient"',projection=pr)
    fig.contour(x=xf,y=yf,z=ww,region=reg,projection=pr,pen="0.5p,black,dashed",label_placement="d7c",levels=[0.65],annotation="0.65,+f6p+r500")
    fig.grdimage(grid=grid1,region=reg,projection=pr, cmap="topo.cpt",nan_transparent=True)
    fig.coast(region=reg, projection=pr,shorelines="0.25,black,solid",borders=["1/0.2p,black", "2/0.5p,grey"],area_thresh='600')
    fig.plot(x=lon_sta, y=lat_sta, style="t0.6c", pen="1p", fill="red")
    #fig.text(text=stat,x=lon_sta-4,y=lat_sta,font="10p")
    if sta=="H21K":
        fig.basemap(region=reg,projection=pr,map_scale="g175/68+c68+w300k+l")
    fig.show()
    fig.savefig("/Users/sebinjohn/Downloads/"+stat+".pdf",dpi=600)
    




