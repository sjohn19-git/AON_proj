#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 20:01:01 2023

@author: sebinjohn
"""
import os 
os.chdir("/Users/sebinjohn/AON_PROJECT/Data/codes")
import pygmt
#import dill
import pandas as pd
from obspy import UTCDateTime
from os.path import isfile, join 
import glob
from scipy.integrate import trapz 
import numpy as np
import pickle
import pygrib
import xarray as xr
import matplotlib.pyplot as plt
import cdsapi
from medfilt import medfilt
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
import matplotlib.dates as mdates
from shapely.geometry import Point, Polygon
import math 
import obspy.signal.cross_correlation as cr
from scipy.stats import gaussian_kde
from matplotlib.ticker import MultipleLocator


#loading station metadata
os.chdir("/Users/sebinjohn/AON_PROJECT/Data/Video Making")
with open("metadta.pkl","rb") as f:
   long,lat,stationo ,env = pickle.load(f) 


#defining different regions
regions=[[(-151.95,70.3),(-151.6,66.5),(-141.6,66.5),(-141.88,69.4),"North east"],[(-165.9,70),(-166.1,64.6),(-154.3,65.1),(-153.88,70),"North west"],[(-157.1,59),(-152.8,57),(-168.4,52.3),(-170.4,54.2),"South west"],
         [(-140,62),(-140.5,59.5),(-131.5,55),(-129,56),"South east"],[(-166.322,62.951),(-153.788,63.587),[-153.820,60.768],(-163.582,59.649),"Central west"],[(-149.853,64.549),(-141.512,64.760),(-142.462,60.537),(-149.370,60.446),"Central east"]]

#defining time_frames and frequencies
datapath="/Users/sebinjohn/AON_PROJECT/Data/*/"
st=UTCDateTime(2018,1,1)
et=UTCDateTime(2021,1,1)
windw=1
tim_len=int((et-st)/(3600*windw))
time_frame=[]
for i in range (int((et-st)/(3600*windw))):
    time_frame.append(st)
    st=st+(3600*windw)

freq=[]
name= pd.read_xml("pdf0.xml", xpath="/PsdRoot/Psds[1]/Psd[1]/value[@freq]")
for i in range (95):
    freq.append(name.iloc[i]['freq'])
freq.append(19.740300000000000)



#####calculating mean SM for a specific region

os.chdir("/Users/sebinjohn/AON_PROJECT/Data/median_Power_time_series")
os.listdir()
median_timeseries=np.load("./median_inerpolated_time_series_secondary2018-01-01-2021-01-01.npy")


def sort_counterclockwise(points, centre = None):
    '''This function sorts a set of points of a polygon in counter 
    clockwise direction.
    :param points: list of points'''
    if centre:
      centre_x, centre_y = centre
    else:
      centre_x, centre_y = sum([x for x,_ in points])/len(points), sum([y for _,y in points])/len(points)
    angles = [math.atan2(y - centre_y, x - centre_x) for x,y in points]
    counterclockwise_indices = sorted(range(len(points)), key=lambda i: angles[i])
    counterclockwise_points = [points[i] for i in counterclockwise_indices]
    return counterclockwise_points


def poly_seis(median_timeseries,long,lat,stationo,*args): 
    ''' This function creates a polygon from set of points
    and checks if each station is within the polygon
    and calculates the mean SM if yes.
    :param median_timeseries: Processed SM timeseries of all station.
    :param long: longitude of corresponding stations
    :param lat: latitude of corresponding stations
    :param stationo: list of station names
    :param *args: expects points
    
    '''
    coords,stas=[],[]
    for point in args:
        coords.append(point)
    counter=sort_counterclockwise(coords, centre = None)
    poly = Polygon(counter)
    wista=[]
    for i in range(np.shape(median_timeseries)[0]):
        p1 = Point(float(long[i]),float(lat[i]))
        if p1.within(poly):
            wista.append(1)
        else:
            wista.append(0)
    c=wista.count(1)
    j=0
    sum_array=np.zeros([c,len(median_timeseries[0,:])])
    for i in range(len(wista)):
        if wista[i]==1:
            stas.append(stationo[i])
            sum_array[j,:]=median_timeseries[i,:]
            j+=1
    sum_tot=np.empty([len(median_timeseries[0,:])])
    for i in range(len(sum_array[0,:])):
        if (c-np.count_nonzero(sum_array[:,i]==0))==0: #if all station is zero at an instant
            sum_tot[i]=0
        else: #average of nonzero station psds
            sum_tot[i]=np.sum(sum_array[:,i])/(c-np.count_nonzero(sum_array[:,i]==0))
    return wista,poly,counter,sum_array,sum_tot,stas


#example usage
wista,poly,coords,sum_array,sum_tot,stas=poly_seis(median_timeseries,long,lat,stationo,(-140,62),(-140.5,59.5),(-131.5,55),(-129,56)) 





######wave__data
#loads significant data
def wave(tim):
    '''This function downloads significant wave height dataset
    from ERA5 and returns the corresponding grid and data.
    
    :param tim: time to download signficant waveheight (obspy.UTCDateTime)'''
    c = cdsapi.Client()
    os.chdir("/Users/sebinjohn/AON_PROJECT/Data/wave")
    files=os.listdir()
    if str(tim)[0:14]+"00"+".grib" not in files:
        print(str(tim)[0:14]+"00"+".grib not found Downloading...")
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': 'significant_height_of_combined_wind_waves_and_swell',
                'year': str(tim.year),
                'month': str(tim.month),
                'day': str(tim.day),
                'time': tim.ctime()[11:14]+"00",
                'format': 'grib',
                },
            str(tim)[0:14]+"00"+".grib")
    tim_wav=str(tim)[0:14]+"00"
    grib=str(tim)[0:14]+"00"+".grib"
    grbs=pygrib.open(grib)
    grb=grbs[1]
    data_wave = grb.values
    latg, long = grb.latlons()
    lat=(latg[:,0])
    lon=(long[0,:])
    grid = xr.DataArray(
        data_wave, dims=["lat", "lon"], coords={"lat": lat, "lon": lon}
        )
    return grid,data_wave

def wave_array(grid,*args):
    '''This function subsets the significant wave height grid
    to Alaska region for faster processing 
    :param grid: significant wave height grid
    :param *args: expects points of bounding box'''
    #args=[(wlon1,wlat1),(wlon1,wlat2),(wlon2,wlat1),(wlon2,wlat2)]
    coordsw=[]
    for point in args:
        coordsw.append((point[0]+360,point[1]))
    lons=[]
    lats=[]
    for point in args:
        if point[0]<0:
            lons.append(point[0]+360)
        else:
            lons.append(point[0])
        lats.append(point[1])
    counterw=sort_counterclockwise(coordsw, centre = None)
    ilat1=np.where(grid.lat==min(lats))[0][0]
    ilat2=np.where(grid.lat==max(lats))[0][0]
    ilon1=np.where(grid.lon==min(lons))[0][0]
    ilon2=np.where(grid.lon==max(lons))[0][0]
    lat=grid.lat[ilat2:ilat1]
    lon=grid.lon[ilon1:ilon2]
    wave_arr=grid[ilat2:ilat1,ilon1:ilon2]
    return wave_arr,counterw,lat,lon

###################################
c=0
i=0
for i in range(len(time_frame)):
    tim=time_frame[i]
    print("searching for wave height for"+str(tim))
    grid,data_wave=wave(tim)
    #sum_w=wave_val(grid,wlat1,wlat2,wlon1,wlon2)
    wave_arr,counterw,latw,lon=wave_array(grid,(150,30),(260,73),(260,30),(150,73))
    a,b=np.shape(wave_arr.data)
    if c==0:
        wave_all=np.zeros([a,b,len(time_frame)])
        c=1
    wave_all[:,:,i]=wave_arr.data

wave_all=np.load("/Users/sebinjohn/AON_PROJECT/Data/wave/wave_all_for cross_correlation_2018-2021_3yr.npy")

#############cross correlating####
def sr(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)

##finding mean wave heights for contouring

wvc=np.load("/Users/sebinjohn/AON_PROJECT/Data/wave/wave_all_for cross_correlation_2018-2021_3yr.npy")
k=np.nanmean(wvc,axis=2)
k[k==0]=np.nan
grid_wave = xr.DataArray(
    k, dims=["lat", "lon"], coords={"lat": latw, "lon": lon}
    )
xmw=grid_wave.lon
ymw=grid_wave.lat
xf,yf=np.meshgrid(xmw,ymw)
xf=xf.flatten()
yf=yf.flatten()
wf=grid_wave.data.flatten()
cm=pygmt.makecpt(cmap="./expll1.cpt",series=[2,4,0.2],output="cntrwave.cpt")

for i in range(len(regions)):
    ll=i
    ps=regions[i]
    wista,poly,coords,sum_array,sum_tot,stas=poly_seis(median_timeseries,long,lat,stationo,ps[0],ps[1],ps[2],ps[3]) 
    print(stas,ps)
    corre_grid=np.zeros([np.shape(wave_all)[0],np.shape(wave_all)[1]])
    corre_inde=np.zeros(np.shape(corre_grid))
    #cross-correlating
    for i in range(np.shape(wave_all)[0]):
        for j in range(np.shape(wave_all)[1]):
            nanind=np.where(~np.isnan(wave_all[i,j,:]))[0]
            w=wave_all[i,j,:][nanind]
            s=sum_tot[nanind]
            k=cr.correlate(w,s,10)
            corre_grid[i,j]=k[10]
            corre_inde[i,j]=cr.xcorr_max(k)[0]
    grid_corre = xr.DataArray(
        corre_grid, dims=["lat", "lon"], coords={"lat": latw, "lon": lon}
        )
    ######calculating mean wave height in primary source region
    wave_all[np.isnan(wave_all)]=0
    grid_corre.shape
    wave_all.shape
    positive_vals=grid_corre>0.65
    index1=np.asarray(positive_vals).nonzero()[0]
    index2=np.asarray(positive_vals).nonzero()[1]
    wave_sumpixels=np.zeros(np.shape(wave_all)[-1])
    for i in range(len(index1)):
        wave_sumpixels=wave_sumpixels+(wave_all[index1[i],index2[i],:].flatten())
    wave_sumpixels=wave_sumpixels/len(index1)
    ####plotting for fig dd.a 
    
    xy = np.vstack([wave_sumpixels,sum_tot[:]])
    z = gaussian_kde(xy)(xy)
    
    fig=plt.figure(dpi=600)
    fig.set_figwidth(3.7)
    fig.set_figheight(2.8)
    plt.scatter(wave_sumpixels,sum_tot[:],s=0.1,c=z,marker="o",alpha=0.7)
    plt.scatter(wave_sumpixels[8370],sum_tot[8370],s=20,marker="o",c="red")
    plt.scatter(wave_sumpixels[8380],sum_tot[8380],s=20,marker="o",c="red")
    plt.xlabel("significant wave height (m)")
    plt.ylim([-147,-108])
    plt.xlim([0.5,8])
    plt.ylabel('dB(rel. 1 (m/s'+sr('2')+')'+sr('2')+'/Hz)',fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(base=20))
    #plt.title(ps[-1],fontsize=10)
    plt.tight_layout() 
    plt.show()
    fig.savefig("/Users/sebinjohn/Downloads/"+ps[-1]+".png",transparent=True)
    
    xre=[]
    for ele in coords:
        xre.append(ele[0])
    xre.append(xre[0])
    yre=[]
    for ele in coords:
        yre.append(ele[1])
    yre.append(yre[0])
    #plotting correlation grids
    grid_corre.data[grid_corre.data==0]=np.nan
    pygmt.makecpt(cmap="hot",output="hot1.cpt", series=[0.5,0.9,0.01,],reverse=True)
    grid1 = '/Users/sebinjohn/AON_PROJECT/Final_codes_paper/grid3.nc'
    fig=pygmt.Figure()
    pygmt.config(MAP_FRAME_TYPE="plain",MAP_FRAME_PEN="0p,white",FONT_LABEL="10p")
    os.chdir("/Users/sebinjohn/AON_PROJECT/Final_codes_paper/")
    pr="L-159/35/33/45/10c"
    reg="175/30/265/68r"
    fig.grdimage(grid=grid_corre,region=reg,projection=pr, cmap="hot5.cpt",nan_transparent=True)
    fig.grdimage(grid=grid1,region=reg,projection=pr, cmap="topo.cpt",nan_transparent=True)
    if ll==3:
        fig.basemap(region=reg,projection=pr,map_scale="g162/65+c68+w500k+l")
    fig.plot(x=xre, y=yre, pen="0.8p,Black")
    fig.plot(x=xre,y=yre,pen="0.8p,Black",fill="241/171/122@50",transparency=50)
    #fig.text(text=ps[-1],x=-160,y=73.5,projection=pr)
    ww=grid_corre.data.flatten()
    fig.contour(x=xf,y=yf,z=ww,region=reg,projection=pr,pen="0.5p,black,dashed",label_placement="d7c",levels=[0.65],annotation="0.65,+f6p+r500")
    fig.contour(x=xf,y=yf,z=wf,region=reg,projection=pr,pen="0.1p,53/99/160",label_placement="d7c",annotation="2.8,2.5,3,3.2+f6p")
    #ig.contour(x=xf,y=yf,z=wf,region=reg,projection=pr,pen="0.1p,black",label_placement="d7c",levels="cntr.txt")
    fig.coast(region=reg, projection=pr,shorelines="0.15p,black,solid",borders=["1/0.2p,black", "2/0.5p,grey"],area_thresh='600')
    #fig.colorbar(cmap="hot1.cpt",frame='af+l"correlation coefficient"',projection=pr) 
    fig.show()
    fig.savefig("/Users/sebinjohn/Downloads/corre/"+ps[-1]+".png",dpi=600,transparent=True)










