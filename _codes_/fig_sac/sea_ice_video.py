#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 17:41:34 2024

@author: sebinjohn
"""

import os 
os.chdir("/Users/sebinjohn/AON_PROJECT/Data/codes")
import pygmt
#import dill
import pandas as pd
from obspy import UTCDateTime
import cv2
from tqdm import tqdm
from os.path import isfile, join 
import glob
import numpy as np
import pickle
import pygrib
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cdsapi
from medfilt import medfilt
import matplotlib.dates as mdates




#####
#load PS01 data
start = UTCDateTime(2008, 5, 1)
end = UTCDateTime(2023, 1, 1)
seis = np.load("/Users/sebinjohn/AON_PROJECT/Data/PS01//2008-05-01T00-2023-01-01T00.npy")
freq = np.load("/Users/sebinjohn/AON_PROJECT/Data/PS01/frequencies.npy")
peri = 1/freq
time = np.arange(start, end, 3600)

# periods using 1 to 2

spsm2 = seis[53:61, :]
spsm1 = np.mean(spsm2, axis=0)
spsm = medfilt(spsm1, 7)
spsm[spsm==0]=np.nan
spsm[spsm>-50]=np.nan
spsm[spsm<-150]=np.nan

plo_time = []
for ele in time:
    plo_time.append(ele.matplotlib_date)
plo_time=np.array(plo_time)

yr=2021

st = UTCDateTime(yr, 10, 1)
et = UTCDateTime(yr, 12, 1)
time_framew=[]
windw = 1
tim_lenw = int((et-st)/(3600*windw))
for i in range(int((et-st)/(3600*windw))):
    time_framew.append(st)
    st = st+(3600*windw)

        
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
##########


def ma_pic_generator(tim,grid,sta):
    
    '''This function plots significant waveheight and
    PSD for each station in a map. Assumes topography grid and 
    cmaps are located in specific directory. Plane fitting 
    function is required.
    
    :param tim: time
    :param grid: significant waveheight grid
    :param median_timeseries: PSD time series of each station'''
    
    print("plotting "+str(tim))
    df1=pd.read_csv("/Users/sebinjohn/AON_PROJECT/Data/station Meta Data.csv")
    stat=sta
    lat_sta=df1[df1["Station\xa0"]==stat]["Latitude\xa0"]
    lon_sta=df1[df1["Station\xa0"]==stat]["Longitude\xa0"]
    grid1 = '/Users/sebinjohn/AON_PROJECT/Final_codes_paper/grid2.nc'
    os.chdir("/Users/sebinjohn/AON_PROJECT/Final_codes_paper/")
    #cpt=pygmt.makecpt(cmap="./expll1.cpt",output="./icewave.cpt",series=[0,3.5])
    fig=pygmt.Figure()
    pygmt.config(MAP_FRAME_TYPE="plain",MAP_FRAME_PEN="0.5p",FONT_LABEL="8p")
    os.chdir("/Users/sebinjohn/AON_PROJECT/Final_codes_paper/")
    pr="L-159/35/33/45/8c"
    reg="188/65/240/75r"
    #fig.basemap(region=[150,260, 30, 73], projection="L-159/35/33/45/22c")
    fig.grdimage(grid=grid,region=reg,projection=pr, cmap="./icewave.cpt",frame="f",nan_transparent=False)
    fig.coast(region=reg, projection=pr,borders=["1/0.5p,black"],area_thresh='600',dcw=["US.AK+gwhite","RU+gwhite","CA+gwhite"])
    fig.grdimage(grid=grid1,region=reg,projection=pr, cmap="topo.cpt",frame="f",nan_transparent=True) 
    fig.coast(region=reg, projection=pr,shorelines=["1/0.15p,black","2/0.25p,grey"],frame="f",borders=["1/0.2p,black", "2/0.5p,grey"],area_thresh='600')
    #fig.grdimage(grid=grid,region=[150,260, 30, 73],projection="L-159/35/33/45/20c", cmap="expll1.cpt",frame="f",nan_transparent=True)
    #fig.coast(region=[150,260, 30, 73], projection="L-159/35/33/45/20c",shorelines=True)
    fig.plot(x=-148.6141, y=70.258, style="t0.4c", pen="1p", fill="red")
    fig.colorbar(projection=pr, cmap="./icewave.cpt",frame=["x+lSignificant Wave Height", "y+lm"],position="JML+o0.1c+w3.5c/0.2c")#position="n0.17/-0.1+w6.5c/0.45c+h")
    fig.plot(x=-157,y=66.2,style="s0.7c", pen="0.5p,black",fill="214/242/238",projection=pr)
    fig.text(x=-155,y=66.2,text="Sea ice",justify="ML",projection=pr) 
    fig.savefig("/Users/sebinjohn/AON_PROJECT/Data/Video Making/ice_video/"+str(tim)+"wave.jpg")



    
def sr(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)
    

def time_series(tim,sta,pltime,lim):
    '''this function plots SPSM time series
    and marks current time
    :param tim: current time
    :param sta: station name
    :param pltime: times in matplolib date
    :param lim: time_frames this is used to set xlimit'''
    
    seis=spsm
    date_format1 = mdates.DateFormatter('%d-%b') 
    seis[seis==0]=np.nan
    fig,ax=plt.subplots(figsize=(5,2.5),dpi=700)
    ax.plot(pltime,seis,color="yellowgreen")
    ax.xaxis.set_major_formatter(date_format1)
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[14,28]))
    #ax.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=[15]))
    ax.set_ylabel('dB (rel. 1 (m/s'+sr('2')+')'+sr('2')+'/Hz',fontsize=10)
    ax.set_xlim([lim[38].matplotlib_date,lim[-1].matplotlib_date])
    ax.axvline(x=tim.matplotlib_date,c="k",ls="--")
    ax.set_ylim(-150, -115)
    ax.tick_params(axis='both', which='both', labelsize=8)
    ax.set_yticks([-150,-140,-130,-120])
    os.chdir("/Users/sebinjohn/AON_PROJECT/Data/Video Making/ice_video")
    fig.savefig(str(tim)+"seis.jpg",bbox_inches='tight')
    plt.close()


from PIL import Image

def merge_images_vertically(image_path1, image_path2, output_path):
    '''this function merges two images vertically
    :param image_path1: path of image1
    :param image_path2: path of image2
    :param output_path: output path'''
    
    # Open the images
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)

    # Determine the maximum width between the two images
    max_width = max(img1.width, img2.width)

    # Resize images to have the same width
    img1 = img1.resize((max_width+50, int(img1.height * (max_width / img1.width))), Image.ANTIALIAS)
    img2 = img2.resize((max_width-50, int(img2.height * (max_width / img2.width))), Image.ANTIALIAS)

    # Calculate the total height of the merged image
    total_height = img1.height + img2.height

    # Create a new image with the same width and the combined height
    merged_image = Image.new('RGB', (max_width, total_height), color='white')

    # Paste the images onto the new image
    merged_image.paste(img1, (0, 0))
    merged_image.paste(img2, (25, img1.height))

    # Save the merged image
    merged_image.save(output_path)

# Example usage

for i in tqdm(range(len(time_framew))):
    tim=time_framew[i]
    sta="PS01"
    dp="/Users/sebinjohn/AON_PROJECT/Data/Video Making/ice_video/"
    image_path1 = dp+str(tim)+'seis.jpg'
    image_path2 = dp+str(tim)+'wave.jpg'
    output_path = dp+str(tim)+'merged_image.jpg'
    grid,data_wave=wave(tim)
    ma_pic_generator(tim,grid,sta)
    time_series(tim,sta,plo_time,time_framew)
    merge_images_vertically(image_path1, image_path2, output_path)
    



###video



def convert_pictures_to_video(pathIn, pathOut, fps, time):
    ''' this function converts images to video
    
    :param pathIn: path for images
    :param pathOut: out path for video
    :param fps: frames per second
    :param time: time'''
    pics=glob.glob(join(directory+"/*merged_image.jpg"))
    frame_array=[]
    files=[f for f in pics if isfile(join(pathIn,f))]
    files.sort()
    pil_image =Image.open(files[0]).convert('RGB')
    height, width = pil_image.size

    # Calculate new dimensions while maintaining the aspect ratio
    target_width = int(width/2.4)
    target_height = int(height/2)

    for i in tqdm(range(len(files))):
        '''reading images'''
        pil_image =Image.open(files[i]).convert('RGB')
        resized_image = pil_image.resize((target_width, target_height))
        open_cv_image = np.array(resized_image) 
        # Convert RGB to BGR 
        img = open_cv_image[:, :, ::-1].copy() 
        for k in range (time):
            frame_array.append(img)
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, (target_width, target_height))
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()         
        
   
# Example:
directory="//Users/sebinjohn/AON_PROJECT/Data/Video Making/ice_video"
pathIn=directory
pathOut=directory+"/"+"w.avi"
fps=8
time=1 # the duration of each picture in the video


convert_pictures_to_video(pathIn, pathOut, fps, time)


#####pic generate

#this is just an adaptation of above code to plot certain time stamps

time=[UTCDateTime(2021,10,20,9),UTCDateTime(2021,11,2,23),UTCDateTime(2021,11,6,14),UTCDateTime(2021,11,22,8)]
avtime=[ele.matplotlib_date for ele in time]


plt.rcParams['font.family'] = 'Helvetica'
seis=spsm
date_format1 = mdates.DateFormatter('%d-%b-%y') 
seis[seis==0]=np.nan
fig,ax=plt.subplots(figsize=(8,3.15),dpi=700)
ax.plot(plo_time,seis,color="yellowgreen")
ax.xaxis.set_major_formatter(date_format1)
#ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
#ax.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=[15]))
ax.set_ylabel('dB (rel. 1 (m/s'+sr('2')+')'+sr('2')+'/Hz',fontsize=12)
ax.set_xlim([time_framew[38].matplotlib_date,time_framew[-1].matplotlib_date])
for i in range(len(avtime)):
    ax.axvline(x=avtime[i],c="k",ls="--")
ax.set_ylim(-150, -115)
ax.tick_params(axis='both', which='both', labelsize=10)
ax.set_yticks([-150,-140,-130,-120])
os.chdir("/Users/sebinjohn/Downloads")

fig.savefig(str(tim)+"seis.pdf",bbox_inches='tight')

plt.close()

for i in range(len(time)):
    tim=time[i]
    grid,data_wave=wave(tim)
    print("plotting "+str(tim))
    df1=pd.read_csv("/Users/sebinjohn/AON_PROJECT/Data/station Meta Data.csv")
    stat=sta
    lat_sta=df1[df1["Station\xa0"]==stat]["Latitude\xa0"]
    lon_sta=df1[df1["Station\xa0"]==stat]["Longitude\xa0"]
    grid1 = '/Users/sebinjohn/AON_PROJECT/Final_codes_paper/grid2.nc'
    os.chdir("/Users/sebinjohn/AON_PROJECT/Final_codes_paper/")
    #cpt=pygmt.makecpt(cmap="./expll1.cpt",output="./icewave.cpt",series=[0,3.5])
    fig=pygmt.Figure()
    pygmt.config(MAP_FRAME_TYPE="plain",MAP_FRAME_PEN="0.5p",FONT_LABEL="8p")
    os.chdir("/Users/sebinjohn/AON_PROJECT/Final_codes_paper/")
    pr="L-159/35/33/45/8c"
    reg="188/65/240/75r"
    #fig.basemap(region=[150,260, 30, 73], projection="L-159/35/33/45/22c")
    fig.grdimage(grid=grid,region=reg,projection=pr, cmap="./icewave.cpt",frame="f",nan_transparent=False)
    fig.coast(region=reg, projection=pr,borders=["1/0.5p,black"],area_thresh='600',dcw=["US.AK+gwhite","RU+gwhite","CA+gwhite"])
    fig.grdimage(grid=grid1,region=reg,projection=pr, cmap="topo.cpt",frame="f",nan_transparent=True) 
    fig.coast(region=reg, projection=pr,shorelines=["1/0.15p,black","2/0.25p,grey"],frame="f",borders=["1/0.2p,black", "2/0.5p,grey"],area_thresh='600')
    #fig.grdimage(grid=grid,region=[150,260, 30, 73],projection="L-159/35/33/45/20c", cmap="expll1.cpt",frame="f",nan_transparent=True)
    #fig.coast(region=[150,260, 30, 73], projection="L-159/35/33/45/20c",shorelines=True)
    fig.plot(x=-148.6141, y=70.258, style="t0.4c", pen="1p", fill="red")
    #fig.colorbar(projection=pr, cmap="./icewave.cpt",frame=["x+lSignificant Wave Height", "y+lm"],position="JBC+o0.1c+w3.5c/0.2c")#position="n0.17/-0.1+w6.5c/0.45c+h")
    #if i ==0:
        #fig.plot(x=-157,y=66.2,style="s0.7c", pen="0.5p,black",fill="214/242/238",projection=pr)
        #fig.text(x=-155,y=66.2,text="Sea ice",justify="ML",projection=pr) 
    fig.show()
    fig.savefig("/Users/sebinjohn/Downloads/"+str(tim)+"wave.png")

    