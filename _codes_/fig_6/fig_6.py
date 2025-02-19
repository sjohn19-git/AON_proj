
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
import numpy as np
import matplotlib.pyplot as plt
import cdsapi
from medfilt import medfilt
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import math 
import obspy.signal.cross_correlation as cr
from scipy.stats import gaussian_kde
from tqdm import tqdm
import cv2
from matplotlib.ticker import MultipleLocator
from PIL import Image


#loading station metadata
os.chdir("/Users/sebinjohn/AON_PROJECT/Data/Video Making")
with open("metadta.pkl","rb") as f:
   long,lati,stationo ,env = pickle.load(f) 


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




os.chdir("/Users/sebinjohn/AON_PROJECT/Data/median_Power_time_series")
os.listdir()
median_timeseries=np.load("./median_inerpolated_time_series_secondary2018-01-010Z-2021-01-01.npy")

#####obtainig seismic data for a specific region
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


def poly_seis(median_timeseries,long,lat,args):
    ''' This function creates a polygon from set of points
    and checks if each station is within the polygon
    and calculates the mean SM if yes.
    :param median_timeseries: Processed SM timeseries of all station.
    :param long: longitude of corresponding stations
    :param lat: latitude of corresponding stations
    :param stationo: list of station names
    :param *args: expects points
    
    '''
    coords=[]
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
            sum_array[j,:]=median_timeseries[i,:]
            j+=1
    sum_tot=np.empty([len(median_timeseries[0,:])])
    for i in range(len(sum_array[0,:])):
        if (c-np.count_nonzero(sum_array[:,i]==0))==0:
            sum_tot[i]=0
        else:
            sum_tot[i]=np.sum(sum_array[:,i])/(c-np.count_nonzero(sum_array[:,i]==0))
    return wista,poly,counter,sum_array,sum_tot

args=[(-165.9,70),(-166.1,64.6),(-154.3,65.1),(-153.88,70)]
wista,poly,coords,sum_array,sum_tot=poly_seis(median_timeseries,long,lati,args) 

###sum_tot is the data



######wave__data
wave_all=np.load("/Users/sebinjohn/AON_PROJECT/Data/wave/wave_all_for cross_correlation_2018-2021_3yr.npy")

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
    

#############cross correlating in loop####
regions=[[(-151.95,70.3),(-151.6,66.5),(-141.6,66.5),(-141.88,69.4),"North_East"],[(-165.9,70),(-166.1,64.6),(-154.3,65.1),(-153.88,70),"North_west"],[(-157.1,59),(-152.8,57),(-168.4,52.3),(-170.4,54.2),"south_west"],
         [(-140,62),(-140.5,59.5),(-131.5,55),(-129,56),"South East Region"],[(-166.322,62.951),(-153.788,63.587),[-153.820,60.768],(-163.582,59.649),"central_west"],[(-149.853,64.549),(-141.512,64.760),(-142.462,60.537),(-149.370,60.446),"central_east"]]




jj=0
args=regions[jj][0:-1]
wista,poly,coords,sum_array,sum_tot=poly_seis(median_timeseries,long,lati,args) 
corre_grid=np.zeros([np.shape(wave_all)[0],np.shape(wave_all)[1]])
corre_inde=np.zeros(np.shape(corre_grid))
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

######finding primary source region
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

####plotting

xy = np.vstack([wave_sumpixels,sum_tot[:]])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots(dpi=300)
ax.scatter(wave_sumpixels,sum_tot[:],c=z,s=0.5)
plt.savefig("/Users/sebinjohn/AON_PROJECT/Results/density plots/1.png",transparent=True)

xre=[]
for ele in coords:
    xre.append(ele[0])
xre.append(xre[0])
yre=[]
for ele in coords:
    yre.append(ele[1])
yre.append(yre[0])
res=np.nonzero(wista)[0]
lat_sta=[lati[i] for i in res]
lon_sta=[long[i] for i in res]
grid_corre.data[grid_corre.data==0]=np.nan
pygmt.makecpt(cmap="hot",output="hot1.cpt", series=[0.5,0.9,0.01,],reverse=True)
fig=pygmt.Figure()
pygmt.config(MAP_FRAME_TYPE="plain")
r=[152,240,30, 73]
pr="L-164/35/33/45/22c"
fig.coast(region=r, projection=pr,shorelines=False,frame="a",land="200/200/200",borders=["1/0.5p,black", "2/0.5p,red"])
fig.grdimage(grid=grid_corre,region=r,projection=pr, cmap="hot1.cpt",frame="a",nan_transparent=True)
fig.plot(x=xre, y=yre, en="1pgrey")
fig.colorbar(cmap="hot1.cpt",frame='af+l"correlation_coeifficent"',projection=pr,region=r) 
fig.plot(x=lon_sta,y=lat_sta,style="i0.3c")
fig.show()


################Finding biggest storm#######

def sr(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)

ims=[]
wf=grid_corre.data.copy().flatten()
lonc=grid_corre.lon.data
latc=grid_corre.lat.data
xf,yf=np.meshgrid(lonc,latc)
xf=xf.flatten()
yf=yf.flatten()

ra=[8355,8400]
for pli in tqdm(range(ra[0],ra[1])):
    grid,data_wave=wave(time_frame[pli])
    wave_arr,counterw,latwe,lonwe=wave_array(grid,(140,20),(260,73),(260,20),(150,73))
    grid1 = '/Users/sebinjohn/AON_PROJECT/Final_codes_paper/grid3.nc'
    fig=pygmt.Figure()
    pygmt.config(MAP_FRAME_TYPE="plain",MAP_FRAME_PEN="0.5p,black",FONT_LABEL="12p",FONT_ANNOT="12p")
    os.chdir("/Users/sebinjohn/AON_PROJECT/Final_codes_paper/")
    pr="L-159/35/33/45/8c"
    reg="175/35/265/68r"
    fig.grdimage(grid=wave_arr,region=reg,projection=pr,frame="lrbt",cmap="expll1.cpt",nan_transparent=True)
    fig.contour(x=xf,y=yf,z=wf,region=reg,projection=pr,pen="0.2p,205/91/92",label_placement="d7c",annotation="0.7,0.6,0.65+f6p")
    fig.grdimage(grid=grid1,region=reg,projection=pr, cmap="topo.cpt",nan_transparent=True)
    fig.coast(region=reg, projection=pr,shorelines="0.15p,black,solid",borders=["1/0.2p,black", "2/0.5p,grey"],area_thresh='600')
    fig.plot(x=xre, y=yre, pen="0.6p,Black",fill="#f4c09cff")
    fig.colorbar(projection=pr, cmap="expll1.cpt",frame=["x+lSignificant Wave Height", "y+lm"],position="n-0.04/0.4+w2.5c/0.2c+v+m")
    fig.text(text=str(time_frame[pli].date)+" "+str(time_frame[pli].hour)+":00",x=-154,y=73,projection=pr,font="10p,Helvetica")
    fig.text(text="NE",x=-146.5,y=68.3,projection=pr)
    #fig.show()
    fig.savefig("/Users/sebinjohn/AON_PROJECT/Results/big_storm/"+str(pli)+"m.png",dpi=900)
    
    
    fig=plt.figure(dpi=700)
    fig.set_figwidth(6)
    fig.set_figheight(4.5)
    plt.scatter(wave_sumpixels,sum_tot[:],s=0.5,marker="o",c=z)
    plt.scatter(wave_sumpixels[pli],sum_tot[pli],s=40,marker="o",c="red")
    plt.xlabel("significant wave height (m)",fontsize=12)
    plt.xlim([0,8])
    plt.ylim([-150,-110])
    plt.ylabel('dB(rel. 1 (m/s'+sr('2')+')'+sr('2')+'/Hz)',fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(base=10))
    #plt.title(ps[-1],fontsize=10)
    plt.tight_layout()
    #plt.grid()
    fig.savefig("/Users/sebinjohn/AON_PROJECT/Results/big_storm/"+str(pli)+"t.png")
    plt.close()
    
    os.chdir("/Users/sebinjohn/AON_PROJECT/Results/big_storm/")
    image1 = Image.open(str(pli) + "m.png")
    image2 = Image.open(str(pli) + "t.png")
    

    # Resize image1 to match the height of image2
    image1_resized = image1.resize((int(image2.width*0.17), int(image2.height*0.18)), Image.LANCZOS)
    image2_resized = image2.resize((int(image2.width*0.17), int(image2.height*0.18)), Image.LANCZOS)
    
    # Create a new image with dimensions to accommodate both images
    width = image1_resized.width + image2_resized.width+10
    height = max(image1_resized.height, image2_resized.height)+20
    combined_image = Image.new("RGB", (width, height),'white')
    
    # Paste image1_resized on the left side of combined_image
    combined_image.paste(image1_resized, (0, 10))
    
    # Paste image2 on the right side of combined_image
    combined_image.paste(image2_resized, (image1_resized.width, 0))
    # Show the combined image
    combined_image.save("/Users/sebinjohn/AON_PROJECT/Results/big_storm/"+str(pli)+"c.png",quality=100)
    os.remove("/Users/sebinjohn/AON_PROJECT/Results/big_storm/"+str(pli)+"t.png")


def convert_pictures_to_video(pathIn, pathOut, fps, time):
    ''' this function converts images to video
    
    :param pathIn: path for images
    :param pathOut: out path for video
    :param fps: frames per second
    :param time: time'''
    pics=glob.glob(join(directory+"*.png"))
    frame_array=[]
    files=[f for f in glob.glob(pathIn) if isfile(join(pathIn,f))]
    files.sort()
    for i in range (len(files)):
        '''reading images'''
        pil_image =Image.open(files[i]).convert('RGB') 
        height, width = pil_image.size
        open_cv_image = np.array(pil_image) 
        # Convert RGB to BGR 
        img = open_cv_image[:, :, ::-1].copy() 
        size=(height,width)
        for k in range (time):
            frame_array.append(img)
    out=cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*"WMV2"), fps,size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()

# Example:
directory="//Users/sebinjohn/AON_PROJECT/Results/big_storm/"
pathIn=directory+'*c.png'
pathOut=directory+"/"+"big_storm.avi"
fps=1
time=1 # the duration of each pictu
convert_pictures_to_video(pathIn, pathOut, fps, time)





# ra=[8370,8380]
# for pli in tqdm(range(len(ra))):
#     grid,data_wave=wave(time_frame[ra[pli]])
#     wave_arr,counterw,latwe,lonwe=wave_array(grid,(140,20),(260,73),(260,20),(150,73))
#     grid1 = '/Users/sebinjohn/AON_PROJECT/Final_codes_paper/grid3.nc'
#     fig=pygmt.Figure()
#     pygmt.config(MAP_FRAME_TYPE="plain",MAP_FRAME_PEN="0.5p,black",FONT_LABEL="12p",FONT_ANNOT="12p")
#     os.chdir("/Users/sebinjohn/AON_PROJECT/Final_codes_paper/")
#     pr="L-159/35/33/45/8c"
#     reg="175/35/265/68r"
#     fig.grdimage(grid=wave_arr,region=reg,projection=pr,frame="lrbt",cmap="expll1.cpt",nan_transparent=True)
#     fig.contour(x=xf,y=yf,z=wf,region=reg,projection=pr,pen="0.2p,205/91/92",label_placement="d7c",annotation="0.7,0.6,0.65+f6p")
#     fig.grdimage(grid=grid1,region=reg,projection=pr, cmap="topo.cpt",nan_transparent=True)
#     fig.coast(region=reg, projection=pr,shorelines="0.15p,black,solid",borders=["1/0.2p,black", "2/0.5p,grey"],area_thresh='600')
#     fig.plot(x=xre, y=yre, pen="0.6p,Black",fill="#f4c09cff")
#     #fig.colorbar(projection=pr, cmap="expll1.cpt",frame=["x+lSignificant Wave Height", "y+lm"],position="n-0.04/0.4+w2.5c/0.2c+v+m")
#     fig.text(text=str(time_frame[ra[pli]].date)+" "+str(time_frame[ra[pli]].hour)+":00",x=-154,y=73,projection=pr,font="10p,Helvetica")
#     fig.text(text="NE",x=-146.5,y=68.3,projection=pr)
#     fig.show()
#     fig.savefig("/Users/sebinjohn/Downloads/"+str(pli)+"m.pdf")
    



# fig=pygmt.Figure()
# #fig.basemap(region=[150,260, 30, 73], projection="L-159/35/33/45/22c")
# fig.grdimage(grid=grid,region=[150,260, 30, 73],projection="L-159/35/33/45/20c", cmap="expll1.cpt",frame="f",nan_transparent=True)
# fig.coast(region=[150,260,30, 73], projection="L-159/35/33/45/20c",shorelines=False,frame="a",land="200/200/200",borders=["1/0.5p,black", "2/0.5p,red"])
# fig.plot(x=xre, y=yre, en="1pgrey")
# fig.colorbar(projection="L-159/35/33/45/20c", cmap="expll1.cpt",frame=["x+lsignificant\twave\theight", "y+lm"],position="n0.57/-0.1+w8c/0.5c+h")
# fig.colorbar(cmap="psd.cpt",frame=["x+lPSD", "y+ldb"],projection="L-159/35/33/45/20c",position="n0/-0.1+w8c/0.5c+h")  
# fig.plot(x=lon_sta,y=lat_sta,style="i0.3c")
# fig.show()

