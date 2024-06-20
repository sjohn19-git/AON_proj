#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 12:26:33 2024

@author: sebinjohn
"""


import pygmt
import os
os.chdir("/Users/sebinjohn/AON_PROJECT/Final_codes_paper")
import numpy as np
import pandas as pd
import pickle
from pyproj import Proj, transform
import geopandas as gpd
from tqdm import tqdm


os.chdir('/Users/sebinjohn/AON_PROJECT/Final_codes_paper/coastal_erosion/shapefiles/')
# Reading a CSV file of shorefast ice width using pandas
df = pd.read_csv('./lfiw_all.csv')
# Display the first few rows of the DataFrame
print(df.head())
#loading shapefile assosciated with shorefast ice width
shapefile_path = './cvecs_Beaufort.shp'
gdf = gpd.read_file(shapefile_path)
# Path to your .prj file
prj_file_path = './cvecs_Beaufort.prj'
# Read the contents of the .prj file
with open(prj_file_path, 'r') as prj_file:
    prj_content = prj_file.read()
# Print the contents of the .prj file
print(prj_content)
# Define the Alaska Albers projection string
alaska_albers_proj_string = (
    '+proj=aea +lat_1=55 +lat_2=65 +lat_0=50 +lon_0=-154 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs'
)

# Define the source and target projections
alaska_albers = Proj(alaska_albers_proj_string)
wgs84 = Proj(init='EPSG:4326')  # WGS84 (latitude, longitude)


#grid1=pygmt.datasets.load_earth_relief(resolution='02m', region=[150,260, 30, 78])
#grid1.to_netcdf("dem.nc")


os.chdir("/Users/sebinjohn/AON_PROJECT/Data/Video Making")
#this .pkl file contains station location, longitude and latitude of each station
#used in this study.
with open("metadta.pkl","rb") as f:
   long,lat,stationo ,env = pickle.load(f) 

###########following lines reformat station names
os.chdir("/Users/sebinjohn/AON_PROJECT/Final_codes_paper")
stas=[]
for ele in stationo:
    sta=ele[6:10]
    if sta[-1]==".":
        sta=sta[:-1]
    print(sta,ele)
    stas.append(sta)

stas.append("PS01")
long.append(-148.6141)
lat.append(70.258)
##################################################


#x= np.array([-156.6175,-157.15,-154.78,-153.48])
#y=np.array([71.32,70,69.15,67.22])
proj="L-155/35/33/85/10c"
fig=pygmt.Figure()
reg="185/46/255/70r"
with pygmt.config(MAP_FRAME_TYPE="plain"):
    fig.basemap(region=reg, projection=proj,frame="lrtb")
pygmt.makecpt(cmap="bath1.cpt",series=[-8000,0])
fig.grdimage(grid="./dem.nc",region=reg,projection=proj, cmap=True,shading='+a300+nt0.8')
#fig.colorbar(frame=["a2000", "x+lElevation", "y+lm"],position="g184/62+w3c/0.25c+v",)
#fig.coast(region=reg, projection=proj,shorelines="0.02p",borders=["1/0.5p,grey"],area_thresh='600',dcw=["RU+g211/211/211@50","CA+g211/211/211@50"])
fig.coast(region=reg, projection=proj,borders=["1/0.5p,black"],area_thresh='600',dcw=["US.AK","RU","CA"],shorelines="0.02p")
#fig.grdimage(grid=temp_grid,region=reg,projection=proj, cmap="./cpts/temperature.cpt",nan_transparent=True)
#fig.plot(x=x, y=y, style="t0.4c", pen="1p", fill="red")
#fig.text(text=["A21K","B20K","C21K","F21K"],x=x-4,y=y,font="8p")
for c in tqdm(range(975,1100)):
    geom=gdf.geometry[c]
    ls=list(geom.coords)
    lon1, lat1 = transform(alaska_albers, wgs84, ls[0][0], ls[0][1])
    lon2, lat2 = transform(alaska_albers, wgs84, ls[1][0], ls[1][1])
    loni=[lon1,lon2]
    lati=[lat1,lat2]
    fig.plot(x=loni, y=lati, pen='1p,blue', label='Line')

fig.text(text="Bering Strait",x=-169,y=65.4,angle=90)
fig.text(text=["Gulf of Alaska", "Bering Sea","Chukchi Sea","Beaufort sea"], x=[-145, -170.5,-171,-145], y=[58, 59,71.5,71.5])
fig.basemap(map_scale="n0.87/0.96+w500k+f+u")
for i in range(len(stas)):
    sta=stas[i]
    lon_p=long[i]
    lat_p=lat[i]
    fig.plot(x=lon_p, y=lat_p, style="t0.3c", pen="1p", fill="red")
    #fig.text(text=sta,x=lon_sta-4,y=lat_sta,font="8p")
fig.show()
fig.savefig("/Users/sebinjohn/Downloads/bathy.pdf",dpi=600)
