import matplotlib.dates as mdates
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.cm as cm
import matplotlib.colors as cl
from matplotlib import gridspec
import pickle
from datetime import datetime
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
import pyproj 

def sr(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)

vmin,vmax=-190,-90
nom=cl.Normalize(vmin=vmin,vmax=vmax)


date_format=mdates.DateFormatter('%b')

plt.rcParams.update({'font.size': 12})
cmap = plt.get_cmap('nipy_spectral').copy()
cmap.set_over(color = 'w')


#plotting colorbar

plt.rcParams.update({'font.size': 8})
fig, ax = plt.subplots(dpi=600)
ax.axis('off')

# Create a ScalarMappable for mapping data values to colors
sm = cm.ScalarMappable(norm=nom, cmap="nipy_spectral")

# Create horizontal colorbar with ticks at specified locations
cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', ticks=[-190, -150, -110, -70],shrink=0.75)

# Set colorbar label
#cbar.set_label('dB (rel. 1 (m/s'+sr('2')+')'+sr('2')+'/Hz)')  # Customize the label
fig.savefig("/Users/sebinjohn/Downloads/clr_bar_horizontal.svg")
plt.show()



#####plotting A21k spectrogram_SPSM_sea_ice_wave_height

sta="A21K"
os.chdir(r"/Users/sebinjohn/AON_PROJECT/Data/"+sta)
with open((str(sta)+".pkl"),"rb") as f:
    sta, starttimeta, endtimeta,starttimeak,endtimeak,sta,cha,loc=pickle.load(f)
fint_frames=np.arange(starttimeta,endtimeak,3600)
for i in range(len(fint_frames)):
    fint_frames[i]=mdates.date2num(fint_frames[i])
fint_frames = np.asarray(fint_frames, dtype=np.float64)
#loading the spectra for A21K
with open("final_"+sta+".npy", 'rb') as g:
    final=np.load(g)
xmin=mdates.date2num(UTCDateTime(2018,1,1))
xmax=mdates.date2num(UTCDateTime(2021,1,1))
xlmi=xmin
xlma=xmax
vmin,vmax=-190,-90
freq=[]
name= pd.read_xml("/Users/sebinjohn/AON_PROJECT/Data/Video Making/pdf0.xml", xpath="/PsdRoot/Psds[1]/Psd[1]/value[@freq]")
for i in range (95):
    freq.append(name.iloc[i]['freq'])
freq.append(19.740)

#loading processed SPSM
final_seis_SPSM=np.load("/Users/sebinjohn/AON_PROJECT/Data/ML_seismic_train/final_seis_ML_SPSM.npy")
seis=final_seis_SPSM[:,-6]
seis[seis==0]=np.nan

st=UTCDateTime(2018,1,1)
et=UTCDateTime(2022,1,1)
windw=1

spsm_time=[]
for i in range (int((et-st)/(3600*windw))):
    spsm_time.append(mdates.date2num(st))
    st=st+(3600*windw)

###sea ice

sea_ice_con=np.load("/Users/sebinjohn/AON_PROJECT/Data/sea_ice_con/sea_ice_con.npy")
dx = dy = 25000

x = np.arange(-3850000, +3750000, +dx)
y = np.arange(+5850000, -5350000, -dy)

X,Y=np.meshgrid(x,y)


NorthPolar_WGS=pyproj.Transformer.from_crs(3413,4326)
WGSvalues=NorthPolar_WGS.transform(X,Y)

def haversine(lat1, lon1, lat2, lon2):
    '''returns the distance between two points
    in the surface of the earth
    :param lat1: latitude of point 1
    :param lon1: longitude of point 1
    :param lat2: latitude of point 2
    :param lon2: longitude of point 2'''
    
    R = 6371  # Earth radius in kilometers

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance

def find_closest_point(lat, lon, lat_grid, lon_grid):
    '''this function return the indices of the closest
    point to the lat lon provided
    :param lat: latitude of the point
    :param lon: longitude of the point
    :param lat_grid: Latitude of the grids
    :param lon_grid: longitude of the grids'''
    
    distances = haversine(lat, lon, lat_grid, lon_grid)
    indices = np.unravel_index(np.argmin(distances), distances.shape)
    return indices

c_s = find_closest_point(71.3221, -156.6175, WGSvalues[0], WGSvalues[1])

closest_sea_ice=NorthPolar_WGS.transform(x[c_s[1]],y[c_s[0]])

st=UTCDateTime(2018,1,1)
et=UTCDateTime(2022,1,1)
windw=1*24
time_frame_sea=np.array([])
for i in range (int((et-st)/(3600*windw))):
    time_frame_sea=np.append(time_frame_sea,st)
    st=st+(3600*windw)
    
plo_time_sea=[]

for ele in time_frame_sea:
    plo_time_sea.append(ele.matplotlib_date)

############################Wave
import xarray as xr

#loading wave data
wave_all=xr.open_dataset("/Users/sebinjohn/AON_PROJECT/Data/wave/wave_2018-2021_fulltime_highlat.nc")
wave_t=wave_all.time
wave_lat=wave_all.lat
wave_lon=wave_all.lon
xw,yw=np.meshgrid(wave_lon,wave_lat)
#finding closest point
c_w = find_closest_point(71.3221, -156.6175, yw, xw)
closest_wave=(wave_lat[c_w[0]],wave_lon[c_w[1]])

############
from matplotlib import rc
# activate latex text rendering
rc('text', usetex=True)

fig,axes=plt.subplots(nrows=3,figsize=(7.5,7.5),sharex=True,gridspec_kw={'hspace': 0.1},dpi=600)
c = axes[0].pcolormesh(fint_frames,freq,final,cmap=cmap,vmin=vmin,vmax=vmax,shading='auto',rasterized=True)
axes[0].axhline(y=1,c="k",linestyle="--")
axes[0].axhline(y=0.5,c="k",linestyle="--")
nom=cl.Normalize(vmin=vmin,vmax=vmax)
axes[0].xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1,interval=6))
axes[0].set_xlim([xmin,xmax])
axes[0].text(0.03, 0.92,r"\textit{A21K}", transform=axes[0].transAxes, fontsize=12, va='top', ha='left',bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.3', alpha=0.9))
# for label in ax1.get_xticklabels():
#     label.set_rotation(40)
#     label.set_horizontalalignment('right')
axes[0].set_yscale('log')
axes[0].xaxis.set_major_formatter(date_format)
axes[0].set_ylabel('Frequency (Hz)')
#axes[inde].title.set_text(sta+"_"+cha+" "+str(UTCDateTime((mdates.num2date(xmin)).strftime('%Y-%m-%d')))[0:10]+" - "+str(UTCDateTime((mdates.num2date(xmax)).strftime('%Y-%m-%d')))[0:10])
#ax1.title.set_text(sta+"_"+cha+" ")
axes[1].plot(spsm_time,seis,c="yellowgreen",label=r"\textit{SPSM}")
axes[1].set_ylabel('dB (rel. 1 (m/s'+sr('2')+')'+sr('2')+'/Hz')
leg0=axes[1].legend(loc="upper right",frameon=True,markerscale=10)
im0=axes[2].plot(plo_time_sea,sea_ice_con[c_s[0],c_s[1],:],c="black",label=r"\textit{Sea ice conc.}")
axes[2].set_ylabel('Sea ice concentration (m)')
ax2 = axes[2].twinx()
axes[2].grid(which='major', axis='x', linestyle='--', linewidth=0.5)
axes[2].grid(which='major', axis='y', linestyle='--', linewidth=0.5)
axes[1].grid(which='major', axis='x', linestyle='--', linewidth=0.5)
axes[1].grid(which='major', axis='y', linestyle='--', linewidth=0.5)
im1=ax2.plot(wave_t,wave_all.wave[c_w[0],c_w[1],:],c='blue',label=r"\textit{Sig. Wave height}",zorder=0)
lns = im0+im1
labs = [l.get_label() for l in lns]
leg=ax2.legend(lns,labs,loc="lower right",frameon=True,markerscale=10)   
ax2.set_ylabel("Significant wave height (m)")
fig.savefig("/Users/sebinjohn/Downloads/pic1.svg")
