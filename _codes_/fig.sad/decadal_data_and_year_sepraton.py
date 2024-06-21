#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 09:11:18 2023

@author: sebinjohn
"""
import os
os.chdir("/Users/sebinjohn/AON_PROJECT/Final_codes_paper/")
from medfilt import medfilt
import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime
import matplotlib.dates as mdates
from tqdm import tqdm
from matplotlib.dates import MONTHLY,MonthLocator
from matplotlib.dates import DayLocator
import datetime
import obspy.signal.filter as flt
from datetime import datetime
from matplotlib.gridspec import GridSpec
import pandas as pd
import matplotlib.patches as patches

#loading PS01 spectra
start = UTCDateTime(2009, 1, 1)
end = UTCDateTime(2024, 1, 1)
seis = np.load("/Users/sebinjohn/AON_PROJECT/Data/PS01//2009-01-01T00-2024-01-01T00.npy")
freq = np.load("/Users/sebinjohn/AON_PROJECT/Data/PS01/frequencies.npy")
peri = 1/freq
time = np.arange(start, end, 3600)

# extracting spsm

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



# splitting to yearly values

datetime_dates = mdates.num2date(plo_time)

years=[]
times=[]
for i in range(len(datetime_dates)):
    date=datetime_dates[i]
    j=date.year
    if f"yr_{j}" not in years:
        years.append(f"yr_{j}")

for ele in years:
    exec(f"{ele} = []")
    exec(f"t{ele} = []")

for i in range(len(datetime_dates)):
    date=datetime_dates[i]
    j=date.year
    exec(f"yr_{j}.append(spsm[i])")
    exec(f"tyr_{j}.append(mdates.date2num(date))")

for ele in years:
    exec(f"{ele}=np.array({ele})")
    exec(f"t{ele}=np.array(t{ele})")
    
#plotting all year together


colors = [
    "blue",
    "green",
    "red",
    "cyan",
    "magenta",
    "yellow",
    "black",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "teal",
    "navy",
    "violet"
]


date_format1 = mdates.DateFormatter('%b')
fig=plt.figure(figsize=(20, 16),dpi=100)
for i in range(len(years)):
    ele=years[i]
    exec(f"x=t{ele}")
    exec(f"y={ele}")
    y=y+i*11
    globals()["ax{}".format(i)]=fig.add_subplot(111,frame_on=False)
    globals()["ax{}".format(i)].plot(x, y,label=ele,c=colors[i])
    startx=mdates.date2num(UTCDateTime(mdates.num2date(x[0]).year,1,1))
    endx=mdates.date2num(UTCDateTime(mdates.num2date(x[0]).year,12,31))
    globals()["ax{}".format(i)].xaxis.set_major_locator(MonthLocator())
    globals()["ax{}".format(i)].xaxis.set_major_formatter(date_format1)
    globals()["ax{}".format(i)].set_ylim(-150, 40)
    globals()["ax{}".format(i)].set_xlim(startx,endx)
    globals()["ax{}".format(i)].set_yticks([])
    if i != 0:
        globals()["ax{}".format(i)].set_xticks([])
    globals()["ax{}".format(i)].legend(loc='upper right', bbox_to_anchor=(1.08,(1-i*0.04)))


fig.savefig("/Users/sebinjohn/AON_PROJECT/Results/")



##loading shore_fast_ice dataset

sfastm=np.load('/Users/sebinjohn/AON_PROJECT/Final_codes_paper/coastal_erosion/shorefastice_width_near_ps01.npy')
df = pd.read_csv('/Users/sebinjohn/AON_PROJECT/Final_codes_paper/coastal_erosion/shore_fast.csv')
mat_timestr=df.columns[2:]
mat_time=[float(ele) for ele in mat_timestr]#available timestamps shorefast ice

s_date = np.datetime64('2009-01-01T00:00:00')
e_date = np.datetime64('2024-12-31T23:00:00')
interval = np.timedelta64(24, 'h')
stime=np.arange(s_date , e_date + interval, interval)#all timestamps
stime_num = mdates.date2num(stime.astype('datetime64[s]').tolist())

boolean_array = np.isin(stime_num, mat_time)

cutoff_date = "2008-12-31T23:00"

# Convert the cutoff date to matplotlib date number
cutoff_num = mdates.date2num(np.datetime64(cutoff_date))

bool_2009=mat_time > cutoff_num#extracting data after 2009
time_cut=np.array(mat_time)[bool_2009]
sfastm_cut=np.array(sfastm)[bool_2009]

sfast_full=np.ones(len(stime))*np.nan # creating a conitnuos data with nan in data gap
sfast_full[boolean_array]=sfastm_cut

mask = boolean_array #data without nans
x_clean = stime[mask] #same as time_cut
y_clean = sfast_full[mask] #same as sfastm_cut

fig,axes=plt.subplots()
axes.plot(x_clean,y_clean,color='black',ls="-",lw=1)
axes.fill_between(x_clean, y_clean, color='grey', alpha=0.4, label='Filled Area')
#axes.set_xlim(datetime(2009,1,1,0),datetime(2010,1,1,0))
axes.set_ylim(0,50)

##plottting
def sr(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica'
})

#plt.rcdefaults()
plt.rcParams.update({'font.size': 12})
z=10
za=np.arange(0,z,1)
#defining dyummy dates
start_date = np.datetime64('1972-01-01T00:00:00')
end_date = np.datetime64('1972-12-31T23:00:00')
interval = np.timedelta64(1, 'h')
xtime=np.arange(start_date, end_date + interval, interval)
xtime = mdates.date2num(xtime)
date_format = mdates.DateFormatter("%b")
fig,axes=plt.subplots(nrows=len(years),ncols=1,sharex=True,figsize=(6,7),dpi=600,gridspec_kw={'hspace': 0.2})
#gs = GridSpec(len(years), 1, figure=fig, hspace=0.8)
vmi=-150
vma=-105
for i in range(len(years)):
    ele=globals()[years[i]]
    time=globals()["t{}".format(years[i])]
    stele=np.vstack([ele]*z)
    stra=np.datetime64(years[i][-4:]+'-01-01')
    stre=np.datetime64(str(int(years[i][-4:])+1)+'-01-01')
    x_clean_yr = x_clean[(x_clean >= stra) & (x_clean < stre)]
    y_clean_yr=y_clean[(x_clean >= stra) & (x_clean < stre)]
    print(years[i],len(x_clean_yr)) 
    x_clean_datetime = x_clean_yr.astype('O')
    ax2 = axes[i].twinx()
    x_clean_1972 =np.array([dt.replace(year=1972) for dt in x_clean_datetime])
    mat_1972=[mdates.date2num(dt) for dt in x_clean_1972]
    im1=ax2.fill_between(x_clean_1972, y_clean_yr, color='silver', alpha=0.95, label='Filled Area')
    ax2.set_ylim([0,60])
    ax2.set_yticks([10,40])
    ax2.tick_params(axis='y', labelsize=10)
    if len(time)==8784:
        X,Y=np.meshgrid(xtime,za)
        im=axes[i].pcolormesh(X,Y,stele,vmin=vmi,vmax=vma,cmap="nipy_spectral",rasterized=True)
    else:#handling leap day
        xtimes=np.hstack((xtime[:1416],xtime[1440:]))
        X,Y=np.meshgrid(xtimes,za)
        im=axes[i].pcolormesh(X,Y,stele,vmin=vmi,vmax=vma,cmap="nipy_spectral",rasterized=True)
    axes[i].set_yticks([])
    axes[i].xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    ylabel=axes[i].set_ylabel(years[i][3:],labelpad=20,rotation=0)
    ylabel.set_y(0.31)
    #axes[i-1].yaxis.label.set_rotation(0) 
axes[i].set_xlim([start_date,end_date])
axes[i].xaxis.set_major_formatter(date_format)
fig.text(.97, 0.5, 'Shorefast ice width (km)', ha='center', va='center', rotation=90)

cax = fig.add_axes([0.5, 0.045, 0.4, 0.015])  # [left, bottom, width, height]
colorbar = fig.colorbar(im, cax=cax, orientation='horizontal')
colorbar.ax.tick_params(labelsize=10)
cax.set_xlabel('dB (rel. 1 (m/s'+sr('2')+')'+sr('2')+'/Hz)',size=12)

rect = patches.Rectangle((0.125, 0.045), 0.055, 0.025, linewidth=1, edgecolor='none', facecolor='silver', alpha=0.95, transform=fig.transFigure)
fig.patches.append(rect)
# Add text next to the rectangle
fig.text(0.15, 0.045, 'Shorefast ice width', ha='left', va='center', fontsize=12)
fig.text(0.14, 0.025, 'km', ha='left', va='center', fontsize=12)

fig.savefig("/Users/sebinjohn/Downloads/decadal.svg",transparent=True)

