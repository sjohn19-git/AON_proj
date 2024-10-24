#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:33:51 2024

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
from matplotlib.dates import MONTHLY,MonthLocator,YearLocator
from matplotlib.dates import DayLocator
import datetime
import obspy.signal.filter as flt
from matplotlib.gridspec import GridSpec
import pandas as pd
import matplotlib.patches as patches
from scipy.optimize import differential_evolution as de
from scipy.signal import find_peaks
import matplotlib.ticker as ticker
from matplotlib.dates import num2date as n2d
from matplotlib.dates import date2num as d2n
from scipy.stats import norm as norm1
from matplotlib.ticker import MultipleLocator


start = UTCDateTime(2009, 1, 1)
end = UTCDateTime(2024, 1, 1)
seis = np.load("/Users/sebinjohn/AON_PROJECT/Data/PS01//2009-01-01T00-2024-01-01T00.npy")
freq = np.load("/Users/sebinjohn/AON_PROJECT/Data/PS01/frequencies.npy")
peri = 1/freq
time = np.arange(start, end, 3600)

# periods using 1.05 to 2.10

spsm2 = seis[53:61, :]
spsm1 = np.mean(spsm2, axis=0)
spsm = medfilt(spsm1, 7)
spsm[spsm==0]=np.nan
spsm[spsm>-100]=np.nan
spsm[spsm<-150]=np.nan

# periods using  5 to 10
sm2 = seis[34:42, :]
sm1 = np.mean(sm2, axis=0)
sm = medfilt(sm1, 7)
sm[sm == 0] = np.nan



plo_time = []
for ele in time:
    plo_time.append(ele.matplotlib_date)
plo_time=np.array(plo_time)


######
%matplotlib inline

# def box(x, *p):
#     height, center, width = p
#     boxcar=height*(center-width/2 < x)*(x < center+width/2)
#     return boxcar

def cum_box(x, *p):
    
    h, start_increase, end_steady, drop_x, drop_val=p
    y = np.full_like(x, drop_val, dtype=np.float64)
    
    # Increase from drop_val to h
    increase_region = (x >= start_increase) & (x <= end_steady)
    y[increase_region] = drop_val + (h - drop_val) * (x[increase_region] - start_increase) / (end_steady - start_increase)
    
    # Stay at height h
    steady_region = (x > end_steady) & (x < drop_x)
    y[steady_region] = h
    
    # Drop to drop_val abruptly
    # The drop_val is already set as initial values in the y array, no additional action needed

    return y

date_example = datetime.date(2019, 3, 1)

def f(p,x,norm_avg):
    resi=np.sum((cum_box(x, *p) - norm_avg)**2)
    pen=(p[2]-p[1])/1.9
    return resi+pen

def con_f(p, x, norm_avg):
    # Ensure the constraint is met
    if p[2] <= p[1]:
        return 1e10 
    if p[3] <= p[2]:
        return 1e10 
    return f(p, x, norm_avg)

date_format1 = mdates.DateFormatter('%d-%b-%y')
storm_peri=[]
amp_peaks=[]
time_of_peaks=[]

#Normalized PSD

norm= np.interp(spsm[~np.isnan(spsm)], (np.nanmin(spsm), np.nanmax(spsm)), (0, 1))
norm_spsm = np.full_like(spsm, np.nan)
norm_spsm[~np.isnan(spsm)] = norm

# cutting each year


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
    exec(f"yr_{j}.append(norm_spsm[i])")
    exec(f"tyr_{j}.append(mdates.date2num(date))")

for ele in years:
    exec(f"{ele}=np.array({ele})")
    exec(f"t{ele}=np.array(t{ele})")


##inv
results=[]
for i in range(len(years)):
    ele=years[i]
    exec(f"x=t{ele}")
    exec(f"y={ele}")
    yr=mdates.num2date(x[0]).year
    seismic_power=pd.Series(y)
    x_nona=x[~np.isnan(seismic_power)]
    norm_avg=seismic_power[~np.isnan(seismic_power)]
    minx1=mdates.date2num(datetime.date(yr,5,1))
    maxx1=mdates.date2num(datetime.date(yr, 7, 10))
    minx2=mdates.date2num(datetime.date(yr,1,1))
    maxx2=mdates.date2num(datetime.date(yr, 12, 1))
    minx3=mdates.date2num(datetime.date(yr,1,2))
    maxx3=mdates.date2num(datetime.date(yr, 12, 1))
    
    p_range=[[0.3,0.8], [minx1, maxx1],[minx2, maxx2],[minx3, maxx3],[0,0.02]]
    result = de(con_f,bounds=p_range, args=(x_nona,norm_avg))
    results.append(result)


results=np.array(results)
np.save("inversion_trape.npy", results)


date_format1 = mdates.DateFormatter('%b')

plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 8



fig,ax=plt.subplots(figsize=(7, 5.67),dpi=600,nrows=5,ncols=3,gridspec_kw={'hspace': 0.1,'wspace': 0.05},sharey=True)
storm_peri=[]
for i in range(len(years)):
    ele=years[i]
    exec(f"x=t{ele}")
    exec(f"y={ele}")
    yr=mdates.num2date(x[0]).year
    seismic_power=pd.Series(y)
    x_nona=x[~np.isnan(seismic_power)]
    norm_avg=seismic_power[~np.isnan(seismic_power)]
    result=results[i]
    
    c = i % 3  
    r = i // 3
    
    ax[r,c].scatter(x_nona,norm_avg,c="yellowgreen",label="SPSM",alpha=0.4,s=0.05,zorder=3,rasterized=True)
    ax[r,c].plot(x_nona, cum_box(x_nona, *result.x),color="red",alpha=0.5,label="Fit")
    ax[r,c].xaxis.set_major_locator(MonthLocator())
    ax[r,c].xaxis.set_major_formatter(date_format1)
    print(yr,result.fun,result.x[1]-result.x[2])
    storm_peri.append([result.x[1],result.x[3]])
    ax[r,c].xaxis.set_major_locator(MonthLocator([1,3,5,7,9,11]))
    ax[r,c].xaxis.set_major_formatter(date_format1)
    xlims=[n2d(x[0]).replace(month=1,day=1,hour=0),n2d(x[0]).replace(month=12,day=31,hour=23)]
    ax[r,c].set_xlim([d2n(xlims[0]),d2n(xlims[1])])
    ax[r, c].yaxis.set_major_locator(MultipleLocator(0.5))
    ax[r,c].axvline(result.x[1],zorder=0,c="grey",ls="--")
    ax[r,c].axvline(result.x[3],zorder=0,c="grey",ls="--")
    #ax[r,c].tick_params(axis='y', labelsize=6)
    ax[r, c].text(
    0.05, 0.925, str(yr), 
    horizontalalignment='left', verticalalignment='top',
    transform=ax[r, c].transAxes,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='square,pad=0.3',linewidth=0.5),
    fontsize=7
)

for i in range(4):  # Iterate through the rows except the last one
    for j in range(3):  # Iterate through all columns
        ax[i, j].set_xticklabels([])  # Remove x-axis labels
plt.legend(markerscale=7,loc="upper right",fontsize=6)
fig.text(0.07, 0.5, 'Normalized Seismic Power', va='center', ha='center', rotation='vertical', fontsize=8)
for axi in ax.flat:
    for spine in axi.spines.values():
            spine.set_linewidth(0.5) 
    axi.tick_params(axis='both', length=2, width=0.5)
plt.show()
fig.savefig("/Users/sebinjohn/Downloads/fits.svg")


#################
storm_bool = np.zeros(len(plo_time), dtype=bool)


for ele in storm_peri:
    a,b=ele[0],ele[1]
    ele_bool=(a <= plo_time) & (plo_time <= b)
    storm_bool[ele_bool]=True


storm_spsm=spsm.copy()
storm_spsm[~storm_bool]=np.nan
fit_storm=storm_spsm[storm_bool][~np.isnan(storm_spsm[storm_bool])]

mean, std_dev = norm1.fit(fit_storm)

# 2. Plot the binned distribution (histogram) with counts
counts, bins, _ = plt.hist(fit_storm, bins=120, density=False, alpha=0.6, color='g', label='Data Histogram')

# 3. Calculate the bin width
bin_width = bins[1] - bins[0]

# 4. Scale the Gaussian fit to match the histogram counts
x = np.linspace(-160, -100, 100)
p = norm1.pdf(x, mean, std_dev) * len(fit_storm) * bin_width

# 5. Plot the scaled Gaussian fit
plt.plot(x, p, 'k', linewidth=2, label='Gaussian Fit')

# 6. Add vertical lines for the mean and standard deviations
plt.axvline(mean, color='r', linestyle='--', label='Mean')
plt.axvline(mean + std_dev, color='b', linestyle='--', label='1 Std Dev')
plt.axvline(mean - std_dev, color='b', linestyle='--')
plt.axvline(mean + 2 * std_dev, color='b', linestyle=':', label='2 Std Dev')
plt.axvline(mean - 2 * std_dev, color='b', linestyle=':')
plt.axvline(mean + 3 * std_dev, color='b', linestyle='-.', label='3 Std Dev')
plt.axvline(mean - 3 * std_dev, color='b', linestyle='-.')

# 7. Add labels, title, and legend
plt.xlabel('Values')
plt.ylabel('Counts')
plt.title(f'Fit results: mean = {mean:.2f}, std_dev = {std_dev:.2f}')
plt.legend()



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

plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 8

storm_index1=[]
storm_index2=[]
storm_index3=[]
thresh1=mean+1*std_dev
thresh2=mean+1.5*std_dev
thresh3=mean+0.25*std_dev
date_format2 = mdates.DateFormatter('%b') 
fig,ax=plt.subplots(figsize=(5,2.5),dpi=600)
for i in range(len(years)):
    ele=years[i]
    exec(f"x=t{ele}")
    exec(f"y={ele}")
    yr=mdates.num2date(x[0]).year
    strm_peri_ele=storm_peri[i]
    a,b=strm_peri_ele[0],strm_peri_ele[1]
    ele_bool=(a <= x) & (x <= b)
    storm_index1.append(np.sum(y[ele_bool]>thresh1))
    storm_index2.append(np.sum(y[ele_bool]>thresh2))
    storm_index3.append(np.sum(y[ele_bool]>thresh3))
    samp_y=strm_peri_ele[0]+(strm_peri_ele[1]-strm_peri_ele[0])/2
    d1,d2=mdates.num2date(samp_y),mdates.num2date(strm_peri_ele[0])
    adj_samp_y = mdates.date2num(d1.replace(year=2021))
    text_y=mdates.date2num(d2.replace(year=2021))
    ax.errorbar(yr, adj_samp_y, yerr=(strm_peri_ele[1]-strm_peri_ele[0])/2, fmt='none', capsize=5, capthick=1,ecolor="black",lw=0.5)
    ax.yaxis.set_major_locator(MonthLocator())
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    x_major_locator = ticker.MultipleLocator(1) 
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_formatter(date_format2)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.errorbar([2050], [adj_samp_y], yerr=(strm_peri_ele[1]-strm_peri_ele[0])/2, fmt='none', capsize=4, capthick=1,ecolor="black",lw=0.5, label="Open ocean period")
ax.set_xlim([2008.5,2023.5])
ax1=ax.twinx()
ax.set_ylim([d2n(datetime.date(2021,5,1)),d2n(datetime.date(2021,12,1))])
ax1.plot(np.arange(2009,2024),storm_index1,color="red",label="Storm index",marker="*",lw=0.5,markersize=1.5)    
#ax1.plot(np.arange(2009,2024),storm_index2,color="green",label="1.5 std dev")
#ax1.plot(np.arange(2009,2024),storm_index3,color="blue",label="0.25 std dev")   
ax1.set_ylabel('Hours',fontsize=7)
ax1.tick_params(axis='y', labelsize=7)
ax1.set_ylim([0,1000])

handles_ax, labels_ax = ax.get_legend_handles_labels()
handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()
# Combine legends and add to plot
handles = handles_ax + handles_ax1
labels = labels_ax + labels_ax1
plt.legend(handles, labels, loc="lower left", fontsize=7.5)

fig.savefig("/Users/sebinjohn/Downloads/figsae.svg")
