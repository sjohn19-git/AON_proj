#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 15:05:06 2022

@author: sebinjohn
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime
import matplotlib.dates as mdates
import pickle
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm

#defining time_frames for seismic data
st=UTCDateTime(2018,1,1)
et=UTCDateTime(2021,1,1)
windw=1
time_frame=np.array([])
for i in range (int((et-st)/(3600*windw))):
    time_frame=np.append(time_frame,st)
    st=st+(3600*windw)
  

os.chdir("/Users/sebinjohn/AON_PROJECT/Data/Video Making")
with open("metadta.pkl","rb") as f:
   long,lat,stationo ,env = pickle.load(f) 

plot_time=[]
for ele in time_frame:
    plot_time.append(ele.matplotlib_date)   

#extracting station names 
stations = [item.split('final_')[1].split(".npy")[0] for item in stationo]


df=pd.DataFrame()

df["longitude"]=long
df["latitude"]=lat
df["stations"]=stations
 

os.chdir("/Users/sebinjohn/AON_PROJECT/Data/median_Power_time_series")
SPSM=np.load("./median_inerpolated_time_series_SPSM2018-2021.npy")
SM=np.load("./median_inerpolated_time_series_secondary2018-01-01-2021-01-01.npy")
wave=np.load("/Users/sebinjohn/AON_PROJECT/Data/wave/wave_mlo_5844.npy")

pl_spsm=SPSM.copy()
pl_sm=SM.copy()
pl_spsm[pl_spsm==0]=np.nan
pl_sm[pl_sm==0]=np.nan

#stations to plot
#extracting sm and spsm for stations
sta_topl=["A21K","B20K","C21K","F21K"]
pl_slsm=np.empty((len(sta_topl),len(plot_time)))
pl_slspsm=np.empty((len(sta_topl),len(plot_time)))
for i in range(len(sta_topl)):
    index = df.index[df['stations'] == sta_topl[i]][0]
    print(index)
    pl_slsm[i,:]=pl_sm[index,:]
    pl_slspsm[i,:]=pl_spsm[index,:]


#finding 5th percentile noise floor in 4 day moving window fo spsm
len_movwindw=4*24
flr_spsm=np.empty((len(sta_topl),int(len(plot_time)-len_movwindw)))
spsm_time=[]
for i in tqdm(range(pl_slspsm.shape[0])):
    ploi=pl_slspsm[i,:].copy()
    c=0
    for j in range(len(plot_time)-len_movwindw):
        plow=ploi[j:j+len_movwindw]
        if i==0:
            spsm_time.append(plot_time[int(j+(len_movwindw/2))])
        flr_spsm[i,c]=np.percentile(plow,5)
        c+=1

#finding 5th percentile noise floor in 4 day moving window fo sm
len_movwindw=4*24
flr_sm=np.empty((len(sta_topl),int(len(plot_time)-len_movwindw)))
sm_time=[]
for i in tqdm(range(pl_slsm.shape[0])):
    ploi=pl_slsm[i,:].copy()
    c=0
    for j in range(len(plot_time)-len_movwindw):
        plow=ploi[j:j+len_movwindw]
        if i==0:
            sm_time.append(plot_time[int(j+(len_movwindw/2))])
        flr_sm[i,c]=np.percentile(plow,5)
        c+=1


def lowess(time,psds,fr):
    '''function to lowess smoothing
    :param time:list of time stamps
    :param psds: corresponding psds
    :param frac: fraction of data to consider for smoothing'''
    low = sm.nonparametric.lowess(psds, time, frac=fr)
    return low

def sr(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)

date_format=mdates.DateFormatter('%b')

#plotting
plt.rcParams['svg.fonttype'] = 'none'
fig,axes=plt.subplots(nrows=len(sta_topl),ncols=1,figsize=(4,6.5),dpi=700,sharex=True,sharey=True)
for i in tqdm(range(len(sta_topl))):
    sc=axes[i].scatter(spsm_time,flr_spsm[i,:],s=0.5,c="yellowgreen",rasterized=True)
    sc1=axes[i].scatter(sm_time,flr_sm[i,:],s=0.5,c="orange",rasterized=True)
    lowspsm=lowess(spsm_time,flr_spsm[i,:],700/len(spsm_time))
    lowsm=lowess(sm_time,flr_sm[i,:],700/len(sm_time))
    pl1=axes[i].plot(lowspsm[:, 0],lowspsm[:, 1],c="black",alpha=0.5)
    pl2=axes[i].plot(lowsm[:, 0],lowsm[:, 1],c="red",alpha=0.5)
    axes[i].xaxis.set_major_formatter(date_format)
    axes[i].xaxis.set_major_formatter(date_format)
    axes[i].set_ylim([-152,-100])
    axes[i].set_ylim([-152,-100])
    axes[i].set_xlim([min(sm_time),max(sm_time)])
    #axes[i,1].set_xlim([min(sm_time),max(sm_time)])
    axes[i].text(0.05, 0.83,sta_topl[i],transform=axes[i].transAxes,bbox=dict(facecolor='white', alpha=0.8),fontsize=9)
    #axes[i,0].axvline(x=mdates.datestr2num('2019-08-01'), color='grey', linestyle='--',zorder=0)
    #axes[i,1].axvline(x=mdates.datestr2num('2020-05-01'), color='grey', linestyle='--',zorder=0)
    #axes[i].axvline(x=mdates.datestr2num('2019-12-01'), color='grey', linestyle='--',zorder=0)
    #axes[i].axvline(x=mdates.datestr2num('2019-12-01'), color='grey', linestyle='--',zorder=0)
    #axes[i,0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[2,7]))
    axes[i].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[12,6]))
    #ax2 =axes[i,1].twinx()
    #ax2.plot(pwave_time,wave[849,:],zorder=0,alpha=0.3)
    #axes[i, 0].axvspan(mdates.datestr2num('2019-08-01'), mdates.datestr2num('2020-05-31'), facecolor='gray', alpha=0.2)
    #axes[i, 1].axvspan(mdates.datestr2num('2019-08-01'), mdates.datestr2num('2020-05-31'), facecolor='gray', alpha=0.2)
    axes[i].grid(which='major', axis='x', linestyle='--', linewidth=0.5)
    axes[i].grid(which='major', axis='y', linestyle='--', linewidth=0.5)
    if i==3:
        sc=axes[i].scatter([],[],s=0.5,label="SPSM floor (1-2s)",c="yellowgreen")
        sc1=axes[i].scatter([],[],s=0.5,label="SM floor (5-10s)",c="orange")
        leg=axes[i].legend(markerscale=8,fontsize=9,handletextpad=0.1,frameon=False,loc="upper right")
        leg1=axes[i].legend(markerscale=8,fontsize=9,handletextpad=0.1,frameon=False,loc="upper right")      
        for text in leg.get_texts():
            text.set_fontstyle('italic')
        for text in leg1.get_texts():
            text.set_fontstyle('italic')
axes[2].plot([], [], c="black", alpha=0.5, label='smoothed SPSM')
axes[2].plot([], [], c="red", alpha=0.5, label='smoothed SM')
leg3=axes[2].legend(markerscale=8, fontsize=9,frameon=False, loc="upper right")
for text in leg3.get_texts():
    text.set_fontstyle('italic')
fig.text(-0.01, 0.5,'dB(rel. 1 (m/s'+sr('2')+')'+sr('2')+'/Hz)',fontsize=12, va='center', rotation='vertical')
fig.tight_layout()
fig.savefig("/Users/sebinjohn/Downloads/fig1.pdf")



##repeat the above step for second set of stations  

sta_topl=["H21K","K20K","SKN","Q19K"]
pl_slsm=np.empty((len(sta_topl),len(plot_time)))
pl_slspsm=np.empty((len(sta_topl),len(plot_time)))
for i in range(len(sta_topl)):
    index = df.index[df['stations'] == sta_topl[i]][0]
    print(index)
    pl_slsm[i,:]=pl_sm[index,:]
    pl_slspsm[i,:]=pl_spsm[index,:]



len_movwindw=4*24
flr_spsm=np.empty((len(sta_topl),int(len(plot_time)-len_movwindw)))
spsm_time=[]
for i in tqdm(range(pl_slspsm.shape[0])):
    ploi=pl_slspsm[i,:].copy()
    c=0
    for j in range(len(plot_time)-len_movwindw):
        plow=ploi[j:j+len_movwindw]
        if i==0:
            spsm_time.append(plot_time[int(j+(len_movwindw/2))])
        flr_spsm[i,c]=np.percentile(plow,5)
        c+=1

len_movwindw=4*24
flr_sm=np.empty((len(sta_topl),int(len(plot_time)-len_movwindw)))
sm_time=[]
for i in tqdm(range(pl_slsm.shape[0])):
    ploi=pl_slsm[i,:].copy()
    c=0
    for j in range(len(plot_time)-len_movwindw):
        plow=ploi[j:j+len_movwindw]
        if i==0:
            sm_time.append(plot_time[int(j+(len_movwindw/2))])
        flr_sm[i,c]=np.percentile(plow,5)
        c+=1


def lowess(time,psds,fr):
    low = sm.nonparametric.lowess(psds, time, frac=fr)
    return low

def sr(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)

date_format=mdates.DateFormatter('%b')


plt.rcParams['svg.fonttype'] = 'none'
fig,axes=plt.subplots(nrows=len(sta_topl),ncols=1,figsize=(4,6.5),dpi=700,sharex=True,sharey=True)
for i in tqdm(range(len(sta_topl))):
    sc=axes[i].scatter(spsm_time,flr_spsm[i,:],s=0.5,c="yellowgreen",rasterized=True)
    sc1=axes[i].scatter(sm_time,flr_sm[i,:],s=0.5,c="orange",rasterized=True)
    lowspsm=lowess(spsm_time,flr_spsm[i,:],700/len(spsm_time))
    lowsm=lowess(sm_time,flr_sm[i,:],700/len(sm_time))
    pl1=axes[i].plot(lowspsm[:, 0],lowspsm[:, 1],c="black",alpha=0.5)
    pl2=axes[i].plot(lowsm[:, 0],lowsm[:, 1],c="red",alpha=0.5)
    axes[i].xaxis.set_major_formatter(date_format)
    axes[i].xaxis.set_major_formatter(date_format)
    axes[i].set_ylim([-152,-100])
    axes[i].set_ylim([-152,-100])
    axes[i].set_xlim([min(sm_time),max(sm_time)])
    #axes[i,1].set_xlim([min(sm_time),max(sm_time)])
    axes[i].text(0.05, 0.83,sta_topl[i],transform=axes[i].transAxes,bbox=dict(facecolor='white', alpha=0.8),fontsize=9)
    #axes[i,0].axvline(x=mdates.datestr2num('2019-08-01'), color='grey', linestyle='--',zorder=0)
    #axes[i,1].axvline(x=mdates.datestr2num('2020-05-01'), color='grey', linestyle='--',zorder=0)
    #axes[i].axvline(x=mdates.datestr2num('2019-12-01'), color='grey', linestyle='--',zorder=0)
    #axes[i].axvline(x=mdates.datestr2num('2019-12-01'), color='grey', linestyle='--',zorder=0)
    #axes[i,0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[2,7]))
    axes[i].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[12,6]))
    #ax2 =axes[i,1].twinx()
    #ax2.plot(pwave_time,wave[849,:],zorder=0,alpha=0.3)
    #axes[i, 0].axvspan(mdates.datestr2num('2019-08-01'), mdates.datestr2num('2020-05-31'), facecolor='gray', alpha=0.2)
    #axes[i, 1].axvspan(mdates.datestr2num('2019-08-01'), mdates.datestr2num('2020-05-31'), facecolor='gray', alpha=0.2)
    axes[i].grid(which='major', axis='x', linestyle='--', linewidth=0.5)
    axes[i].grid(which='major', axis='y', linestyle='--', linewidth=0.5)
    if i==3:
        sc=axes[i].scatter([],[],s=0.5,label="SPSM floor (1-2s)",c="yellowgreen")
        sc1=axes[i].scatter([],[],s=0.5,label="SM floor (5-10s)",c="orange")
        leg=axes[i].legend(markerscale=8,fontsize=9,handletextpad=0.1,frameon=False,loc="upper right")
        leg1=axes[i].legend(markerscale=8,fontsize=9,handletextpad=0.1,frameon=False,loc="upper right")      
        for text in leg.get_texts():
            text.set_fontstyle('italic')
        for text in leg1.get_texts():
            text.set_fontstyle('italic')
axes[2].plot([], [], c="black", alpha=0.5, label='smoothed SPSM')
axes[2].plot([], [], c="red", alpha=0.5, label='smoothed SM')
leg3=axes[2].legend(markerscale=8, fontsize=9,frameon=False, loc="upper right")
for text in leg3.get_texts():
    text.set_fontstyle('italic')
fig.text(-0.01, 0.5,'dB(rel. 1 (m/s'+sr('2')+')'+sr('2')+'/Hz)',fontsize=12, va='center', rotation='vertical')
fig.tight_layout()
fig.savefig("/Users/sebinjohn/Downloads/fig2.pdf")





