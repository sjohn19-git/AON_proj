import os 
os.chdir("/Users/sebinjohn/AON_PROJECT/Data/codes")
import pygmt
#import dill
import pandas as pd
from obspy import UTCDateTime
import cv2
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
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from global_land_mask import globe

os.chdir("/Users/sebinjohn/AON_PROJECT/Data/Video Making")

#load station metadata
with open("metadta.pkl","rb") as f:
   long,lat,stationo ,env = pickle.load(f)

#reading xml files of PDF for loading frequency bins on wich 
#PSD is defined
freq=[]
name= pd.read_xml("pdf0.xml", xpath="/PsdRoot/Psds[1]/Psd[1]/value[@freq]")
for i in range (95):
    freq.append(name.iloc[i]['freq'])
freq.append(19.740300000000000)
len(freq)


pygmt.makecpt(cmap="hot",output="psd.cpt", series=[-150,-90,0.005],reverse=True)
pygmt.makecpt(cmap="bathy",output="expll1.cpt", series=[0,14],reverse=True)


#grid = pygmt.datasets.load_earth_relief(resolution="10m", region=[-172, -135, 52, 73])
datapath="/Users/sebinjohn/AON_PROJECT/Data/*/"
st=UTCDateTime(2019,11,1)#start time
et=UTCDateTime(2019,12,1)#endtime
windw=1
tim_len=int((et-st)/(3600*windw))
smth_intg=np.zeros((len(stationo),tim_len))

#one hour interval time stamps with in st and et range
time_frame=[]
for i in range (int((et-st)/(3600*windw))):
    time_frame.append(st)
    st=st+(3600*windw)
    
###these functions produces median filtered and interpolated
##SM time series  



def mean_integ(tim):
    '''This function returns the mean and integral PSD
    in 5-10s band for all station at a particular time instance time
    
    :param tim:time
    '''
    def mean(res):
        su=0
        for i in range(len(res)):
            su+=res[i]
        mea=su/len(res)
        return mea
    def integ(res):
        intg=trapz(res,freq[34:42])
        return intg  
    mea_appen=np.zeros(3).reshape(3,1)
    integ_appen=np.zeros(3).reshape(3,1)
    for i in range(len(long)):
        os.chdir(join("/Users/sebinjohn/AON_PROJECT/Data",env[i].split("_")[-1].split(".")[-2]))
        sta=env[i].split("_")[-1].split(".")[-2]
        with open((str(sta)+".pkl"),"rb") as f:
           sta, starttimeta, endtimeta,starttimeak,endtimeak,sta,cha,loc = pickle.load(f) 
        j=int((tim-starttimeta)/(3600))
        with open((glob.glob(join(datapath+stationo[i]))[0]), 'rb') as g:
            final=np.load(g)
        res=final[34:43,j]
        out_mean=mean(res)
        out_integ=integ(res) 
        mea_appen=np.append(mea_appen,np.array([out_mean,long[i],lat[i]]).reshape(3,1),axis=1)
        integ_appen=np.append(integ_appen,np.array([out_integ,long[i],lat[i]]).reshape(3,1),axis=1)   
    return mea_appen,integ_appen
   
    
def map_plo(st,et):
    '''This function median filters and interpolates the data 
    if datagap is less than 12 hour then for 
    each time sample this function invoke ma_pic_generator function
    which plots the map
    
    :param st: start time (obspy.UTCDateTime)
    :param et: end time (obspy.UTCDateTime)'''
    
    mean_timeseries=np.zeros((len(stationo)+1)*3).reshape(3,len(stationo)+1)
    time_frame=[]
    for i in range (int((et-st)/(3600*windw))):
        time_frame.append(st)
        st=st+(3600*windw)
    for i in range(len(time_frame)):
        tim=time_frame[i]
        print(tim)
        mea_appen,integ_appen=mean_integ(tim)
        mean_timeseries=np.dstack((mean_timeseries,mea_appen))   
        mean_timeseries = mean_timeseries.astype('float64')
        median_timeseries=np.copy(mean_timeseries)
        #median_timeseries[median_timeseries==0]=np.nan
        np.count_nonzero(np.isnan(median_timeseries))
    for j in range(1,np.shape(mean_timeseries)[1]):
        median_timeseries[0,j,1:]=medfilt(median_timeseries[0,j,1:],7)
        print("interpolating station "+stationo[j-1][6:])
        interp_inde=np.array([])
        interp_x=median_timeseries[0,j,1:]
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
            median_timeseries[0,j,interp_inde]=interp
        else:
            continue
    # median_timeseries=np.nan_to_num(median_timeseries)
    median_timeseries[median_timeseries==0]=-180
    for i in range(len(time_frame)):
        tim=time_frame[i]
        print("searching for wave height for "+str(tim))
        grid,data_wave=wave(tim)
        ma_pic_generator(tim,grid,median_timeseries,i)
        #ma_pic_generator_samp(tim,grid,median_timeseries,i)
        #fig_merge(tim)
    return mean_timeseries,median_timeseries

        
    

#loading processed SM 3 year time series   
os.chdir("/Users/sebinjohn/AON_PROJECT/Data/median_Power_time_series")
with open(("median and mean"+str(time_frame[0])+"-"+str(time_frame[-1])+"Secondary.pkl"), 'rb') as f:  # Python 3: open(..., 'wb')
          mean_timeseries,median_timeseries=pickle.load(f)
#setting no data value to -180 so that it is outside cmap scale
median_timeseries[median_timeseries==0]=-180 
os.chdir("/Users/sebinjohn/AON_PROJECT/Data/Video Making")




        
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
   
    

def plane_fitting(tim,grid,median_timeseries,z):
    '''fits a plane to the PSD grid using least squares and returns the
    angle of the maximum gradient direction. PSD below -160 db and above 
    -90 db are considered outliers and excluded from invrsion
    
    :param time: time 
    :param median_timeseries: PSD grid
    :param z: index position of time in time_frame'''
    
    xs=median_timeseries[1,1:,z+1].copy()
    ys=median_timeseries[2,1:,z+1].copy()
    zs=median_timeseries[0,1:,z+1].copy()
    dat_gap=np.where(np.logical_or(zs>=-90,zs<=-160))
    xs = np.delete(xs,dat_gap)
    ys=np.delete(ys,dat_gap )
    zs=np.delete(zs,dat_gap)
    # plot raw data
    # do fit
    tmp_A = []
    tmp_b = []
    for i in range(len(xs)):
        tmp_A.append([xs[i], ys[i], 1])
        tmp_b.append(zs[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
        # Manual solution
    fit = (A.T * A).I * A.T * b
    errors = b - A * fit
    residual = np.linalg.norm(errors)
    print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
        #print("errors: \n", errors)
        #print("residual:", residual)
    vector_1 = [fit[0,0],fit[1,0]]
    vector_2 = [-1, 0]
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)*(180/3.14159)
    print("angle is =",angle)
    plt.figure(figsize=(12,12))
    ax = plt.subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, color='b')
    # plot plane
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                      np.arange(ylim[0], ylim[1]))
    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
    # ax.plot_wireframe(X,Y,Z, color='k')
    # ax.set_xlabel('lat')
    # ax.set_ylabel('lon')
    # ax.set_zlabel('power')
    # ax.set_zlim(-170,-80)
    # ax.set_title(str(tim))
    #plt.savefig(str(tim)+"plane.jpg",dpi=700)
    plt.close()
    return angle,vector_1   


    
def ma_pic_generator(tim,grid,median_timeseries,z):
    '''This function plots significant waveheight and
    PSD for each station in a map. Assumes topography grid and 
    cmaps are located in specific directory. Plane fitting 
    function is required.
    
    :param tim: time
    :param grid: significant waveheight grid
    :param median_timeseries: PSD time series of each station'''
    
    print("plotting "+str(tim),z)
    #this is topography grid
    grid1 = '/Users/sebinjohn/AON_PROJECT/Final_codes_paper/grid3.nc'
    fig=pygmt.Figure()
    pygmt.config(MAP_FRAME_TYPE="plain",MAP_FRAME_PEN="0.5p",FONT_LABEL="10p")
    os.chdir("/Users/sebinjohn/AON_PROJECT/Final_codes_paper/")
    pr="L-159/35/33/45/10c"
    reg="178/44/260/69r"
    #fig.basemap(region=[150,260, 30, 73], projection="L-159/35/33/45/22c")
    fig.grdimage(grid=grid,region=reg,projection=pr, cmap="expll1.cpt",frame="f",nan_transparent=True)
    fig.grdimage(grid=grid1,region=reg,projection=pr, cmap="topo.cpt",frame="f",nan_transparent=True)
    fig.coast(region=reg, projection=pr,shorelines=["1/0.15p,black","2/0.25p,grey"],frame="f",borders=["1/0.2p,black", "2/0.5p,grey"],area_thresh='600') 
    fig.plot(x=median_timeseries[1,1:,z+1],y=median_timeseries[2,1:,z+1],fill=median_timeseries[0,1:,z+1],style="c0.25c",cmap="psd.cpt",pen="black", projection=pr)
    angle,vector_1=plane_fitting(tim,grid,median_timeseries,i)
    #fig.grdimage(grid=grid,region=[150,260, 30, 73],projection="L-159/35/33/45/20c", cmap="expll1.cpt",frame="f",nan_transparent=True)
    #fig.coast(region=[150,260, 30, 73], projection="L-159/35/33/45/20c",shorelines=True)
    fig.colorbar(cmap="psd.cpt",frame=["x+lSeismic Power", "y+ldb"],projection=pr,position="n0.05/-0.1+w3.5c/0.3c+h")  
    fig.colorbar(projection=pr, cmap="expll1.cpt",frame=["x+lSignificant Wave Height", "y+lm"],position="n0.57/-0.1+w3.5c/0.3c+h")
    fig.plot(
        region=reg,
        projection=pr,
        x=[-149.5432],
        y=[65.8251],
        frame="f",
        style="v0.4c+e",
        direction=[[angle], [-1.5*np.linalg.norm(vector_1)]],
        pen="1p",
        color="grey",
    )
    fig.text(text=str(tim)[:-11],x=-155,y=73.5,projection=pr)
    os.chdir('/Users/sebinjohn/AON_PROJECT/Data/Video Making/one_panel/')
    fig.savefig(str(tim)+"wave.png",transparent=True)
    if z<10:
        fig.show()






# def fig_merge(tim):
#     os.chdir("/Users/sebinjohn/AON_PROJECT/Data/Video Making")
#     image1 = Image.open(str(tim)+"wave.jpg")
#     image1_size = image1.size
#     new_image = Image.new('RGB',(7423,3575), (250,250,250))
#     new_image.paste(image1,(0,0))
#     new_image.save(str(tim)+".jpg","JPEG")



def convert_pictures_to_video(pathIn, pathOut, fps, time):
    ''' this function converts images to video
    
    :param pathIn: path for images
    :param pathOut: out path for video
    :param fps: frames per second
    :param time: time'''
    pics=glob.glob(join(directory+"/*wave.jpg"))
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
    out=cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'VP80'), fps,size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()




# Example:
directory="//Users/sebinjohn/AON_PROJECT/Data/Video Making/one_panel"
pathIn=directory+'/*.png'
pathOut=directory+"/"+str(st)+"-"+str(et)+"w.avi"
fps=8
time=1 # the duration of each picture in the video


st=st=UTCDateTime(2019,11,12,20)
et=UTCDateTime(2019,11,12,21)

mean_timeseries,median_timeseries=map_plo(st,et)



convert_pictures_to_video(pathIn, pathOut, fps, time)


###################


##following code is just an adaptation of above functions to plot just
##snapshots in figure.aa.

def map_plo_samp(st,et):
    ies=[0,613,609,299,196,165]
    for ll in range(len(ies)):
        i=ies[ll]
        if ll>3:
            flag=True
        else:
            flag=False
        mean_timeseries=np.zeros((len(stationo)+1)*3).reshape(3,len(stationo)+1)
        tim=time_frame[i]
        print(tim)
        mea_appen=np.zeros(3).reshape(3,1)
        integ_appen=np.zeros(3).reshape(3,1)
        mea_appen,integ_appen=mean_integ(tim,mea_appen,integ_appen)
        mean_timeseries=np.dstack((mean_timeseries,mea_appen))   
        mean_timeseries = mean_timeseries.astype('float64')
        median_timeseries=np.copy(mean_timeseries)
        #median_timeseries[median_timeseries==0]=np.nan
        np.count_nonzero(np.isnan(median_timeseries))
        for j in range(1,np.shape(mean_timeseries)[1]):
            median_timeseries[0,j,1:]=medfilt(median_timeseries[0,j,1:],7)
            print("interpolating station "+stationo[j-1][6:])
            interp_inde=np.array([])
            interp_x=median_timeseries[0,j,1:]
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
                median_timeseries[0,j,interp_inde]=interp
            else:
                continue
        # median_timeseries=np.nan_to_num(median_timeseries)
        median_timeseries[median_timeseries==0]=-180
        print("searching for wave height for "+str(tim))
        grid,data_wave=wave(tim)
        ma_pic_generator_samp(tim,grid,median_timeseries,i,flag)
        #ma_pic_generator_samp(tim,grid,median_timeseries,i)
        #fig_merge(tim)
    return mean_timeseries,median_timeseries

def rm_sea(grid):
    '''function to remove bathymetry values from topography grid'''
    lat=grid.lat.values
    lon=grid.lon.values
    lon1=(lon + 180) % 360 - 180
    grid_x,grid_y=np.meshgrid(lon1,lat)
    mask=~globe.is_land(grid_y, grid_x)
    mod_grid=grid.copy()
    mod_grid.data[mask]=np.nan
    return mod_grid

grid1=pygmt.datasets.load_earth_relief(resolution='20m', region="g")
grid2=rm_sea(grid1)
output_grid_file = '/Users/sebinjohn/AON_PROJECT/Final_codes_paper/grid3.nc'  # Replace with your desired output file path
grid2.to_netcdf(output_grid_file)

def ma_pic_generator_samp(tim,grid,median_timeseries,z,flag):
    if flag==False:
        print("plotting "+str(tim),z)
        grid1 = '/Users/sebinjohn/AON_PROJECT/Final_codes_paper/grid2.nc'
        fig=pygmt.Figure()
        pygmt.config(MAP_FRAME_TYPE="plain",MAP_FRAME_PEN="0.5p,black",FONT_LABEL="10p")
        os.chdir("/Users/sebinjohn/AON_PROJECT/Final_codes_paper/")
        pr="L-159/35/33/45/10c"
        #pr="G259.5/90/10c"
        re="178/44/260/69r"
        #re=[180,340, 35, 75]
        #fig.basemap(region=[150,260, 30, 73], projection="L-159/35/33/45/22c")
        fig.grdimage(grid=grid,region=re,projection=pr, cmap="expll1.cpt",nan_transparent=True)
        fig.grdimage(grid=grid1,region=re,projection=pr, cmap="topo.cpt",nan_transparent=True)
        fig.coast(region=re, projection=pr,shorelines=["1/0.15p,black","2/0.25p,grey"],frame="lrtb",borders=["1/0.2p,black", "2/0.5p,grey"],area_thresh='600') 
        fig.plot(x=median_timeseries[1,1:,0+1],y=median_timeseries[2,1:,0+1],fill=median_timeseries[0,1:,0+1],style="c0.25c",cmap="psd.cpt",pen="black", projection=pr)
        angle,vector_1=plane_fitting(tim,grid,median_timeseries,0)
        #fig.grdimage(grid=grid,region=re,projection=pr, cmap="expll1.cpt",frame="f",nan_transparent=True)
        #fig.coast(region=[150,260, 30, 73], projection="L-159/35/33/45/20c",shorelines=True)
        fig.colorbar(cmap="psd.cpt",frame=["x+lSeismic Power", "y+ldb"],projection=pr,position="n0.05/-0.1+w3.5c/0.3c+h")  
        fig.colorbar(projection=pr, cmap="expll1.cpt",frame=["x+lSignificant Wave Height", "y+lm"],position="n0.57/-0.1+w3.5c/0.3c+h")
        fig.plot(
            region=re,
            projection=pr,
            x=[-149.5432],
            y=[65.8251],
            frame="lrtb",
            style="v0.4c+e",
            direction=[[angle], [-1.5*np.linalg.norm(vector_1)]],
            pen="1p",
            color="grey",
        )
        fig.text(text=str(tim)[:-11],x=-160,y=73.5,projection=pr)
        os.chdir('/Users/sebinjohn/Downloads')
        fig.savefig(str(tim)+"wave.png")
        fig.show()
    else:
        print("plotting "+str(tim),z)
        grid1 = '/Users/sebinjohn/AON_PROJECT/Final_codes_paper/grid3.nc'
        fig=pygmt.Figure()
        pygmt.config(MAP_FRAME_TYPE="plain",MAP_FRAME_PEN="0p,white",FONT_LABEL="10p")
        os.chdir("/Users/sebinjohn/AON_PROJECT/Final_codes_paper/")
        #pr="L-130/35/33/45/10c"
        pr="G255/90/10c"
        #re=[155,250, 35, 75]
        re=[0,360, 30, 90]
        fig.grdimage(grid=grid1,region=re,projection=pr, cmap="topo.cpt",nan_transparent=True)
        #fig.basemap(region=[150,260, 30, 73], projection="L-159/35/33/45/22c")
        fig.grdimage(grid=grid,region=re,projection=pr, cmap="expll1.cpt",nan_transparent=True)
        fig.coast(region=re, projection=pr,shorelines=["1/0.15p,black","2/0.25p,grey"],frame="lrtb",borders=["1/0.2p,black", "2/0.5p,grey"],area_thresh='600') 
        fig.plot(x=median_timeseries[1,1:,0+1],y=median_timeseries[2,1:,0+1],color=median_timeseries[0,1:,0+1],style="c0.15c",cmap="psd.cpt",pen="black", projection=pr)
        angle,vector_1=plane_fitting(tim,grid,median_timeseries,0)
        #fig.grdimage(grid=grid,region=[150,260, 30, 73],projection="L-159/35/33/45/20c", cmap="expll1.cpt",frame="f",nan_transparent=True)
        #fig.coast(region=[150,260, 30, 73], projection="L-159/35/33/45/20c",shorelines=True)
        #fig.colorbar(cmap="psd.cpt",frame=["x+lSeismic Power", "y+ldb"],projection=pr,position="n0.05/-0.1+w3.5c/0.3c+h")  
        #fig.colorbar(projection=pr, cmap="expll1.cpt",frame=["x+lSignificant Wave Height", "y+lm"],position="n0.57/-0.1+w3.5c/0.3c+h")
        # fig.plot(
        #     region=re,
        #     projection=pr,
        #     x=[-149.5432],
        #     y=[65.8251],
        #     style="v0.4c+e",
        #     direction=[[angle-55], [-1.5*np.linalg.norm(vector_1)]],
        #     pen="1p",
        #     color="grey",
        # )
        # fig.text(text=str(tim)[:-11],x=-98,y=85,projection=pr)
        fig.show()
        os.chdir('/Users/sebinjohn/Downloads')
        fig.savefig(str(tim)+"wave.pdf")
       
