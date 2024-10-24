import math
from os.path import isfile, join
import obspy.signal.cross_correlation as cr
from shapely.geometry import Point, Polygon
import matplotlib.dates as mdates
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
import cdsapi
import matplotlib.pyplot as plt
import xarray as xr
import pygrib
import pickle
import numpy as np
from obspy import UTCDateTime
import pandas as pd
import pygmt
import os
from global_land_mask import globe
os.chdir("/Users/sebinjohn/AON_PROJECT/Data/codes")
# import dill
from medfilt import medfilt


#loading station metadata
os.chdir("/Users/sebinjohn/AON_PROJECT/Data/Video Making")
with open("metadta.pkl", "rb") as f:
   long, lat, stationo, env = pickle.load(f)

datapath = "/Users/sebinjohn/AON_PROJECT/Data/*/"

st = UTCDateTime(2019, 11, 1)
et = UTCDateTime(2019, 12, 1)
windw = 1
tim_len = int((et-st)/(3600*windw))
smth_intg = np.zeros((len(stationo), tim_len))

#loading frequency bins
freq = []
name = pd.read_xml("pdf0.xml", xpath="/PsdRoot/Psds[1]/Psd[1]/value[@freq]")
for i in range(95):
    freq.append(name.iloc[i]['freq'])
freq.append(19.740300000000000)

time_frame = []
for i in range(int((et-st)/(3600*windw))):
    time_frame.append(st)
    st = st+(3600*windw)

#loading median filtered and interpolated PSD 
os.chdir("/Users/sebinjohn/AON_PROJECT/Data/median_Power_time_series")
# Python 3: open(..., 'wb')
with open(("median and mean"+str(time_frame[0])+"-"+str(time_frame[-1])+"Secondary.pkl"), 'rb') as f:
          mean_timeseries, median_timeseries = pickle.load(f)




def sort_counterclockwise(points, centre=None):
    '''This function sorts a set of points of a polygon in counter 
    clockwise direction.
    :param points: list of points'''
    
    if centre:
      centre_x, centre_y = centre
    else:
      centre_x, centre_y = sum([x for x, _ in points]) / \
                               len(points), sum(
                                   [y for _, y in points])/len(points)
    angles = [math.atan2(y - centre_y, x - centre_x) for x, y in points]
    counterclockwise_indices = sorted(
        range(len(points)), key=lambda i: angles[i])
    counterclockwise_points = [points[i] for i in counterclockwise_indices]
    return counterclockwise_points


def poly_seis(median_timeseries, mean_timeseries, *args):
    ''' This function creates a polygon from set of points
    and checks if each station is within the polygon
    and calculates the mean SM of that region.
    
    :param median_timeseries: Processed SM timeseries of all station.
    :param mean_timeseries: non processed mean SM timeseries in 5-10s band
    :param *args: points of the polygon
    '''
    
    coords = []
    for point in args:
        coords.append(point)
    counter = sort_counterclockwise(coords, centre=None)
    poly = Polygon(counter)
    wista = []
    for i in range(np.shape(median_timeseries)[1]):
        p1 = Point(median_timeseries[1, i, 5], median_timeseries[2, i, 5])
        if p1.within(poly):
            wista.append(1)
        else:
            wista.append(0)
    c = wista.count(1)
    j = 0
    sum_array = np.empty([c, 2, len(median_timeseries[0, 0, 1:])])
    for i in range(len(wista)):
        if wista[i] == 1:
            sum_array[j, 0, :] = median_timeseries[0, i, 1:]
            sum_array[j, 1, :] = mean_timeseries[0, i, 1:]
            j += 1
    sum_tot = np.empty([2, len(median_timeseries[0, 0, 1:])])
    for i in range(len(sum_array[0, 0, :])):
        sum_tot[0, i] = np.sum(sum_array[:, 0, i]) / \
                               (c-np.count_nonzero(sum_array[:, 0, i] == 0))
        sum_tot[1, i] = np.sum(sum_array[:, 1, i]) / \
                               (c-np.count_nonzero(sum_array[:, 1, i] == 0))
    return wista, poly, counter, sum_array, sum_tot


def cross_correlate(time_frame, sum_tot, itera, *argv):
    ''' calculates mean SM time 
    series of different regions and cross correlates it with
    significant wave height and return the correlation values
    :param time_frame: list of time frames
    :param sum_tot: Mean SM PSD from a region
    :para itera: 0/1, 0 if wave_all is provided as an argument
    '''
    c = 0
    if itera == 1:
        for i in range(len(time_frame)):
            tim = time_frame[i]
            print("searching for wave height for"+str(tim))
            grid, data_wave = wave(tim)
            # sum_w=wave_val(grid,wlat1,wlat2,wlon1,wlon2)
            wave_arr, counterw, lat, lon = wave_array(
                grid, (150, 30), (260, 73), (260, 30), (150, 73))
            if c == 0:
               a, b = np.shape(wave_arr.data)
               wave_all = np.zeros([a, b, len(time_frame)])
               c = 1
            wave_all[:, :, i] = wave_arr.data
    else:
        for i in range(1):
            tim = time_frame[i]
            print("searching for wave height for"+str(tim))
            grid, data_wave = wave(tim)
            # sum_w=wave_val(grid,wlat1,wlat2,wlon1,wlon2)
            wave_arr, counterw, lat, lon = wave_array(
                grid, (150, 30), (260, 73), (260, 30), (150, 73))
        wave_all = argv[0]
    corre_grid = np.zeros([np.shape(wave_all)[0], np.shape(wave_all)[1]])
    corre_inde = np.zeros(np.shape(corre_grid))
    for i in range(np.shape(wave_all)[0]):
        for j in range(np.shape(wave_all)[1]):
            k = cr.correlate(wave_all[i, j, :], sum_tot[0, :], 720)
            corre_grid[i, j] = k[720]
            corre_inde[i, j] = cr.xcorr_max(k)[0]
    return corre_grid, corre_inde, wave_all, counterw, lat, lon



regions = [[(-151.95, 70.3), (-151.6, 66.5), (-141.6, 66.5), (-141.88, 69.4), "north_east"], [(-165.9, 70), (-166.1, 66.5), (-154.3, 66.5), (-153.88, 70), "North_west"], [(-157.1, 59), (-152.8, 57), (-168.4, 52.3), (-170.4, 54.2), "south_west"],
         [(-140, 62), (-140.5, 59.5), (-131.5, 55), (-129, 56), "south_east"], [(-166.322, 62.951), (-153.788, 63.587), [-153.820, 60.768], (-163.582, 59.649), "central_west"], [(-149.853, 64.549), (-141.512, 64.760), (-142.462, 60.537), (-149.370, 60.446), "central_east"]]

def wave(tim):
    '''This function downloads significant wave height dataset
    from ERA5 and returns the corresponding grid and data.
    
    :param tim: time to download signficant waveheight (obspy.UTCDateTime)'''
    
    c = cdsapi.Client()
    os.chdir("/Users/sebinjohn/AON_PROJECT/Data/wave")
    files = os.listdir()
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
    tim_wav = str(tim)[0:14]+"00"
    grib = str(tim)[0:14]+"00"+".grib"
    grbs = pygrib.open(grib)
    grb = grbs[1]
    data_wave = grb.values
    latg, long = grb.latlons()
    lat = (latg[:, 0])
    lon = (long[0, :])
    grid = xr.DataArray(
        data_wave, dims=["lat", "lon"], coords={"lat": lat, "lon": lon}
        )
    return grid, data_wave


def wave_array(grid,*args):
    
    '''This function subsets the significant wave height grid
    to Alaska region for faster processing 
    :param grid: significant wave height grid
    :param *args: expects points of bounding box'''
    # args=[(wlon1,wlat1),(wlon1,wlat2),(wlon2,wlat1),(wlon2,wlat2)]
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

def rm_sea(grid):
    '''This function masks ocean of geographical grid
    :param grid: grid'''
    
    lat=grid.lat.values
    lon=grid.lon.values
    lon1=(lon + 180) % 360 - 180
    grid_x,grid_y=np.meshgrid(lon1,lat)
    mask=~globe.is_land(grid_y, grid_x)
    mod_grid=grid.copy()
    mod_grid.data[mask]=np.nan
    return mod_grid


median_timeseries[median_timeseries==0]=-180
wave_all=[]
#corre_grid,corre_inde,wave_all,counterw,lat,lon=cross_correlate(time_frame,sum_tot,1,wave_all)

z=2
tim=time_frame[2]
grid,data_wave=wave(tim)
grid1=pygmt.datasets.load_earth_relief(resolution='30s', region=[155,250, 40, 75])
grid2=rm_sea(grid1)
output_grid_file = '/Users/sebinjohn/AON_PROJECT/Final_codes_paper/grid2.nc'  # Replace with your desired output file path
grid2.to_netcdf(output_grid_file)
proj="Cyl_stere/200/60/10c"
r=[175,240, 50, 75]

wave_all=np.load("/Users/sebinjohn/AON_PROJECT/Data/wave/wave_all_for cross_correlation_2018-2021_3yr.npy")

#plotting

fig=pygmt.Figure()
os.chdir("/Users/sebinjohn/AON_PROJECT/Final_codes_paper/")
pygmt.config(MAP_FRAME_TYPE="plain",MAP_FRAME_WIDTH="0.5p",MAP_FRAME_PEN="1p")
pygmt.makecpt(cmap="bhw1_deadwood.cpt", series=[0,4000, 1],output="topo.cpt",reverse=True)
# fig.basemap(region=[150,260, 30, 73], projection="L-159/35/33/45/22c")
fig.grdimage(grid=grid,region=r,projection=proj, cmap="expll1.cpt",frame="lrbt",nan_transparent=True) 
fig.grdimage(grid="./grid2.nc",region=r,projection=proj, cmap="topo.cpt",frame="lrbt",nan_transparent=True)
fig.coast(region=r, projection=proj,shorelines=True,frame="lrbt",borders=["1/0.5p,black", "2/0.5p,grey"],area_thresh='600')
fig.text(text=str(tim)[:13],x=205,y=73.8,projection=proj)
fig.colorbar(projection=proj, cmap="expll1.cpt",frame=["x+lsignificant\twave\theight", "y+lm"],position="n0.6/-0.07+w3.5c/0.3c+h")
fig.text(text="Bering Strait",x=-169.4,y=65.4,angle=90)
fig.colorbar(cmap="psd.cpt",frame=["x+lPSD", "y+ldB"],projection=proj,position="n0.05/-0.07+w3.5c/0.3c+h") 
for i in range(len(regions)):
    p1=regions[i][0]
    p2=regions[i][1]
    p3=regions[i][2]
    p4=regions[i][3]
    reg=regions[i][-1]
    wista,poly,coords,sum_array,sum_tot=poly_seis(median_timeseries,mean_timeseries,p1,p2,p3,p4)
    corre_grid,corre_inde,wave_all,counterw,lat,lon=cross_correlate(time_frame,sum_tot,1,wave_all)
    xre=[]
    for ele in coords:
        xre.append(ele[0])
    xre.append(xre[0])
    yr=[]
    for ele in coords:
        yr.append(ele[1])
    yr.append(yr[0])

    xw=[]
    for ele in counterw:
        xw.append(ele[0])
    xw.append(xw[0])
    yw=[]
    for ele in counterw:
        yw.append(ele[1])
    yw.append(yw[0])
    fig.plot(x=xre,y=yr,pen="0.8p,Black",fill="241/171/122@50",transparency=50)
os.chdir("/Users/sebinjohn/AON_PROJECT/Final_codes_paper/")
fig.plot(x=median_timeseries[1,1:,z+1],y=median_timeseries[2,1:,z+1],color=median_timeseries[0,1:,z+1],style="c0.20c",cmap="psd.cpt",pen="black", projection=proj)
fig.text(text=["Gulf of Alaska", "Bering Sea","Chukchi Sea","Beaufort sea"], x=[-145, -178,-171,-145], y=[58, 57,71.5,71.5])
fig.show()
fig.savefig('/Users/sebinjohn/Downloads/map.pdf')

#converting to matplotlib_date for plotting


mattime=[]
for ele in time_frame:
    mattime.append(ele.matplotlib_date)
    


def sr(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)

plt.rcdefaults()
plt.rcParams['ytick.labelsize'] = 10
fig,axs=plt.subplots(nrows=1,ncols=1,figsize=(4,1.5),dpi=300)
axs.plot(mattime,wave_all[30,120],color="blue",label="mean wave height (m)")
axs.xaxis.set_major_locator(mdates.MonthLocator())
leg=axs.legend(frameon=False,markerscale=6,handletextpad=0.6)
for text in leg.get_texts():
    text.set_fontstyle('italic')
# axs.xaxis.set_major_formatter(mdates.DateFormatter())
axs.set_xticks([])
axs.set_yticks([])
axs.set_xlim([min(mattime),max(mattime)])
fig.savefig("/Users/sebinjohn/Downloads/wave.svg",transparent=True)


plt.rcdefaults()
plt.rcParams['ytick.labelsize'] = 9
p1=regions[0][0]
p2=regions[0][1]
p3=regions[0][2]
p4=regions[0][3]
reg=regions[0][-1]
wista,poly,coords,sum_array,sum_tot=poly_seis(median_timeseries,mean_timeseries,p1,p2,p3,p4)
fig,axs=plt.subplots(nrows=1,ncols=1,figsize=(4,1.5),dpi=300)
axs.plot(mattime,sum_tot[0,:],color="orange",label="mean PSD (dB)")
axs.xaxis.set_major_locator(mdates.MonthLocator())
leg=axs.legend(frameon=False,markerscale=6,handletextpad=0.6)
for text in leg.get_texts():
    text.set_fontstyle('italic')
# axs.xaxis.set_major_formatter(mdates.DateFormatter())
axs.set_xticks([])
axs.set_yticks([])
axs.set_xlim([min(mattime),max(mattime)])
fig.savefig("/Users/sebinjohn/Downloads/psd.svg",transparent=True)
