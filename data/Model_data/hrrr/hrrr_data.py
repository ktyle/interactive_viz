#!/usr/bin/env python
# coding: utf-8

# In[6]:


import wget
import requests
from bs4 import BeautifulSoup
import datetime
import os
import glob
import xarray as xr
for file_path in glob.glob("hrrr_data_*"):
    os.remove(file_path)

url = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl'
reqs = requests.get(url)
soup = BeautifulSoup(reqs.text, 'html.parser')

urls_gfs = []
for link in soup.find_all('a'):
    #print(link.get('href'))
    urls_gfs.append(link.get('href'))
    
date = urls_gfs[0][-8::]

# Get the current UTC time
current_time = datetime.datetime.utcnow()

hour = current_time.hour - 1
if hour == -1:
    hour = 23
if hour == 23:
    date = int(date)-1
    


# Format the new time in the format of "HHMM"
time_str = hour
if len(str(time_str)) == 1:
    time_str = f'0{time_str}'
else:
    time_str = time_str
    
if hour == 0 or hour == 6 or hour == 12 or hour == 18:
    end = 48+1
else:
    end = 18+1

for x in range(1,end,1):
    if len(str(x)) == 1:
        x2 = f'0{x}'
    if len(str(x)) == 2:
        x2 = f'{x}'
    print(x2)
    ds_hrrr = wget.download(f'https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl?file=hrrr.t{time_str}z.wrfsfcf{x2}.grib2&lev_1000_m_above_ground=on&lev_mean_sea_level=on&lev_surface=on&var_GUST=on&var_APCP=on&var_CFRZR=on&var_CICEP=on&var_CRAIN=on&var_CSNOW=on&var_REFD=on&var_MSLMA=on&var_TMP=on&var_UGRD=on&var_VGRD=on&subregion=&leftlon=280&rightlon=295&toplat=47&bottomlat=37&dir=%2Fhrrr.{date}%2Fconus', out =f'hrrr_data_{x2}.grb2')
    ds_hrrr_hght = wget.download(f'https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl?file=hrrr.t{time_str}z.wrfsfcf{x2}.grib2&lev_2_m_above_ground=on&var_TMP=on&var_UGRD=on&var_VGRD=on&subregion=&leftlon=280&rightlon=295&toplat=47&bottomlat=37&dir=%2Fhrrr.{date}%2Fconus', out =f'hrrr_data_hght_{x2}.grb2')

    


# In[ ]:




