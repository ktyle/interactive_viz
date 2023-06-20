#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import json
import time
import datetime as dt
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import pydeck as pdk
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
from pyproj import Proj, transform

st.set_page_config(page_title="Map Example", page_icon=":guardsman:", layout="wide")

valid_time = datetime.now()
valid_time2 = valid_time - timedelta(days=1)

now = dt.datetime.now()
valid_time3 = now.time()
end_time = dt.time(hour=23, minute=59)


# In[2]:


precip_types_map = {'Unknown Precip': 0,
                    'Heavy Unknown Precip': 0,
                    'Unknown Precip with Thunderstorm': 0,
                  'Snow and/or Graupel': 1,
                    'Heavy Snow and/or Graupel': 1,
                    'Snow and/or Graupel with Thunderstorm': 1,
                  'Ice Pellets/Sleet': 2,
                    'Heavy Ice Pellets/Sleet': 2,
                    'Ice Pellets/Sleet with Thunderstorm': 2,
                  'Mixed Ice Pellets and Snow': 3,
                    'Heavy Mixed Ice Pellets and Snow': 3,
                    'Mixed Ice Pellets and Snow with Thunderstorm': 3,
                  'Freezing Rain': 4,
                    'Heavy Freezing Rain': 4,
                    'Freezing Rain with Thunderstorm': 4,
                  'Freezing Drizzle': 4, #don't have separate category for this currently
                  'Mixed Freezing Rain and Ice Pellets': 5,
                    'Heavy Mixed Freezing Rain and Ice Pellets': 5,
                    'Mixed Freezing Rain and Ice Pellets with Thunderstorm': 5,
                  'Rain': 6, 
                    'Heavy Rain': 6, 
                    'Rain with Thunderstorm': 6, 
                  'Drizzle': 6, #don't have separate category for this
                  'Mixed Rain and Snow': 7,
                    'Heavy Mixed Rain and Snow': 7,
                    'Mixed Rain and Snow with Thunderstorm': 7,
                  'Mixed Rain and Ice Pellets': 8,
                    'Heavy Mixed Rain and Ice Pellets': 8,
                    'Mixed Rain and Ice Pellets with Thunderstorm': 8,
                  }

precip_types_color = {'Unknown Precip': '#707070',
                      'Heavy Unknown Precip': '#707070',
                      'Unknown Precip with Thunderstorm': '#707070',
                  'Snow and/or Graupel': '#1f48cf',
                      'Heavy Snow and/or Graupel': '#1f48cf',
                      'Snow and/or Graupel with Thunderstorm': '#1f48cf',
                  'Ice Pellets/Sleet': '#ac6cd9',
                      'Heavy Ice Pellets/Sleet': '#ac6cd9',
                      'Ice Pellets/Sleet with Thunderstorm': '#ac6cd9',
                  'Mixed Ice Pellets and Snow': '#56419c',
                      'Heavy Mixed Ice Pellets and Snow': '#56419c',
                      'Mixed Ice Pellets and Snow with Thunderstorm': '#56419c',
                  'Freezing Rain': '#e30bc6',
                      'Heavy Freezing Rain': '#e30bc6',
                      'Freezing Rain with Thunderstorm': '#e30bc6',
                  'Freezing Drizzle': '#e30bc6', #don't have separate category for this currently
                  'Mixed Freezing Rain and Ice Pellets': '#8502b5',
                      'Heavy Mixed Freezing Rain and Ice Pellets': '#8502b5',
                      'Mixed Freezing Rain and Ice Pellets with Thunderstorm': '#8502b5',
                  'Rain': '#169c2f', 
                      'Heavy Rain': '#169c2f', 
                      'Rain with Thunderstorm': '#169c2f', 
                  'Drizzle': '#169c2f', #don't have separate category for this
                  'Mixed Rain and Snow': '#0be3df',
                      'Heavy Mixed Rain and Snow': '#0be3df',
                      'Mixed Rain and Snow with Thunderstorm': '#0be3df',
                  'Mixed Rain and Ice Pellets': '#42ffca',
                      'Heavy Mixed Rain and Ice Pellets': '#42ffca',
                      'Mixed Rain and Ice Pellets with Thunderstorm': '#42ffca',
                  }


# ### Downloading Latest MPING Data

# In[3]:


#set up an interval for the MPING obs
interval_min = 1440

#Setup variables
var_name = 'mping' #used in plot filename

api_key = '3accb44957d69b28a7e1fd4411da3fb94a07d971'

def get_mping_obs(valid_time, interval_min = interval_min, time_window = 'center'):
    '''Retrieve mPING observations and parse into a pandas DataFrame
    Inputs: 
        valid_time (datetime object) - desired observation time
        interval_min (int) - range of time in minutes to get observations  
        time_window ("begin", "center" or "end") 
            - "begin": get obs for interval_min beginning at valid_time
            - "center": get obs centered on valid_time
            - "end": get obs for interval_min ending at valid_time
    Return:
        pandas DataFrame with nicely parsed obs'''

    reqheaders = {
    'content-type': 'application/json',
    'Authorization': f'Token {api_key}',
    }

    #Form API query URL
    mping_url_base = 'http://mping.ou.edu/mping/api/v2/reports'

    #Add filters to base URL
    if time_window == 'begin':
        #get all reports for time interval beginning at valid time
        mping_start = hr
        mping_end = hr + timedelta(minutes = interval_min)
        mping_url = f'{mping_url_base}?obtime_gte={mping_start:%Y-%m-%d %H:%M:%S}&obtime_lt={mping_end:%Y-%m-%d %H:%M:%S}'
        #print (mping_url)
        print (f'getting mPING reports from {interval_min} min beginning at {hr:%H:%Mz %d %b %Y}')
    elif time_window == 'end':
        #get all reports for 1h preceding valid time
        #mping_valid = valid_time - timedelta(minutes = interval_min)
        #mping_url = f'{mping_url_base}?year={mping_valid:%Y}&month={mping_valid:%-m}&day={mping_valid:%-d}&hour={mping_valid:%-H}'

        #get all reports for time interval ending at valid time
        mping_start = valid_time - timedelta(minutes = interval_min)
        mping_end = valid_time
        mping_url = f'{mping_url_base}?obtime_gt={mping_start:%Y-%m-%d %H:%M:%S}&obtime_lte={mping_end:%Y-%m-%d %H:%M:%S}'
        #print (mping_url)
        print (f'getting mPING reports from {interval_min} min ending at {valid_time:%H:%Mz %d %b %Y}')
    elif time_window == 'center':
        #get all reports for time interval centered on valid time
        mping_start = valid_time - timedelta(minutes = interval_min//2)
        mping_end = valid_time + timedelta(minutes = interval_min//2)
        mping_url = f'{mping_url_base}?obtime_gte={mping_start:%Y-%m-%d %H:%M:%S}&obtime_lt={mping_end:%Y-%m-%d %H:%M:%S}'
        #print (mping_url)
        print (f'getting mPING reports from {interval_min} min centered on {valid_time:%H:%Mz %d %b %Y}')

    #Retrieve JSON data
    response = requests.get(mping_url, headers = reqheaders)
    if response.status_code != 200:
        print (f'request failed with status code {response.status_code}')
        return
    else:
        data = response.json()
        print (f'retrieved {data["count"]} reports')

    #Read mPING json into dataframe for easier filtering
    df = pd.DataFrame.from_dict(data['results'])
    #Parse out lat/lon data
    df['longitude'] = [geom['coordinates'][0] for geom in df['geom']]
    df['latitude'] = [geom['coordinates'][1] for geom in df['geom']]

    #could stop here
    #return df

    #Also map mPING p-types to p-type values/colors used in colorbar
    mping_types_map_m = {'NULL': 0,
                      'Snow and/or Graupel': 1,
                      'Ice Pellets/Sleet': 2,
                      'Mixed Ice Pellets and Snow': 3,
                      'Freezing Rain': 4,
                      'Freezing Drizzle': 4, #don't have separate category for this currently
                      'Mixed Freezing Rain and Ice Pellets': 5,
                      'Rain': 6, 
                      'Drizzle': 6, #don't have separate category for this
                      'Mixed Rain and Snow': 7,
                      'Mixed Rain and Ice Pellets': 8,
                      }
    #map indexes to colors (optional: only works if continuous value HRRRE colorbar used)
    #mping_colors_map = {k:ptype_colors[int(v)] for k,v in mping_types_map.items()}

    #Subtract 0.01 to make p-type categories correct
    df['ptype'] = df['description'].map(mping_types_map_m)
    #df['ptype_colors'] = df['description'].map(mping_colors_map)

    return df

MPING_data = get_mping_obs(valid_time, interval_min = interval_min, time_window = 'end')


# ### Filtering MPING DATA to remove NULL/Impacts/Fog Reports

# In[4]:


MPING_Filter = MPING_data[MPING_data['description'] != 'NULL']
MPING_Filter = MPING_Filter.dropna(subset='ptype')


# ### Replacing Some codes that I/ASOS don't have categories for and changing column names to line up with ASOS.

# In[5]:


MPING_Filter['description'] =MPING_Filter['description'].replace(['Drizzle'],
                                                                ['Rain']) 

MPING_Filter['description'] =MPING_Filter['description'].replace(['Freezing Drizzle'],
                                                                ['Freezing Rain']) 

MPING_Filter = MPING_Filter.rename(columns={'description':'wxcodes'})
MPING_Filter = MPING_Filter.rename(columns={'obtime':'valid'})
MPING_Filter['valid'] = MPING_Filter['valid'].str.replace('T',' ')


# ### Adding a station column to let the user know it's an MPING report

# In[6]:


MPING_Filter = MPING_Filter.assign(station='MPING')


# ### Adding the Colormap

# In[7]:


MPING_Filter['color'] = MPING_Filter['wxcodes'].map(precip_types_color)


# ### Rounding time to make a timeseries

# In[8]:


MPING_Filter['valid_rounded'] =pd.to_datetime(MPING_Filter['valid'])
MPING_Filter['valid_rounded'] = MPING_Filter['valid_rounded'].apply(lambda x: x + pd.Timedelta(minutes=60) if x.minute >= 1 and x.minute < 30 else x)
MPING_Filter['valid_rounded'] = MPING_Filter['valid_rounded'].apply(lambda x: x + pd.Timedelta(hours=1) if x.minute >= 30 else x)
MPING_Filter['valid_rounded'] = MPING_Filter['valid_rounded'].dt.strftime('%Y-%m-%d %H:00Z')


# In[9]:


### Renaming Columns


# In[10]:


MPING_data = MPING_Filter.rename(columns={'valid':'OBS Time','station' : 'Station','longitude':'Lon','latitude':'Lat','wxcodes': 'Current WX','valid_rounded':'Time'})


# In[11]:


MPING_data['Time'] = pd.to_datetime(MPING_data['Time'])


# ### Downloading Latest ASOS Data

# In[13]:


"""
Example script that scrapes data from the IEM ASOS download service
"""

# Python 2 and 3: alternative 4
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen
    
MAX_ATTEMPTS = 6


# HTTPS here can be problematic for installs that don't have Lets Encrypt CA
SERVICE = "http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"

def download_data(uri):
    """Fetch the data from the IEM
    The IEM download service has some protections in place to keep the number
    of inbound requests in check.  This function implements an exponential
    backoff to keep individual downloads from erroring.
    Args:
      uri (string): URL to fetch
    Returns:
      string data
    """
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        try:
            data = urlopen(uri, timeout=300).read().decode("utf-8")
            if data is not None and not data.startswith("ERROR"):
                return data
        except Exception as exp:
            print("download_data(%s) failed with %s" % (uri, exp))
            time.sleep(5)
        attempt += 1

    print("Exhausted attempts to download, returning empty data")
    return ""


def download_alldata():
    """An alternative method that fetches all available data.

    Service supports up to 24 hours worth of data at a time."""
    # timestamps in UTC to request data for
    startts = valid_time - timedelta(hours=1)
    endts = valid_time
    interval = timedelta(hours=1)

    service = SERVICE + "data=all&tz=Etc/UTC&format=comma&latlon=yes&"

    now = endts

    thisurl = service
    thisurl += now.strftime("year1=%Y&month1=%m&day1=%d&")
    thisurl += (now).strftime("year2=%Y&month2=%m&day2=%d&")
    print("Downloading: %s" % (now,))
    data = download_data(thisurl)
    outfn = "ASOS_pre.csv"
    with open(outfn, "w") as fh:
        fh.write(data)
    now += interval
    
def download_alldata2():
    """An alternative method that fetches all available data.

    Service supports up to 24 hours worth of data at a time."""
    # timestamps in UTC to request data for
    startts = valid_time2 - timedelta(hours=1)
    endts = valid_time2
    interval = timedelta(hours=1)

    service = SERVICE + "data=all&tz=Etc/UTC&format=comma&latlon=yes&"

    now = endts

    thisurl = service
    thisurl += now.strftime("year1=%Y&month1=%m&day1=%d&")
    thisurl += (now).strftime("year2=%Y&month2=%m&day2=%d&")
    print("Downloading: %s" % (now,))
    data = download_data(thisurl)
    outfn = "ASOS_pre2.csv"
    with open(outfn, "w") as fh:
        fh.write(data)
    now += interval


if __name__ == "__main__":
    download_alldata()
    download_alldata2()
    #main()


# ### Opening the ASOS data

# In[14]:


valid_time_str = valid_time.strftime('%Y%m%d')

#deleting first 5 lines because formtting is weird when downloading the data
with open("ASOS_pre.csv", 'r') as f:
    lines = f.readlines()
with open("ASOS_pre.csv", 'w') as f:
    f.writelines(lines[5:])
    
    #opening the data with pandas
ASOS_data = pd.read_csv("ASOS_pre.csv", delimiter=',')

#deleting first 5 lines because formtting is weird when downloading the data
with open("ASOS_pre2.csv", 'r') as f:
    lines = f.readlines()
with open("ASOS_pre2.csv", 'w') as f:
    f.writelines(lines[5:])
    
    #opening the data with pandas
ASOS_data2 = pd.read_csv("ASOS_pre2.csv", delimiter=',')


# ### Filtering old dataset to correct times and merging them together

# In[17]:


ASOS_data2['valid'] = pd.to_datetime(ASOS_data2['valid'])
mask = (ASOS_data2['valid'].dt.time >= valid_time3) & (ASOS_data2['valid'].dt.time <= end_time)
ASOS_data2 = ASOS_data2[mask]

ASOS_data['valid'] = pd.to_datetime(ASOS_data['valid'])
mask = (ASOS_data['valid'].dt.time <= valid_time3)
ASOS_data = ASOS_data[mask]


# In[18]:


ASOS_data_full = pd.concat([ASOS_data2,  ASOS_data])


# ### Filter data to just USA

# In[20]:


lat_min, lat_max = 20, 50
lon_min, lon_max = -130, -60


# In[21]:


ASOS_data_full = ASOS_data_full[(ASOS_data_full['lat'] >= lat_min) & (ASOS_data_full['lat'] <= lat_max) & (ASOS_data_full['lon'] >= lon_min) & (ASOS_data_full['lon'] <= lon_max)]


# In[22]:


#ASOS_data_full.columns


# In[24]:


#ASOS_data_full[ASOS_data_full['station'] == 'ALB']


# ### Removing NA/M values from just the data we want

# In[25]:


#WX_Codes_Filter = ASOS_data_full[ASOS_data_full['wxcodes'] != 'M']
#WX_Codes_Filter = WX_Codes_Filter.dropna(subset='wxcodes')
WX_Codes_Filter = ASOS_data_full[['station','valid','lat','lon','wxcodes','tmpf', 'sknt','peak_wind_gust','feel','p01i']]
WX_Codes_Filter['wxcodes_full'] = WX_Codes_Filter['wxcodes']


# In[26]:


#WX_Codes_Filter[WX_Codes_Filter['station'] == 'ALB']


# ### Adding the PTYPE Map to the ASOS observations

# In[1]:


#Replacing all of the metar codes with the easier to read map language, this is also used to coencide with the MPING data better as we will use the same language between both
WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace(['HZ BLDU','BCFZFG BR','VCSS','-BLSN','BCFZFG', 'GR', 'BLSN VCSH', 'VCMIFG', 'HZ PRFG','FZFG BR','MIFG BCFG', 'PRFG MIFG','-', 'GS','HZ DRSN', 'DRSN BR','DRSN BR''HZ DRSN','FU PRFG', 'DRSN HZ','VCPO','PO', 'SHGR','VCFG MIFG', 'VCVA','FG VCSH','BLSN FZFG','VCBR','BR FU','BR BCFG','BR PRFG','BCFG BR','BR MIFG','BR VCFG','PRFG BR', 'VCFG BR','MIFG BR','BR HZ','BLSN BR','BR BLSN','BR DRSN','FU BR','BR','+FC','VCBLSN','FC','HZ BLSN','DRSN VCFG','PRFZFG','-DS','FZFG BLSN','HZ SQ','DRSN VCSH','FG HZ', 'DRSN VCBLSN','VCFG HZ','-HZ','DS','NSW','-DRSN','FG FU','FU HZ','BLSN DRSN','SS','DRDU', 'BLSA', 'SA','DU','HZ FU','BLDU','BCFG FU','VCHZ','PRFG','NP','M','NaN','DRSN', 'FG', 'HZ', 'FZFG','BLSN','FU','FZFG FU','MIFG','PRFG FU','PRFG FU','VCFG','BCFG','VCBLDU','+FG','BR DU','HZ BR',],
                                                   ['NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL', 'NULL', 'NULL','NULL', 'NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL','NULL'])


#SNOW#############################################################################################################################################################################
WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace(['-SN BR SQ', 'SN FG SQ','-SN FZFG DRSN','-SNSN','SHSNGS BLSN','-SN DRSN SN','SN BLSN BR','-SGSN', '-SHSN HZ','-SHSN BLSN BR', 'SHSN BLSN BR','-FZUP -SN','SHSN VCSH','-SNSG DRSN','-SN BLSN BR','SNSG','-SN FZUP','-SHSN BLSN DRSN','-SHSN FU', 'SHSN FU','-SN DRSN BR', '-SHSN DRSN BR', 'SN DRSN BR','SN +BLSN', '-SHSN +BLSN','SN FZFG BLSN', '-SN DRSN HZ', '+SHSN BR', '-SN +BLSN', '-SN -FZUP','SHSNGS','SHSN FZFG','-SN FU PRFG','-SHSN FG', 'SG FZFG','SG','-SG','-SG DRSN','-SG VCFG','SHSG','-SHSG','SG BR','-SG BR','BLSN -SHSN', '-SG BLSN', '-SHSN FZFG','-SHSNGS BR', '-SNGS', '-SN -','-SNSG BR','SN BCFG','BCFG BR -SHSN','-SN BR FU', '-SHGSSN', '-SHSN PRFG','SN SQ', 'VCFG SHSN','FG -SHSN', 'BR -SN', '-SN VCSH','-SN DRSN VCBLSN','+SHSN DRSN','VCSN BR','-SN PRFG','VCSHSN','SNFG','-SNSG','-SN BCFG','-SN BCFG BR','-SN BR BCFG','SN FU', '-SN FU','SHSN BR','-SHSNGS','-SN UP','-SHSN BCFG','-SHSN VCFG','-SHSN','SNBR','-SN FZFG BLSN','-SN -UP','+SHSN BLSN', 'SN DRSN','-SN HZ', '-SN HZ DRSN','-SN VCFG','-SHSN BLSN','-SN BR DRSN','-SN SQ','BLSN SHSN','-SHSN DRSN','SHSN BLSN','SN BR','+SHSN','-SG FZFG','-SHSN BR','SHSN','-SN','-SN BR','SN','-SN DRSN','-SN BLSN','SN BLSN','-SNBR','SN FZFG','SN FG','-SN FG','-SN FZFG'],
                                                     ['Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel','Snow and/or Graupel', 'Snow and/or Graupel','Snow and/or Graupel'])

WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace(['VCTSSN', 'VCTS SN BR', 'TSSN FG','+VCTSSN', '+TSSN','VCTS SN FZFG','TSSN BLSN','-VCTSSN','SN VCTS','+SN VCTS','-SN VCTS','-VCTSSN BR','+TSGSSN', '-TSSNGS', '+TSGSSN', '-TSSNGS','TSSNGS','TSSN','-TSSN BR','VCTS -SN','VCTS -SN FZFG', 'VCTS -SN BR','-TSSN'],
                                                                ['Snow and/or Graupel with Thunderstorm','Snow and/or Graupel with Thunderstorm','Snow and/or Graupel with Thunderstorm','Snow and/or Graupel with Thunderstorm','Snow and/or Graupel with Thunderstorm','Snow and/or Graupel with Thunderstorm','Snow and/or Graupel with Thunderstorm','Snow and/or Graupel with Thunderstorm','Snow and/or Graupel with Thunderstorm','Snow and/or Graupel with Thunderstorm','Snow and/or Graupel with Thunderstorm','Snow and/or Graupel with Thunderstorm','Snow and/or Graupel with Thunderstorm','Snow and/or Graupel with Thunderstorm','Snow and/or Graupel with Thunderstorm','Snow and/or Graupel with Thunderstorm','Snow and/or Graupel with Thunderstorm','Snow and/or Graupel with Thunderstorm','Snow and/or Graupel with Thunderstorm','Snow and/or Graupel with Thunderstorm','Snow and/or Graupel with Thunderstorm','Snow and/or Graupel with Thunderstorm','Snow and/or Graupel with Thunderstorm'])

WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace(['+SN FG SQ','-FZRA +SN','+SN BCFG','+SNSG','+SN -UP','+SN DRSN','+SN +BLSN','+SN BR','+SN BLSN','+SN','+SN FG','+SN FZFG'],
                                                                ['Heavy Snow and/or Graupel','Heavy Snow and/or Graupel','Heavy Snow and/or Graupel','Heavy Snow and/or Graupel','Heavy Snow and/or Graupel','Heavy Snow and/or Graupel','Heavy Snow and/or Graupel','Heavy Snow and/or Graupel','Heavy Snow and/or Graupel','Heavy Snow and/or Graupel','Heavy Snow and/or Graupel','Heavy Snow and/or Graupel',])
#SLEET#############################################################################################################################################################################
WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace(['IC BCFG BR','IC PRFG','IC BR','IC -SHGS DRSN','IC DRDU DRSN','IC HZ','-IC BR VCFG','IC BR DRSN','IC DRSA','IC BCFG','IC FZFG','+SHPL','PLBR','PL FZFG','PL BR',  'IC DRSN VCBLSN','-IC BR','-PL FZFG','IC BLSN','BR PL','IC VCFG','IC HZ DRSN','IC DRSN','-IC','-IP','IP','IC','-PL','PL',],
                                                                ['Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet','Ice Pellets/Sleet']) 

WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace(['VCTS PL','+TSPL FZFG','TSPL','-TSPL BR','TSPL BR','VCTS PL BR'],
                                                                ['Ice Pellets/Sleet with Thunderstorm','Ice Pellets/Sleet with Thunderstorm','Ice Pellets/Sleet with Thunderstorm','Ice Pellets/Sleet with Thunderstorm','Ice Pellets/Sleet with Thunderstorm','Ice Pellets/Sleet with Thunderstorm',])

WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace(['+PL BR','+PL FZFG','+PL FG','+IC','+IP','+PL'],
                                                                ['Hevay Ice Pellets/Sleet','Hevay Ice Pellets/Sleet','Hevay Ice Pellets/Sleet','Hevay Ice Pellets/Sleet','Hevay Ice Pellets/Sleet','Hevay Ice Pellets/Sleet',])
#RAIN#############################################################################################################################################################################
WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace(['SQ','-DZBR','RA SQ','VCTS RA HZ', 'RA BR SQ','-SHGS BR','RA UP BR', 'RA MIFG','-RAGS', '+SH', 'RA VCFG', '-RA SQ','-RA BLDU','-DZ VCFG BR','-DZ PRFG BR','DZSG', '-DZSG VCSH', '-SHRA PRFG BR', 'RA DU','RA -UP BR','-SHRA BR BCFG','-RA DU', 'SHRA BCFG BR','-SHRA BCFG BR','-RA BR BCFG','BR RADZ', '+SHGSRA', '-RA -', '-DZSG','SHRAGR', '-RA FZFG','DZRABR','BR FG','FG BR','-SGRAGS','RA BR VCSH','-SHRAGR','RADZ BCFG','-RA -UP BR','BR -RA', '-RA -UP BR''SHSGRA','SHGR BR', '+SHGR BR','SHGR BR', '+SHGR BR','-SHGS','-GS BR','-SHRAGS BR','-SHRAGS','SHRAGS','-SHGSRA','SHGSRA','+SHRAGS','GS BR','-SHGR','VCFC VCSH','FG -DZ','BCFG VCSH','-VCSH','FC VCSH','DZ VCFG','-VCRA', '-SHRA DZ', '-SHRA FU','VCSH BR', 'DR', '+DZ FG','VCSHRA','-DZ FU', 'RA VCSH','- SHRA','SHRA BR FU', '-RADZ FG', '-SHRA +DZ','SHRA HZ', '-SHRA -DS', '-FG DZ', 'FG RA', 'VC RA','SH','-RA BCFG BR', '-DZ PRFG','VCDZ', '- DZ', 'RABR', '-SHRAHZ',  '-DZ BCFG', '+SHRA BR','VCFG VCSH','FZFG VCSH','-SHRA BR FU','RA HZ', '-DZ VCSH', '-RADZ VCFG', '-DZ VCFG', '+DZ BR','DZ VCSH','VCSH HZ', '-SH','-RA RA','-SHRA HZ','RA HZ', '-DZRA BR', 'RADZ FG','BR -RADZ','-RA BR FU','-RA PRFG', '-RA FG FU', 'RA -UP','-RA -UP', 'SHRA BR','-RADZ BR','FG DZ','-RABR','-RA VCSH', 'RADZ BR', '- RA', 'BR DZ','-DZRA VCFG','-RA SHRA', 'BR VCSH','SHRA VCSH','RA BCFG','FG -RA','DZRA BR','-RA BR VCFG','-RA HZ', '-SHRA VCSH', '- RA BR','-SHGRRA','VCRA','-SHRA PRFG','DRSA','-SHRA BCFG','SHGS', 'DZRA', '-SHRA FG','-RADZ BCFG','-RA VCFG','-RA BCFG', '+SHRA','RADZ','-RA BR','+DZRA', '+DZ', '-SHRA BR','-RA MIFG','BR -DZ', 'DZ', '-DZRA','-SHRA','-RADZ','-RA','RA','RA BR','-RA FG','-DZ BR','SHRA','-DZ FG','-DZ','DZ BR','SHRA''-RA BR','DZ FG','RA FG','VCSH'],
                                                    ['Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain','Rain']) 

WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace(['+TSRAGR','VCTS GR','+SHGR', '+TSGR','-SHGR VCTS','VCTS +RA FG','+TSRA BR SQ', '+VCTSRA BR','VCTSRA BR', '+VCTSDZ','RA FG VCTS', 'VCTSHZ', '-RA BR VCTS','-VCTSRA BR','VCTSBR',  'SHRAGS TS','+TS','TSDZ', '+SHRA VCTS','-VCTSDZ', 'TSGS','TSGR', '-TS SHRA','BR TS','TSRA FG','TS RABR','-DZ VCTS','TS FZFG','VCTS FZFG','-TSSH','-TSGRRA','TSRAGR','-TSGR','TS GR','TSGSRA','TSGSRA','+TSGSRA','-TSRAGS', 'TSRAGS','VCTS DZ', 'TSRA BCFG','BR TSRA', 'BR -TSRA','-TSRA PRFG','+TSBR','VCTSDZ','-TSSHRA','VCTS -TSRA','TSBR', '-TS RA','-TSDZ','VCTS HZ','VCTS +SHRA','-SHRA TS','+RA BR VCTS','SQ -TSRA','-TSRA HZ','SHRA VCTS','VCTS -DZ','VCTS RA','RA VCTS', 'TSRA VCFG','+RA VCTS','-SHRA VCTS','-TSRA VCSH','VCTS VCSH','TS -DZ','TS SHRA','TS HZ','TSSHRA','TSHZ', 'TS FG','TSSHRA','VCTS BR','VCTS -RA','+TSRA FG','VCTSSH','TS RA','RA BR VCTS','VCTS +RA','+RA FG VCTS','-TSRA FG', 'TSRA BR', 'VCTS RA FG', '-VCTSRABR','VCTS +RA BR','-TS','VCTS FG','VCTSRA','SHRA TS','TS BR','-TSRA BR', 'VCSH VCTS','+VCTSRA','VCTS -RA BR','-RA VCTS','VCTS RA BR','-VCTSRA','+TSRA','TS VCSH', '+TSRA BR', 'VCTS SHRA', 'TSRA','-RA TS','VCTS','VCTS -SHRA','TS','VCSH TS','-TSRA',],
                                                                ['Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm','Rain with Thunderstorm',])

WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace(['+RA SQ','+RA FG SQ','+RA BR SQ','+RA BCFG VCTS','+RA FG VCFG','+RA VCFG','+RABR','+RA BCFG','+RA HZ','+RA FG','+RA','+RA BR',],
                                                                ['Heavy Rain','Heavy Rain','Heavy Rain','Heavy Rain','Heavy Rain','Heavy Rain','Heavy Rain','Heavy Rain','Heavy Rain','Heavy Rain','Heavy Rain','Heavy Rain',])
#FREEZING RAIN#############################################################################################################################################################################
WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace([ '-FZRA -RA','FZRA -SN','FZRASN BLSN','-FZRADZ','-FZRA -SN BLSN','-FZRA -SN BLSN','-FZDZ -SN DRSN','-FZRA BCFG','-FZRA FG','FZRA SN BR', 'FZRASN BR','FZRA SN', '-FZDZ FG','FZRA FZFG','-FZRA SN', '+FZDZ FZFG','+FZDZ BR','-FZRASN BR','FZDZ BR','+FZDZ','-FZDZ BCFG','- FZ RA','-FZRA FZFG','-FZRA -SN','-FZDZ -SN','-FZDZ -SN BR','-FZDZ FZFG','FZDZ FZFG','FZRA FG','FZRA','-FZDZ','-FZDZ BR','-FZRA','-FZRA BR','FZDZ','FZRA BR','-FZDZSN','-FZRASN'],
                                                    ['Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain','Freezing Rain'])

WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace(['TS FZRA BR','VCTS FZRA FZFG','VCTS FZRA BR','TSFZRA BR','VCTS FZRA','TS -FZRA','VCTS -FZRA',],
                                                                ['Freezing Rain with Thunderstorm','Freezing Rain with Thunderstorm','Freezing Rain with Thunderstorm','Freezing Rain with Thunderstorm','Freezing Rain with Thunderstorm','Freezing Rain with Thunderstorm','Freezing Rain with Thunderstorm'])

WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace(['+FZRA BR','+FZRA',],
                                                                ['Heavy Freezing Rain','Heavy Freezing Rain',])
#MIXED ZR/IP#############################################################################################################################################################################
WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace(['-FZRA +SHPL BLSN','FZRA PL','FZRAPL','-FZDZPL','FZRA PL FZFG','FZRA PL BR','-FZRAPL','-FZRA -PL -SN','-FZRASNPL','-FZDZ -PL','-FZRA -PL','-FZRAPL BR','-FZRA -PL DRSN','-FZRA -PL BR'],
                                                                ['Mixed Freezing Rain and Ice Pellets','Mixed Freezing Rain and Ice Pellets','Mixed Freezing Rain and Ice Pellets','Mixed Freezing Rain and Ice Pellets','Mixed Freezing Rain and Ice Pellets','Mixed Freezing Rain and Ice Pellets','Mixed Freezing Rain and Ice Pellets','Mixed Freezing Rain and Ice Pellets','Mixed Freezing Rain and Ice Pellets','Mixed Freezing Rain and Ice Pellets','Mixed Freezing Rain and Ice Pellets','Mixed Freezing Rain and Ice Pellets','Mixed Freezing Rain and Ice Pellets','Mixed Freezing Rain and Ice Pellets'])

WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace(['TS -FZRAPL','TS FZRAPL FZFG','TS FZRAPL BR',],
                                                                ['Mixed Freezing Rain and Ice Pellets with Thunderstorm','Mixed Freezing Rain and Ice Pellets with Thunderstorm','Mixed Freezing Rain and Ice Pellets with Thunderstorm'])

WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace([],
                                                                [])
#MIXED RA/SN#############################################################################################################################################################################
WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace([ '-RA -SN -','RASN -UP','SHRASN BR', '-SN RA','-SG VCSH','-SNRA BLSN','-RASN BR VCFG','-SHSNRAGS','SHSNRA FG''-SN RA','RA -SN','-SHRASN BCFG','+SHRASN','BR -SNRA','-SNDZ','-SNRA FG','SNRA BR','-SNRA VCFG','SHRASNGS', '-RAPLSN','SHSNRA BR','-SNRA PRFG','-RASNGS','+SHSNRA','SNRA FG','-SNRA FZFG','SHSNRA','-RASN DRSN','-SHRASNGS','SHRASN','-SHSNRA BR','SHRAGSSN','RA SN','SHGSSN', 'SHSN DRSN','-DZSN','-SHRASN BR','VCSH DRSN','-SNRA -UP','SNRA','RASN BR','-SHSNRA','-RA -SN','-RA -SN BR','-RASN BR','-FZRA -SN DRSN','-RASN','RASN','-FZRA -SN BR','-SNRA BR','-SHRASN','-SNRA'],
                                                                ['Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow','Mixed Rain and Snow']) 

WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace(['TSRASN','-TSSNRA', ],
                                                                ['Mixed Rain and Snow with Thunderstorm','Mixed Rain and Snow with Thunderstorm',])

WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace(['+SNRA BR','+RASN FG VCFG','+RASN','+SNRA','+RA SN',],
                                                                ['Heavy Mixed Rain and Snow','Heavy Mixed Rain and Snow','Heavy Mixed Rain and Snow','Heavy Mixed Rain and Snow','Heavy Mixed Rain and Snow',])
#MIXED IP/SN#############################################################################################################################################################################
WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace(['-PLSN BR','+SHPL BLSN','IC -SN DRSN','IC -SG','-PLSN', 'IC -SG VCBLSN DRSN','-SNPL BLSN','-DZSNPL','-SN PL','SNPL','-SNPL','-PL -SN BR','-PL -SN','-SNPL BR','-SNPL DRSN', '-PLSN DRSN','IC -SN'],
                                                                ['Mixed Ice Pellets and Snow','Mixed Ice Pellets and Snow','Mixed Ice Pellets and Snow','Mixed Ice Pellets and Snow','Mixed Ice Pellets and Snow','Mixed Ice Pellets and Snow','Mixed Ice Pellets and Snow','Mixed Ice Pellets and Snow','Mixed Ice Pellets and Snow','Mixed Ice Pellets and Snow','Mixed Ice Pellets and Snow','Mixed Ice Pellets and Snow','Mixed Ice Pellets and Snow','Mixed Ice Pellets and Snow','Mixed Ice Pellets and Snow','Mixed Ice Pellets and Snow','Mixed Ice Pellets and Snow']) 

WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace(['-TSSNPL',],
                                                                ['Mixed Ice Pellets and Snow with Thunderstorm',])

WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace([],
                                                                [])
#MIXED IP/RA#############################################################################################################################################################################
WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace(['-PLRA BR','-RAPLSN BR','-SHRA -PL','RAPL BR','-PLRA','-RASNPL', '-SNPLRA','-RAPL', '-RA -PL', '-RA -PL BR','-RA -PL -SN','RAPL','-RA -PL -SN BR','-RAPL BR','-SHPL BR', '-PL BR'],
                                                                ['Mixed Rain and Ice Pellets','Mixed Rain and Ice Pellets','Mixed Rain and Ice Pellets','Mixed Rain and Ice Pellets','Mixed Rain and Ice Pellets','Mixed Rain and Ice Pellets','Mixed Rain and Ice Pellets','Mixed Rain and Ice Pellets','Mixed Rain and Ice Pellets','Mixed Rain and Ice Pellets','Mixed Rain and Ice Pellets','Mixed Rain and Ice Pellets','Mixed Rain and Ice Pellets','Mixed Rain and Ice Pellets','Mixed Rain and Ice Pellets','Mixed Rain and Ice Pellets']) 

WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace([],
                                                                [])

WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace(['+RAPL',],
                                                                ['Heavy Mixed Rain and Ice Pellets',])
#UNKNOWN PRECIP#############################################################################################################################################################################
WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace([ '-UP FG','-FZUP -SG','FZUP','FZUP FZFG','-FZUP BR','-FZUP FZFG', '-FZUP', 'FZUP BR','-SHUP','UP HZ','-UP FZFG', '-UP BR','UP FG','UP','-UP','UP BR','UP FZFG',],
                                                                ['Unknown Precip','Unknown Precip','Unknown Precip','Unknown Precip','Unknown Precip','Unknown Precip','Unknown Precip','Unknown Precip','Unknown Precip','Unknown Precip','Unknown Precip','Unknown Precip','Unknown Precip','Unknown Precip','Unknown Precip','Unknown Precip','Unknown Precip']) 

WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace(['VCTS UP BR','VCTSUP','TSUP','VCTS UP','VCTS UP HZ', 'VCTS UP FZFG','UP VCTS','TSUP BR'],
                                                                ['Unknown Precip with Thunderstorm','Unknown Precip with Thunderstorm','Unknown Precip with Thunderstorm','Unknown Precip with Thunderstorm','Unknown Precip with Thunderstorm','Unknown Precip with Thunderstorm','Unknown Precip with Thunderstorm','Unknown Precip with Thunderstorm',])

WX_Codes_Filter['wxcodes'] = WX_Codes_Filter['wxcodes'].replace(['+FZUP','+UP BR','+UP',],
                                                                ['Heavy Unknown Precip','Heavy Unknown Precip','Heavy Unknown Precip',])
WX_Codes_Filter['ptype'] = WX_Codes_Filter['wxcodes'].map(precip_types_map)
WX_Codes_Filter['color'] = WX_Codes_Filter['wxcodes'].map(precip_types_color)


# In[28]:


#WX_Codes_Filter = WX_Codes_Filter[WX_Codes_Filter['wxcodes'] != 'NULL']
#WX_Codes_Filter[WX_Codes_Filter['station'] == 'ALB']


# ### Uncomment out to view other wxcodes

# In[29]:


#WX_Codes_Filter[WX_Codes_Filter['ptype'].isna()]
pd.unique(WX_Codes_Filter['wxcodes'])


# ### Replace all nan values with -1

# In[30]:


WX_Codes_Filter['ptype'] = WX_Codes_Filter['ptype'].fillna(0)


# ### Make Datetime and Object to use when hovered over

# In[31]:


WX_Codes_Filter['valid']=WX_Codes_Filter['valid'].astype(str)
WX_Codes_Filter['valid'] = WX_Codes_Filter['valid'].apply(lambda x: x + 'Z')


# ### Round times to be used in making a timeseries

# In[33]:


WX_Codes_Filter['valid_rounded'] = pd.to_datetime( WX_Codes_Filter['valid'])
WX_Codes_Filter['valid_rounded'] = WX_Codes_Filter['valid_rounded'].apply(lambda x: x + pd.Timedelta(minutes=60) if x.minute >= 1 and x.minute < 30 else x)
WX_Codes_Filter['valid_rounded'] = WX_Codes_Filter['valid_rounded'].apply(lambda x: x + pd.Timedelta(hours=1) if x.minute >= 30 else x)
WX_Codes_Filter['valid_rounded'] = WX_Codes_Filter['valid_rounded'].dt.strftime('%Y-%m-%d %H:00Z')


# ### Renaming Columns

# In[77]:


WX_Codes_Filter = WX_Codes_Filter.fillna('#000000')
WX_Codes_Filter = WX_Codes_Filter.rename(columns={'valid':'OBS Time','station' : 'Station','lon':'Lon','lat':'Lat','tmpf': 'Temperature (F)', 'wxcodes': 'Current WX','wxcodes_full':'WX Code','valid_rounded':'Time','sknt':'Wind (mph)','feel':'Real Feel Temperature','p01i':'1 Hour Precipitation'})
WX_Codes_Filter['Time'] = pd.to_datetime(WX_Codes_Filter['Time'])


# ### Finding the max precip per hour

# In[78]:


precip_filter = WX_Codes_Filter[WX_Codes_Filter['1 Hour Precipitation'] != 'M']
precip_filter = precip_filter[precip_filter['1 Hour Precipitation'] != '0.00']
#WX_Codes_Filter2[WX_Codes_Filter2['Station'] == 'FYV']
precip_filter['1 Hour Precipitation'] = precip_filter['1 Hour Precipitation'].astype(float)
ASOS_precip_hr = precip_filter.groupby(['Station','Time','Lat','Lon'])['1 Hour Precipitation'].max().reset_index()


# In[79]:


ASOS_precip_hr['1 Hour Precipitation'] = ASOS_precip_hr['1 Hour Precipitation'].astype(float).round(2)
ASOS_precip_hr = ASOS_precip_hr[ASOS_precip_hr['1 Hour Precipitation'] != 0]


# ### Just keep the last instance for each hour

# In[80]:


Temp_Filter = WX_Codes_Filter[WX_Codes_Filter['Temperature (F)'] != 'M']
ASOS_data_temp = Temp_Filter.drop_duplicates(subset=['Time', 'Station'], keep='last')

Real_Filter = WX_Codes_Filter[WX_Codes_Filter['Real Feel Temperature'] != 'M']
ASOS_data_real = Real_Filter.drop_duplicates(subset=['Time', 'Station'], keep='last')
ASOS_data_real['Real Feel Temperature'] = ASOS_data_real['Real Feel Temperature'].astype(float).round()


# ### Calculating max gusts

# In[81]:


ASOS_data_gust = WX_Codes_Filter

ASOS_data_gust['peak_wind_gust'] = pd.to_numeric(ASOS_data_gust['peak_wind_gust'], errors='coerce')
ASOS_data_gust['Wind (mph)'] = pd.to_numeric(ASOS_data_gust['Wind (mph)'], errors='coerce')
ASOS_data_gust['Max Gust (mph)'] = ASOS_data_gust[['peak_wind_gust', 'Wind (mph)']].max(axis=1)
ASOS_data_gust_grouped = ASOS_data_gust.groupby(['Station','Time','Lat','Lon'])['Max Gust (mph)'].max().reset_index()


# In[82]:


ASOS_data_gust = ASOS_data_gust_grouped.dropna()
ASOS_data_gust['Max Gust (mph)'] = (ASOS_data_gust['Max Gust (mph)']*1.15078).round().astype(int)


# ### Making datetime functions the same throughout datasets so the slider can work

# In[83]:


ASOS_data_temp['Temperature (F)_r'] = round(ASOS_data_temp['Temperature (F)'].astype(float)).astype(int).astype(str).astype('U')


# ### Removing null from ptype

# In[84]:


ASOS_data_ptype = WX_Codes_Filter[WX_Codes_Filter['WX Code'] != 'NULL']
ASOS_data_ptype = ASOS_data_ptype[ASOS_data_ptype['WX Code'] != 'M']
ASOS_data_ptype = ASOS_data_ptype[ASOS_data_ptype['WX Code'] != '#000000']
ASOS_data_ptype = ASOS_data_ptype[ASOS_data_ptype['Current WX'] != 'NULL']
ASOS_data_ptype = ASOS_data_ptype[ASOS_data_ptype['Current WX'] != '#000000']


# In[85]:


ASOS_data_ptype = ASOS_data_ptype.drop_duplicates(subset=['Time', 'Station'], keep='last')


# ### NYSM_data

# In[86]:


nysm_sites = pd.read_csv('/spare11/atm533/data/nysm_sites.csv')
nysm_sites = nysm_sites[['stid','lat','lon']]
nysm_sites = nysm_sites.rename(columns={'stid':'station'})
NYSM_data_full = pd.read_csv('/data1/nysm/latest.csv',parse_dates=['time']) 
NYSM_cols = ['station','time','temp_2m [degC]','max_wind_speed_prop [m/s]','relative_humidity [percent]','precip_local [mm]']
NYSM_data = NYSM_data_full[NYSM_cols]

NYSM_data = pd.merge(nysm_sites, NYSM_data, on='station', how='left')
NYSM_data = NYSM_data.rename(columns={'lat':'Lat', 'lon':'Lon','time':'OBS Time','station':'Station','temp_2m [degC]':'Temperature (F)','max_wind_speed_prop [m/s]':'Wind (mph)','relative_humidity [percent]': 'Relative Humidity','precip_local [mm]':'1 Hour Precipitation'})
NYSM_data = NYSM_data.dropna()

NYSM_data['Time'] = pd.to_datetime(NYSM_data['OBS Time'])
NYSM_data['Time'] = NYSM_data['Time'].apply(lambda x: x + pd.Timedelta(minutes=60) if x.minute >= 1 and x.minute < 30 else x)
NYSM_data['Time'] = NYSM_data['Time'].apply(lambda x: x + pd.Timedelta(hours=1) if x.minute >= 30 else x)
NYSM_data['Time'] = NYSM_data['Time'].dt.strftime('%Y-%m-%d %H:00Z')
NYSM_data['Time'] = pd.to_datetime(NYSM_data['Time'])
NYSM_data['OBS Time'] = NYSM_data['OBS Time'].dt.strftime('%Y-%m-%d %H:%MZ')


# ### Precipitation 

# In[87]:


precip_filter_NYSM = NYSM_data[NYSM_data['1 Hour Precipitation'] != 0]
precip_filter_NYSM = precip_filter_NYSM.sort_values(['Station','Time','Lat','Lon'])
precip_NYSM_grouped = precip_filter_NYSM.groupby(['Station', 'Lat','Lon','Time'])


# In[88]:


first = precip_NYSM_grouped.first()
last = precip_NYSM_grouped.last()


# In[89]:


precip_NYSM = last['1 Hour Precipitation'] - first['1 Hour Precipitation']


# In[90]:


precip_NYSM = precip_NYSM.groupby(['Station', 'Lat','Lon','Time']).last().reset_index()


# In[91]:


precip_NYSM['1 Hour Precipitation'] = precip_NYSM['1 Hour Precipitation'].astype(float)*0.0393701


# In[92]:


#NYSM_data.columns


# ### Converting NYSM temp and adding wind chill

# In[93]:


NYSM_data['Temperature (F)'] = (NYSM_data['Temperature (F)']*9/5)+32
NYSM_data['Wind (mph)'] = (NYSM_data['Wind (mph)']*2.237).round().astype(object)
NYSM_Temp = NYSM_data['Temperature (F)'].values*units.degF
NYSM_Wind =NYSM_data['Wind (mph)'].values*units.mph
NYSM_data['Real Feel Temperature'] = mpcalc.apparent_temperature(NYSM_Temp.astype(float), NYSM_data['Relative Humidity'], NYSM_Wind.astype(float), mask_undefined=False)


# In[94]:


NYSM_data['Temperature (F)_r'] = round(NYSM_data['Temperature (F)'].astype(float)).astype(int).astype(str).astype('U')
NYSM_data['Real Feel Temperature'] = NYSM_data['Real Feel Temperature'].round().astype(int)


# In[95]:


NYSM_data_gust = NYSM_data
NYSM_data_gust['Max Gust (mph)'] = NYSM_data_gust['Wind (mph)'].astype(int)
NYSM_data_gust_grouped = NYSM_data_gust.groupby(['Station','Time','Lat','Lon'])
NYSM_max_gust = NYSM_data_gust_grouped['Max Gust (mph)'].max()
NYSM_max_gust = NYSM_max_gust.reset_index()
NYSM_max_gust['Max Gust (mph)'] =NYSM_max_gust['Max Gust (mph)'].round().astype(int)


# In[96]:


NYSM_data = NYSM_data.drop_duplicates(subset=['Time', 'Station'], keep='last')


# ### Merging Datasets

# In[97]:


precip_NYSM['1 Hour Precipitation'] = precip_NYSM['1 Hour Precipitation'].astype(object)
Gust_data = pd.merge(NYSM_max_gust, ASOS_data_gust,  how='outer')
Precip_data = pd.merge(ASOS_precip_hr, precip_NYSM, how='outer')


# In[98]:


Precip_data = Precip_data[Precip_data['1 Hour Precipitation'] != 0.0001]
Precip_data = Precip_data[Precip_data['1 Hour Precipitation'] != 0.0]


# ### Changing Dataset columns so they're uniform throughout NYSM and ASOS

# In[99]:


NYSM_data['Wind (mph)'] = NYSM_data['Wind (mph)'].astype(object)
NYSM_data['Real Feel Temperature'] = NYSM_data['Real Feel Temperature'].astype(object)
NYSM_data['1 Hour Precipitation'] = NYSM_data['1 Hour Precipitation'].astype(object)
NYSM_data['Max Gust (mph)'] = NYSM_data['Max Gust (mph)'].astype(object)
NYSM_data['Temperature (F)'] = NYSM_data['Temperature (F)'].astype(object)


# ### Merging the datasets together

# In[100]:


Temp_data = pd.merge(ASOS_data_temp, NYSM_data, how='outer')
Real_data = pd.merge(ASOS_data_real, NYSM_data, how='outer')
Ptype_data = pd.merge(ASOS_data_ptype, MPING_data, how='outer')


# In[102]:


# Define function to map precipitation types to short codes
def get_short_code(precip_type):
    if precip_type == 'Unknown Precip':
        return 'UP'
    elif precip_type == 'Heavy Unknown Precip':
        return '+UP'
    elif precip_type == 'Unknown Precip with Thunderstorm':
        return 'TS UP'
    elif precip_type == 'Snow and/or Graupel':
        return 'SN'
    elif precip_type == 'Heavy Snow and/or Graupel':
        return '+SN'
    elif precip_type ==  'Snow and/or Graupel with Thunderstorm':
        return 'TS SN'
    elif precip_type == 'Ice Pellets/Sleet':
        return 'IP'
    elif precip_type == 'Heavy Ice Pellets/Sleet':
        return '+IP'
    elif precip_type == 'Ice Pellets/Sleet with Thunderstorm':
        return 'TS IP'
    elif precip_type == 'Mixed Ice Pellets and Snow':
        return 'IP/SN'
    elif precip_type == 'Heavy Mixed Ice Pellets and Snow':
        return '+IP/SN'
    elif precip_type ==  'Mixed Ice Pellets and Snow with Thunderstorm':
        return 'TS IP/SN'
    elif precip_type == 'Freezing Rain':
        return 'ZR'
    elif precip_type == 'Heavy Freezing Rain':
        return '+ZR'
    elif precip_type == 'Freezing Rain with Thunderstorm':
        return 'TS ZR'
    elif precip_type == 'Freezing Drizzle':
        return 'ZR'
    elif precip_type == 'Mixed Freezing Rain and Ice Pellets':
        return 'ZR/IP'
    elif precip_type ==  'Heavy Mixed Freezing Rain and Ice Pellets':
        return '+ZR/IP'
    elif precip_type == 'Mixed Freezing Rain and Ice Pellets with Thunderstorm':
        return 'TS ZR/IP'
    elif precip_type == 'Rain':
        return 'RA'
    elif precip_type == 'Heavy Rain':
        return '+RA'
    elif precip_type == 'Rain with Thunderstorm':
        return 'TS RA'
    elif precip_type == 'Drizzle':
        return 'RA'
    elif precip_type ==  'Mixed Rain and Snow':
        return 'RA/SN'
    elif precip_type == 'Heavy Mixed Rain and Snow':
        return '+RA/SN'
    elif precip_type == 'Mixed Rain and Snow with Thunderstorm':
        return 'TS RA/SN'
    elif precip_type == 'Mixed Rain and Ice Pellets':
        return 'RA/IP'
    elif precip_type ==  'Heavy Mixed Rain and Ice Pellets':
        return '+RA/IP'
    elif precip_type == 'Mixed Rain and Ice Pellets with Thunderstorm':
        return 'TS RA/IP'
    else:
        return ''

# Apply the function to create a new column
Ptype_data['Precipitation Label'] = Ptype_data['Current WX'].apply(get_short_code)


# ### Adding X and Y coordinates to the data

# In[104]:


inProj = Proj(init='epsg:4326')
outProj = Proj(init='epsg:3857')


# In[105]:


Temp_data['x'], Temp_data['y'] = transform(inProj,outProj,Temp_data['Lon'].values,Temp_data['Lat'].values)
Ptype_data['x'], Ptype_data['y'] = transform(inProj,outProj,Ptype_data['Lon'].values,Ptype_data['Lat'].values)
Real_data['x'], Real_data['y'] = transform(inProj,outProj,Real_data['Lon'].values,Real_data['Lat'].values)
Gust_data['x'], Gust_data['y'] = transform(inProj,outProj,Gust_data['Lon'].values,Gust_data['Lat'].values)
Precip_data['x'],Precip_data['y'] = transform(inProj,outProj,Precip_data['Lon'].values,Precip_data['Lat'].values)


# In[107]:


Precip_data_24 = Precip_data.groupby(['Station','Lat','Lon','x','y'])['1 Hour Precipitation'].sum()
Precip_data_24 = Precip_data_24.reset_index()
Precip_data_24 = Precip_data_24.rename(columns={'1 Hour Precipitation':'24 Hour Precipitation'})


# In[108]:


Real_data['Real Feel Temperature'] = Real_data['Real Feel Temperature'].astype(float)


# In[109]:


Ptype_data.dropna(subset=['Current WX'], inplace=True)


# In[111]:


Gust_data = Gust_data[Gust_data['Max Gust (mph)'] >= 10]


# In[112]:


Ptype_data['Label'] = 'Precipitation Type'
Gust_data['Label'] = 'Wind Gust'
Temp_data['Label'] = '2m Temperature'
Real_data['Label'] = 'Real Feel Temperature'
Precip_data['Label'] = '1hr Precipitation'
Precip_data_24['Label'] = '24hr Precipitation'


# In[114]:


Ptype = Ptype_data.loc[:,['Station','OBS Time','Current WX','WX Code', 'Temperature (F)','color','Lat','Lon','Time','Label','Precipitation Label']]
Gust = Gust_data.loc[:,['Station','Max Gust (mph)','Lat','Lon','Time','Label']]
Temp = Temp_data.loc[:,['Station','OBS Time','Temperature (F)','Temperature (F)_r','Lat','Lon','Time','Label']]
Real_Feel = Real_data.loc[:,['Station','OBS Time','Real Feel Temperature','Temperature (F)','Lat','Lon','Time','Label']]
Precip = Precip_data.loc[:,['Station','1 Hour Precipitation','Lat','Lon','Time','Label']]
Precip_24 = Precip_data_24.loc[:,['Station','24 Hour Precipitation','Lat','Lon','Label']]


# In[116]:


Temp['Temperature (F)']= Temp['Temperature (F)'].astype(float).round(2)


# In[117]:


Temp['Temperature (F)']= Temp['Temperature (F)'].astype(float)


# In[118]:


All_data = pd.concat([Ptype, Gust, Temp, Real_Feel, Precip, Precip_24])


# ### Save the Data

# In[119]:


All_data.to_csv('All_data.csv', index=False)


# In[120]:


All_data[All_data['Station'] =='MYGF']


# In[ ]:




