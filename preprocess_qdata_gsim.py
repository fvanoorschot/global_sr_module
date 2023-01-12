# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 11:05:39 2021

@author: fransjevanoors

For all GSIM catchments preprocess yearly files
- create csv with metadata per catchment -> saved P/GSIM_processed_data/catchment_metadata
- create csv with yearly timeseries per catchment P/GSIM_processed_data/catchment_yearly/m3_s or mm_day


COLUMN MEANINGS (see paper Gudmundsson part2)

"MEAN"' -> mean yearly streamflow (m3/s)
'"SD"' -> standard deviation of daily streamflow (m3/s)
'"CV"' -> coefficient of variation of daily streamflow
'"IQR"' -> inter quartile range of day-to-day streamflow variability (m3/s)
'"MIN"' -> yearly minimum streamflow (m3/s)
'"MAX"' -> yearly maximum streamflow (m3/s)
'"MIN7"' -> minimum 7-day mean streamflow (m3/s)
'"MAX7"' -> maximum 7-day mean streamflow (m3/s)
'"P10"' -> percentiles of daily streamflow (m3/s)
'"P20"'
'"P30"'
'"P40"'
'"P50"'
'"P60"'
'"P70"'
'"P80"'
'"P90"'
'"GINI"' -> gini coefficient
'"CT"' -> centre timing
'"DOYMIN"' -> day of year of minimum streamflow (doy)
'"DOYMAX"' -> day of year of maximum streamflow (doy)
'"DOYMIN7"' -> day of year of minimum 7daily streamflow (doy)
'"DOYMAX7"' -> day of year of maximum 7daily streamflow (doy)
'"n.missing"' -> number of missing/suspect daily values (#days)
"n.available" -> number of used daily values (#days)

"""

#%%
import numpy as np
import pandas as pd
import glob
import math
import datetime
import geopandas as gpd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import shutil
# import cartopy.crs as ccrs # basemap is oud, cartopy nieuwer
folder = '/home/vanoorschot/Documents/'

#%% make dataframes for all catchments of metadata and timeseries -> DONE

for filepath in glob.iglob(f'{folder}GSIM_indices/TIMESERIES/yearly/*'):
    print(filepath)
    # save catchment meta data in dataframe
    b = pd.read_csv(filepath,delimiter=':',skiprows=10,nrows=7, index_col=0)
    c = pd.DataFrame(index=[0],columns=['gsim.no','river','station','country','lat_deg','lon_deg','alt_m','area_km2'])
    c['gsim.no'] = str.strip(b.columns[0])
    c['river'] = str.strip(b.iloc[0,0])
    c['station'] = str.strip(b.iloc[1,0])
    c['country'] = str.strip(b.iloc[2,0])
    c['lat_deg'] = str.strip(b.iloc[3,0])
    c['lon_deg'] = str.strip(b.iloc[4,0])
    c['alt_m'] = str.strip(b.iloc[5,0])
    c['area_km2'] = str.strip(b.iloc[6,0])
    c = c.replace('',np.nan)
    c[['lat_deg','lon_deg','alt_m','area_km2']] = c[['lat_deg','lon_deg','alt_m','area_km2']].astype(float)
    
    
    # save catchment timeseries in dataframe
    a = pd.read_csv(filepath, skiprows=21, delimiter=',',index_col=0)
    a.index = pd.to_datetime(a.index)
    a.columns = a.columns.str.strip() # remove \t (=tab))    
    # drop columns we don't need
    a = a.drop(columns=['"GINI"','"CV"','"CT"','"P10"', '"P20"', '"P30"', '"P40"', '"P50"', '"P60"', '"P70"', '"P80"', '"P90"'])
    a = a.astype(str)
    
    # remove all tabs
    for i in range(len(a)): #loop over rows
        for j in range(len(a.iloc[0])): # loop over columns
            a.iloc[i,j] = a.iloc[i,j].strip()
            
    # replace NA with 999
    a = a.replace('NA',np.nan)
    
    #rename columns
    a = a.rename(columns={'"MEAN"': 'mean_m3s', '"SD"': 'sd_m3s', '"IQR"': 'iqr_m3s', '"MIN"': 'min_m3s', '"MAX"': 'max_m3s',
            '"MIN7"': 'min7_m3s', '"MAX7"': 'max7_m3s', '"DOYMIN"':'doymin','"DOYMAX"':'doymax', '"DOYMIN7"':'doy7min', '"DOYMAX7"':'doy7max',
            '"n.missing"':'nr_mis_days','"n.available"':'nr_av_days'})
    a = a.astype(float)
    
    #convert values to mm/day
    area_m2 = c.area_km2.values[0] * (10**6)
    a_mm = a
    a_mm = a_mm.rename(columns={'mean_m3s':'mean_mmd', 'sd_m3s':'sd_mmd', 'iqr_m3s':'iqr_mmd', 'min_m3s':'min_mmd', 'max_m3s':'max_mmd', 'min7_m3s':'min7_mmd','max7_m3s':'max7_mmd'})
    a_mm[['mean_mmd', 'sd_mmd', 'iqr_mmd', 'min_mmd', 'max_mmd', 'min7_mmd','max7_mmd']] = a[['mean_m3s', 'sd_m3s', 'iqr_m3s', 'min_m3s', 'max_m3s', 'min7_m3s','max7_m3s']] * (1/area_m2) * 1000 * 86400
    a_mm = a_mm.rename(columns={'mean_m3s':'mean_mmd', 'sd_m3s':'sd_mmd', 'iqr_m3s':'iqr_mmd', 'min_m3s':'min_mmd', 'max_m3s':'max_mmd', 'min7_m3s':'min7_mmd','max7_m3s':'max7_mmd'})
    a_mm = a_mm.astype(float)
    
    #catchid
    catch_id = c['gsim.no'].values[0]
    
    # add length timeseries to c
    c['start_year'] = a_mm.index[0]
    c['end_year'] = a_mm.index[-1]
    c['length_ts_yr'] = int((a_mm.index[0] - a_mm.index[-1]).days / -365)
    c['start_year'] = pd.to_datetime(c['start_year'])
    c['end_year'] = pd.to_datetime(c['end_year'])
    
    # save to 
    c.to_csv(str(folder)+'GSIM_processed_data/catchment_metadata/'+str(catch_id)+'.csv')
    a.to_csv(str(folder)+'GSIM_processed_data/catchment_yearly/m3_s/'+str(catch_id)+'.csv')
    a_mm.to_csv(str(folder)+'GSIM_processed_data/catchment_yearly/mm_day/'+str(catch_id)+'.csv')
    
        
#%%# select catchments
# - area quality high or medium
# - timeseries after 1980
# - amount of days <250 -> nan
# - remove nans
# - >10 years data

# make list with catchment ids that are ok:
catch_id = []
df = pd.read_csv(f'{folder}GSIM_metadata/GSIM_catalog/GSIM_catchment_characteristics.csv',index_col=0) 

for filepath in glob.iglob(f'{folder}GSIM_processed_data/catchment_metadata/*'):
    print(filepath)
    a = pd.read_csv(filepath,index_col=0)
    a['start_year'] = pd.to_datetime(a['start_year'])
    a['end_year'] = pd.to_datetime(a['end_year'])
    k = a['gsim.no'][0]   
    
    # if end year earlier than 1980 go to next iteration
    if a['end_year'][0]< datetime.datetime(year=1980,month=1,day=1):
        continue
        
    filepath2 = f'{folder}GSIM_processed_data/catchment_yearly/mm_day/'+str(k)+'.csv'
    b = pd.read_csv(filepath2,index_col=0)
    
    # later than 1980
    b = b.loc['1980-12-31':]
    
    # set to nan if nr available days <250
    b.mean_mmd[b['nr_av_days']<250] = np.nan
    
    # drop nan years
    b = b.dropna(axis=0)
    
    # check length of timeseries
    len_b = len(b)
    
    # >10 years data
    if len_b>10:
        # check area quality
        if (df.loc[k]['quality']=='High') or (df.loc[k]['quality']=='Medium'): # exclude areas with 'caution' and 'low' area quality
            catch_id.append(a['gsim.no'].values[0])
            b.to_csv(str(folder)+'GSIM_processed_data/catchment_yearly_selected/'+str(k)+'.csv')
s = np.asarray(catch_id)
s = s.astype(str)
np.savetxt(f'{folder}GSIM_processed_data/catch_id_selected_mmd.txt',s,fmt = '%s')

#%% read catch_id_selected
a = np.loadtxt(f'{folder}GSIM_processed_data/catch_id_selected_mmd.txt',dtype=str)   

# convert upper case letters to lower case because shapefiles use lower case
z = np.zeros(len(a),dtype='<U10')
for i in range(len(z)):
    k = a[i].astype(str)
    k = k.lower()
    z[i] = k

np.savetxt(f'{folder}GSIM_processed_data/catch_id_selected_mmd_lowercase.txt',z,fmt = '%s')


#%% map the catchments with useful timeseries

# append all shapefilenames of selected catchments
shapefiles = []
for i in range(len(z)):
    shapefiles.append(f'{folder}GSIM_metadata/GSIM_catchments/'+str(z[i])+'.shp')
    o = f'{folder}GSIM_metadata/GSIM_catchments/'+str(z[i])+'.shp'
    t = f'{folder}GSIM_processed_data/shapes_selected/'+str(z[i])+'.shp'
    shutil.copy(o,t)

shapefiles2=[]
for i in range(len(shapefiles)):
    myfile = Path(str(shapefiles[i]))
    if myfile.is_file():
        shapefiles2.append(myfile)
        

# concat shapefiles into one geodataframe  
gdf = pd.concat([
    gpd.read_file(shp)
    for shp in shapefiles2
]).pipe(gpd.GeoDataFrame)

gdf.to_file(str(folder)+'GSIM_processed_data/merged_gpd_selected_catchments.shp')

# make map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
gdf = gpd.read_file(str(folder)+'GSIM_processed_data/merged_gpd_selected_catchments.shp')

fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111)
world.plot(ax=ax, edgecolor='black',linestyle=':', facecolor='white')

gdf.plot(edgecolor='black',
        linewidth=0.2,
        ax=ax)
ax.set_title('#catchments: '+str(len(gdf)),size=30)
fig.savefig(str(folder)+'/GSIM_processed_data/figures/mapping_catchments.jpg',bbox_inches='tight')

#%% split gdf in pieces for GEE analyses
a=[0,1000,2000,3000,4000,5000,6000,7000,8437]
for i in range(len(a)-1):
    gdf1 = gdf[a[i]:a[i+1]]
    print(gdf1)
    gdf1.to_file(f'{folder}GSIM_processed_data/merged_gpd_selected_catchments_{a[i]}_{a[i+1]}.shp')

    
#%% add australia shapes to gdf
gdf = gpd.read_file(str(folder)+'GSIM_processed_data/merged_gpd_selected_catchments.shp')
aus = gpd.read_file(str(folder)+'/CAMELS_AUS/02_location_boundary_area/shp/CAMELS_AUS_Boundaries_adopted.shp')

gdf = gdf.rename(columns={'FILENAME':'CatchID'})
gdf_aus = pd.concat([gdf,aus])    
gdf_aus.to_file(f'{folder}GSIM_processed_data/merged_gpd_selected_catchments_aus.shp')

# split into single shapefiles
for j in range(len(aus)):
    g = aus.loc[[j]]
    g.to_file(f'{folder}CAMELS_AUS_processed/shapes/{g.CatchID.values[0]}.shp')
    print(j)
    o = f'{folder}/CAMELS_AUS_processed/shapes/{g.CatchID.values[0]}.shp'
    t = f'{folder}/GSIM_processed_data/shapes_selected/{g.CatchID.values[0]}.shp'
    shutil.copy(o,t)

# plot    
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111)
world.plot(ax=ax, edgecolor='black',linestyle=':', facecolor='white')

gdf_aus.plot(edgecolor='black',
        linewidth=0.2,
        ax=ax)
ax.set_title('#catchments: '+str(len(gdf_aus)),size=30)
fig.savefig(str(folder)+'/GSIM_processed_data/figures/mapping_catchments_aus.jpg',bbox_inches='tight')

#%% add australia catch_id to id_list  
a = np.loadtxt(f'{folder}GSIM_processed_data/catch_id_selected_mmd.txt',dtype=str)
b = np.loadtxt(f'{folder}GSIM_processed_data/catch_id_selected_mmd_lowercase.txt',dtype=str)
c = np.loadtxt('/home/vanoorschot/Documents/CAMELS_AUS_processed/catch_id_aus.txt',dtype=str)

d=np.concatenate([a,c])
e=np.concatenate([b,c])

np.savetxt(f'{folder}GSIM_processed_data/catch_id_selected_mmd_aus.txt',d,fmt='%s')   
np.savetxt(f'{folder}GSIM_processed_data/catch_id_selected_mmd_lowercase_aus.txt',e,fmt='%s')   
    
