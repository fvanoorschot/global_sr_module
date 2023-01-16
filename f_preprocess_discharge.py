"""
f_preprocess_discharge.py
-------------------------
This function contains functions to preprocess the GSIM discharge data

The columns in the data have the following meaning (see table 3 and 4 paper Gudmundsson part2 - https://essd.copernicus.org/articles/10/787/2018/):
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


# import packages
import glob
from pathlib import Path
import os
import numpy as np
from datetime import datetime
from datetime import timedelta
import pandas as pd
import calendar
import geopandas as gpd
import cartopy
import matplotlib.pyplot as plt
import math
from pathos.threading import ThreadPool as Pool
from scipy.optimize import least_squares
import sklearn
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import random

def preprocess_gsim_discharge(catch_id, fol_in, fol_out):

    """
    This function preprocesses the GSIM discharge data to readable and usable tables
    input:
    catch_id:   str, catchment ID 
    fol_in:     str, dir, folder with GSIM discharge timeseries 
    fol_out:    str, dir, folder for output tables 

    returns:
    c:     pandas dataframe with catchment characteristics, stored as csv
    a_mm:  pandas dataframe with discharge timeseries, stored as csv
    """
    # open pandas dataframe from GSIM raw files in fol_in
    catch_id_c = catch_id.upper() # lower case to capital letters
    filepath = glob.glob(f'{fol_in}/{catch_id_c}*')[0] # find catch id's timeseries
    b = pd.read_csv(filepath,delimiter=':',skiprows=10,nrows=7, index_col=0) # read pandas dataframe catch id's timeseries

    # make dataframe with catchment characteristics
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

    # make dataframe with catchment timeseries
    a = pd.read_csv(filepath, skiprows=21, delimiter=',',index_col=0)
    a.index = pd.to_datetime(a.index)
    a.columns = a.columns.str.strip() # remove \t (=tab))    
    a = a.drop(columns=['"GINI"','"CV"','"CT"','"P10"', '"P20"', '"P30"', '"P40"', '"P50"', '"P60"', '"P70"', '"P80"', '"P90"']) # drop columns we don't need
    a = a.astype(str)

    # remove all tabs
    for i in range(len(a)): #loop over rows
        for j in range(len(a.iloc[0])): # loop over columns
            a.iloc[i,j] = a.iloc[i,j].strip()

    a = a.replace('NA',np.nan) # replace NA with np.nan

    #rename columns
    a = a.rename(columns={'"MEAN"': 'mean_m3s', '"SD"': 'sd_m3s', '"IQR"': 'iqr_m3s', '"MIN"': 'min_m3s', '"MAX"': 'max_m3s',
            '"MIN7"': 'min7_m3s', '"MAX7"': 'max7_m3s', '"DOYMIN"':'doymin','"DOYMAX"':'doymax', '"DOYMIN7"':'doy7min', '"DOYMAX7"':'doy7max',
            '"n.missing"':'nr_mis_days','"n.available"':'nr_av_days'})
    a = a.astype(float)

    #convert values to mm/day using the catchment area saved in dataframe c
    area_m2 = c.area_km2.values[0] * (10**6) #convert km2 to m2
    a_mm = a
    a_mm = a_mm.rename(columns={'mean_m3s':'mean_mmd', 'sd_m3s':'sd_mmd', 'iqr_m3s':'iqr_mmd', 'min_m3s':'min_mmd', 'max_m3s':'max_mmd', 'min7_m3s':'min7_mmd','max7_m3s':'max7_mmd'})
    a_mm[['mean_mmd', 'sd_mmd', 'iqr_mmd', 'min_mmd', 'max_mmd', 'min7_mmd','max7_mmd']] = a[['mean_m3s', 'sd_m3s', 'iqr_m3s', 'min_m3s', 'max_m3s', 'min7_m3s','max7_m3s']] * (1/area_m2) * 1000 * 86400
    a_mm = a_mm.rename(columns={'mean_m3s':'mean_mmd', 'sd_m3s':'sd_mmd', 'iqr_m3s':'iqr_mmd', 'min_m3s':'min_mmd', 'max_m3s':'max_mmd', 'min7_m3s':'min7_mmd','max7_m3s':'max7_mmd'})
    a_mm = a_mm.astype(float)
    a_mm = a_mm.rename(columns={'mean_mmd':'Q'})

    # add length of timeseries to dataframe c
    c['start_year'] = a_mm.index[0]
    c['end_year'] = a_mm.index[-1]
    c['length_ts_yr'] = int((a_mm.index[0] - a_mm.index[-1]).days / -365)
    c['start_year'] = pd.to_datetime(c['start_year'])
    c['end_year'] = pd.to_datetime(c['end_year'])

    # save files as csv in fol_out
    c.to_csv(f'{fol_out}/characteristics/{catch_id}.csv')
    # a.to_csv(f'{fol_out}/timeseries/{catch_id}.csv')
    a_mm.to_csv(f'{fol_out}/timeseries/{catch_id}.csv')
    
    return c, a_mm
    
def run_function_parallel(
    catch_list=list,
    fol_in_list=list,
    fol_out_list=list,
    # threads=None
    threads=100
):
    """
    Runs function preprocess_gsim_discharge  in parallel.

    catch_list:  str, list, list of catchmet ids
    fol_in_list:     str, list, list of input folders
    fol_out_list:   str, list, list of output folders
    threads:         int,       number of threads (cores), when set to None use all available threads

    Returns: None
    """
    # Set number of threads (cores) used for parallel run and map threads
    if threads is None:
        pool = Pool()
    else:
        pool = Pool(nodes=threads)
    # Run parallel models
    results = pool.map(
        preprocess_gsim_discharge,
        catch_list,
        fol_in_list,
        fol_out_list,
    )

    
def select_catchments(data_dir,out_dir,catch_id):
    # load df with all catchments with characteristics
    df = pd.read_csv(f'{data_dir}/GSIM_data/GSIM_metadata/GSIM_catalog/GSIM_catchment_characteristics.csv',index_col=0) 
    
    # load catchment characteristic file
    a = pd.read_csv(f'{out_dir}/gsim/characteristics/{catch_id}.csv',index_col=0)
    a['start_year'] = pd.to_datetime(a['start_year'])
    a['end_year'] = pd.to_datetime(a['end_year'])
    k = a['gsim.no'][0]   

    # check if end year later than 1980
    t_1980 = a['end_year'][0]< datetime(year=1980,month=1,day=1)

    # check timeseries
    b = pd.read_csv(f'{out_dir}/gsim/timeseries/{catch_id}.csv',index_col=0)

    # later than 1980
    b = b.loc['1980-12-31':]

    # set to nan if nr available days <250
    b.Q[b['nr_av_days']<250] = np.nan

    # drop nan years
    b = b.dropna(axis=0)

    # check length of timeseries
    len_b = len(b)

    # >10 years data
    t_length = len_b>10

    # area quality
    a_qual = (df.loc[k]['quality']=='High') or (df.loc[k]['quality']=='Medium')

    # if three criteria are met -> save catchment timeseries in timeseries_selected
    if (t_length) & (a_qual) & (t_1980):
        b.to_csv(f'{out_dir}/gsim/timeseries_selected/{catch_id}.csv')
        
def run_function2_parallel(
    data_dir_list=list,
    out_dir_list=list,
    catch_list=list,
    # threads=None
    threads=100
):
    """
    Runs function select_catchments in parallel.

    data_dir_list: str, list, list of data dirs
    out_dir_list: str, list, list of output directories
    catch_list: str, list, list of catchment ids
    threads:         int,       number of threads (cores), when set to None use all available threads

    Returns: None
    """
    # Set number of threads (cores) used for parallel run and map threads
    if threads is None:
        pool = Pool()
    else:
        pool = Pool(nodes=threads)
    # Run parallel models
    results = pool.map(
        select_catchments,
        data_dir_list,
        out_dir_list,
        catch_list,
    )