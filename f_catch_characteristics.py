"""
f_catch_characteristics
------------------------
calculate and organize catchment characteristics based on timeseries of Q P and Ep

TO DO -> add more variables (ST-index, other EE products,....)

1. compute catchment characteristics (p_mean, ep_mean, t_mean, ai, si_p, si_ep, phi, q_mean)
2. catch_characteristics
3. geo_catchments


"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from scipy.optimize import least_squares
import geopandas as gpd
from pathos.threading import ThreadPool as Pool

## 1
def p_mean(df):
    """
    calculate mean precipitation
    df: pandas dataframe, P timeseries
    returns: mean P [mm/day]
    """
    m = df['p'].mean()
    return m

def ep_mean(df):
    """
    calculate mean potential evaporation
    df: pandas dataframe, Ep timeseries
    returns: mean Ep [mm/day]
    """
    m = df['ep'].mean()
    return m

def t_mean(df):
    """
    calculate mean temperature
    df: pandas dataframe, T timeseries
    returns: mean T [deg C]
    """
    m = df['tas'].mean()
    return m

def ai(df):
    """
    calculate aridity index (P/Ep)
    df: pandas dataframe, P and Ep timeseries
    returns: aridity index AI [-]
    """
    ai = df['p'].mean()/df['ep'].mean()
    return ai

def si_p(df):
    """
    calculate seasonality index of precipitation (see https://esd.copernicus.org/articles/12/725/2021/ for equation)
    df: pandas dataframe, P timeseries
    returns: seasonality index SI_P [-]
    """
    p = df['p']
    for j in p.index:
        if(j.month==1 and j.day==1):
            start_date = j
            break
    for j in p.index:
        if(j.month==12 and j.day==31):
            end_date = j

    p_annual = p.loc[start_date:end_date].groupby(pd.Grouper(freq="Y")).sum()
    pa = p_annual.mean()
    p_monthly = p.loc[start_date:end_date].groupby(pd.Grouper(freq="M")).sum()
    pm = p_monthly.groupby([p_monthly.index.month]).mean()

    a = np.zeros(12)
    for k in range(12):
        a[k] = np.abs(pm[k + 1]-(pa/12))
    if (pa>0):
        sip = (1/pa)*np.sum(a)
    else:
        sip = np.nan
    return sip
    
def si_ep(df):
    """
    calculate seasonality index of potential evaporation (see https://esd.copernicus.org/articles/12/725/2021/ for equation)
    df: pandas dataframe, Ep timeseries
    returns: seasonality index SI_Ep [-]
    """
    ep = df['ep']
    for j in ep.index:
        if(j.month==1 and j.day==1):
            start_date = j
            break
    for j in ep.index:
        if(j.month==12 and j.day==31):
            end_date = j
        
    ep_annual = ep.loc[start_date:end_date].groupby(pd.Grouper(freq="Y")).sum()
    epa = ep_annual.mean()
    ep_monthly = ep.loc[start_date:end_date].groupby(pd.Grouper(freq="M")).sum()
    epm = ep_monthly.groupby([ep_monthly.index.month]).mean()
    
    a = np.zeros(12)
    for k in range(12):
        a[k] = np.abs(epm[k + 1]-(epa/12))
    if (epa>0):
        siep = (1/epa)*np.sum(a)
    else:
        siep=np.nan
    return siep

def phi(df):
    """
    calculate phase lag (timing shift) between max Ep and max P 
    df: pandas dataframe, P and Ep timeseries
    returns: phase lag phi [months]
    """
    p = df['p']
    ep = df['ep']
    for j in p.index:
        if(j.month==1 and j.day==1):
            start_date = j
            break
    for j in p.index:
        if(j.month==12 and j.day==31):
            end_date = j
            
    p_annual = p.loc[start_date:end_date].groupby(pd.Grouper(freq="Y")).sum()
    pa = p_annual.mean()
    p_monthly = p.loc[start_date:end_date].groupby(pd.Grouper(freq="M")).sum()
    pm = p_monthly.groupby([p_monthly.index.month]).mean()
    ep_annual = ep.loc[start_date:end_date].groupby(pd.Grouper(freq="Y")).sum()
    epa = ep_annual.mean()
    ep_monthly = ep.loc[start_date:end_date].groupby(pd.Grouper(freq="M")).sum()
    epm = ep_monthly.groupby([p_monthly.index.month]).mean()
    
    epm_max_month = epm.idxmax()
    pm_max_month = pm.idxmax()
    phi = np.abs(epm_max_month - pm_max_month)
    if(phi>6):
        phi = 12 + min(epm_max_month,pm_max_month) - max(epm_max_month,pm_max_month)
    return phi

def q_mean(df_q):
    """
    calculate mean discharge
    df: pandas dataframe, Q timeseries
    returns: mean Q [mm/day]
    """
    m = df_q['Q'].mean()
    return m

## 2 
def catch_characteristics(var, catch_id, fol_in, fol_out):
    """
    calculate catchment characteristics and store in dataframe
    var:             str, list, list of variables (options: p_mean, q_mean, ep_mean, t_mean, ai, si_p, si_ep, phi, tc)
    catch_id_list:   str, list, list of catchment ids
    fol_in:          str, dir, directory with timeseries data
    fol_out:         str, dir, directory where to store the output tables
    
    returns: table (cc) with catchment characteristics for all catchments       
    """
    # make cc dataframe
    cc = pd.DataFrame(index=[catch_id], columns=var)
    j = catch_id
    l = glob.glob(f'{fol_in}/forcing_timeseries/processed/daily/{j}*.csv') #find daily forcing (P Ep T) timeseries for catchment 
    df = pd.read_csv(l[0], index_col=0)
    df.index = pd.to_datetime(df.index)

    l_q = glob.glob(f'{fol_in}/q_timeseries_selected/{j}*.csv') # find discharge data for catchment
    df_q = pd.read_csv(l_q[0], index_col=0)
    df_q.index = pd.to_datetime(df_q.index)

    # calculate catchment characteristics using functions in (1)
    if 'p_mean' in var:
        cc.loc[j,'p_mean'] = p_mean(df)

    if 'q_mean' in var:
        cc.loc[j,'q_mean'] = q_mean(df_q)

    if 'ep_mean' in var:
        cc.loc[j,'ep_mean'] = ep_mean(df)

    if 't_mean' in var:
        cc.loc[j,'t_mean'] = t_mean(df)

    if 'ai' in var:
        cc.loc[j,'ai'] = ai(df)

    if 'si_p' in var:
        cc.loc[j,'si_p'] = si_p(df)

    if 'si_ep' in var:
        cc.loc[j,'si_ep'] = si_ep(df)    

    if 'phi' in var:
        cc.loc[j,'phi'] = phi(df)

    # get tree cover statistics 
    if 'tc' in var:
        # l = glob.glob(f'{fol_in}/treecover/{j}*.csv') #find treecover tables for catchment
        # dft = pd.read_csv(l[0], index_col=0)
        # cc.loc[j,'tc'] = dft.loc[j,'mean_tc']
        # cc.loc[j,'ntc'] = dft.loc[j,'mean_ntc']
        # cc.loc[j,'nonveg'] = dft.loc[j,'mean_nonveg']
        dft = pd.read_csv(f'{fol_in}/treecover/gsim_shapes_treecover.csv',index_col=0) #find treecover tables for catchment
        cc.loc[j,'tc'] = dft.loc[j,'mean_tc']
        cc.loc[j,'ntc'] = dft.loc[j,'mean_ntc']
        cc.loc[j,'nonveg'] = dft.loc[j,'mean_nonveg']

    a = pd.read_csv(f'{fol_in}/catchment_area.csv',index_col=0)
    cc.loc[j,'area'] = a.loc[j,'area']
            
    cc.to_csv(f'{fol_out}/catchment_characteristics/{j}.csv') #store cc dataframe
    return cc

def run_function_parallel_catch_characteristics(
    var_list=list,
    catch_list=list,
    fol_in_list=list,
    fol_out_list=list,
    # threads=None
    threads=100
):
    """
    Runs function preprocess_gsim_discharge  in parallel.

    var_list: str,list, list of variables
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
        catch_characteristics,
        var_list,
        catch_list,
        fol_in_list,
        fol_out_list,
    )
    

    
## 3
def geo_catchments(shape_dir,out_dir):
    """
    merge all catchment shapefiles into one
    
    shape_dir:   str, dir, directory with shapefiles
    out_dir:     out, dir, output directory for merged shapefile
    
    Stores merged shapefile as .shp
    
    """
    # list al shapefiles    
    shapefiles = glob.glob(f"{shape_dir}/*shp")
    li=[] #empty list
    for filename in shapefiles:
        df = gpd.read_file(filename, index_col=None, header=0) #read shapefile as geopandas dataframe
        li.append(df) #append shapefile to list
    f = pd.concat(li, axis=0) #concatenate lists
    f = f.rename(columns={'FILENAME':'catch_id'})
    f.index = f['catch_id']
    f = f.drop(columns={'catch_id','Id'})
    f.to_file(f'{out_dir}/geo_catchments.shp') #store geopandas dataframe as .shp
