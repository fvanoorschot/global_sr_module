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
from scipy.optimize import minimize

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

def hai(df):
    """
    calculate holdridge aridity index 
    df: pandas dataframe, P and T timeseries
    returns:holdrige aridity index HAI [-]
    """
    pmean = df['p'].mean()
    t = df['tas']
    temp_adjusted = np.zeros(len(t))

    for i in range(1,len(t)):
        if t[i] < 0:
            temp_adjusted[i-1] = 0
        elif t[i] > 30:
            temp_adjusted[i-1] = 30
        else:
            temp_adjusted[i-1] = t[i]            

    T = np.sum(temp_adjusted)
    HAI = (58.93 * (T/len(t)))/(pmean*365)
    
    return HAI

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

def idu_mean(df):
    """
    interstorm duration: mean consecutive days of p<1mm
    df: pandas dataframe, P and Ep timeseries
    returns: idu_mean [days]
    """
    p = df['p']
    interstorm = []
    count = 0

    for j in range(len(p)):
        if p[j] < 1:
            count += 1
        elif p[j] >= 1 and count > 0:
            interstorm.append(count)
            count = 0 
    if (len(interstorm))>0:
        idu_mean = int(np.mean(interstorm))
    else:
        idu_mean=0     
    return idu_mean

def idu_max(df):
    """
    interstorm duration: mean annual maximum consecutive days of p<1mm
    df: pandas dataframe, P and Ep timeseries
    returns: idu_max [days]
    """
    df['year'] = pd.DatetimeIndex(df.index).year
    yearstart = df['year'][0]
    yearend = df['year'][-1]
    a = []
    for year in range(yearstart,yearend+1):
        dfy = df[df['year']==year]
        p = dfy['p']
        interstorm=[]
        count = 0
        for j in range(len(p)):
            if p[j] < 1:
                count += 1
            elif p[j] >= 1 and count > 0:
                interstorm.append(count)
                count = 0 
        if (len(interstorm))>0:
            a.append(max(interstorm))
        else:
            a.append(0)
    idu_max = int(np.mean(a))

    return idu_max

def hpd_mean(df):
    """
    high precipitation days: mean consecutive days of p>5*pmean
    df: pandas dataframe, P and Ep timeseries
    returns: hpd_mean [days]
    """
    p = df['p']
    pmean = df['p'].mean()
    high_p = []
    count = 0

    for j in range(len(p)):
        if p[j] > (5*pmean):
            count += 1
        elif p[j] <= (5*pmean) and count > 0:
            high_p.append(count)
            count = 0 
    if (len(high_p))>0:
        hpd_mean = int(np.mean(high_p))
    else:
        hpd_mean=0        
    return hpd_mean

def hpd_max(df):
    """
    high precipitation days: mean annual max consecutive days of p>5*pmean
    df: pandas dataframe, P and Ep timeseries
    returns: hpd_max [days]
    """
    df['year'] = pd.DatetimeIndex(df.index).year
    yearstart = df['year'][0]
    yearend = df['year'][-1]
    a = []
    for year in range(yearstart,yearend+1):
        dfy = df[df['year']==year]
        p = dfy['p']
        pmean = dfy['p'].mean()
        high_p=[]
        count = 0
        for j in range(len(p)):
            if p[j] > (5*pmean):
                count += 1
            elif p[j] <= (5*pmean) and count > 0:
                high_p.append(count)
                count = 0 
        if (len(high_p))>0:
            a.append(max(high_p))
        else:
            a.append(0)
    hpd_max = int(np.mean(a))
    return hpd_max

def hpf(df):
    """
    high precipitation frequency: days with p>5*pmean / total days
    df: pandas dataframe, P and Ep timeseries
    returns: hpf [-]
    """
    p = df['p']
    pmean = df['p'].mean()
    count = 0

    for j in range(len(p)):
        if p[j] > (5*pmean):
            count += 1
    hpf = count/len(p)
    return hpf

def lpf(df):
    """
    low precipitation frequency: days with p<1mm / total days
    df: pandas dataframe, P and Ep timeseries
    returns: lpf [-]
    """
    p = df['p']
    count = 0

    for j in range(len(p)):
        if p[j] <1:
            count += 1
    lpf = count/len(p)
    return lpf

def ftf(df):
    """
    freezing temperatures frequency: days with T<0 degreeC / total days
    df: pandas dataframe, tas timeseries
    returns: ftf [-]
    """
    t = df['tas']
    count = 0
    for j in range(len(t)):
        if t[j] < 0:
            count += 1
    ftf = count/len(t)
    return ftf

def tdiff_mean(df):
    """
    mean temperature difference: monthly mean t max - monthly mean t min
    df: pandas dataframe, tas timeseries
    returns: tdiff_mean [-]
    """
    df['year'] = pd.DatetimeIndex(df.index).year
    yearstart = df['year'][0]
    yearend = df['year'][-1]
    tdiff = []
    for year in range(yearstart,yearend+1):
        dfy = df[df['year']==year]
        dfy = dfy.groupby(pd.Grouper(freq="M")).mean()
        t = dfy['tas']
        tmax = dfy['tas'].max()
        tmin = dfy['tas'].min()
        tdiff.append(tmax-tmin)
    tdiff_mean = np.mean(tdiff)
    return tdiff_mean

def tdiff_max(df):
    """
    max temperature difference: monthly mean t max - monthly mean t min
    df: pandas dataframe, tas timeseries
    returns: tdiff_max [-]
    """
    df['year'] = pd.DatetimeIndex(df.index).year
    yearstart = df['year'][0]
    yearend = df['year'][-1]
    tdiff = []
    for year in range(yearstart,yearend+1):
        dfy = df[df['year']==year]
        dfy = dfy.groupby(pd.Grouper(freq="M")).mean()
        t = dfy['tas']
        tmax = dfy['tas'].max()
        tmin = dfy['tas'].min()
        tdiff.append(tmax-tmin)
    tdiff_max = np.max(tdiff)
    return tdiff_max

#Function for Seasonality Timing Index
def ST_calc(dP,dT):
    days = 366
    ST = dP[0] * np.sign(dT[0]) * np.cos((np.pi * (dP[1] - dT[1]))/days)
    return ST

#Functions to compute Seasonal variability indexes
def T_daily(dT):
    t = np.linspace(1,366,366)
    days = 366
    T = T_mean + dT[0] * np.sin((2*np.pi * (t-dT[1]))/days)
    return T

def Cal_T_daily(dT):
    days = 366
    T_calc = T_daily(dT)
    
    return (np.sum(np.abs(T_calc - T_obs)))/days

def P_daily(dP):
    t = np.linspace(1,366,366)
    days = 366
    P = P_mean * (1 + dP[0] * np.sin((2*np.pi * (t-dP[1]))/days))
    return P

def Cal_P_daily(dP):
    days = 366
    P_calc = P_daily(dP)
    
    return (np.sum(np.abs(P_calc - P_obs)))/days

def E_daily(dE):
#    t = np.linspace(1,366,366)
    t = np.linspace(1,366,366)

    days = 366
    E = E_mean * (1 + dE[0] * np.sin((2*np.pi * (t-dE[1]))/days))
    return E

def Cal_E_daily(dE):
    days = 366
    E_calc = E_daily(dE)
    
    return (np.sum(np.abs(E_calc - E_obs)))/days

#Compute the seasonality variability indexes
def seas_var_indices(df):
    """
    calculate seasonality variability indices from Berghuijs 2014
    de, dp, dt: seasonal ep, p and t amplitudes
    sp, st, se: phase shifts of p, t and ep
    sd: phase difference between p and ep
    sti: seasonality timing index
    df: pandas dataframe, P and Ep timeseries
    returns: de,dp,dt,sp,st,se,sd,sti
    """
    global T_mean
    global P_mean
    global E_mean
    global T_obs
    global P_obs
    global E_obs
    
    data_d = df.resample('d').mean().bfill()
    daily_sliced_mean = df.groupby([data_d.index.month, data_d.index.day]).agg(np.mean)

    T1 = np.zeros((366))
    P1 = np.zeros((366))
    E1 = np.zeros((366))
    count = 0

    for k in range(1,13):
        for j in range(1,len(daily_sliced_mean['tas'][k])+1):
            T1[count] = daily_sliced_mean['tas'][k,j]
            P1[count] = daily_sliced_mean['p'][k,j]
            E1[count] = daily_sliced_mean['ep'][k,j]
            count += 1
    t = np.linspace(1,366,366)

    T_obs = T1
    P_obs = P1
    E_obs = E1

    T_mean = np.nanmean(T_obs)
    P_mean = np.nanmean(P_obs)
    E_mean = np.nanmean(E_obs)

    x0_T = [5, 110]
    x0_P = [0.3, 40]
    x0_E = [0.4, 40]
    lb = [0, np.inf]
    ub = [0, 366]

    res_T = minimize(Cal_T_daily, x0_T,method='Powell', bounds=(lb,ub))
    res_P = minimize(Cal_P_daily, x0_P, method='Powell', bounds=(lb,ub))
    res_E = minimize(Cal_E_daily, x0_E, method='Powell', bounds=(lb,ub))

    dp = res_P.x[0]
    sp = res_P.x[1] / 366
    dt = res_T.x[0]
    st = res_T.x[1] / 366

    if abs(sp - st) <= 0.5:
        sd = sp - st

    elif (sp - st) > 0.5:
        sd = -1 + (sp - st)

    else:
        sd = 1 + (sp - st)

    de = res_E.x[0]
    se = res_E.x[1] / 366

    #Compute Seasonality Timing index
    sti = ST_calc(res_P.x,res_T.x)

    return de,dp,dt,sp,st,se,sd,sti


def catch_characteristics_climate(var_cl, catch_id,work_dir):
    """
    calculate catchment characteristics - climate variables
    var_cl: define list of climate variables
    catch_id: catchment id
    returns: dataframe with climate variables for catchment
    """
    cc_cl = pd.DataFrame(index=[catch_id], columns=var_cl)
    j = catch_id
    l = glob.glob(f'{work_dir}/output/forcing_timeseries/processed/daily/{j}*.csv') #find daily forcing (P Ep T) timeseries for catchment 
    df = pd.read_csv(l[0], index_col=0)
    df.index = pd.to_datetime(df.index)

    l_q = glob.glob(f'{work_dir}/output/q_timeseries_selected/{j}*.csv') # find discharge data for catchment
    df_q = pd.read_csv(l_q[0], index_col=0)
    df_q.index = pd.to_datetime(df_q.index)

    # calculate catchment characteristics using functions in (1)
    if 'p_mean' in var_cl:
        cc_cl.loc[j,'p_mean'] = p_mean(df)

    if 'q_mean' in var_cl:
        cc_cl.loc[j,'q_mean'] = q_mean(df_q)

    if 'ep_mean' in var_cl:
        cc_cl.loc[j,'ep_mean'] = ep_mean(df)

    if 't_mean' in var_cl:
        cc_cl.loc[j,'t_mean'] = t_mean(df)

    if 'ai' in var_cl:
        cc_cl.loc[j,'ai'] = ai(df)

    if 'si_p' in var_cl:
        cc_cl.loc[j,'si_p'] = si_p(df)

    if 'si_ep' in var_cl:
        cc_cl.loc[j,'si_ep'] = si_ep(df)    

    if 'phi' in var_cl:
        cc_cl.loc[j,'phi'] = phi(df)

    if 'tdiff_mean' in var_cl:
        cc_cl.loc[j,'tdiff_mean'] = tdiff_mean(df)
    if 'tdiff_max' in var_cl:
        cc_cl.loc[j,'tdiff_max'] = tdiff_max(df)

    if 'hai' in var_cl:
        cc_cl.loc[j,'hai'] = hai(df)
    if 'ftf' in var_cl:
        cc_cl.loc[j,'ftf'] = ftf(df)

    if 'idu_mean' in var_cl:
        cc_cl.loc[j,'idu_mean'] = idu_mean(df)
    if 'idu_max' in var_cl:
        cc_cl.loc[j,'idu_max'] = idu_max(df)

    if 'hpd_mean' in var_cl:
        cc_cl.loc[j,'hpd_mean'] = hpd_mean(df)
    if 'hpd_max' in var_cl:
        cc_cl.loc[j,'hpd_max'] = hpd_max(df)

    if 'hpf' in var_cl:
        cc_cl.loc[j,'hpf'] = hpf(df)
    if 'lpf' in var_cl:
        cc_cl.loc[j,'lpf'] = lpf(df)

    if 'sti' in var_cl:
        cc_cl.loc[j,['de','dp','dt','sp','st','se','sd','sti']] = seas_var_indices(df)
        
    return cc_cl

def catch_characteristics_landscape(var_lc,catch_id,work_dir):
    """
    calculate catchment characteristics - landscape variables
    var_lc: define list of landscape variables
    catch_id: catchment id
    returns: dataframe with landscape variables for catchment
    """
    cc_lc = pd.DataFrame(index=[catch_id], columns=var_lc)
    j = catch_id
    l = glob.glob(f'{work_dir}/output/forcing_timeseries/processed/daily/{j}*.csv') #find daily forcing (P Ep T) timeseries for catchment 
    df = pd.read_csv(l[0], index_col=0)
    df.index = pd.to_datetime(df.index)

    if 'tc' in var_lc:
        dft = pd.read_csv(f'{work_dir}/output/treecover/gsim_shapes_treecover.csv',index_col=0) #find treecover tables for catchment
        cc_lc.loc[j,'tc'] = dft.loc[j,'mean_tc'] /100
        cc_lc.loc[j,'ntc'] = dft.loc[j,'mean_ntc']/100
        cc_lc.loc[j,'nonveg'] = dft.loc[j,'mean_nonveg']/100

    if 'area' in var_lc:
        a = pd.read_csv(f'{work_dir}/output/catchment_area.csv',index_col=0)
        cc_lc.loc[j,'area'] = a.loc[j,'area']

    # add gsim variables
    gsim_var=['ir.mean','ele.mean','ele.min','ele.max','dr.mean','slp.mean','scl.mean','snd.mean','slt.mean','tp.mean']
    df_gsim = pd.read_csv(f'{work_dir}/data/GSIM_data/GSIM_metadata/GSIM_catalog/GSIM_catchment_characteristics.csv',index_col=0)
    k = j.upper()
    if k in df_gsim.index.values:
        cc_lc.loc[j,['ir_mean','el_mean','el_min','el_max','drd','slp_mean','cla','snd','slt','tpi']] = df_gsim.loc[k,gsim_var].values
    else:
        # use aus information
        df_aus = pd.read_csv(f'{work_dir}/data/CAMELS_AUS/CAMELS_AUS_Attributes-Indices_MasterTable.csv',index_col=0)
        cc_lc.loc[j,['el_mean','el_min','el_max','drd','slp_mean','cla','snd']] = df_aus.loc[j,['elev_mean','elev_min','elev_max','strdensity','mean_slope_pct','claya','sanda']].values  
        cc_lc.loc[j,['ir_mean','slt','tpi']] = np.nan # not available for camels aus    
    return cc_lc

def catch_characteristics(var_lc,var_cl, catch_id, work_dir):
    """
    combine climate and landscape variables in one dataframe
    returns: catchment characteristics dataframe cc - saved as csv file
    """
    cc_lc = catch_characteristics_landscape(var_lc,catch_id,work_dir)
    cc_cl = catch_characteristics_climate(var_cl, catch_id,work_dir)
    cc = pd.concat([cc_cl,cc_lc],axis=1)
    cc.to_csv(f'{work_dir}/output/catchment_characteristics/{catch_id}.csv') #store cc dataframe
    

def run_function_parallel_catch_characteristics(
    var_lc_list=list,
    var_cl_list=list,
    catch_list=list,
    work_dir_list=list,
    # threads=None
    threads=100
    ):
    """
    Runs function preprocess_gsim_discharge  in parallel.

    var_cl_list: str,list, list of climate variables
    var_lc_list: str,list, list of landscape variables
    catch_list:  str, list, list of catchmet ids
    work_dir_list:     str, list, list of work dir
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
        var_lc_list,
        var_cl_list,
        catch_list,
        work_dir_list,
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
