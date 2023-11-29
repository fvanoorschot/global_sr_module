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
from datetime import datetime
from datetime import timedelta

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
    calculate aridity index (Ep/P)
    df: pandas dataframe, P and Ep timeseries
    returns: aridity index AI [-]
    """
    ai = df['ep'].mean()/df['p'].mean()
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

def ppd(df):
    '''
    potential precipitation deficit
    monthly mean p and ep
    for months with ep>p: sum values (use longest consecutive time period only)
    return: sum
    '''
    p = df['p']
    ep = df['ep']
    for j in p.index:
        if(j.month==1 and j.day==1):
            start_date = j
            break
    for j in p.index:
        if(j.month==12 and j.day==31):
            end_date = j
            
    df_m = df.loc[start_date:end_date].groupby(pd.Grouper(freq="M")).sum()
    df_mm = df_m.groupby([df_m.index.month]).mean()

    df_mm[df_mm.ep<df_mm.p] = np.nan
    a = df_mm.p.values  # Extract out relevant column from dataframe as array
    if (len(df_mm.dropna())==0):
        difsum=0
    else:
        m = np.concatenate(( [True], np.isnan(a), [True] ))  # Mask
        ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)   # Start-stop limits
        start,stop = ss[(ss[:,1] - ss[:,0]).argmax()]  # Get max interval, interval limits
        if (start==0)&(pd.isna(df_mm.p.iloc[-1])==False):
            df_mm2 = df_mm
            dif = df_mm2.p - df_mm2.ep
            difsum = dif.sum()
        else:
            df_mm2 = df_mm.iloc[start:stop]
            dif = df_mm2.p - df_mm2.ep
            difsum = dif.sum()
    return abs(difsum)

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
        idu_mean = np.round(np.mean(interstorm),2)
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
    idu_max = np.round(np.mean(a),2)

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
        hpd_mean = np.round(np.mean(high_p),2)
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
    hpd_max = np.round(np.mean(a),2)
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
    low precipitation frequency: days with p<1mm
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

# calculate asynchronity index of p and ep - Feng 2019
def asi(df):
    p = df.p
    ep = df.ep
    p_monthly = p.groupby(pd.Grouper(freq="M")).sum()
    pm = p_monthly.groupby([p_monthly.index.month]).mean()
    ep_monthly = ep.groupby(pd.Grouper(freq="M")).sum()
    epm = ep_monthly.groupby([p_monthly.index.month]).mean()

    ppm = pm/pm.sum()
    pepm = epm/epm.sum()
    asi_var = get_synchronicity(ppm,pepm)
    return asi_var

def get_JSDiv(prcp_pdf, temp_pdf):    
    M = 0.5*(temp_pdf + prcp_pdf)
    pre = prcp_pdf[prcp_pdf>0]; M1 = M[prcp_pdf>0]
    tmp = temp_pdf[temp_pdf>0]; M2 = M[temp_pdf>0]
    DivRM = np.sum(pre * np.log2(pre/M1))
    DivPM = np.sum(tmp * np.log2(tmp/M2))
    JSDiv = np.round( 0.5*(DivRM + DivPM), 10)
    return JSDiv 

def get_synchronicity(prcp_pdf, temp_pdf):
    JSDiv_m = np.zeros(12)
    for m_shift in range(12): 
        temp_rolled = np.roll(temp_pdf, m_shift)
        JSDiv_m[m_shift] = get_JSDiv(prcp_pdf, temp_rolled)
    diffJSDiv = np.sqrt( JSDiv_m[0] - np.min(JSDiv_m) ) 
    return diffJSDiv    
    
def cvp(df):
    p = df.p
    p_annual = p.groupby(pd.Grouper(freq="Y")).sum()
    ps = p_annual.std()
    pa = p_annual.mean()
    cvp_var = ps/pa
    return cvp_var

# combine climate catch characteristics
def catch_characteristics_climate(var_cl,var_sn, catch_id,work_dir,data_sources):
    cc_cl = pd.DataFrame(index=[catch_id], columns=var_cl)
    j = catch_id
    if (data_sources=='gswp-p_gleam-ep_gswp-t'):
        l = glob.glob(f'{work_dir}/output/forcing_timeseries/processed/daily/{j}*.csv') #find daily forcing (P Ep T) timeseries for catchment 
        df = pd.read_csv(l[0], index_col=0)
        df.index = pd.to_datetime(df.index)

    l_q = glob.glob(f'{work_dir}/output/q_timeseries_selected/{j}*.csv') # find discharge data for catchment
    df_q = pd.read_csv(l_q[0], index_col=0)
    df_q.index = pd.to_datetime(df_q.index)

    # find longest timeseries for Q and P-T-Ep
    p_ep_start_year = df.index.year[0]
    q_start_year = int(df_q.index[0].year)
    p_ep_end_year = df.index.year[-1]
    q_end_year = int(df_q.index[-1].year)

    if q_start_year>p_ep_end_year:
        a=1
    elif p_ep_start_year>q_end_year:
        a=1
    else:
        a=0
        start_year = max(q_start_year,p_ep_start_year)
        end_year = min(q_end_year,p_ep_end_year)
        start_date = datetime(start_year,1,1)
        start_date_q = datetime(start_year,12,31)
        end_date = datetime(end_year,12,31)

    cc_cl['start_year'] = start_year
    cc_cl['end_year'] = end_year
    cc_cl['years'] = end_year-start_year

    # select matching timeseries
    df_q = df_q.loc[start_date_q:end_date]
    df = df.loc[start_date:end_date]

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
    
    if 'asi' in var_cl:
        cc_cl.loc[j,'asi'] = asi(df)
        
    if 'cvp' in var_cl:
        cc_cl.loc[j,'cvp'] = cvp(df)
    if 'ppd' in var_cl:
        cc_cl.loc[j,'ppd'] = ppd(df)
    
    # SNOW
    snow_list=np.genfromtxt(f'{work_dir}/output/snow/catch_id_list_snow_t_and_p_italy.txt',dtype='str')
    cc_sn = pd.DataFrame(index=[catch_id], columns=var_sn)
    j = catch_id
    if (data_sources=='gswp-p_gleam-ep_gswp-t'):
        pdata='gswp'
        if j in snow_list: # if snow, use liquid input instead of p
            l = glob.glob(f'{work_dir}/output/snow/timeseries_{pdata}/{j}*.csv') #find daily forcing (P Ep T) timeseries for catchment liquid input
            df = pd.read_csv(l[0], index_col=0)
            df.index = pd.to_datetime(df.index)
            df = df.loc[start_date:end_date]

            # set df p column to df pl 
            df['p']=df['pl']

            # calculate catchment characteristics using functions in (1)
            if 'si_pl' in var_sn:
                cc_sn.loc[j,'si_pl'] = si_p(df) 

            if 'phi_l' in var_sn:
                cc_sn.loc[j,'phi_l'] = phi(df)

            if 'idu_mean_l' in var_sn:
                cc_sn.loc[j,'idu_mean_l'] = idu_mean(df)
            if 'idu_max_l' in var_sn:
                cc_sn.loc[j,'idu_max_l'] = idu_max(df)

            if 'hpd_mean_l' in var_sn:
                cc_sn.loc[j,'hpd_mean_l'] = hpd_mean(df)
            if 'hpd_max_l' in var_sn:
                cc_sn.loc[j,'hpd_max_l'] = hpd_max(df)

            if 'hpf_l' in var_sn:
                cc_sn.loc[j,'hpf_l'] = hpf(df)
            if 'lpf_l' in var_sn:
                cc_sn.loc[j,'lpf_l'] = lpf(df)

            if 'sti_l' in var_sn:
                cc_sn.loc[j,['de_l','dp_l','dt_l','sp_l','st_l','se_l','sd_l','sti_l']] = seas_var_indices(df)
                
            if 'asi_l' in var_sn:
                cc_sn.loc[j,'asi_l'] = asi(df)
            
            if 'ppd_l' in var_sn:
                cc_sn.loc[j,'ppd_l'] = ppd(df)

        else: # if no snow, then liquid p variables same as normal p variables
            l = glob.glob(f'{work_dir}/output/forcing_timeseries/processed/daily/{j}*.csv') #find daily forcing (P Ep T) timeseries for catchment 
            df = pd.read_csv(l[0], index_col=0)
            df.index = pd.to_datetime(df.index)
            df = df.loc[start_date:end_date]

            # calculate catchment characteristics using functions in (1)
            if 'si_pl' in var_sn:
                cc_sn.loc[j,'si_pl'] = si_p(df) 

            if 'phi_l' in var_sn:
                cc_sn.loc[j,'phi_l'] = phi(df)

            if 'idu_mean_l' in var_sn:
                cc_sn.loc[j,'idu_mean_l'] = idu_mean(df)
            if 'idu_max_l' in var_sn:
                cc_sn.loc[j,'idu_max_l'] = idu_max(df)

            if 'hpd_mean_l' in var_sn:
                cc_sn.loc[j,'hpd_mean_l'] = hpd_mean(df)
            if 'hpd_max_l' in var_sn:
                cc_sn.loc[j,'hpd_max_l'] = hpd_max(df)

            if 'hpf_l' in var_sn:
                cc_sn.loc[j,'hpf_l'] = hpf(df)
            if 'lpf_l' in var_sn:
                cc_sn.loc[j,'lpf_l'] = lpf(df)

            if 'sti_l' in var_sn:
                cc_sn.loc[j,['de_l','dp_l','dt_l','sp_l','st_l','se_l','sd_l','sti_l']] = seas_var_indices(df)
            
            if 'asi_l' in var_sn:
                cc_sn.loc[j,'asi_l'] = asi(df)
            
            if 'ppd_l' in var_sn:
                cc_sn.loc[j,'ppd_l'] = ppd(df)
        
        cc = pd.concat([cc_cl,cc_sn],axis=1)
        
        return cc


def catch_characteristics(var_cl,var_sn, catch_id, work_dir,data_sources):
    """
    combine climate and snow variables in one dataframe
    returns: catchment characteristics dataframe cc - saved as csv file
    """
    cc = catch_characteristics_climate(var_cl,var_sn, catch_id,work_dir,data_sources)
    cc.to_csv(f'{work_dir}/output/catchment_characteristics/{data_sources}/climate/{catch_id}.csv') #store cc dataframe

    
def run_function_parallel_catch_characteristics(
    var_cl_list=list,
    var_sn_list=list,
    catch_list=list,
    work_dir_list=list,
    data_sources_list=list,
    # threads=None
    threads=100
    ):
    """
    Runs function preprocess_gsim_discharge  in parallel.

    var_cl_list: str,list, list of climate variables
    var_sn_list: str,list, list of climate snow variables
    catch_list:  str, list, list of catchmet ids
    work_dir_list:     str, list, list of work dir
    data_sources: combination of data used for P, Ep and T (gswp-p_gleam-ep_gswp-t for example)
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
        var_cl_list,
        var_sn_list,
        catch_list,
        work_dir_list,
        data_sources_list,
    )
    
    
    
def catch_characteristics_landscape(var_lc,catch_id,work_dir):
    """
    calculate catchment characteristics - landscape variables
    var_lc: define list of landscape variables
    catch_id: catchment id
    returns: dataframe with landscape variables for catchment
    """
    cc_lc = pd.DataFrame(index=[catch_id], columns=var_lc)
    j = catch_id

    if 'tc' in var_lc:
        dft = pd.read_csv(f'{work_dir}/output/treecover/gsim_shapes_treecover_italy.csv',index_col=0) #find treecover tables for catchment
        cc_lc.loc[j,'tc'] = dft.loc[j,'mean_tc'] /100
        cc_lc.loc[j,'ntc'] = dft.loc[j,'mean_ntc']/100
        cc_lc.loc[j,'nonveg'] = dft.loc[j,'mean_nonveg']/100

    if 'area' in var_lc:
        a = pd.read_csv(f'{work_dir}/output/catchment_area.csv',index_col=0)
        cc_lc.loc[j,'area'] = a.loc[j,'area']

    if 'el_mean' in var_lc:
        e = pd.read_csv(f'{work_dir}/output/elevation/stats_hydrosheds/ele_{j}.csv',index_col=0)
        e.index = e.index.map(str)
        cc_lc.loc[j,'el_mean'] = e.loc[j,'mean_ele']
        cc_lc.loc[j,'el_max'] = e.loc[j,'max_ele']
        cc_lc.loc[j,'el_min'] = e.loc[j,'min_ele']
        cc_lc.loc[j,'el_std'] = e.loc[j,'std_ele']
    if 'slp_mean' in var_lc:
        e = pd.read_csv(f'{work_dir}/output/elevation/stats_hydrosheds/slope_{j}.csv',index_col=0)
        e.index = e.index.map(str)
        cc_lc.loc[j,'slp_mean'] = e.loc[j,'mean_slope']
        cc_lc.loc[j,'slp_max'] = e.loc[j,'max_slope']
        cc_lc.loc[j,'slp_min'] = e.loc[j,'min_slope']
        cc_lc.loc[j,'slp_std'] = e.loc[j,'std_slope']

    if 'iwu' in var_lc:
        e = pd.read_csv(f'{work_dir}/output/irrigation/processed2/mean/{j}.csv',index_col=0)
        e.index = e.index.map(str)
        cc_lc.loc[j,'iwu'] = e.loc[j,'iwu_mean_mmday']

    if 'ia' in var_lc:
        e = pd.read_csv(f'{work_dir}/data/irrigated_area/output/combined_ia.csv',index_col=0)
        e.index = e.index.map(str)
        cc_lc.loc[j,'ia'] = e.loc[j,'hi']

    if 'kg' in var_lc:
        e = pd.read_csv(f'{work_dir}/data/koppen_climates/all_catch_table_kg_climates.csv',index_col=0)
        e.index = e.index.map(str)
        cc_lc.loc[j,'kg'] = e.loc[j,'kg']

    if 'lat' in var_lc:
        e = pd.read_csv(f'{work_dir}/output/lat_lon_catchment_outlets.csv', index_col=0)
        e.index = e.index.map(str)
        cc_lc.loc[j,['lat','lon']] = e.loc[j,['lat','lon']]

    if 'bp' in var_lc:
        e = pd.read_csv(f'{work_dir}/output/bedrock_depth/0_01deg/{j}.csv',index_col=0)
        e.index = e.index.map(str)
        cc_lc.loc[j,'bp'] = e.loc[j,'bd_perc']
    if 'dtb' in var_lc:
        e = pd.read_csv(f'{work_dir}/output/bedrock_depth/0_05deg/{j}.csv',index_col=0)
        e.index = e.index.map(str)
        cc_lc.loc[j,'dtb'] = e.loc[j,'bd_mean']

    if 'pclay' in var_lc:
        e = pd.read_csv(f'{work_dir}/output/soil_types/processed/clay.csv',index_col=0)
        e.index = e.index.map(str)
        cc_lc.loc[j,'pclay'] = e.loc[j,'mean']

        e = pd.read_csv(f'{work_dir}/output/soil_types/processed/sand.csv',index_col=0)
        e.index = e.index.map(str)
        cc_lc.loc[j,'psand'] = e.loc[j,'mean']

        e = pd.read_csv(f'{work_dir}/output/soil_types/processed/carb.csv',index_col=0)
        e.index = e.index.map(str)
        cc_lc.loc[j,'pcarb'] = e.loc[j,'mean']

        e = pd.read_csv(f'{work_dir}/output/soil_types/processed/bulk.csv',index_col=0)
        e.index = e.index.map(str)
        cc_lc.loc[j,'bulk'] = e.loc[j,'mean']

        e = pd.read_csv(f'{work_dir}/output/soil_types/processed/text.csv',index_col=0)
        e.index = e.index.map(str)
        cc_lc.loc[j,'stext'] = e.loc[j,'med_text']
        
    if 'fc_mean' in var_lc:
        if os.path.exists(f'{work_dir}/output/lai_fcover/lai_timeseries/{j}_lai_mean_2010_2010.csv'):
            e = pd.read_csv(f'{work_dir}/output/lai_fcover/lai_timeseries/{j}_lai_mean_2010_2010.csv',index_col=0)
            cc_lc.loc[j,'lai_mean'] = e.leaf_area_index.mean()
            if (e.leaf_area_index.mean()>0):
                cc_lc.loc[j,'lai_rsd'] = e.leaf_area_index.std()/e.leaf_area_index.mean()
            else:
                cc_lc.loc[j,'lai_rsd'] = 0

            e = pd.read_csv(f'{work_dir}/output/lai_fcover/fcover_timeseries/{j}_fcover_mean_2010_2010.csv',index_col=0)
            cc_lc.loc[j,'fc_mean'] = e.vegetation_area_fraction.mean()
            if (e.vegetation_area_fraction.mean()>0):
                cc_lc.loc[j,'fc_rsd'] = e.vegetation_area_fraction.std()/e.vegetation_area_fraction.mean()
            else:
                cc_lc.loc[j,'fc_rsd'] = 0  
        else:
            cc_lc.loc[j,'lai_mean'] = np.nan
            cc_lc.loc[j,'lai_rsd'] = np.nan
            cc_lc.loc[j,'fc_mean'] = np.nan
            cc_lc.loc[j,'fc_rsd'] = np.nan
        
        if os.path.exists(f'{work_dir}/output/snow_cover/timeseries/{j}_snowcover_mean_2010_2010.csv'):
            e = pd.read_csv(f'{work_dir}/output/snow_cover/timeseries/{j}_snowcover_mean_2010_2010.csv',index_col=0)
            cc_lc.loc[j,'sc_mean'] = e['Monthly snow cover extent, 5km'].mean()
      
            if (e['Monthly snow cover extent, 5km'].mean()>0):
                cc_lc.loc[j,'sc_rsd'] = e['Monthly snow cover extent, 5km'].std()/e['Monthly snow cover extent, 5km'].mean()
            else:
                cc_lc.loc[j,'sc_rsd'] = 0
        else:
            cc_lc.loc[j,'sc_mean'] = np.nan
            cc_lc.loc[j,'sc_rsd'] = np.nan   
    return cc_lc


def catch_characteristics_lc(var_lc,catch_id, work_dir,data_sources):
    """
    combine climate and snow variables in one dataframe
    returns: catchment characteristics dataframe cc - saved as csv file
    """
    cc = catch_characteristics_landscape(var_lc, catch_id,work_dir)
    cc.to_csv(f'{work_dir}/output/catchment_characteristics/{data_sources}/landscape/{catch_id}.csv') #store cc dataframe

    
def run_function_parallel_catch_characteristics_lc(
    var_lc_list=list,
    catch_list=list,
    work_dir_list=list,
    data_sources_list=list,
    # threads=None
    threads=100
    ):
    """
    Runs function preprocess_gsim_discharge  in parallel.

    var_cl_list: str,list, list of climate variables
    var_sn_list: str,list, list of climate snow variables
    catch_list:  str, list, list of catchmet ids
    work_dir_list:     str, list, list of work dir
    data_sources: combination of data used for P, Ep and T (gswp-p_gleam-ep_gswp-t for example)
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
        catch_characteristics_lc,
        var_lc_list,
        catch_list,
        work_dir_list,
        data_sources_list,
    )
    

# def catch_characteristics(var_lc,var_cl,var_sn, catch_id, work_dir,data_sources):
#     """
#     combine climate and landscape variables in one dataframe
#     returns: catchment characteristics dataframe cc - saved as csv file
#     """
#     cc_lc = catch_characteristics_landscape(var_lc,catch_id,work_dir)
#     cc_cl = catch_characteristics_climate(var_cl, catch_id,work_dir,data_sources)
#     cc_sn = catch_characteristics_climate_snow(var_sn, catch_id,work_dir,data_sources)
#     cc = pd.concat([cc_cl,cc_sn,cc_lc],axis=1)
    
#     # merge with existing dataframe
#     if os.path.exists(f'{work_dir}/output/catchment_characteristics/{data_sources}/{catch_id}.csv'):
#         cce = pd.read_csv(f'{work_dir}/output/catchment_characteristics/{data_sources}/{catch_id}.csv',index_col=0)
#         cc2 = pd.concat([cc,cce],axis=1)
#     else: 
#         cc2 = cc
#     cc2.to_csv(f'{work_dir}/output/catchment_characteristics/{data_sources}/{catch_id}.csv') #store cc dataframe
    


    

    
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
