"""

to do fransje -> add clear explanations per function
-> add more variables (ST-index, other EE products,....)


"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from scipy.optimize import least_squares
import geopandas as gpd


def p_mean(df):
    m = df['P'].mean()
    return m

def ep_mean(df):
    m = df['Ep'].mean()
    return m

def t_mean(df):
    m = df['T'].mean()
    return m

def ai(df):
    ai = df['P'].mean()/df['Ep'].mean()
    return ai

def si_p(df):
    p = df['P']
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
    sip = (1/pa)*np.sum(a)
    return sip
    
def si_ep(df):
    ep = df['Ep']
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
    siep = (1/epa)*np.sum(a)
    
    return siep

def phi(df):
    p = df['P']
    ep = df['Ep']
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
    m = df_q['Q'].mean()
    return m


def catch_characteristics(var, catch_id_list, fol_in, fol_out):
    cc = pd.DataFrame(index=catch_id_list, columns=var)
    for j in catch_id_list:
        l = glob.glob(f'{fol_in}/forcing_timeseries/processed/daily/{j}*.csv')
        df = pd.read_csv(l[0], index_col=0)
        df.index = pd.to_datetime(df.index)
        l_q = glob.glob(f'{fol_in}/discharge/timeseries/{j}*.csv')
        df_q = pd.read_csv(l_q[0], index_col=0)
        df_q.index = pd.to_datetime(df_q.index)

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
        
        # treecover
        if 'tc' in var:
            l = glob.glob(f'{fol_in}/earth_engine_timeseries/treecover/{j}*.csv')
            dft = pd.read_csv(l[0], index_col=0)
            cc.loc[j,'tc'] = dft.loc[j,'mean_tc']
            cc.loc[j,'ntc'] = dft.loc[j,'mean_ntc']
            cc.loc[j,'nonveg'] = dft.loc[j,'mean_nonveg']
            
    cc.to_csv(f'{fol_out}/catchment_characteristics.csv')
    return cc

def geo_catchments(shape_dir,out_dir):
    shapefiles = glob.glob(f"{shape_dir}/*shp")
    li=[]
    for filename in shapefiles:
        df = gpd.read_file(filename, index_col=None, header=0)
        li.append(df)
    f = pd.concat(li, axis=0)
    f = f.rename(columns={'FILENAME':'catch_id'})
    f.index = f['catch_id']
    f = f.drop(columns={'catch_id','Id'})
    f.to_file(f'{out_dir}/geo_catchments.shp')