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
from scipy.optimize import minimize



def snow_calculation(catch_id,work_dir):
    # load p-ep-tas timeseries
    file = glob.glob(f'{work_dir}/output/forcing_timeseries/processed/daily/{catch_id}*.csv')[0]
    ts = pd.read_csv(f'{file}',index_col=0)
    # ts = ts.loc[start_date:end_date] #this should be based on sr - or do this for the full timeseries???
    p = ts.p.values
    tas = ts.tas.values

    # load mean elevation
    file = glob.glob(f'{work_dir}/output/elevation/stats_hydrosheds/ele_{catch_id}.csv')[0]
    elm = pd.read_csv(f'{file}',index_col=0)

    # load elevation zones
    file = glob.glob(f'{work_dir}/output/elevation/el_zones/{catch_id}*.csv')[0]
    elz = pd.read_csv(f'{file}',index_col=0)
    
    # snow parameters
    TT = 0
    MF = 2
    
    # make empty matrices with timseries rows and elevation zones columns
    Pm_el = np.zeros((len(p),len(elz))) # melt water
    Pl_el = np.zeros((len(p),len(elz))) # liquid precipitation
    Ps_el = np.zeros((len(p),len(elz))) # solid precipitation

    for j in range(len(elz)):
        Ss = np.zeros(len(p)) # snow storage
        Pm = np.zeros(len(p)) # melt water
        Pl = np.zeros(len(p)) # liquid precipitation
        Ps = np.zeros(len(p)) # solid precipitation
        T = np.zeros(len(p)) # temperature

        # temperature difference for elevation zone
        el_dif = elm.mean_ele - elz.loc[j]['mean_el']
        dt = (el_dif/1000.)*6.4

        # loop over timesteps and compute ps, pm and pl
        for i in range(0,len(p)):               
            Tmean = tas[i]
            T[i] = Tmean+dt

            if T[i]>TT:
                Pl[i] = p[i]
                Ps[i] = 0

                Pm[i] = min(Ss[i],MF*(T[i]-TT))
                Ss[i] = max(0, Ss[i]-Pm[i])
            else:
                Ps[i] = p[i]
                Pl[i] = 0
                Ss[i] = Ss[i]+Ps[i]
            if i<len(p)-1:
                Ss[i+1] = Ss[i]

        # scale to coverage of elevation zone and add to matrix
        Ps = Ps * elz.loc[j]['frac']
        Pm = Pm * elz.loc[j]['frac']
        Pl = Pl * elz.loc[j]['frac']
        Pm_el[:,j] = Pm
        Pl_el[:,j] = Pl
        Ps_el[:,j] = Ps

    # sum fractions of pm, pl and ps
    pm = Pm_el.sum(axis=1)
    pl = Pl_el.sum(axis=1)
    ps = Ps_el.sum(axis=1)

    # add to forcing dataframe
    ts['pm'] = pm
    ts['pl'] = pl
    ts['ps'] = ps
    ts.to_csv(f'{work_dir}/output/snow/timeseries/{catch_id}.csv')
    
    
    
def run_function_parallel_snow(
    catch_list=list,
    work_dir_list=list,
    # threads=None
    threads=100
    ):
    """
    Runs function snow_calculation  in parallel.

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
        snow_calculation,
        catch_list,
        work_dir_list,
    )
    