import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import copy
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import os
import plotly.express as px
from matplotlib import cm
import matplotlib as mpl
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import partial_dependence
import matplotlib as mpl
from matplotlib import cm
import numpy.ma as ma
import cartopy.crs as ccrs
import cartopy
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import random
import sklearn
import xarray as xr
import scipy.stats
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from scipy.stats import gaussian_kde
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from sklearn import tree
from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score
import geopandas as gpd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from pathos.threading import ThreadPool as Pool
from sklearn import datasets, linear_model
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib
from scipy.stats import spearmanr

work_dir=Path("/mnt/u/LSM root zone/global_sr/data_to_share")

def compute_corr_tables():
    pp = xr.open_dataset(f'{work_dir}/sr_predicted_map.nc')
    def load_p():
        p = xr.open_dataset(f'{work_dir}/sr_predicted_map.nc')
        p = p.sr_p
        return p
    def load_f():
        f = xr.open_dataset(f'{work_dir}/reference_root_products/fan2017.nc') 
        f = f.root_depth
        pp = xr.open_dataset(f'{work_dir}/sr_predicted_map.nc')
        f = f.where(pp.sr_p[:,:]>=0)
        return f
    def load_ya():
        ya = xr.open_dataset(f'{work_dir}/reference_root_products/yang2016.nc')
        ya = ya.Band1
        pp = xr.open_dataset(f'{work_dir}/sr_predicted_map.nc')
        ya = ya.where(pp.sr_p[:,:]>=0)
        return ya
    def load_l():
        l = xr.open_dataset(f'{work_dir}/reference_root_products/wangerlandsson2016.nc')
        l = l.sr_cru_20yrs[0]
        pp = xr.open_dataset(f'{work_dir}/sr_predicted_map.nc')
        l = l.where(pp.sr_p[:,:]>=0)
        return l
    def load_s():
        s = xr.open_dataset(f'{work_dir}/reference_root_products/stocker2023.nc')
        s = s.cwdx80
        pp = xr.open_dataset(f'{work_dir}/sr_predicted_map.nc')
        s = s.where(pp.sr_p[:,:]>=0)
        return s
    def load_sc():
        sc = xr.open_dataset(f'{work_dir}/reference_root_products/schenk2009.nc') 
        sc = sc['95ecosys_rootdepth_1d'][0]
        pp = xr.open_dataset(f'{work_dir}/sr_predicted_map.nc')
        sc = sc.where(pp.sr_p[:,:]>=0)
        return sc
    def load_ko():
        ko = xr.open_dataset(f'{work_dir}/reference_root_products/kleidon2004.nc')
        ko = ko['rootOptMap150_m']
        return ko

    def correlations(x,y):
        x1 = x.stack(flat=['lat','lon']).values
        y1 = y.stack(flat=['lat','lon']).values
        nan_indices = np.logical_or(np.isnan(x1), np.isnan(y1))
        x1[nan_indices] = np.nan
        y1[nan_indices] = np.nan
        x2 = x1[~np.isnan(x1)]
        y2 = y1[~np.isnan(y1)]
        # print(len(x2),len(y2))
        rp = np.round(np.corrcoef(x2,y2)[0,1],2)
        rs = np.round(spearmanr(x2,y2)[0],2)
        return(len(x2),len(y2),rp,rs)

    rp_table = np.zeros((8,8))
    rs_table = np.zeros((8,8))
    lens = []

    k=0
    xs,ys = load_p(),load_p()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,0] = rp
    rs_table[k,0] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_p(),load_s()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,1] = rp
    rs_table[k,1] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_p(),load_l()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,2] = rp
    rs_table[k,2] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_p(),load_f()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,3] = rp
    rs_table[k,3] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_p(),load_sc()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,4] = rp
    rs_table[k,4] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_p(),load_ya()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,5] = rp
    rs_table[k,5] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_p(),load_ko()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,6] = rp
    rs_table[k,6] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    k=1
    xs,ys = load_s(),load_p()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,0] = rp
    rs_table[k,0] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_s(),load_s()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,1] = rp
    rs_table[k,1] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_s(),load_l()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,2] = rp
    rs_table[k,2] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_s(),load_f()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,3] = rp
    rs_table[k,3] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_s(),load_sc()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,4] = rp
    rs_table[k,4] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_s(),load_ya()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,5] = rp
    rs_table[k,5] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_s(),load_ko()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,6] = rp
    rs_table[k,6] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    k=2
    xs,ys = load_l(),load_p()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,0] = rp
    rs_table[k,0] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_l(),load_s()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,1] = rp
    rs_table[k,1] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_l(),load_l()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,2] = rp
    rs_table[k,2] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_l(),load_f()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,3] = rp
    rs_table[k,3] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_l(),load_sc()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,4] = rp
    rs_table[k,4] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_l(),load_ya()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,5] = rp
    rs_table[k,5] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_l(),load_ko()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,6] = rp
    rs_table[k,6] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    k=3
    xs,ys = load_f(),load_p()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,0] = rp
    rs_table[k,0] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_f(),load_s()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,1] = rp
    rs_table[k,1] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_f(),load_l()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,2] = rp
    rs_table[k,2] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_f(),load_f()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,3] = rp
    rs_table[k,3] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_f(),load_sc()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,4] = rp
    rs_table[k,4] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_f(),load_ya()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,5] = rp
    rs_table[k,5] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_f(),load_ko()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,6] = rp
    rs_table[k,6] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    k=4
    xs,ys = load_sc(),load_p()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,0] = rp
    rs_table[k,0] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_sc(),load_s()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,1] = rp
    rs_table[k,1] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_sc(),load_l()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,2] = rp
    rs_table[k,2] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_sc(),load_f()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,3] = rp
    rs_table[k,3] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_sc(),load_sc()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,4] = rp
    rs_table[k,4] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_sc(),load_ya()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,5] = rp
    rs_table[k,5] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_sc(),load_ko()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,6] = rp
    rs_table[k,6] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    k=5
    xs,ys = load_ya(),load_p()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,0] = rp
    rs_table[k,0] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_ya(),load_s()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,1] = rp
    rs_table[k,1] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_ya(),load_l()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,2] = rp
    rs_table[k,2] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_ya(),load_f()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,3] = rp
    rs_table[k,3] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_ya(),load_sc()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,4] = rp
    rs_table[k,4] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_ya(),load_ya()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,5] = rp
    rs_table[k,5] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_ya(),load_ko()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,6] = rp
    rs_table[k,6] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    k=6
    xs,ys = load_ko(),load_p()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,0] = rp
    rs_table[k,0] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_ko(),load_s()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,1] = rp
    rs_table[k,1] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_ko(),load_l()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,2] = rp
    rs_table[k,2] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_ko(),load_f()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,3] = rp
    rs_table[k,3] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_ko(),load_sc()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,4] = rp
    rs_table[k,4] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_ko(),load_ya()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,5] = rp
    rs_table[k,5] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    xs,ys = load_ko(),load_ko()
    rp,rs = correlations(xs,ys)[2],correlations(xs,ys)[3]
    rp_table[k,6] = rp
    rs_table[k,6] = rs
    lens.append([correlations(xs,ys)[0],correlations(xs,ys)[1]])

    def correlations_sjj(x,y):
        x1 = x
        y1 = y
        nan_indices = np.logical_or(np.isnan(x1), np.isnan(y1))
        x1[nan_indices] = np.nan
        y1[nan_indices] = np.nan
        x2 = x1[~np.isnan(x1)]
        y2 = y1[~np.isnan(y1)]
        rp = np.round(np.corrcoef(x2,y2)[0,1],2)
        rs = np.round(spearmanr(x2,y2)[0],2)
        return(len(x2),len(y2),rp,rs)

    k=7
    sjj = pd.read_csv(f'{work_dir}/reference_root_products/schenkjackson2003.csv',index_col=0)
    xs = sjj.D95_extrapolated
    ys = sjj.D95_extrapolated
    rp,rs = correlations_sjj(xs,ys)[2],correlations_sjj(xs,ys)[3]
    rp_table[k,7] = rp
    rs_table[k,7] = rs
    lens.append([correlations_sjj(xs,ys)[0],correlations_sjj(xs,ys)[1]])

    xs = sjj.D95_extrapolated
    ys = sjj.sr_p
    rp,rs = correlations_sjj(xs,ys)[2],correlations_sjj(xs,ys)[3]
    rp_table[k,0] = rp
    rs_table[k,0] = rs
    rp_table[0,k] = rp
    rs_table[0,k] = rs
    lens.append([correlations_sjj(xs,ys)[0],correlations_sjj(xs,ys)[1]])

    xs = sjj.D95_extrapolated
    ys = sjj.stocker
    rp,rs = correlations_sjj(xs,ys)[2],correlations_sjj(xs,ys)[3]
    rp_table[k,1] = rp
    rs_table[k,1] = rs
    rp_table[1,k] = rp
    rs_table[1,k] = rs
    lens.append([correlations_sjj(xs,ys)[0],correlations_sjj(xs,ys)[1]])

    xs = sjj.D95_extrapolated
    ys = sjj.lan
    rp,rs = correlations_sjj(xs,ys)[2],correlations_sjj(xs,ys)[3]
    rp_table[k,2] = rp
    rs_table[k,2] = rs
    rp_table[2,k] = rp
    rs_table[2,k] = rs
    lens.append([correlations_sjj(xs,ys)[0],correlations_sjj(xs,ys)[1]])

    xs = sjj.D95_extrapolated
    ys = sjj.fan
    rp,rs = correlations_sjj(xs,ys)[2],correlations_sjj(xs,ys)[3]
    rp_table[k,3] = rp
    rs_table[k,3] = rs
    rp_table[3,k] = rp
    rs_table[3,k] = rs
    lens.append([correlations_sjj(xs,ys)[0],correlations_sjj(xs,ys)[1]])

    xs = sjj.D95_extrapolated
    ys = sjj.schenk2009
    rp,rs = correlations_sjj(xs,ys)[2],correlations_sjj(xs,ys)[3]
    rp_table[k,4] = rp
    rs_table[k,4] = rs
    rp_table[4,k] = rp
    rs_table[4,k] = rs
    lens.append([correlations_sjj(xs,ys)[0],correlations_sjj(xs,ys)[1]])

    xs = sjj.D95_extrapolated
    ys = sjj.yang
    rp,rs = correlations_sjj(xs,ys)[2],correlations_sjj(xs,ys)[3]
    rp_table[k,5] = rp
    rs_table[k,5] = rs
    rp_table[5,k] = rp
    rs_table[5,k] = rs
    lens.append([correlations_sjj(xs,ys)[0],correlations_sjj(xs,ys)[1]])

    xs = sjj.D95_extrapolated
    ys = sjj.kleidon_opt
    rp,rs = correlations_sjj(xs,ys)[2],correlations_sjj(xs,ys)[3]
    rp_table[k,6] = rp
    rs_table[k,6] = rs
    rp_table[6,k] = rp
    rs_table[6,k] = rs
    lens.append([correlations_sjj(xs,ys)[0],correlations_sjj(xs,ys)[1]])
    
    return rp_table,rs_table



def compute_corr_tables_weighted():
    pp = xr.open_dataset(f'{work_dir}/sr_predicted_map.nc')
    
    # get grid cell areas
    R = 6371.0
    # Latitude and longitude intervals in degrees for a 0.5-degree grid
    delta_phi = 0.5
    delta_lambda = 0.5

    # Convert degrees to radians
    delta_phi_rad = np.radians(delta_phi)
    delta_lambda_rad = np.radians(delta_lambda)

    # Calculate the area for each latitude
    latitudes = pp.lat.values
    lon_len = len(pp.lon.values)
    lat_len = len(pp.lat.values)
    grid_cell_areas = R**2 * delta_lambda_rad * (np.sin(np.radians(latitudes + delta_phi)) - np.sin(np.radians(latitudes)))

    ag = np.zeros((lat_len,lon_len))
    for i in range(lon_len):
        ag[:,i] = grid_cell_areas
    
    def load_p():
        p = xr.open_dataset(f'{work_dir}/sr_predicted_map.nc')
        p = p.sr_p
        return p
    def load_f():
        f = xr.open_dataset(f'{work_dir}/reference_root_products/fan2017.nc') 
        f = f.root_depth
        pp = xr.open_dataset(f'{work_dir}/sr_predicted_map.nc')
        f = f.where(pp.sr_p[:,:]>=0)
        return f
    def load_ya():
        ya = xr.open_dataset(f'{work_dir}/reference_root_products/yang2016.nc')
        ya = ya.Band1
        pp = xr.open_dataset(f'{work_dir}/sr_predicted_map.nc')
        ya = ya.where(pp.sr_p[:,:]>=0)
        return ya
    def load_l():
        l = xr.open_dataset(f'{work_dir}/reference_root_products/wangerlandsson2016.nc')
        l = l.sr_cru_20yrs[0]
        pp = xr.open_dataset(f'{work_dir}/sr_predicted_map.nc')
        l = l.where(pp.sr_p[:,:]>=0)
        return l
    def load_s():
        s = xr.open_dataset(f'{work_dir}/reference_root_products/stocker2023.nc')
        s = s.cwdx80
        pp = xr.open_dataset(f'{work_dir}/sr_predicted_map.nc')
        s = s.where(pp.sr_p[:,:]>=0)
        return s
    def load_sc():
        sc = xr.open_dataset(f'{work_dir}/reference_root_products/schenk2009.nc') 
        sc = sc['95ecosys_rootdepth_1d'][0]
        pp = xr.open_dataset(f'{work_dir}/sr_predicted_map.nc')
        sc = sc.where(pp.sr_p[:,:]>=0)
        return sc
    def load_ko():
        ko = xr.open_dataset(f'{work_dir}/reference_root_products/kleidon2004.nc')
        ko = ko['rootOptMap150_m']
        return ko

    def weighted_pearson_correlation(x, y, weights):
        # Calculate weighted covariance matrix
        cov_matrix = np.cov(x, y, aweights=weights)
        # Extract the diagonal elements (variances)
        var_x = cov_matrix[0, 0]
        var_y = cov_matrix[1, 1]
        # Calculate the weighted Pearson correlation coefficient
        weighted_corr = cov_matrix[0, 1] / np.sqrt(var_x * var_y)
        return weighted_corr

    def weighted_spearmanr(x, y, weights):
        ranks_x = scipy.stats.rankdata(x)
        ranks_y = scipy.stats.rankdata(y)
        # Calculate weighted covariance matrix
        cov_matrix = np.cov(ranks_x, ranks_y, aweights=weights)
        # Extract the diagonal elements (variances)
        var_x = cov_matrix[0, 0]
        var_y = cov_matrix[1, 1]
        # Calculate the weighted Pearson correlation coefficient
        weighted_corr = cov_matrix[0, 1] / np.sqrt(var_x * var_y)
        # # Calculate the ranks of x and y
        # ranks_x = np.argsort(x)
        # ranks_y = np.argsort(y)
        # # Calculate the weighted ranks
        # weighted_ranks_x = np.average(ranks_x, weights=weights)
        # weighted_ranks_y = np.average(ranks_y, weights=weights)
        # # Calculate the Spearman correlation coefficient
        # correlation, p_value = spearmanr(ranks_x, ranks_y)
        return weighted_corr
    
    def weighted_correlations(x,y,ag):
        x1 = x.stack(flat=['lat','lon']).values
        y1 = y.stack(flat=['lat','lon']).values
        a = ag.flatten()
        nan_indices = np.logical_or(np.isnan(x1), np.isnan(y1))
        x1[nan_indices] = np.nan
        y1[nan_indices] = np.nan
        a[nan_indices] = np.nan
        x2 = x1[~np.isnan(x1)]
        y2 = y1[~np.isnan(y1)]
        a = a[~np.isnan(a)]
        wp = weighted_pearson_correlation(x2, y2, a)
        ws = weighted_spearmanr(x2, y2, a)
        rp = np.round(wp,2)
        rs = np.round(ws,2)
        return(len(x2),len(y2),rp,rs)

    rp_table = np.zeros((8,8))
    rs_table = np.zeros((8,8))
    lens = []

    k=0
    xs,ys = load_p(),load_p()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,0] = rp
    rs_table[k,0] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_p(),load_s()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,1] = rp
    rs_table[k,1] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_p(),load_l()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,2] = rp
    rs_table[k,2] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_p(),load_f()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,3] = rp
    rs_table[k,3] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_p(),load_sc()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,4] = rp
    rs_table[k,4] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_p(),load_ya()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,5] = rp
    rs_table[k,5] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_p(),load_ko()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,6] = rp
    rs_table[k,6] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    k=1
    xs,ys = load_s(),load_p()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,0] = rp
    rs_table[k,0] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_s(),load_s()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,1] = rp
    rs_table[k,1] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_s(),load_l()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,2] = rp
    rs_table[k,2] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_s(),load_f()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,3] = rp
    rs_table[k,3] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_s(),load_sc()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,4] = rp
    rs_table[k,4] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_s(),load_ya()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,5] = rp
    rs_table[k,5] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_s(),load_ko()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,6] = rp
    rs_table[k,6] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    k=2
    xs,ys = load_l(),load_p()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,0] = rp
    rs_table[k,0] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_l(),load_s()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,1] = rp
    rs_table[k,1] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_l(),load_l()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,2] = rp
    rs_table[k,2] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_l(),load_f()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,3] = rp
    rs_table[k,3] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_l(),load_sc()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,4] = rp
    rs_table[k,4] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_l(),load_ya()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,5] = rp
    rs_table[k,5] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_l(),load_ko()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,6] = rp
    rs_table[k,6] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    k=3
    xs,ys = load_f(),load_p()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,0] = rp
    rs_table[k,0] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_f(),load_s()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,1] = rp
    rs_table[k,1] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_f(),load_l()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,2] = rp
    rs_table[k,2] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_f(),load_f()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,3] = rp
    rs_table[k,3] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_f(),load_sc()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,4] = rp
    rs_table[k,4] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_f(),load_ya()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,5] = rp
    rs_table[k,5] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_f(),load_ko()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,6] = rp
    rs_table[k,6] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    k=4
    xs,ys = load_sc(),load_p()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,0] = rp
    rs_table[k,0] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_sc(),load_s()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,1] = rp
    rs_table[k,1] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_sc(),load_l()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,2] = rp
    rs_table[k,2] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_sc(),load_f()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,3] = rp
    rs_table[k,3] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_sc(),load_sc()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,4] = rp
    rs_table[k,4] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_sc(),load_ya()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,5] = rp
    rs_table[k,5] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_sc(),load_ko()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,6] = rp
    rs_table[k,6] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    k=5
    xs,ys = load_ya(),load_p()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,0] = rp
    rs_table[k,0] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_ya(),load_s()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,1] = rp
    rs_table[k,1] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_ya(),load_l()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,2] = rp
    rs_table[k,2] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_ya(),load_f()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,3] = rp
    rs_table[k,3] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_ya(),load_sc()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,4] = rp
    rs_table[k,4] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_ya(),load_ya()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,5] = rp
    rs_table[k,5] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_ya(),load_ko()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,6] = rp
    rs_table[k,6] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    k=6
    xs,ys = load_ko(),load_p()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,0] = rp
    rs_table[k,0] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_ko(),load_s()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,1] = rp
    rs_table[k,1] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_ko(),load_l()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,2] = rp
    rs_table[k,2] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_ko(),load_f()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,3] = rp
    rs_table[k,3] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_ko(),load_sc()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,4] = rp
    rs_table[k,4] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_ko(),load_ya()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,5] = rp
    rs_table[k,5] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    xs,ys = load_ko(),load_ko()
    rp,rs = weighted_correlations(xs,ys,ag)[2],weighted_correlations(xs,ys,ag)[3]
    rp_table[k,6] = rp
    rs_table[k,6] = rs
    lens.append([weighted_correlations(xs,ys,ag)[0],weighted_correlations(xs,ys,ag)[1]])

    def correlations_sjj(x,y):
        x1 = x
        y1 = y
        nan_indices = np.logical_or(np.isnan(x1), np.isnan(y1))
        x1[nan_indices] = np.nan
        y1[nan_indices] = np.nan
        x2 = x1[~np.isnan(x1)]
        y2 = y1[~np.isnan(y1)]
        rp = np.round(np.corrcoef(x2,y2)[0,1],2)
        rs = np.round(spearmanr(x2,y2)[0],2)
        return(len(x2),len(y2),rp,rs)

    k=7
    sjj = pd.read_csv(f'{work_dir}/reference_root_products/schenkjackson2003.csv',index_col=0)
    xs = sjj.D95_extrapolated
    ys = sjj.D95_extrapolated
    rp,rs = correlations_sjj(xs,ys)[2],correlations_sjj(xs,ys)[3]
    rp_table[k,7] = rp
    rs_table[k,7] = rs
    lens.append([correlations_sjj(xs,ys)[0],correlations_sjj(xs,ys)[1]])

    xs = sjj.D95_extrapolated
    ys = sjj.sr_p
    rp,rs = correlations_sjj(xs,ys)[2],correlations_sjj(xs,ys)[3]
    rp_table[k,0] = rp
    rs_table[k,0] = rs
    rp_table[0,k] = rp
    rs_table[0,k] = rs
    lens.append([correlations_sjj(xs,ys)[0],correlations_sjj(xs,ys)[1]])

    xs = sjj.D95_extrapolated
    ys = sjj.stocker
    rp,rs = correlations_sjj(xs,ys)[2],correlations_sjj(xs,ys)[3]
    rp_table[k,1] = rp
    rs_table[k,1] = rs
    rp_table[1,k] = rp
    rs_table[1,k] = rs
    lens.append([correlations_sjj(xs,ys)[0],correlations_sjj(xs,ys)[1]])

    xs = sjj.D95_extrapolated
    ys = sjj.lan
    rp,rs = correlations_sjj(xs,ys)[2],correlations_sjj(xs,ys)[3]
    rp_table[k,2] = rp
    rs_table[k,2] = rs
    rp_table[2,k] = rp
    rs_table[2,k] = rs
    lens.append([correlations_sjj(xs,ys)[0],correlations_sjj(xs,ys)[1]])

    xs = sjj.D95_extrapolated
    ys = sjj.fan
    rp,rs = correlations_sjj(xs,ys)[2],correlations_sjj(xs,ys)[3]
    rp_table[k,3] = rp
    rs_table[k,3] = rs
    rp_table[3,k] = rp
    rs_table[3,k] = rs
    lens.append([correlations_sjj(xs,ys)[0],correlations_sjj(xs,ys)[1]])

    xs = sjj.D95_extrapolated
    ys = sjj.schenk2009
    rp,rs = correlations_sjj(xs,ys)[2],correlations_sjj(xs,ys)[3]
    rp_table[k,4] = rp
    rs_table[k,4] = rs
    rp_table[4,k] = rp
    rs_table[4,k] = rs
    lens.append([correlations_sjj(xs,ys)[0],correlations_sjj(xs,ys)[1]])

    xs = sjj.D95_extrapolated
    ys = sjj.yang
    rp,rs = correlations_sjj(xs,ys)[2],correlations_sjj(xs,ys)[3]
    rp_table[k,5] = rp
    rs_table[k,5] = rs
    rp_table[5,k] = rp
    rs_table[5,k] = rs
    lens.append([correlations_sjj(xs,ys)[0],correlations_sjj(xs,ys)[1]])

    xs = sjj.D95_extrapolated
    ys = sjj.kleidon_opt
    rp,rs = correlations_sjj(xs,ys)[2],correlations_sjj(xs,ys)[3]
    rp_table[k,6] = rp
    rs_table[k,6] = rs
    rp_table[6,k] = rp
    rs_table[6,k] = rs
    lens.append([correlations_sjj(xs,ys)[0],correlations_sjj(xs,ys)[1]])
    
    return rp_table,rs_table
