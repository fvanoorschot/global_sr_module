import numpy as np
import pandas as pd
import glob

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

from f_sr_calculation import *

work_dir=Path("/scratch/fransjevanoors/global_sr")
#work_dir=Path('/tudelft.net/staff-umbrella/LSM root zone/global_sr')
data_dir=Path(f'{work_dir}/data')
out_dir = Path(f"{work_dir}/output")

print(work_dir)
print(out_dir)
print(data_dir)

pep_dir = f'{work_dir}/output/forcing_timeseries/processed/daily'
q_dir = f'{work_dir}/output/q_timeseries_selected'
out_dir = f'{work_dir}/output/sr_calculation/sd_catchments'
snow_ids = np.genfromtxt(f'{work_dir}/output/snow/catch_id_list_snow_t_and_p.txt',dtype='str')
snow_dir = f'{work_dir}/output/snow/timeseries_gswp'
irri_ids = np.loadtxt(f'{work_dir}/output/irrigation/catchment_irri_area_5percent.txt',dtype=str)

# check which catchments are missing
catch_list = np.genfromtxt(f'{work_dir}/output/irrigation/catchment_irri_area_5percent.txt',dtype='str')[:]
el_id_list=[]
for filepath in glob.iglob(f'{work_dir}/output/sr_calculation/sd_catchments/irri/sd/*1.0ia.csv'):
    f = os.path.split(filepath)[1] # remove full path
    f = f[:-10] # remove .year extension
    el_id_list.append(f)
dif = list(set(catch_list) - set(el_id_list))
irri_ids=dif

catch_id_list = catch_list[:]
pep_dir_list = [pep_dir] * len(catch_id_list)
q_dir_list = [q_dir] * len(catch_id_list)
out_dir_list = [out_dir] * len(catch_id_list)
snow_id_list = [snow_ids] * len(catch_id_list)
snow_dir_list = [snow_dir] * len(catch_id_list)
irri_id_list = [irri_ids] * len(catch_id_list)
work_dir_list = [work_dir] * len(catch_id_list)

run_sd_calculation_parallel(catch_id_list,pep_dir_list,q_dir_list,out_dir_list,snow_id_list,irri_id_list,snow_dir_list,work_dir_list)







# # define directories
# pep_dir = f'{work_dir}/output/forcing_timeseries/processed/daily'
# q_dir = f'{work_dir}/output/q_timeseries_selected'
# out_dir = f'{work_dir}/output/sr_calculation/sd_catchments'
# snow_ids = np.genfromtxt(f'{work_dir}/output/snow/catch_id_list_snow_t_and_p.txt',dtype='str')
# snow_dir = f'{work_dir}/output/snow/timeseries'

# c = np.loadtxt(f'{work_dir}/output/gsim_aus_catch_id_list_lo_sel_area_wb.txt',dtype=str) 
# catch_id_list = c[:]
# pep_dir_list = [pep_dir] * len(catch_id_list)
# q_dir_list = [q_dir] * len(catch_id_list)
# out_dir_list = [out_dir] * len(catch_id_list)
# snow_id_list = [snow_ids] * len(catch_id_list)
# snow_dir_list = [snow_dir] * len(catch_id_list)

# run_sd_calculation_parallel(catch_id_list,pep_dir_list,q_dir_list,out_dir_list,snow_id_list,snow_dir_list)

# # # define directories
# pep_dir = f'{work_dir}/output/forcing_timeseries/processed/daily'
# q_dir = f'{work_dir}/output/q_timeseries_selected'
# out_dir = f'{work_dir}/output/sr_calculation/sd_catchments'

# c = np.loadtxt(f'{work_dir}/output/sr_calculation/sd_list.txt',dtype=str) 
# catch_id_list = c[:]
# pep_dir_list = [pep_dir] * len(catch_id_list)
# q_dir_list = [q_dir] * len(catch_id_list)
# out_dir_list = [out_dir] * len(catch_id_list)

# run_sd_calculation_parallel(catch_id_list,pep_dir_list,q_dir_list,out_dir_list)
