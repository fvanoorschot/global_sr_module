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

from f_grid_to_catchments import *


work_dir=Path('/scratch/fransjevanoors/global_sr')
#work_dir=Path('/tudelft.net/staff-umbrella/LSM root zone/global_sr')
data_dir=Path(f'{work_dir}/data')
out_dir = Path(f"{work_dir}/output")

print(work_dir)
print(out_dir)
print(data_dir)

# define input directory
fol_in=f'{work_dir}/output/forcing_timeseries/raw'
# define output directory
fol_out=f'{work_dir}/output/forcing_timeseries/processed'

# get catch_id_list
catch_id_list = np.genfromtxt(f'{work_dir}/output/gsim_aus_catch_id_list_up_sel.txt',dtype='str')

# define variables
var = ['ep','p','tas']

# make lists for parallel computation
catch_list = catch_id_list
fol_in_list = [fol_in] * len(catch_list)
fol_out_list = [fol_out] * len(catch_list)
var_list = [var] * len(catch_list)
run_processing_function_parallel(catch_list,fol_in_list,fol_out_list,var_list)


# # check catchments that are not processed yet
# gsim_id_list_lo_done = []
# for filepath in glob.iglob(f'{out_dir}/gsim/timeseries/*'):
#     f = os.path.split(filepath)[1] # remove full path
#     f = f[:-4] # remove .csv extension
#     gsim_id_list_lo_done.append(f)

# dif_list=list(set(gsim_id_list_lo)-set(gsim_id_list_lo_done))
# print(len(dif_list))
# gsim_id_list_lo=dif_list


