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

from f_catch_characteristics import *

work_dir=Path('/scratch/fransjevanoors/global_sr')
#work_dir=Path('/tudelft.net/staff-umbrella/LSM root zone/global_sr')
data_dir=Path(f'{work_dir}/data')
out_dir = Path(f"{work_dir}/output")

print(work_dir)
print(out_dir)
print(data_dir)

# define in and output folder
fol_in=f'{work_dir}/output/'
fol_out=f'{work_dir}/output/'

# define variables of interest
var=['p_mean','ep_mean','q_mean','t_mean','ai','si_p','si_ep','phi','tc','ntc','nonveg','area']

catch_id_list = np.genfromtxt(f'{work_dir}/output/gsim_aus_catch_id_list_lo_sel.txt',dtype='str')[:] # test for 3 catchments -> run on delftblue for all catchments
#catch_id_list = np.genfromtxt(f'/scratch/fransjevanoors/dif.csv',dtype='str')[:] # test for 3 catchments -> run on delftblue for all catchments

# make lists for parallel computation
catch_list = catch_id_list.tolist()
var_list = [var] * len(catch_list)
fol_in_list = [fol_in] * len(catch_list)
fol_out_list = [fol_out] * len(catch_list)

# run catch_characteristics (defined in f_catch_characteristics.py) for the catchments in your catch_id_list parallel
run_function_parallel_catch_characteristics(var_list,catch_list,fol_in_list,fol_out_list)
print('done')
