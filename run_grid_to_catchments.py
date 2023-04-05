import glob
from pathlib import Path
import os
import geopandas as gpd
import iris
import iris.pandas
import numpy as np
from esmvalcore import preprocessor
from iris.coords import DimCoord
from iris.cube import Cube
from pathos.threading import ThreadPool as Pool
from datetime import datetime
from datetime import timedelta
import pandas as pd

from f_grid_to_catchments import *

work_dir=Path('/scratch/fransjevanoors/global_sr')
#work_dir=Path('/tudelft.net/staff-umbrella/LSM root zone/global_sr')
data_dir=Path(f'{work_dir}/data')
out_dir = Path(f"{work_dir}/output")

print(work_dir)
print(out_dir)
print(data_dir)


# # define directories 
# nc_cru = f'{work_dir}/data/cru_p/cru_ts4.06_1961_2010_pre.nc' # dir of netcdf forcing files
# out_dir = f'{work_dir}/output/forcing_timeseries/cru_p/nearest' # output dir
# operator = 'mean'

# shape_dir = Path(f'{work_dir}/output/selected_shapes/')
# shapefile_list = glob.glob(f'{shape_dir}/*.shp')[:]

# netcdf_list = [nc_cru]*len(shapefile_list)
# output_dir_list = [out_dir]*len(shapefile_list)
# operator_list = [operator]*len(shapefile_list)


# # run function parallel
# run_cru_function_parallel(shapefile_list, netcdf_list, operator_list, output_dir_list)

# # define directories 
# nc_bd = f'{work_dir}/data/IWU_irri_data/IWU_2011_2018.nc' # dir of netcdf forcing files
# out_dir = f'{work_dir}/output/irrigation/raw/' # output dir
# operator = 'mean'

# shape_dir = Path(f'{work_dir}/output/selected_shapes/')
# shapefile_list = glob.glob(f'{shape_dir}/*.shp')[:]

# netcdf_list = [nc_bd]*len(shapefile_list)
# output_dir_list = [out_dir]*len(shapefile_list)

# # run function parallel
# run_iwu_function_parallel(shapefile_list, netcdf_list, output_dir_list)

# define directories 
nc_bd = f'{work_dir}/data/mswep_p/daily_peryear/1990.nc' # dir of netcdf forcing files
out_dir = f'{work_dir}/output/forcing_timeseries/mswep_p/raw/' # output dir
operator = 'mean'

shape_dir = Path(f'{work_dir}/output/selected_shapes/')
shapefile_list = glob.glob(f'{shape_dir}/*.shp')[0:3]

netcdf_list = [nc_bd]*len(shapefile_list)
output_dir_list = [out_dir]*len(shapefile_list)
operator_list = [operator]*len(shapefile_list)