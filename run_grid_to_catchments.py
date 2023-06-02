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

# define directories 
nc_bd = f'{work_dir}/data/IWU_irri_data/IWU_2011_2018.nc' # dir of netcdf forcing files
out_dir = f'{work_dir}/output/irrigation/raw2/' # output dir
operator = 'mean'

shape_dir = Path(f'{work_dir}/output/selected_shapes/')
shapefile_list = glob.glob(f'{shape_dir}/*.shp')[:]

netcdf_list = [nc_bd]*len(shapefile_list)
output_dir_list = [out_dir]*len(shapefile_list)

# run function parallel
run_iwu_function_parallel(shapefile_list, netcdf_list, output_dir_list)

# MSWEP - done on tintin 
# out_dir = f'{work_dir}/output/raw/' # output dir
# operator = 'mean'

# shape_dir = Path(f'{work_dir}/selected_shapes/')
# shapefiles = glob.glob(f'{shape_dir}/*.shp')[:]

# # check which catchments already done
# shape_list=[]
# for filepath in glob.iglob(f'{work_dir}/selected_shapes/*.shp'):
#     f = os.path.split(filepath)[1] # remove full path
#     f = f[:-4] 
#     shape_list.append(f)

# el_id_list=[]
# for filepath in glob.iglob(f'{work_dir}/output/raw/*.csv'):
#     f = os.path.split(filepath)[1] # remove full path
#     f = f[:-21]
#     el_id_list.append(f)

# el_id_list2=[]
# for c in shape_list:
#     count = el_id_list.count(c)
#     if (count<5):
#         el_id_list2.append(c)

# print(len(el_id_list2))
# #el_id_list=np.unique(el_id_list)
# #dif = list(set(shape_list) - set(el_id_list))

# shapefiles2=[]
# for i in range(len(shapefiles)):
#     j = os.path.split(shapefiles[i])[1]
#     if ('_' in j):
#         j = j[:-4].lower()
#     else:
#         j = j[:-4]
#     if (j in el_id_list2):
#         shapefiles2.append(shapefiles[i])

# #print(shapefiles2)

# netcdfs = glob.glob(f'/home/vanoorschot/work/fransje/DATA/MSWEP/daily/years_processed/merged/0_25deg/*.nc')

# shapefile_list = shapefiles2 * len(netcdfs)
# shapefile_list.sort()
# netcdf_list = netcdfs * len(shapefiles2)
# output_dir_list = [out_dir] * len(shapefile_list)
# operator_list = [operator]*len(shapefile_list)

# run_function_parallel(shapefile_list, netcdf_list, operator_list, output_dir_list)


# IRRIGATED AREA
# catchment_netcdf = f'{work_dir}/data/irrigated_area/AEI_HYDE_FINAL_IR_2005_fraction.nc'
# catchment_netcdf = f'{work_dir}/data/irrigated_area/AEI_HYDE_FINAL_CP_2005_fraction.nc'
# catchment_netcdf = f'{work_dir}/data/irrigated_area/AEI_EARTHSTAT_CP_2005_fraction.nc'

# irri_data = 'AEI_EARTHSTAT_IR_2005'
# irri_data = 'AEI_EARTHSTAT_CP_2005'
# irri_data = 'AEI_HYDE_FINAL_CP_2005'
# irri_data = 'AEI_HYDE_FINAL_IR_2005'
# nc = f'{work_dir}/data/irrigated_area/{irri_data}_fraction.nc'
# shape_dir = Path(f'{work_dir}/output/selected_shapes/')
# out_dir =  f'{work_dir}/data/irrigated_area/output/{irri_data}/'

# shape_dir = Path(f'{work_dir}/output/selected_shapes/')
# shapefile_list = glob.glob(f'{shape_dir}/*.shp')[:]

# netcdf_list = [nc]*len(shapefile_list)
# out_dir_list = [out_dir]*len(shapefile_list)

# run_irri_area_parallel(shapefile_list, netcdf_list, out_dir_list) # run all catchments as job file