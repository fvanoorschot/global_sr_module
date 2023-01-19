"""
f_grid_to_catchments.py
------------------------
These functions are used to extract catchment timeseries from gridded netcdf climate data

1. regridding_target_cube
2. area_weighted_shapefile_rasterstats
3. construct_lists_for_parallel_function
4. run_functions_parallel
5. process_forcing_timeseries
ddd
"""

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

## 1 
def regridding_target_cube(catchment_shapefile, spatial_resolution, buffer=1):
    
    """
    Define the target cube for regridding the input netcdf data
    catchment_shapefile:  str, catchment shapefile
    spatial_resolution:   float, target resolution
    buffer:               int, buffer
    
    returns:
    target cube for regridding
    
    """    
    catchment_bounds = gpd.read_file(catchment_shapefile).bounds

    buffer = 1
    minx = int(catchment_bounds.minx.values[0]) - buffer
    maxx = int(catchment_bounds.maxx.values[0]) + buffer
    miny = int(catchment_bounds.miny.values[0]) - buffer
    maxy = int(catchment_bounds.maxy.values[0]) + buffer

    latitude = DimCoord(
        np.linspace(
            miny,
            maxy,
            int(np.divide(abs(miny-maxy), spatial_resolution)),
            dtype=float,
        ),
        standard_name="latitude",
        units="degrees",
    )
    latitude.guess_bounds()
    
    longitude = DimCoord(
        np.linspace(
            minx,
            maxx,
            int(np.divide(abs(minx-maxx), spatial_resolution)),
            dtype=float,
        ),
        standard_name="longitude",
        units="degrees",
    )
    longitude.guess_bounds()
    
    target_cube = Cube(
        np.zeros((len(latitude.points), len(longitude.points)), np.float32),
        dim_coords_and_dims=[(latitude, 0), (longitude, 1)],
    )

    return target_cube


## 2
def area_weighted_shapefile_rasterstats(
    catchment_shapefile,
    catchment_netcdf,
    statistical_operator,
    output_dir,
    output_csv=True,
    return_cube=False,
    regrid_first=True,
    grid_resolution=0.1
):
    
    """
    Calculate area weighted zonal statistics of netcdfs using a shapefile to extract netcdf data.

    catchment_shapefile:  str, catchment shapefile
    catchment_netcdf:     str, netcdf file
    statistical_operator: str, (mean, median (NOT area weighted), sum, variance, min, max, rms)
    - https://docs.esmvaltool.org/projects/esmvalcore/en/latest/api/esmvalcore.preprocessor.html#esmvalcore.preprocessor.area_statistics
    output_csv:          bool, True stores csv output and False stores netcdf output
    regrid_first:        bool, True regrid cube first before extracting shape, False do not regrid first
    grid_resolution:    float, grid cell size of target cube in degrees
    Returns: iris cube, stores .csv file or .nc file
    """
    
    # Load iris cube of netcdf
    cube = iris.load_cube(catchment_netcdf)
    cube.dim_coords[1].guess_bounds()
    cube.dim_coords[2].guess_bounds()
    
    # extract dates of netcdf timeseries to be used as filename
    time_start,time_end = cube.coord('time')[0],cube.coord('time')[-1]
    point_start, point_end = time_start.points, time_end.points
    unit = time_start.units
    l = unit.num2date(0)
    d = datetime(year=l.year, month=l.month, day=l.day)
    date_start, date_end = d + timedelta(days=point_start[0]), d + timedelta(days=point_end[0])
    y_start, y_end = date_start.year, date_end.year
    
    # Create target grid and regrid cube
    if regrid_first is True:
        target_cube = regridding_target_cube(catchment_shapefile, grid_resolution, buffer=1) #create the regrid target cube
        cube = preprocessor.regrid(cube, target_cube, scheme="area_weighted") #regrid the netcdf file (conservative) to a higher resolution

    # From cube extract shapefile shape
    cube = preprocessor.extract_shape(cube, catchment_shapefile, method="contains") #use all grid cells that lie >50% inside the catchment shape

    # Calculate area weighted statistics of extracted grid cells (inside catchment shape)
    cube_stats = preprocessor.area_statistics(cube, statistical_operator)

    if output_csv is True: #save the timeseries as csv
        # Convert cube to dataframe
        df = iris.pandas.as_data_frame(cube_stats)

        # Change column names of timeseries dataframe
        df = df.reset_index()
        df = df.set_axis(["time", cube_stats.name()], axis=1)
        
        var=0
        if (catchment_netcdf.split('/')[-1].split('_')[0] == 'pr'): #pr is gswp daily precipitation
            var='P'
        if (catchment_netcdf.split('/')[-1].split('_')[0] == 'Ep'): #Ep is GLEAM potential evaporation
            var='Ep'
        if (catchment_netcdf.split('/')[-1].split('_')[0] == 'tas'): #tas is gswp mean daily temperature
            var='T'

        # Write csv as output
        df.to_csv(f"{output_dir}/{Path(catchment_shapefile).name.split('.')[0]}_{var}_{statistical_operator}_{y_start}_{y_end}.csv")
       
    # if output_csv is False -> save the netcdf cube
    else:
        iris.save(cube_stats, f"{output_dir}/{Path(catchment_shapefile).name.split('.')[0]}_{var}_{statistical_operator}_{y_start}_{y_end}.nc")

    # return cube yes or no
    if return_cube == True:
        return cube
    else:
        return

## 3
def construct_lists_for_parallel_function(NC4_DIR, SHAPE_DIR, OUT_DIR):
    """
    This functions constructs list for running code in parallel.

    NC4_DIR:              str, dir containing input netcdf files for area weighted statitic calculations
    SHAPE_DIR:            str, dir containing catchment shapefiles
    OUT_DIR:              str, dir for output storage

    Returns: list of netcdfs, shapefiles, output directories
    """
    
    # make list of files
    netcdfs = glob.glob(f"{NC4_DIR}/*nc")
    shapefiles = glob.glob(f"{SHAPE_DIR}/*shp")

    output_dir = [OUT_DIR]

    # combine lists to get all needed combinations
    shapefile_list = shapefiles * len(netcdfs)
    shapefile_list.sort()
    netcdf_list = netcdfs * len(shapefiles)
    output_dir_list = output_dir * len(shapefile_list)

    operator_list = []

    # define statistical operator - in this case we always need the mean but if have accumulated data you should take the sum
    for netcdf in netcdf_list:
        if "tas" in netcdf:
            operator_list.append("mean")
        else:
            operator_list.append("mean")

    return shapefile_list, netcdf_list, operator_list, output_dir_list    

## 4
def run_function_parallel(
    shapefile_list=list,
    netcdf_list=list,
    operator_list=list,
    output_dir_list=list,
    threads=None
    #threads=100
):
    """
    Runs function area_weighted_shapefile_rasterstats in parallel.

    shapefile_list:  str, list, list of input catchment shapefiles
    netcdf_list:     str, list, list of input netcdf files
    operator_list:   str, list, list of statistical operators (single operator)
    output_dir_list: str, list, list of output directories
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
        area_weighted_shapefile_rasterstats,
        shapefile_list,
        netcdf_list,
        operator_list,
        output_dir_list,
    )

    return results

## 5
def process_forcing_timeseries(catch_id,fol_in,fol_out,var):
    """
    postprocess the catchment timeseries to usable pandas dataframes
    catch_id:   str, catchment id
    fol_in:     str, dir, folder with raw output csvs from grid-catchment extraction
    fol_out:    str, dir, folder where to storre the processed dataframes
    var:        str, list, list of variables calculated
    
    returns:
    stores csvs of daily, monthly, yearly, climatology and mean timeseries    
    
    """
    # make empty dataframe
    d = pd.DataFrame()

    # for j in variable list - list the timeseries csvs for the catch id
    l = glob.glob(fol_in + f"*/{catch_id}*.csv")

    # combine variable timeseries in one dataframe
    li=[] #make empty list
    for filename in l:
        df = pd.read_csv(filename, index_col=0, header=0)
        # df = df.drop(columns=['Unnamed: 0'])
        df.index = pd.to_datetime(df.index)
        df = df.loc['1981-01-01':'2010-12-31']
        li.append(df) #append dataframe to list

    d = pd.DataFrame()
    frame = pd.concat(li, axis=1, ignore_index=False) #concatenate dataframes in li
    col=frame.columns #get column names 
    y_start,y_end = frame.index[0].year, frame.index[-1].year #add columns with start and end years
    d[col] = frame #add frame data to dataframe d
    d = d.rename(columns={col.values[0]:f'{var[0]}'}) #rename column names to variable list names
    d = d.rename(columns={col.values[1]:f'{var[1]}'})
    d = d.rename(columns={col.values[2]:f'{var[2]}'})

    # get daily timeseries and store as csv
    if not os.path.exists(f'{fol_out}/daily'):
         os.makedirs(f'{fol_out}/daily')
    d.to_csv(f'{fol_out}/daily/{catch_id}_{y_start}_{y_end}.csv')

    # get monthly timeseries and store as csv
    if not os.path.exists(f'{fol_out}/monthly'):
         os.makedirs(f'{fol_out}/monthly')
    df_m = d.groupby(pd.Grouper(freq='M')).mean()
    y_start,y_end = df_m.index[0].year, df_m.index[-1].year
    df_m.to_csv(f'{fol_out}/monthly/{catch_id}_{y_start}_{y_end}.csv')    

    # get climatology and store as csv
    if not os.path.exists(f'{fol_out}/climatology'):
         os.makedirs(f'{fol_out}/climatology')
    df_m = df_m.groupby([df_m.index.month]).mean()
    df_m.to_csv(f'{fol_out}/climatology/{catch_id}_{y_start}_{y_end}.csv')

    # get yearly timeseries and store as csv
    if not os.path.exists(f'{fol_out}/yearly'):
         os.makedirs(f'{fol_out}/yearly')
    df_y = d.groupby(pd.Grouper(freq='Y')).mean()
    y_start,y_end = df_y.index[0].year, df_y.index[-1].year
    df_y.to_csv(f'{fol_out}/yearly/{catch_id}_{y_start}_{y_end}.csv')

    # get mean of timeseries and store as csv
    if not os.path.exists(f'{fol_out}/mean'):
         os.makedirs(f'{fol_out}/mean')
    dm = d.mean()
    dm.to_csv(f'{fol_out}/mean/{catch_id}_{y_start}_{y_end}.csv')
    

        
def run_processing_function_parallel(
    catch_list=list,
    fol_in_list=list,
    fol_out_list=list,
    var_list=list,
    # threads=None
    threads=100
):
    """
    Runs function preprocess_gsim_discharge  in parallel.
​
    catch_list:  str, list, list of catchmet ids
    fol_in_list:     str, list, list of input folders
    fol_out_list:   str, list, list of output folders
    var_list: str,list, list of var list 
    threads:         int,       number of threads (cores), when set to None use all available threads
​
    Returns: None
    """
    # Set number of threads (cores) used for parallel run and map threads
    if threads is None:
        pool = Pool()
    else:
        pool = Pool(nodes=threads)
    # Run parallel models
    results = pool.map(
        process_forcing_timeseries,
        catch_list,
        fol_in_list,
        fol_out_list,
        var_list,
    )