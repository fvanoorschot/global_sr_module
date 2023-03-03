# import packages
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
import xarray as xr

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


def stocker_to_shape_values(catch_id,work_dir):
    catchment_shapefile =  glob.glob(f'{work_dir}/output/selected_shapes/{catch_id}.shp')[0]
    
    # STOCKER CWDX80
    catchment_netcdf= glob.glob(f'{work_dir}/data/reference_sr/stocker/cwdx80_units.nc')[0]

    # Load iris cube of netcdf
    cube = iris.load_cube(catchment_netcdf)
    cube.dim_coords[0].guess_bounds()
    cube.dim_coords[1].guess_bounds()

    # Create target grid and regrid cube
    # if regrid_first is True:
    #     target_cube = regridding_target_cube(catchment_shapefile, grid_resolution, buffer=1) #create the regrid target cube
    #     cube = preprocessor.regrid(cube, target_cube, scheme="area_weighted") #regrid the netcdf file (conservative) to a higher resolution

    # From cube extract shapefile shape
    cube = preprocessor.extract_shape(cube, catchment_shapefile, method="contains") #use all grid cells that lie >50% inside the catchment shape

    # Calculate area weighted statistics of extracted grid cells (inside catchment shape)
    cube_stats = preprocessor.area_statistics(cube, 'mean')

    # from cube to xarray dataarray
    a=xr.DataArray.from_iris(cube_stats)

    # STOCKER ROOT DEPTH
    catchment_netcdf= glob.glob(f'{work_dir}/data/reference_sr/stocker/zroot_cwd80_units.nc')[0]

    # Load iris cube of netcdf
    cube = iris.load_cube(catchment_netcdf)
    cube.dim_coords[0].guess_bounds()
    cube.dim_coords[1].guess_bounds()

    # Create target grid and regrid cube
    # if regrid_first is True:
    #     target_cube = regridding_target_cube(catchment_shapefile, grid_resolution, buffer=1) #create the regrid target cube
    #     cube = preprocessor.regrid(cube, target_cube, scheme="area_weighted") #regrid the netcdf file (conservative) to a higher resolution

    # From cube extract shapefile shape
    cube = preprocessor.extract_shape(cube, catchment_shapefile, method="contains") #use all grid cells that lie >50% inside the catchment shape

    # Calculate area weighted statistics of extracted grid cells (inside catchment shape)
    cube_stats = preprocessor.area_statistics(cube, 'mean')

    # from cube to xarray dataarray
    b=xr.DataArray.from_iris(cube_stats)


    df = pd.DataFrame(index=[catch_id], columns=['stocker_cwd80x_mm','stocker_zroot_cwd80x_mm'])
    df['stocker_cwd80x_mm'] = a.values
    df['stocker_zroot_cwd80x_mm'] = b.values
    df.to_csv(f'{work_dir}/output/sr_calculation/stocker/{catch_id}.csv')
    
    
def run_stocker_sr_parallel(
    catch_id_list=list,
    work_dir_list=list,
    # threads=None
    threads=100
):
    """
    Runs function area_weighted_shapefile_rasterstats in parallel.

    catch_list:  str, list, list of catchment ids
    work_dir_list: str, list, list of work directories
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
        stocker_to_shape_values,
        catch_id_list,
        work_dir_list,
    )
    
    
def lan_to_shape_values(catch_id,work_dir):
    catchment_shapefile =  glob.glob(f'{work_dir}/output/selected_shapes/{catch_id}.shp')[0]    
    # LAN CRU
    rp = ['2yrs','5yrs','10yrs','20yrs','30yrs','40yrs','50yrs','60yrs','max']
    df_cru = pd.DataFrame(index=[catch_id], columns=[])
    for i in rp:
        catchment_netcdf= glob.glob(f'{work_dir}/data/reference_sr/lan/sr_cru_{i}.nc')[0]

        # Load iris cube of netcdf
        cube = iris.load_cube(catchment_netcdf)
        cube.dim_coords[1].guess_bounds()
        cube.dim_coords[2].guess_bounds()

        # Create target grid and regrid cube to higher resolution (0.1) because 0.5 is coarse
        grid_resolution = 0.1
        target_cube = regridding_target_cube(catchment_shapefile, grid_resolution, buffer=1) #create the regrid target cube
        cube = preprocessor.regrid(cube, target_cube, scheme="nearest") #regrid the netcdf file with nearest neighbour (most logical for sr values I think)

        # From cube extract shapefile shape
        cube = preprocessor.extract_shape(cube, catchment_shapefile, method="contains") #use all grid cells that lie >50% inside the catchment shape

        # Calculate area weighted statistics of extracted grid cells (inside catchment shape)
        cube_stats = preprocessor.area_statistics(cube, 'mean')

        # from cube to xarray dataarray
        a=xr.DataArray.from_iris(cube_stats)
        df_cru[f'lan_cru_{i}'] = a.values

    # LAN CHIRPS
    rp = ['2yrs','5yrs','10yrs','20yrs','30yrs','40yrs','50yrs','60yrs','max']
    df_chirps = pd.DataFrame(index=[catch_id], columns=[])
    for i in rp:
        catchment_netcdf= glob.glob(f'{work_dir}/data/reference_sr/lan/sr_chirps_{i}.nc')[0]

        # Load iris cube of netcdf
        cube = iris.load_cube(catchment_netcdf)
        cube.dim_coords[1].guess_bounds()
        cube.dim_coords[2].guess_bounds()

        # Create target grid and regrid cube to higher resolution (0.1) because 0.5 is coarse
        grid_resolution = 0.1
        target_cube = regridding_target_cube(catchment_shapefile, grid_resolution, buffer=1) #create the regrid target cube
        cube = preprocessor.regrid(cube, target_cube, scheme="nearest") #regrid the netcdf file with nearest neighbour (most logical for sr values I think)

        # From cube extract shapefile shape
        cube = preprocessor.extract_shape(cube, catchment_shapefile, method="contains") #use all grid cells that lie >50% inside the catchment shape

        # Calculate area weighted statistics of extracted grid cells (inside catchment shape)
        cube_stats = preprocessor.area_statistics(cube, 'mean')

        # from cube to xarray dataarray
        a=xr.DataArray.from_iris(cube_stats)
        df_chirps[f'lan_chirps_{i}'] = a.values   

    df = pd.concat([df_chirps,df_cru],axis=1)
    df.to_csv(f'{work_dir}/output/sr_calculation/lan/{catch_id}.csv')
    
    
def run_lan_sr_parallel(
    catch_id_list=list,
    work_dir_list=list,
    # threads=None
    threads=100
):
    """
    Runs function area_weighted_shapefile_rasterstats in parallel.

    catch_list:  str, list, list of catchment ids
    work_dir_list: str, list, list of work directories
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
        lan_to_shape_values,
        catch_id_list,
        work_dir_list,
    )