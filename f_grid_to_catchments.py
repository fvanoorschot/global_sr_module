"""
to do fransje -> clearly write in-out of functions and steps
"""

import glob
from pathlib import Path

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


def regridding_target_cube(catchment_shapefile, spatial_resolution, buffer=1):
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
    
    # extract dates for filename
    time_start,time_end = cube.coord('time')[0],cube.coord('time')[-1]
    point_start, point_end = time_start.points, time_end.points
    unit = time_start.units
    l = unit.num2date(0)
    d = datetime(year=l.year, month=l.month, day=l.day)
    date_start, date_end = d + timedelta(days=point_start[0]), d + timedelta(days=point_end[0])
    y_start, y_end = date_start.year, date_end.year
    
    # Create target grid and regrid cube
    if regrid_first is True:
        target_cube = regridding_target_cube(
            catchment_shapefile, grid_resolution, buffer=1
        )
  
        cube = preprocessor.regrid(cube, target_cube, scheme="area_weighted")

    # From cube extract shapefile shape
    cube = preprocessor.extract_shape(
        cube, catchment_shapefile, method="contains"
    )

    # Calculate area weighted statistics
    cube_stats = preprocessor.area_statistics(cube, statistical_operator)

    if output_csv is True:
        # Convert cube to dataframe
        df = iris.pandas.as_data_frame(cube_stats)

        # Change column names
        df = df.reset_index()
        df = df.set_axis(["time", cube_stats.name()], axis=1)
        
        var=0
        if (catchment_netcdf.split('/')[-1].split('_')[0] == 'pr'):
            var='P'
        if (catchment_netcdf.split('/')[-1].split('_')[0] == 'Ep'):
            var='Ep'
        if (catchment_netcdf.split('/')[-1].split('_')[0] == 'tas'):
            var='T'

        # Write csv as output
        df.to_csv(
            f"{output_dir}/{Path(catchment_shapefile).name.split('.')[0]}_{var}_{statistical_operator}_{y_start}_{y_end}.csv"
        )
    else:
        iris.save(
            cube_stats,
            f"{output_dir}/{Path(catchment_shapefile).name.split('.')[0]}_{var}_{statistical_operator}_{y_start}_{y_end}.nc",
        )
    #     # Write csv as output
    #     df.to_csv(
    #         f"{output_dir}/{Path(catchment_shapefile).name.split('.')[0]}_{catchment_netcdf.split('/')[-1].split('_')[0]}_{statistical_operator}_{y_start}_{y_end}.csv"
    #     )
    # else:
    #     iris.save(
    #         cube_stats,
    #         f"{output_dir}/{Path(catchment_shapefile).name.split('.')[0]}_{catchment_netcdf.split('/')[-1].split('_')[0]}_{statistical_operator}_{y_start}_{y_end}.nc",
    #     )

    if return_cube == True:
        return cube
    else:
        return
    

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

def construct_lists_for_parallel_function(NC4_DIR, SHAPE_DIR, OUT_DIR):
    """
    This functions constructs list for running code in parallel.

    NC4_DIR:              str, dir containing input netcdf files for area weighted statitic calculations
    SHAPE_DIR:            str, dir containing catchment shapefiles
    OUT_DIR:              str, dir for output storage

    Returns: list of netcdfs, shapefiles, output directories
    """
    netcdfs = glob.glob(f"{NC4_DIR}/*nc")
    shapefiles = glob.glob(f"{SHAPE_DIR}/*shp")

    output_dir = [OUT_DIR]

    shapefile_list = shapefiles * len(netcdfs)
    shapefile_list.sort()
    netcdf_list = netcdfs * len(shapefiles)
    output_dir_list = output_dir * len(shapefile_list)

    operator_list = []

    for netcdf in netcdf_list:
        if "tas" in netcdf:
            operator_list.append("mean")
        else:
            operator_list.append("mean")

    return shapefile_list, netcdf_list, operator_list, output_dir_list