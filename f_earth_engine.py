"""
f_earth_engine
----------------
These functions are used to get catchment average values from satellite products using Google Earth Engine

TO DO -> add other products (NDVI, DEM)

1. write_dict
2. preprocess_treecover_data
3. catchment_treecover

"""
import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import glob
from pathlib import Path
from pathos.threading import ThreadPool as Pool
import datetime
from dateutil.relativedelta import relativedelta

## 1
def write_dict(fc):
    """
    write dictionary from list
    
    fc:    feature collection
    returns: dict of feature collection
    
    """
    names = fc.first().propertyNames()
    lists = fc.reduceColumns(
    reducer=ee.Reducer.toList().repeat(names.size()),
    selectors=names).get('list')
    return ee.Dictionary.fromLists(names, lists)

## 2
def preprocess_treecover_data(start_date,end_date):
    """
    Preprocess image collection of treecover (MODIS 44B https://modis.gsfc.nasa.gov/data/dataprod/mod44.php)
    
    start_date: str, start date of analysed period
    end_date:   str, end date of analysed period
    
    return: MODIS tree and non-tree cover images resampled and averaged over analysed timeperiod
    """
    # get image collection
    MOD44B = ee.ImageCollection('MODIS/006/MOD44B')

    # tree cover > image
    MOD44B_tree = MOD44B.select('Percent_Tree_Cover').filterDate(f'{start_date}', f'{end_date}') #select timeperiod
    MOD44B_tree = MOD44B_tree.mean() #take mean over timeperiod
    MOD44B_tree_res = MOD44B_tree.resample('bilinear').reproject(crs= MOD44B_tree.projection().crs(), scale= 1110) #resample to 1110->1degree using bilinear resampling

    # non tree cover > image
    MOD44B_nontree = MOD44B.select('Percent_NonTree_Vegetation').filterDate(f'{start_date}', f'{end_date}') #select timeperiod
    MOD44B_nontree = MOD44B_nontree.mean() #take mean over timeperiod
    MOD44B_nontree_res = MOD44B_nontree.resample('bilinear').reproject(crs= MOD44B_nontree.projection().crs(), scale= 1110) #resample to 1110->1degree using bilinear resampling

    return(MOD44B_tree_res, MOD44B_nontree_res)

## 3
def catchment_treecover(MOD44B_tree_res, MOD44B_nontree_res, catch_id, shape_dir, out_dir):
    """
    get catchment averaged treecover
    
    MOD44B_tree_res:     image of processed tree cover from (2)
    MOD44B_nontree_res:  image of processed non-tree cover from (2)
    catch_id:            str, catchment id 
    shape_dir:           str, dir, directory of shapefiles
    out_dir:             str, dir, output directory for csv
    
    returns:
    df: pandas dataframe with catchment tree cover statistics, stored as csv
        
    """
    # read shapefile as ee feature
    f = glob.glob(f'{shape_dir}/{catch_id}.shp')[0]
    shapefile = gpd.read_file(f)
    features = []
    for j in range(shapefile.shape[0]):
        geom = shapefile.iloc[j:j+1,:] 
        jsonDict = eval(geom.to_json()) 
        geojsonDict = jsonDict['features'][0] 
        features.append(ee.Feature(geojsonDict))
    
    # area of interest = feature = catchment geometry
    aoi = ee.FeatureCollection(features).geometry()
    
    # specifiy reducer statistics (here mean, stdev, max and min)
    reducers = ee.Reducer.mean().combine(
      reducer2= ee.Reducer.stdDev(),
      sharedInputs= True
    ).combine(
      reducer2= ee.Reducer.max(),
      sharedInputs= True
    ).combine(
      reducer2= ee.Reducer.min(),
      sharedInputs= True
    )
    
    # reduce the tree and non tree cover images with the reducers and the aoi 
    t = MOD44B_tree_res.reduceRegions(collection= aoi,reducer=reducers, scale=1110) #scale 1110 m > 1 degree
    nt = MOD44B_nontree_res.reduceRegions(collection= aoi,reducer=reducers, scale=1110)
    
    # write to dictionary (1)
    dt = write_dict(t).getInfo()
    dnt = write_dict(nt).getInfo()
    
    # convert dictionary to pandas dataframe
    dft = pd.DataFrame(dt)
    dfnt = pd.DataFrame(dnt)
    
    # organize dataframe and store output
    df = pd.DataFrame(index=[0], columns=['max_tc','mean_tc','min_tc','std_tc','max_ntc','mean_ntc','min_ntc','std_ntc','mean_nonveg'])
    df[['max_tc','mean_tc','min_tc','std_tc']] = dft[['max','mean','min','stdDev']]
    df[['max_ntc','mean_ntc','min_ntc','std_ntc']] = dfnt[['max','mean','min','stdDev']]
    df['mean_nonveg'] = 100 - df['mean_tc'] - df['mean_ntc']
    df.index = [catch_id]
    df.to_csv(f'{out_dir}/{catch_id}.csv')
    
    return(df)



def elevation_stats(catch_id,work_dir):
    # get elevation image
    # DEM = ee.Image('CGIAR/SRTM90_V4')
    # ele = DEM.select('elevation')
    DEM = ee.Image("WWF/HydroSHEDS/15CONDEM") #here we load a DEM
    ele = DEM.select('b1') # select the elevation band
    slope = ee.Terrain.slope(ele); # compute slope
    ele_res = ele.resample('bilinear').reproject(crs=ele.projection().crs(), scale=450) # get coarser resolution
    slope_res = slope.resample('bilinear').reproject(crs=slope.projection().crs(), scale=450) # get coarser resolution
    
    # make feature from catchment shape
    shape_dir = Path(f'{work_dir}/output/selected_shapes/') #directory with shapefiles
    f = glob.glob(f'{shape_dir}/{catch_id}.shp')[0] #load specific catchment id shapefile
    shapefile = gpd.read_file(f) # read shapefile
    features = []
    for j in range(shapefile.shape[0]): #preprocess shapefile
        geom = shapefile.iloc[j:j+1,:] 
        jsonDict = eval(geom.to_json()) 
        geojsonDict = jsonDict['features'][0] 
        features.append(ee.Feature(geojsonDict))

    # area of interest = feature = catchment geometry
    aoi = ee.FeatureCollection(features).geometry() #take pixels inside shapefile

    # specifiy reducer statistics (here mean, stdev, max and min)
    reducers = ee.Reducer.mean().combine(
      reducer2= ee.Reducer.stdDev(),
      sharedInputs= True
    ).combine(
      reducer2= ee.Reducer.max(),
      sharedInputs= True
    ).combine(
      reducer2= ee.Reducer.min(),
      sharedInputs= True
    )

    # reduce the elevation images with the reducers and the aoi - match grid with the shapefile
    ele_mean = ele_res.reduceRegions(collection= aoi,reducer=reducers, scale=450) #scale 1110 m > 1 degree
    slope_mean = slope_res.reduceRegions(collection= aoi,reducer=reducers, scale=450) #scale 1110 m > 1 degree

    # write to dictionary (1) - to get readable output
    ele = write_dict(ele_mean).getInfo()
    slope = write_dict(slope_mean).getInfo()
    
    if (len(ele['max'])>0):
        # convert dictionary to pandas dataframe
        df_ele = pd.DataFrame(ele)

        # organize dataframe and store output
        df = pd.DataFrame(index=[0], columns=['max_ele','mean_ele','min_ele','std_ele'])
        df[['max_ele','mean_ele','min_ele','std_ele']] = df_ele[['max','mean','min','stdDev']]
        df.index = [catch_id]
        
        # convert dictionary to pandas dataframe
        df_slope = pd.DataFrame(slope)
        # organize dataframe and store output
        dfs = pd.DataFrame(index=[0], columns=['max_slope','mean_slope','min_slope','std_slope'])
        dfs[['max_slope','mean_slope','min_slope','std_slope']] = df_slope[['max','mean','min','stdDev']]
        dfs.index = [catch_id]
        
    else: #catchments at lat>60
        DEM = ee.Image("MERIT/DEM/v1_0_3")
        ele = DEM.select('dem')
        slope = ee.Terrain.slope(ele)
        ele_res = ele.resample('bilinear').reproject(crs=ele.projection().crs(), scale=450)
        slope_res = slope.resample('bilinear').reproject(crs=slope.projection().crs(), scale=450)
        # reduce the elevation images with the reducers and the aoi 
        ele_mean = ele_res.reduceRegions(collection= aoi,reducer=reducers, scale=450) #scale 1110 m > 1 degree
        slope_mean = slope_res.reduceRegions(collection= aoi,reducer=reducers, scale=450) #scale 1110 m > 1 degree

        # write to dictionary (1)
        ele = write_dict(ele_mean).getInfo()
        slope = write_dict(slope_mean).getInfo()
        
        # convert dictionary to pandas dataframe
        df_ele = pd.DataFrame(ele)

        # organize dataframe and store output
        df = pd.DataFrame(index=[0], columns=['max_ele','mean_ele','min_ele','std_ele'])
        df[['max_ele','mean_ele','min_ele','std_ele']] = df_ele[['max','mean','min','stdDev']]
        df.index = [catch_id]
        
        # convert dictionary to pandas dataframe
        df_slope = pd.DataFrame(slope)
        # organize dataframe and store output
        dfs = pd.DataFrame(index=[0], columns=['max_slope','mean_slope','min_slope','std_slope'])
        dfs[['max_slope','mean_slope','min_slope','std_slope']] = df_slope[['max','mean','min','stdDev']]
        dfs.index = [catch_id]

    df.to_csv(f'{work_dir}/output/elevation/stats_hydrosheds/ele_{catch_id}.csv')
    dfs.to_csv(f'{work_dir}/output/elevation/stats_hydrosheds/slope_{catch_id}.csv')
    
    
def elevation_zones(catch_id,work_dir):
    # get elevation image
    # DEM = ee.Image('CGIAR/SRTM90_V4')
    # ele = DEM.select('elevation')
    DEM = ee.Image("WWF/HydroSHEDS/15CONDEM")
    ele = DEM.select('b1')
    ele_res = ele.resample('bilinear').reproject(crs=ele.projection().crs(), scale=100)
    
    # make feature from catchment shape
    shape_dir = Path(f'{work_dir}/output/selected_shapes/')
    f = glob.glob(f'{shape_dir}/{catch_id}.shp')[0]
    shapefile = gpd.read_file(f)
    features = []
    for j in range(shapefile.shape[0]):
        geom = shapefile.iloc[j:j+1,:] 
        jsonDict = eval(geom.to_json()) 
        geojsonDict = jsonDict['features'][0] 
        features.append(ee.Feature(geojsonDict))
    
    # area of interest = feature = catchment geometry
    aoi = ee.FeatureCollection(features).geometry()
    
    # reduce DEM using autohistogram - get histogram with frequency of elevation values inside catchment
    ele_shape = ele_res.reduceRegions(collection=aoi, reducer=ee.Reducer.autoHistogram(),scale=100)

    # process ele_shape to dataframe df with elevation and frequency columns
    a = write_dict(ele_shape).getInfo()
    
    if (len(a.keys())==2):
        b = pd.DataFrame(a)
        b = b.histogram.values
        df = pd.DataFrame(index=np.arange(0,len(b[0]),1),columns=['elevation','frequency'])
        ev = np.zeros(len(b[0]))
        freq = np.zeros(len(b[0]))
        for j in range(len(b[0])):
            ev[j]=b[0][j][0]
            freq[j]=b[0][j][1]
        df['elevation'] = ev
        df['frequency'] = freq

        # make dataframe with elevation zones and frequency inside elevation zone
        f_sum = df.frequency.sum()
        el=[]
        zones = np.arange(0,7500,250) # use fixed elevation zones
        for j in range(1,len(zones)):
            e = df[(df['elevation']>zones[j-1])&(df['elevation']<zones[j])]
            ef = e['frequency'].sum() / f_sum
            el.append(ef)
        df_el = pd.DataFrame()
        df_el['min_el'] = zones[0:-1]
        df_el['max_el'] = zones[1:]
        df_el['mean_el'] = np.mean( np.array([ df_el.min_el.values, df_el.max_el.values ]), axis=0 )
        df_el['frac'] = el

    else: #catchments at lat>60
        DEM = ee.Image("MERIT/DEM/v1_0_3")
        ele = DEM.select('dem')
        ele_res = ele.resample('bilinear').reproject(crs=ele.projection().crs(), scale=100)
    
        # reduce DEM using autohistogram - get histogram with frequency of elevation values inside catchment
        ele_shape = ele_res.reduceRegions(collection=aoi, reducer=ee.Reducer.autoHistogram(),scale=100)

        # process ele_shape to dataframe df with elevation and frequency columns
        a = write_dict(ele_shape).getInfo()
        
        b = pd.DataFrame(a)
        b = b.histogram.values
        df = pd.DataFrame(index=np.arange(0,len(b[0]),1),columns=['elevation','frequency'])
        ev = np.zeros(len(b[0]))
        freq = np.zeros(len(b[0]))
        for j in range(len(b[0])):
            ev[j]=b[0][j][0]
            freq[j]=b[0][j][1]
        df['elevation'] = ev
        df['frequency'] = freq

        # make dataframe with elevation zones and frequency inside elevation zone
        f_sum = df.frequency.sum()
        el=[]
        zones = np.arange(0,7500,250) # use fixed elevation zones
        for j in range(1,len(zones)):
            e = df[(df['elevation']>zones[j-1])&(df['elevation']<zones[j])]
            ef = e['frequency'].sum() / f_sum
            el.append(ef)
        df_el = pd.DataFrame()
        df_el['min_el'] = zones[0:-1]
        df_el['max_el'] = zones[1:]
        df_el['mean_el'] = np.mean( np.array([ df_el.min_el.values, df_el.max_el.values ]), axis=0 )
        df_el['frac'] = el

    df_el.to_csv(f'{work_dir}/output/elevation/el_zones/{catch_id}.csv')
    
    
def run_function_parallel_elevation_stats(
    catch_list=list,
    work_dir_list=list,
    # threads=None
    threads=100
    ):
    # Set number of threads (cores) used for parallel run and map threads
    if threads is None:
        pool = Pool()
    else:
        pool = Pool(nodes=threads)
    # Run parallel models
    results = pool.map(
        elevation_stats,
        catch_list,
        work_dir_list,
    )
    
def run_function_parallel_elevation_zones(
    catch_list=list,
    work_dir_list=list,
    # threads=None
    threads=100
    ):
    # Set number of threads (cores) used for parallel run and map threads
    if threads is None:
        pool = Pool()
    else:
        pool = Pool(nodes=threads)
    # Run parallel models
    results = pool.map(
        elevation_zones,
        catch_list,
        work_dir_list,
    )
    

# CHIRPS PRECIPITATION TIMESERIES

#%% reducer function
def reducer_function(geometry,reducer=ee.Reducer.mean(), scale=1000,crs='EPSG:4326', maxPixels=1e10,):
    def reduce_region_function(image):
        stats = image.reduceRegion(reducer=reducer,geometry=geometry,scale=scale,crs=crs,maxPixels=maxPixels)
        return ee.Feature(geometry, stats).set({'millis': image.date().millis()})
    return reduce_region_function

def write_dict(fc):
    names = fc.first().propertyNames()
    lists = fc.reduceColumns(
        reducer=ee.Reducer.toList().repeat(names.size()),
        selectors=names).get('list')
    return ee.Dictionary.fromLists(names, lists)
    
def ee_chirps_timeseries(catch_id,start_date,work_dir):
    sd = datetime.datetime.strptime(start_date,'%Y-%m-%d')
    ed = sd + relativedelta(years=2)
    end_date = ed.strftime('%Y-%m-%d')
    
    CH = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
    p = CH.select('precipitation').filterDate(f'{start_date}', f'{end_date}')
    
    # make feature from catchment shape
    shape_dir = Path(f'{work_dir}/output/selected_shapes/')
    f = glob.glob(f'{shape_dir}/{catch_id}.shp')[0]
    shapefile = gpd.read_file(f)
    features = []
    for j in range(shapefile.shape[0]):
        geom = shapefile.iloc[j:j+1,:] 
        jsonDict = eval(geom.to_json()) 
        geojsonDict = jsonDict['features'][0] 
        features.append(ee.Feature(geojsonDict))

    # area of interest = feature = catchment geometry
    aoi = ee.FeatureCollection(features).geometry()
    
    reduce_p = reducer_function(geometry=aoi, reducer=ee.Reducer.mean(), scale=1000, crs='EPSG:4326')
    p_stat_fc = ee.FeatureCollection(p.map(reduce_p)).filter(ee.Filter.notNull(p.first().bandNames()))

    p_dict = write_dict(p_stat_fc).getInfo()
    p_df = pd.DataFrame(p_dict)
    
    p_df['date'] = pd.to_datetime(p_df['millis'], unit = 'ms')
    p_df['p'] = p_df['precipitation']
    p_df.index = p_df['date']
    p_df = p_df.drop(columns=['date','precipitation','millis','system:index'])
    
    p_df.to_csv(f'{work_dir}/output/p_chirps_timeseries_selected_catchments/daily_ee/{catch_id}_{start_date}_{end_date}.csv')
    # return p_df
    
def ee_chirps_parallel(
    catch_list=list,
    start_date_list=list,
    work_dir_list=list,
    # threads=None
    threads=100
    ):
    # Set number of threads (cores) used for parallel run and map threads
    if threads is None:
        pool = Pool()
    else:
        pool = Pool(nodes=threads)
    # Run parallel models
    results = pool.map(
        ee_chirps_timeseries,
        catch_list,
        start_date_list,
        work_dir_list,
    )