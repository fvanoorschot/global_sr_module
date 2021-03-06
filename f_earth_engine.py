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