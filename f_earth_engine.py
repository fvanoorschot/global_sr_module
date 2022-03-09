"""
to do fransje -> clearly write function details

"""


import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import glob
from pathlib import Path

#%% reducer function
def reducer_function(geometry,reducer=ee.Reducer.mean(), scale=10,crs='EPSG:4326', maxPixels=1e10,):
    def reduce_region_function(image):
        stats = image.reduceRegion(reducer=reducer,geometry=geometry,scale=scale,crs=crs,maxPixels=maxPixels)
        return ee.Feature(geometry, stats).set({'millis': image.date().millis()})
    return reduce_region_function

# function to write python dictionary
def write_dict(fc):
    names = fc.first().propertyNames()
    lists = fc.reduceColumns(
    reducer=ee.Reducer.toList().repeat(names.size()),
    selectors=names).get('list')
    return ee.Dictionary.fromLists(names, lists)

def preprocess_treecover_data(start_date,end_date):
    MOD44B = ee.ImageCollection('MODIS/006/MOD44B')

    MOD44B_tree = MOD44B.select('Percent_Tree_Cover').filterDate(f'{start_date}', f'{end_date}')
    MOD44B_tree = MOD44B_tree.mean()
    MOD44B_tree_res = MOD44B_tree.resample('bilinear').reproject(crs= MOD44B_tree.projection().crs(), scale= 1110) 

    MOD44B_nontree = MOD44B.select('Percent_NonTree_Vegetation').filterDate(f'{start_date}', f'{end_date}')
    MOD44B_nontree = MOD44B_nontree.mean()
    MOD44B_nontree_res = MOD44B_nontree.resample('bilinear').reproject(crs= MOD44B_nontree.projection().crs(), scale= 1110)

    return(MOD44B_tree_res, MOD44B_nontree_res)


def catchment_treecover(MOD44B_tree_res, MOD44B_nontree_res, catch_id, shape_dir, out_dir):
    f = glob.glob(f'{shape_dir}/{catch_id}.shp')[0]
    shapefile = gpd.read_file(f)
    features = []
    for j in range(shapefile.shape[0]):
        geom = shapefile.iloc[j:j+1,:] 
        jsonDict = eval(geom.to_json()) 
        geojsonDict = jsonDict['features'][0] 
        features.append(ee.Feature(geojsonDict))
        
    aoi = ee.FeatureCollection(features).geometry()
    
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
    
    t = MOD44B_tree_res.reduceRegions(collection= aoi,reducer=reducers, scale=1110)
    nt = MOD44B_nontree_res.reduceRegions(collection= aoi,reducer=reducers, scale=1110)
    
    dt = write_dict(t).getInfo()
    dnt = write_dict(nt).getInfo()
    
    dft = pd.DataFrame(dt)
    dfnt = pd.DataFrame(dnt)
    
    df = pd.DataFrame(index=[0], columns=['max_tc','mean_tc','min_tc','std_tc','max_ntc','mean_ntc','min_ntc','std_ntc','mean_nonveg'])
    df[['max_tc','mean_tc','min_tc','std_tc']] = dft[['max','mean','min','stdDev']]
    df[['max_ntc','mean_ntc','min_ntc','std_ntc']] = dfnt[['max','mean','min','stdDev']]
    df['mean_nonveg'] = 100 - df['mean_tc'] - df['mean_ntc']
    df.index = [catch_id]
    df.to_csv(f'{out_dir}/{catch_id}.csv')
    
    return(df)