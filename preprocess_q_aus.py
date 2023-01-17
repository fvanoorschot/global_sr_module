#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:46:37 2021

@author: vanoorschot
"""

#%%
import numpy as np
import pandas as pd
import geopandas as gpd
import glob
from pathlib import Path

work_dir2=Path('/scratch/fransjevanoors/global_sr')
work_dir=Path('/tudelft.net/staff-umbrella/LSM root zone/global_sr')
data_dir=Path(f'{work_dir}/data')
out_dir = Path(f"{work_dir2}/output")

print(work_dir)
print(out_dir)
print(data_dir)

#%% save q timeseries for camels aus catchments
m = pd.read_csv(f'{data_dir}/CAMELS_AUS/CAMELS_AUS_Attributes-Indices_MasterTable.csv',index_col=0)
catch_id = m.index
c_array = np.array(catch_id)
np.savetxt(f'{out_dir}/camels_aus/catch_id_aus.txt',c_array,fmt='%s')

q_mmd = pd.read_csv(f'{data_dir}/CAMELS_AUS/03_streamflow/streamflow_mmd.csv')
q_mmd.index = pd.to_datetime(q_mmd[['year','month','day']])

for i in range(len(catch_id)):
    df = pd.DataFrame(index=q_mmd.index, columns=['q_mmd'])
    df.q_mmd = q_mmd[catch_id[i]]
    df.q_mmd[df.q_mmd==-99.990000] = np.nan
    df.to_csv(f'{out_dir}/camels_aus/q_timeseries_daily/{catch_id[i]}.csv')
    
    df_y = df.groupby(pd.Grouper(freq='A')).mean()
    df_y = df_y.dropna()
    df.to_csv(f'{out_dir}/camels_aus/q_timeseries_yearly/{catch_id[i]}.csv')
    
    print(i)
    
    