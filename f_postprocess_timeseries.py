import glob
from pathlib import Path
import os

import numpy as np
from datetime import datetime
from datetime import timedelta
import pandas as pd


def process_forcing_timeseries(catch_id,fol_in,fol_out,var):
    d = pd.DataFrame()

    for j in var:
        l = glob.glob(fol_in + f"*/{catch_id}*{j}*.csv")
        if len(l)==0:
            continue
        li=[]

        for filename in l:
            df = pd.read_csv(filename, index_col=1, header=0)
            df = df.drop(columns=['Unnamed: 0'])
            df.index = pd.to_datetime(df.index)
            li.append(df)
        frame = pd.concat(li, axis=0, ignore_index=False)
        col=frame.columns
        y_start,y_end = frame.index[0].year, frame.index[-1].year
        d[col] = frame
        d = d.rename(columns={col.values[0]:f'{j}'})

        # get daily
        if not os.path.exists(f'{fol_out}/daily'):
             os.makedirs(f'{fol_out}/daily')
        d.to_csv(f'{fol_out}/daily/{catch_id}_{y_start}_{y_end}.csv')
    
        # get monthly
        if not os.path.exists(f'{fol_out}/monthly'):
             os.makedirs(f'{fol_out}/monthly')
        df_m = d.groupby(pd.Grouper(freq='M')).mean()
        y_start,y_end = df_m.index[0].year, df_m.index[-1].year
        df_m.to_csv(f'{fol_out}/monthly/{catch_id}_{y_start}_{y_end}.csv')    

        # get climatology
        if not os.path.exists(f'{fol_out}/climatology'):
             os.makedirs(f'{fol_out}/climatology')
        df_m = df_m.groupby([df_m.index.month]).mean()
        df_m.to_csv(f'{fol_out}/climatology/{catch_id}_{y_start}_{y_end}.csv')
        
        # get yearly
        if not os.path.exists(f'{fol_out}/yearly'):
             os.makedirs(f'{fol_out}/yearly')
        df_y = d.groupby(pd.Grouper(freq='Y')).mean()
        y_start,y_end = df_y.index[0].year, df_y.index[-1].year
        df_y.to_csv(f'{fol_out}/yearly/{catch_id}_{y_start}_{y_end}.csv')
        
        # get mean
        if not os.path.exists(f'{fol_out}/mean'):
             os.makedirs(f'{fol_out}/mean')
        dm = d.mean()
        dm.to_csv(f'{fol_out}/mean/{catch_id}_{y_start}_{y_end}.csv')
    