"""
f_sr_calculation
-----------------
functions to calculate catchment sr using the memory method
ddd
1. sd_initial
2. sr_return_periods_minmax_rzyear
3. run_sd_calculation
4. plot_sd
5. run_sr_calculation
6. merge_sr_catchments
7. plot_sr 

"""

import glob
from pathlib import Path
import os
import numpy as np
from datetime import datetime
from datetime import timedelta
import pandas as pd
import calendar
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy
from pathos.threading import ThreadPool as Pool


## 1
def sd_initial(df, si_0, si_max, q_mean):
    """
    calculate timeseries of storage deficits

    df:       pandas dataframe, daily values for P, and Ep and date_start and date_end (defined in (2))
    si_0:     int, initial interception storage = 0
    si_max:   int, maximum interception storage = 2.5 mm
    q_mean:   df, catchment mean discharge

    returns: 
    b:        int, 1>non closing water balance, 0>closing water balance
    df:       pandas dataframe, daily values for P, Pe, Ep, Ei, Et, Sd
    """
    
    # add year if the start date is earlier than the timeseries (e.g. startdate 02-01, timeseries starts 02-28) 
    if df.index[0]>df.date_start[0]:
        df.date_start = df.date_start[0] + relativedelta(years=1)
    
    # select time period of interest
    df = df.loc[df.date_start[0]:df.date_end[0]]
    
    # add empty columns for interception storage calculation
    df.loc[:,'Si_1'] = np.nan
    df.loc[:,'Pe'] = np.nan
    df.loc[:,'Si_2'] = np.nan
    df.loc[:,'Ei'] = np.nan
    df.loc[:,'Si_3'] = np.nan
    df.loc[:,'Et'] = np.nan
    df.loc[:,'Sd'] = np.nan
    
    # convert to numpy arrays (to speed up calculations)
    p = np.array(df.p.values)
    ep = np.array(df.ep.values)
    
    si1 = np.zeros(len(df))
    pe = np.zeros(len(df))
    si2 = np.zeros(len(df))
    ei = np.zeros(len(df))
    si3 = np.zeros(len(df))
    et = np.zeros(len(df))
    sd = np.zeros(len(df))
    
    #calculate interception storage and effective precipitation for all timesteps
    for l in range(1,len(si1)):
        # first timestep l=0
        si1[0] = p[0] + si_0
        pe[0] = max(0,si1[0]-si_max)
        si2[0] = si1[0] - pe[0]
        ei[0] = min(si2[0],ep[0])
        si3[0] = si2[0] - ei[0]
    
        # timestep 1 to end
        si1[l] = p[l] + si3[l-1]
        pe[l] = max(0,si1[l]-si_max)
        si2[l] = si1[l] - pe[l]
        ei[l] = min(si2[l],ep[l])
        si3[l] = si2[l] - ei[l]
    
    #calculate Et from the catchment water balance (Et = Pe-Q)
    Pe_mean = np.mean(pe)
    EP_mean = np.mean(ep)
    Q_mean = q_mean #q_mean from other file than p and e because yearly timeseries
    Et_mean = Pe_mean - Q_mean
    
    #check if water balance is ok
    if Et_mean<0: # if this is the case, it is not possible to calculate sd
        b = 1 # wb not ok
    else:
        b = 0 # wb ok
        #calculate daily Et (EP(daily)*(Et_sum/EP_sum)) and Sd
        for l in range(1,np.size(df.index)):
            # sd for timestep 0
            et[0] = ep[0]/EP_mean * Et_mean
            sd[0] = min(0,pe[0] - et[0])

            # sd for timestep 1 - end
            et[l] = ep[l]/EP_mean * Et_mean
            sd[l] = min(0,sd[l-1]+pe[l]-et[l])

        # add numpy arrays to dataframe
        df.Si_1 = si1
        df.Si_2 = si2
        df.Si_3 = si3
        df.Pe = pe
        df.Ei = ei
        df.Sd = sd
        df.Et = et
    
    # if(df.Sd.mean()==0):
    #     df.Sd=np.nan
    
    return b, df

## 2
def run_sd_calculation(catch_id, pep_dir, q_dir, out_dir):
    """
    run calculation of storage deficits (1)
    
    catch_id:    str, catchment id
    pep_dir:     str, dir, directory of P and Ep timeseries
    q_dir:       str, dir, directory of Q timeseries
    out_dir:     str, dir, output directory
    
    returns: out:sd timeseries, stores out dataframe (Sd calculation) as csv
    """
    
    # get P Ep and Q files for catch id
    f_pep = glob.glob(f'{pep_dir}/{catch_id}*.csv')
    f_q = glob.glob(f'{q_dir}/{catch_id}*.csv')

    # read files as dataframes
    q_ts = pd.read_csv(f_q[0],index_col=0)
    q_ts.index = pd.to_datetime(q_ts.index)
    pep_ts = pd.read_csv(f_pep[0],index_col=0)
    pep_ts.index = pd.to_datetime(pep_ts.index)

    # convert to monthly dataframes
    df_monthly = pd.DataFrame(index=pd.date_range(pep_ts.index[0],pep_ts.index[-1],freq='M'), columns=['p','ep'])
    df_monthly[['p','ep']] = pep_ts[['p','ep']].groupby(pd.Grouper(freq="M")).sum()

    # calculate start hydroyear -> month after on average the wettest month
    df_monthly_mean = df_monthly.groupby([df_monthly.index.month]).mean()
    wettest_month = (df_monthly_mean['p']-df_monthly_mean['ep']).idxmax()
    hydro_year_start_month = wettest_month+1
    if hydro_year_start_month==13:
        hydro_year_start_month=1

    # find the start and end date for the sr calculation based on P, Ep, Q timeseries and hydroyear
    p_ep_start_year = pep_ts.index.year[0]
    q_start_year = int(q_ts.index[0].year)
    p_ep_end_year = pep_ts.index.year[-1]
    q_end_year = int(q_ts.index[-1].year)

    # test if timeseries have overlap -> if not don't continue the sd calculation
    if q_start_year>p_ep_end_year:
        a=1
    elif p_ep_start_year>q_end_year:
        a=1
    else:
        a=0
        start_year = max(q_start_year,p_ep_start_year)
        end_year = min(q_end_year,p_ep_end_year)
        start_date = datetime(start_year,hydro_year_start_month,1)
        end_date = datetime(end_year,hydro_year_start_month,1)
        end_date = end_date - timedelta(days=1)

        #calculate mean Q for startdate enddate timeseries
        q_mean = q_ts.loc[start_date:end_date,'Q'].mean()

        # prepare input dataframe for sd calculation
        sd_input = pd.DataFrame(index=pd.date_range(start_date,end_date,freq='d'), columns=['p','ep','date_start','date_end'])
        sd_input[['p','ep']] = pep_ts[['p','ep']]
        sd_input[['date_start','date_end']] = start_date, end_date
        si_0 = 0 #initial interception storage
        si_max = 2.5 #maximum interception storage

        # run sd calculation
        b = sd_initial(sd_input, si_0, si_max, q_mean)[0] #b==0: closing wb, b==1: non-closing wb > no sd calculation
        if b==0:      
            # save output dataframe from sd calculation
            out = sd_initial(sd_input, si_0, si_max, q_mean)[1]
            out.to_csv(f'{out_dir}/{catch_id}.csv')
            return out
        
## 3
def run_sd_calculation_parallel(
    catch_id_list=list,
    pep_dir_list=list,
    q_dir_list=list,
    out_dir_list=list,
    # threads=None
    threads=100
):
    """
    Runs function area_weighted_shapefile_rasterstats in parallel.

    catch_list:  str, list, list of catchment ids
    pep_dir_list:     str, list, list of input folders for pep forcing data
    q_dir_list:   str, list, list of folder with q timeseries
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
        run_sd_calculation,
        catch_id_list,
        pep_dir_list,
        q_dir_list,
        out_dir_list,
    )

    # return results
        
## 4
def plot_sd(catch_id, sd_dir):
    """
    plot timeseries of storage deficits for catchment catch id
    catch_id:   str, catchment id
    sd_dir:     str, dir, directory where you find the sd table from (2)
    
    returns:    None, shows figure of sd
    """
    df = pd.read_csv(f'{sd_dir}/{catch_id}.csv',index_col=0)
    df.index = pd.to_datetime(df.index)
    
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(111)
    ax.plot(df.index, df.Sd*-1)
    ax.set_ylim(300,0)
    ax.set_ylabel('storage deficit (mm)')
    ax.set_title(f'catchment {catch_id}')
            
            
## 5
def sr_return_periods_minmax_rzyear(rp_array,Sd,year_start,year_end,date_start,date_end):
    """
    calculate sr for different return periods - min max root zone year approach from Stijn??
    
    rp_array:   int, array, list of return periods
    sd:         pandas df, storage deficits
    year_start: str, start year
    year_end:   str, end year
    date_start: str, month-day start
    date_end:   str, month-day end
    
    returns:
    Sd_T:       list of storage deficit for return periods in rp_array
    
    """

    # inverse of sd
    Sd = Sd*-1
    
    # count years
    total_years = year_end - year_start
    years = range(year_start,year_end+1,1)

    # calculate annual max Sd - without iterations for hydro years
    # CHECK THIS PROCEDURE AGAIN FRANSJE
    Sd_max=[]
    Sd_maxmin = []
    for i in range(0,total_years,1):
        sd_max_i = max(Sd.loc[str(years[i])+'-'+str(date_start):str(years[i+1])+'-'+str(date_end)]) #max value
        Sd_max.append(sd_max_i) #append max deficit per year

        sd_max_ix = Sd.loc[str(years[i])+'-'+str(date_start):str(years[i+1])+'-'+str(date_end)].idxmax() #find index of max value
        sd_hystart_maxvalue = Sd.loc[str(years[i])+'-'+str(date_start):sd_max_ix] #timeseries from start hydroyear to index of max value
        min_value = min(sd_hystart_maxvalue) #find min value in timeseries before max value
        Sd_maxmin.append(sd_max_i-min_value) #append max-min sd per year

    # define root zone year
    sd_max_month = Sd.groupby(pd.Grouper(freq='M')).max() #calculate maximum sd per month
    sd_max_month_sum =  sd_max_month.groupby([sd_max_month.index.month]).sum() #sum max sd per month for full timeseries per month
    start_rz_year = sd_max_month_sum.idxmin() #define month where rz year starts
    date_start_rz_year = str(start_rz_year)+'-1'        
    if(start_rz_year==1):
        start_rz_year=13
    day_end_rz_year = calendar.monthrange(2010,start_rz_year-1)[1] #find last day of end month rz year
    date_end_rz_year = str(start_rz_year-1)+'-'+str(day_end_rz_year)

    # calculate annual max Sd - without iterations for rootzone years -> CHECK THIS APPROACH
    Sd_max_rz_year = []
    Sd_maxmin_rz_year = []
    for i in range(0,total_years,1):
        sd_max_i = max(Sd.loc[str(years[i])+'-'+str(date_start_rz_year):str(years[i+1])+'-'+str(date_end_rz_year)])
        Sd_max_rz_year.append(sd_max_i) #append max deficit per year

        sd_max_ix = Sd.loc[str(years[i])+'-'+str(date_start_rz_year):str(years[i+1])+'-'+str(date_end_rz_year)].idxmax() #find index of max value
        sd_hystart_maxvalue = Sd.loc[str(years[i])+'-'+str(date_start_rz_year):sd_max_ix] #timeseries from start rzyear to index of max value
        min_value = min(sd_hystart_maxvalue) #find min value in timeseries before max value
        Sd_maxmin_rz_year.append(sd_max_i-min_value) #append max-min sd per year
            
    # gumbel function
    def gumbel_r_mom(x):
        """
        gumbel extreme value analysis
        x:        list of max sd values per year
        returns:  loc and scale of gumbel distribution
        
        """
        scale = np.sqrt(6)/np.pi * np.std(x)
        loc = np.mean(x) - np.euler_gamma*scale
        return loc, scale    

    # calculate gumbel parameters
    loc1, scale1 = gumbel_r_mom(Sd_maxmin_rz_year)

    # find Sd value corresponding with return period
    Sd_T = []
    for i in np.arange(0,len(rp_array),1):
        p = 1-(1/rp_array[i])
        y = -np.log(-np.log(p))
        x = scale1 * y + loc1
        Sd_T.append(x)
         
    return(Sd_T)   

       
## 6
def run_sr_calculation(catch_id, rp_array, sd_dir, out_dir):
    """
    run sr calculation
    
    catch_id:   str, catchment id
    rp_array:   int, array, array of return periods
    sd_dir:     str, dir, directory with sd dataframes
    out_dir:    str, dir, output directory
    
    returns:    sr_df, dataframe with sr for catchment, stored as csv
    
    """
    # check if sd exists for catchment id
    if(os.path.exists(f'{sd_dir}/{catch_id}.csv')==True):  
        
        # read storage deficit table
        sd_table = pd.read_csv(f'{sd_dir}/{catch_id}.csv',index_col=0)
        sd_table.index = pd.to_datetime(sd_table.index)
        
        # get sd, start and end year and date from sd_table
        Sd = sd_table.Sd
        year_start = sd_table.index[0].year
        year_end = sd_table.index[-1].year
        date_start = str(sd_table.index[0].month)+'-'+str(sd_table.index[0].day)
        date_end = str(sd_table.index[-1].month)+'-'+str(sd_table.index[-1].day)
        if(date_end=='2-29'):
            date_end='2-28'
        
        # calculate sr for different return periods using (4)
        sr_T = sr_return_periods_minmax_rzyear(rp_array, Sd, year_start, year_end, date_start, date_end)
        
        # store dataframe with catchment sr values
        sr_df = pd.DataFrame(index=[catch_id], columns=rp_array)
        sr_df.loc[catch_id]=sr_T
        sr_df.to_csv(f'{out_dir}/{catch_id}.csv')
        
        return(sr_df)

## 7
def merge_sr_catchments(sr_dir,out_dir):
    """
    merge sr from individual catchments into one dataframe
    
    sr_dir:   str, dir, directory with sr csvs for all catchments
    out_dir:  str, dir, output directory for table
    
    returns: df with sr of all catchments, stores csv with sr of all catchments
    
    """
    # get all sr files
    all_files = glob.glob(f'{sr_dir}/*.csv')
    
    # make empty list
    li = []

    # loop over sr files
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)
    
    # merge files and store as csv
    frame = pd.concat(li, axis=0)
    frame = frame.rename(columns={'Unnamed: 0':'catch_id'})
    frame.index = frame['catch_id']
    frame = frame.drop(columns={'catch_id'})
    frame.to_csv(f'{out_dir}/sr_all_catchments.csv')
    
    return frame
    
## 7
def plot_sr(shp_file, sr_file, rp):
    """
    plot sr estimates in map
    
    shp_file:   str, file, shapefile of catchments
    sr_file:    str, file, csv file from (6) with catchment sr values
    rp:         int, value for return period
    
    returns: None, creates map of sr estimates
    
    """
    # read sr file
    df = pd.read_csv(sr_file,index_col=0)
    
    # read shapefile
    sh = gpd.read_file(shp_file,index_col=0)
    sh.index = sh.catch_id
    sh = sh.drop(columns=['catch_id'])
    
    # add sr values to geodataframe with shapes
    sh['sr'] = df[str(rp)]
    
    # get catchment centroid instead of shape as geometry -> ignore warning about CRS
    sh['centroid'] = sh.centroid
    sh = sh.drop(columns='geometry')
    sh = sh.rename(columns={'centroid':'geometry'})

    # make plot
    fig = plt.figure(figsize=(12,12))
    cm = plt.cm.get_cmap('jet')
    ax = plt.axes(projection=cartopy.crs.PlateCarree())
    ax.coastlines(linestyle=':')
    ax.set_xlim(-180,180)
    ax.set_ylim(-70,90)
    lvls = np.linspace(0,800,17)
    # plot centroid points, color based on Sr value
    pl = sh.plot(column='sr',ax=ax,markersize=20, cmap=cm,
               k=10,vmin=20,vmax=600,
               legend=True,
               legend_kwds={'label': "Sr(mm)", 'orientation': "horizontal", 'pad':0.02,'ticks':lvls})
    ax.set_title(f'Sr catchments, RP={rp}',size=20)