#%% LOAD PACKAGES
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


#%% SR CALCULATION
# INPUT
# sd_input: dataframe with daily catchment values for P, Ep, Q
# Si_0: initial interception storage = 0
# Si_max: maximum interception storage = 2.5mm
# date_start, date_end: start and end 'month-day' of time-series (depending on hydro-year)
# year_start, year_end: start and end year of time-series

# OUTPUT
# catchment: pandas dataframe with daily catchment values for P, Ep, Q, Pe, Et and Sd (based on initial Et estimate)

# SD
def sd_initial(sd_input, Si_0, Si_max, q_mean):

    #read csv file for catchment of interest
    # catchment = pd.read_csv(filename, sep=',', skiprows=0, index_col=0, skipinitialspace=True)
    # catchment.index = pd.to_datetime(catchment.index)
    sd_input = sd_input.loc[sd_input.date_start[0]:sd_input.date_end[0]]
    
    # soms is de start date eg 02-01 maar begint de timeseries pas 02-28: dan een jaar erbij optellen
    if sd_input.index[0]>sd_input.date_start[0]:
        sd_input.date_start = sd_input.date_start[0] + relativedelta(years=1)
    
    sd_input = sd_input.loc[sd_input.date_start[0]:sd_input.date_end[0]]
    
    # add columns for interception storage calculation
    sd_input['Si_1'] = np.nan
    sd_input['Pe'] = np.nan
    sd_input['Si_2'] = np.nan
    sd_input['Ei'] = np.nan
    sd_input['Si_3'] = np.nan
    sd_input['Et'] = np.nan
    sd_input['Sd'] = np.nan
    
    # convert to numpy arrays
    p = np.array(sd_input.P.values)
    # q = np.array(sd_input.Q.values)
    ep = np.array(sd_input.Ep.values)
    
    si1 = np.zeros(len(sd_input))
    pe = np.zeros(len(sd_input))
    si2 = np.zeros(len(sd_input))
    ei = np.zeros(len(sd_input))
    si3 = np.zeros(len(sd_input))
    et = np.zeros(len(sd_input))
    sd = np.zeros(len(sd_input))
    
    #calculate interception storage and effective precipitation for all timesteps
    for l in range(1,len(si1)):
        si1[0] = p[0] + Si_0
        pe[0] = max(0,si1[0]-Si_max)
        si2[0] = si1[0] - pe[0]
        ei[0] = min(si2[0],ep[0])
        si3[0] = si2[0] - ei[0]
    
        si1[l] = p[l] + si3[l-1]
        pe[l] = max(0,si1[l]-Si_max)
        si2[l] = si1[l] - pe[l]
        ei[l] = min(si2[l],ep[l])
        si3[l] = si2[l] - ei[l]
    
    #water balance Et calculation (Et = Pe-Q)
    Pe_mean = np.mean(pe)
    EP_mean = np.mean(ep)
    Q_mean = q_mean
    Et_mean = Pe_mean - Q_mean
    
    if Et_mean<0: 
        b = 1
    else:
        b = 0
        #calculate daily Et (EP(daily)*(Et_sum/EP_sum)) and Sd
        for l in range(1,np.size(sd_input.index)):
            #if Pe < Q -> kan niet!            
            et[0] = ep[0]/EP_mean * Et_mean
            sd[0] = min(0,pe[0] - et[0])

            et[l] = ep[l]/EP_mean * Et_mean
            sd[l] = min(0,sd[l-1]+pe[l]-et[l])

        sd_input.Si_1 = si1
        sd_input.Si_2 = si2
        sd_input.Si_3 = si3
        sd_input.Pe = pe
        sd_input.Ei = ei
        sd_input.Sd = sd
        sd_input.Et = et
    
    # if(sd_input.Sd.mean()==0):
    #     sd_input.Sd=np.nan
    
    return b, sd_input


#%% function 2: Sr calculation based on return periods - INCLUDE MIN MAX APPROACH LIKE STIJN

# INPUT
# T: array of return periods of interest T=[2,5,10,15,20,30,40]
# Sd: dataframe of Sd calculated in sd_iterations function
# date_start, date_end: start and end 'month-day' of time-series (depending on hydro-year)
# year_start, year_end: start and end year of time-series
# it: amount of iterations
    
# OUTPUT
# Sd_T: storage deficits corresponding with return periods T
    

def sr_return_periods_minmax_rzyear(RP,Sd,year_start,year_end,date_start,date_end):

    # for j in range(len(RP)):
    Sd = Sd*-1
    total_years = year_end - year_start
    years = range(year_start,year_end+1,1)

    # calculate annual max Sd - without iterations for hydro years
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
        scale = np.sqrt(6)/np.pi * np.std(x)
        loc = np.mean(x) - np.euler_gamma*scale
        return loc, scale    

    loc1, scale1 = gumbel_r_mom(Sd_maxmin_rz_year)

    # find Sd value corresponding with return period
    Sd_T = []
    for i in np.arange(0,len(RP),1):
        p = 1-(1/RP[i])
        y = -np.log(-np.log(p))
        x = scale1 * y + loc1
        Sd_T.append(x)
         
    return(Sd_T)   



def run_sd_calculation(catch_id, pep_dir,q_dir,out_dir,Si_max):
    
    f_pep = glob.glob(f'{pep_dir}/{catch_id}*.csv')
    f_q = glob.glob(f'{q_dir}/{catch_id}*.csv')

    q_ts = pd.read_csv(f_q[0],index_col=0)
    q_ts.index = pd.to_datetime(q_ts.index)
    pep_ts = pd.read_csv(f_pep[0],index_col=0)
    pep_ts.index = pd.to_datetime(pep_ts.index)

    df_monthly = pd.DataFrame(index=pd.date_range(pep_ts.index[0],pep_ts.index[-1],freq='M'), columns=['P','Ep'])
    df_monthly[['P','Ep']] = pep_ts[['P','Ep']].groupby(pd.Grouper(freq="M")).sum()

    # calculate start hydroyear
    df_monthly_mean = df_monthly.groupby([df_monthly.index.month]).mean()
    wettest_month = (df_monthly_mean['P']-df_monthly_mean['Ep']).idxmax()
    hydro_year_start_month = wettest_month+1
    if hydro_year_start_month==13:
        hydro_year_start_month=1

    p_ep_start_year = pep_ts.index.year[0]
    q_start_year = int(q_ts.index[0].year)
    p_ep_end_year = pep_ts.index.year[-1]
    q_end_year = int(q_ts.index[-1].year)
    start_year = max(q_start_year,p_ep_start_year)
    end_year = min(q_end_year,p_ep_end_year)
    start_date = datetime(start_year,hydro_year_start_month,1)
    end_date = datetime(end_year,hydro_year_start_month,1)
    end_date = end_date - timedelta(days=1)

    # qmean
    q_mean = q_ts.loc[start_date:end_date,'Q'].mean()

    # p ep data
    sd_input = pd.DataFrame(index=pd.date_range(start_date,end_date,freq='d'), columns=['P','Ep','date_start','date_end'])
    sd_input[['P','Ep']] = pep_ts[['P','Ep']]
    sd_input[['date_start','date_end']] = start_date, end_date
    Si_0 = 0
    Si_max = 2.5
    b = sd_initial(sd_input, Si_0, Si_max, q_mean)[0]
    if b==0:      
        out = sd_initial(sd_input, Si_0, Si_max, q_mean)[1]
        out.to_csv(f'{out_dir}/{catch_id}.csv')
        
        

def run_sr_calculation(catch_id,RP,sd_dir,out_dir):
    if(os.path.exists(f'{sd_dir}/{catch_id}.csv')==True):  
        out = pd.read_csv(f'{sd_dir}/{catch_id}.csv',index_col=0)
        out.index = pd.to_datetime(out.index)
        # run SR calculation based on intial Sd calculation (without iterations)
        Sd = out.Sd
        year_start = out.index[0].year
        year_end = out.index[-1].year
        date_start = str(out.index[0].month)+'-'+str(out.index[0].day)
        date_end = str(out.index[-1].month)+'-'+str(out.index[-1].day)
        if(date_end=='2-29'):
            date_end='2-28'
        sr_T = sr_return_periods_minmax_rzyear(RP,Sd,year_start,year_end,date_start,date_end)
        sr_df = pd.DataFrame(index=[catch_id], columns=RP)
        sr_df.loc[catch_id]=sr_T
        sr_df.to_csv(f'{out_dir}/{catch_id}.csv')
        return(sr_df)
    
def merge_sr_catchments(sr_dir,out_dir):
    all_files = glob.glob(f'{sr_dir}/*.csv')
    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)
    frame = pd.concat(li, axis=0)
    frame = frame.rename(columns={'Unnamed: 0':'catch_id'})
    frame.index = frame['catch_id']
    frame = frame.drop(columns={'catch_id'})
    frame.to_csv(f'{out_dir}/sr_all_catchments.csv')
    
def plot_sr(shp_file,sr_file):
    df = pd.read_csv(sr_file,index_col=0)
    
    sh = gpd.read_file(shp_file,index_col=0)
    sh.index = sh.catch_id
    sh = sh.drop(columns=['catch_id'])
    sh['sr'] = df['20']
    sh['centroid'] = sh.centroid
    sh = sh.drop(columns='geometry')
    sh = sh.rename(columns={'centroid':'geometry'})

    fig = plt.figure(figsize=(12,12))
    cm = plt.cm.get_cmap('jet')
    ax = plt.axes(projection=cartopy.crs.PlateCarree())
    ax.coastlines(linestyle=':')
    ax.set_xlim(-180,180)
    ax.set_ylim(-70,90)
    lvls = np.linspace(0,800,17)
    pl = sh.plot(column='sr',ax=ax,markersize=20, cmap=cm,
               k=10,vmin=20,vmax=600,
               legend=True,
               legend_kwds={'label': "Sr(mm)", 'orientation': "horizontal", 'pad':0.02,'ticks':lvls})
    ax.set_title(f'Sr catchments',size=20)