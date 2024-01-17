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
def sd_initial(df, si_0, si_max, q_mean,s):
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
    df.loc[:,'se'] = np.nan

    # convert to numpy arrays (to speed up calculations)
    p = np.array(df.p.values)
    ep = np.array(df.ep.values)

    if (s==1):
        pl = np.array(df.pl.values)
        pm = np.array(df.pm.values)

    si1 = np.zeros(len(df))
    pe = np.zeros(len(df))
    pef = np.zeros(len(df))
    si2 = np.zeros(len(df))
    ei = np.zeros(len(df))
    si3 = np.zeros(len(df))
    et = np.zeros(len(df))
    sd = np.zeros(len(df))
    se = np.zeros(len(df))

    if (s==1): # snow
        #calculate interception storage and effective precipitation for all timesteps
        for l in range(1,len(si1)):
            # first timestep l=0
            si1[0] = pl[0] + si_0
            pef[0] = max(0,si1[0]-si_max)
            si2[0] = si1[0] - pef[0]
            ei[0] = min(si2[0],ep[0])
            si3[0] = max(0,si2[0]-ei[0])

            pe[0] = pef[0]+pm[0]

            # timestep 1 to end
            si1[l] = pl[l] + si3[l-1]
            pef[l] = max(0,si1[l]-si_max)
            si2[l] = si1[l] - pef[l]
            ei[l] = min(si2[l],ep[l])
            si3[l] = max(0,si2[l]-ei[l])

            pe[l] = pef[l]+pm[l]

    else: # no snow
        #calculate interception storage and effective precipitation for all timesteps
        for l in range(1,len(si1)):
            # first timestep l=0
            si1[0] = p[0] + si_0
            pe[0] = max(0,si1[0]-si_max)
            si2[0] = si1[0] - pe[0]
            ei[0] = min(si2[0],ep[0])
            si3[0] = max(0,si2[0]-ei[0])

            # timestep 1 to end
            si1[l] = p[l] + si3[l-1]
            pe[l] = max(0,si1[l]-si_max)
            si2[l] = si1[l] - pe[l]
            ei[l] = min(si2[l],ep[l])
            si3[l] = max(0,si2[l]-ei[l])

    #calculate Et from the catchment water balance (Et = Pe-Q)
    Pe_mean = np.mean(pe)
    EP_mean = np.mean(ep)
    Q_mean = q_mean #q_mean from other file than p and e because yearly timeseries
    Et_mean = Pe_mean - Q_mean
    
    #check if water balance is ok
    if Et_mean<0: # if this is the case, it is not possible to calculate sd
        b = 1 # wb not ok
        sd[:] = np.nan
        et[:]=np.nan
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

            if (sd[l]==0):
                se[l] = pe[l]-et[l]
    
    # add numpy arrays to dataframe
    df.Si_1 = si1
    df.Si_2 = si2
    df.Si_3 = si3
    df.Pe = pe
    df.Ei = ei
    df.Sd = sd
    df.Et = et
    df.se = se
                
    return b, df

## 2
def run_sd_calculation(catch_id, pep_dir, q_dir, out_dir,snow_id_list,snow_dir,work_dir,ir_case):
    """
    run calculation of storage deficits (1)
    
    catch_id:    str, catchment id
    pep_dir:     str, dir, directory of P and Ep timeseries
    q_dir:       str, dir, directory of Q timeseries
    out_dir:     str, dir, output directory
    
    returns: out:sd timeseries, stores out dataframe (Sd calculation) as csv
    """
    
    if catch_id in snow_id_list:
        s = 1 # snow is yes
        f_pep = glob.glob(f'{snow_dir}/{catch_id}*.csv')

    else:
        s = 0 # snow is no
        # get P Ep and Q files for catch id
        f_pep = glob.glob(f'{pep_dir}/{catch_id}*.csv')
        
    cc = pd.read_csv(f'{work_dir}/output/catchment_characteristics/gswp-p_gleam-ep_gswp-t/landscape/{catch_id}.csv',index_col=0)
    ir_area = cc.ia.values

    # read q df
    f_q = glob.glob(f'{q_dir}/{catch_id}*.csv')

    # read files as dataframes
    q_ts = pd.read_csv(f_q[0],index_col=0)
    q_ts.index = pd.to_datetime(q_ts.index)
    pep_ts = pd.read_csv(f_pep[0],index_col=0)
    pep_ts.index = pd.to_datetime(pep_ts.index)

    if (s==1): #snow
        # convert to monthly dataframes
        df_monthly = pd.DataFrame(index=pd.date_range(pep_ts.index[0],pep_ts.index[-1],freq='M'), columns=['p','ep','ps','pm','pl'])
        df_monthly[['p','ep','ps','pm','pl']] = pep_ts[['p','ep','ps','pm','pl']].groupby(pd.Grouper(freq="M")).sum()

        # calculate start hydroyear -> month after on average the wettest month
        df_monthly_mean = df_monthly.groupby([df_monthly.index.month]).mean()
        df_monthly_mean['liq'] = df_monthly_mean['pm'] + df_monthly_mean['pl']
        wettest_month = (df_monthly_mean['liq']-df_monthly_mean['ep']).idxmax()
        hydro_year_start_month = wettest_month+1
        if hydro_year_start_month==13:
            hydro_year_start_month=1

    else: # no snow
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

        if (s==1): # snow
            # prepare input dataframe for sd calculation
            sd_input = pd.DataFrame(index=pd.date_range(start_date,end_date,freq='d'), columns=['p','ep','ps','pm','pl','date_start','date_end'])
            sd_input[['p','ep','ps','pm','pl']] = pep_ts[['p','ep','ps','pm','pl']]
            sd_input[['date_start','date_end']] = start_date, end_date
            si_0 = 0 #initial interception storage
            si_max = 2.5 #maximum interception storage

        else: # no snow
            # prepare input dataframe for sd calculation
            sd_input = pd.DataFrame(index=pd.date_range(start_date,end_date,freq='d'), columns=['p','ep','date_start','date_end'])
            sd_input[['p','ep']] = pep_ts[['p','ep']]
            sd_input[['date_start','date_end']] = start_date, end_date
            si_0 = 0 #initial interception storage
            si_max = 2.5 #maximum interception storage

        # run sd calculation
        b = sd_initial(sd_input, si_0, si_max, q_mean,s)[0] #b==0: closing wb, b==1: non-closing wb > no sd calculation
        if b==0:      
            # save output dataframe from sd calculation
            out = sd_initial(sd_input, si_0, si_max, q_mean,s)[1]
            if (ir_case=='ni'):
                out.to_csv(f'{out_dir}/no_irri/sd/{catch_id}.csv')
            else:
                if ir_area>0:
                    ir = irrigation_sd(out,catch_id,work_dir,ir_case)
                    out = ir[0] 
                    se_out = ir[1]
                    f = ir[2]
                    if (ir_case=='iaf'):
                        # se_out.to_csv(f'{out_dir}/irri/f0.9ia/se/{catch_id}_f0.9ia.csv')
                        # out.to_csv(f'{out_dir}/irri/f0.9ia/sd/{catch_id}_f0.9ia.csv')
                        out.to_csv(f'{out_dir}/sd/{catch_id}.csv')
                    if (ir_case=='iwu'):
                        # se_out.to_csv(f'{out_dir}/irri/fiwu2/se/{catch_id}_fiwu2.csv')
                        out.to_csv(f'{out_dir}/irri/fiwu2/sd/{catch_id}_fiwu2.csv')
                else: 
                    if (ir_case=='iaf'):
                        # out.to_csv(f'{out_dir}/irri/f0.9ia/sd/{catch_id}_f0.9ia.csv')
                        out.to_csv(f'{out_dir}/sd/{catch_id}.csv')
                    if (ir_case=='iwu'):
                        out.to_csv(f'{out_dir}/irri/fiwu2/sd/{catch_id}_fiwu2.csv')

            return out
        
        
def irrigation_sd(df,catch_id,work_dir,ir_case):
    split_dates = [] # dates of annual maximum deficit
    start_date = df.index[0]
    split_dates.append(start_date) # add start date to split_dates

    se_l = [] # accumulated Se available
    se_used = [] # used Se
    ldd_l = [] # last day of deficit
    lde_l = [] # first day of deficit
    days_l = [] # amount of deficit days
    f_ar = [] # f values used
    
    s=df
    s2 = df #make new dataframe, copy of s which is output of initial sd calculation
    s2['p_total'] = np.zeros(len(s2)) # add zeros column of total p = pe+irri
    s2['p_irri'] = np.zeros(len(s2)) # add zeros column of irri
    s2['sd2']=df.Sd # set sd2 column intially equal to sd
    s2['se2']=df.se # set se2 column intially equal to se
    s2['se_cum'] = np.zeros(len(s2)) # set zeros column for cumulative se

    iwu = pd.read_csv(f'{work_dir}/output/irrigation/processed2/monthly_mean/{catch_id}.csv',index_col=0) # read iwu data
    iwu_mean = iwu.mean().values[0]*365 # annual mean IWU
    ir2 = pd.read_csv(f'{work_dir}/data/irrigated_area/output/combined_ia.csv',index_col=0) #read ia data
    ir_area = ir2.loc[catch_id].hi # irrigated area fraction

    ldd_l.append(df.index[0]) # add first day of timeseries to ldd_l list

    years=len(np.unique(s.index.year)) #count years
    # years = 5
    for i in range(years): # loop over years
        ss = s2.iloc[i*365:(i*365)+365] #select 1 year from s2 dataframe
        min_date = ss[ss.Sd==ss.Sd.min()].index.values[0] #select date where Sd minimizes
        split_dates.append(min_date) #append min date to split_dates list
        sp = s2.loc[split_dates[i]:split_dates[i+1]] # split timeseries from min-date year i until year i+1

        # find start + end dates of deficit and surplus periods
        if (i==0): # for first year
            sss = ss.loc[:min_date] # select timeseries until minimum sd date
            if (len(sss.se[sss.se>0])>0): 
                lde = sss.se[sss.se>0].index[-1] # find last day of se>0 before minimum deficit (=start day deficit)
            else:
                lde = sss.index[0]   

            ldd_l.append(lde) # append to list

            sss = ss.loc[min_date:] # select timeseries from minimum sd date to end
            if (len(sss.Sd[sss.Sd==0])>0):
                ldd = sss[sss.Sd==0].index[0] # find first day of zero deficit after minimum deficit (=end day deficit)
            else:
                ldd = sss.Sd.index[-1]
            ldd_l.append(ldd) # append to list
        else: # for other years than i=0
            sss = ss.loc[:min_date] # select timeseries until minimum sd date
            if (len(sss.se[sss.se>0])>0):
                lde = sss.se[sss.se>0].index[-1] # find last day of se>0 before minimum deficit (=start day deficit)
            else:
                lde = sss.index[0]   
            ldd_l.append(lde)

            # lde = ldd_l[i]
            sss = ss.loc[min_date:]  # select timeseries from minimum sd date to end
            if (len(sss.Sd[sss.Sd==0])>0):
                ldd = sss[sss.Sd==0].index[0] # find first day of zero deficit after minimum deficit (=end day deficit)
            else:
                ldd = sss.Sd.index[-1]
            # lde_l.append(lde)
            ldd_l.append(ldd)
        
        # get length of irrigation timeseries
        if (i==0): # first year
            dd = ss.loc[ldd_l[i+1]+timedelta(days=1):ldd_l[i+2]] # set irrigation period from start deficit until end deficit
        else: # other years than i=0
            dd = ss.loc[ldd_l[2*i+1]+timedelta(days=1):ldd_l[2*i+2]] # select from start of deficit until end of deficit period
        days = len(dd) # length of deficit period = length of irrigation period
        days_l.append(days) # append deficit days to days_l list

        # accumulate Se for accumulation period
        dfse2 = pd.DataFrame(index=ss.index,columns=['se','se_cum']) # make new dataframe for year i
        dfse2['se'] = ss['se2'] # set se to se2 from ss dataset
        dfse2['se_cum'] = np.zeros(len(ss)) # make zeros se_cum column

        if(i==0): # for first year
            if (len(dd)>0): # all years except last year
                cum = dfse2['se'].loc[ldd_l[0]:ldd_l[1]].cumsum() # cumsum of se from start timeseries to start deficit
                dfse2.se_cum.loc[ldd_l[0]:ldd_l[1]] = cum # add cumsum to dataframe

                dfse2.se_cum.loc[ldd_l[1]:ldd_l[2]] = np.zeros(len(dfse2.se_cum.loc[ldd_l[1]:ldd_l[2]])) # se_cum for deficit period is zero

                cum = dfse2['se'].loc[ldd_l[2]:].cumsum() # cumsum of se from end deficit to end timeseries
                dfse2.se_cum.loc[ldd_l[2]:] = cum # add cumsum to dataframe       

                s2.se_cum.iloc[i*365:(i*365)+365] = dfse2.se_cum # add cumsum to s2 dataframe
        else: # other years than i=0
            if (len(dd)>0): # all years except last year
                s_index = dfse2.index[0]-timedelta(days=1) # get last day of previous year
                ss_s = s2.se_cum.loc[s_index] # get se_cum value for last day of previous year
                cum = dfse2['se'].loc[:ldd_l[2*i+1]].cumsum() + ss_s # cumsum of se from start year to start deficit + se_cum value from last day of previous year
                dfse2.se_cum.loc[:ldd_l[2*i+1]] = cum # add to dataframe

                dfse2.se_cum.loc[ldd_l[2*i+1]:ldd_l[2*i+2]] = np.zeros(len(dfse2.se_cum.loc[ldd_l[2*i+1]:ldd_l[2*i+2]])) # se_cum for deficit period is zero

                cum = dfse2['se'].loc[ldd_l[2*i+2]:].cumsum() # cumsum of se from end deficit to end timeseries
                dfse2.se_cum.loc[ldd_l[2*i+2]:] = cum # add cumsum to dataframe         

                s2.se_cum.iloc[i*365:(i*365)+365] = dfse2.se_cum # add cumsum to s2 dataframe

        if (len(dd)>0):
            if (i==0):
                if (ldd_l[1]-timedelta(days=1)) in s2.index: # if deficit starts directly, the day before deficit does not exist in the first year dataframe
                    se_sum = s2.se_cum.loc[ldd_l[1]-timedelta(days=1)] # total se is value of se_cum last day before deficit period
                else:
                    se_sum=0
            else:
                se_sum = s2.se_cum.loc[ldd_l[2*i+1]-timedelta(days=1)] # total se is value of se_cum last day before deficit period  
        else:
            se_sum=0


        # DEFINE f-VARIABLE
        if (ir_case=='iaf'):
            # f based on fixed factor and irrigated area fraction
            f = 0.9
            f2 = min(f*ir_area, 1) 
            f_ar.append(f2) 
            if (days>0):
                irri = f2 * se_sum/days # calculate the irrigation fraction per day, equally distributed over the deficit period
                se_used.append(f2*se_sum)
            else:
                irri=0
                se_used.append(0)
            se_l.append(se_sum) # append se_sum to se_l list
        
        if (ir_case=='iwu'):
            # f based on IWU directly
            if (se_sum>0):
                f = iwu_mean/se_sum
            else: 
                f=0
            if (f>1):
                f=1
            f2=f
            f_ar.append(f2)       
            if (days>0):
                irri = f2 * se_sum/days # calculate the irrigation rates per day, equally distributed over the deficit period
                se_used.append(f2*se_sum) # used se is f2*se_sum
            else:
                irri=0
                se_used.append(0)
            se_l.append(se_sum) # append se_sum to se_l list

        # CALCULATE IRRIGATION
        p_irri = dd['Pe'] + irri # precipitation+irrigation for deficit period
        dd['irri'] = irri

        dfp2 = pd.DataFrame(index=ss.index, columns=['p_irri']) # make new p-irri dataframe for year i
        dfp2.p_irri = p_irri
        dfp2 = dfp2.fillna(-1) # if no irrigation, set to nan
        dfp2.p_irri.loc[dfp2[dfp2.p_irri<0].index] = ss.Pe.loc[dfp2[dfp2.p_irri<0].p_irri.index] # set nan values (set to -1) in dfp2 (no irrigation) to original p values
        ss['p_irri'] = dd['irri'] # only irrigation
        ss['p_irri'] = ss['p_irri'].fillna(0)
        ss['p_total'] = dfp2['p_irri'] # sum of Pe and irrigation
        s2.p_total.iloc[i*365:(i*365)+365] = ss['p_total'] #update p_total = irri+pe in s2
        s2.p_irri.iloc[i*365:(i*365)+365] = ss['p_irri'] #update p_total = irri+pe in s2

        #update sd and se in full timeseries from year -> end
        for l in range(i*365,len(s2)):
            if (i==0)&(l==0):
                s2['sd2'].iloc[l]=0
            else:
                s2['sd2'].iloc[l] = min(0,s2['sd2'].iloc[l-1]+s2['p_total'][l]-s2['Et'][l])
            if (s2['sd2'].iloc[l]==0):
                s2['se2'].iloc[l] = s2.p_total[l]-s2.Et[l]

    # make irrigation dataframe
    df_se = pd.DataFrame(index=range(len(se_used)), columns=['start_date_irri','end_date_irri','se','f','se_used','iwu_mean','days_irri'])
    df_se['se'] = se_l
    df_se['se_used'] = se_used
    df_se['f'] = f_ar
    df_se['start_date_irri'] = ldd_l[1::2] #odd values
    df_se['end_date_irri'] = ldd_l[0::2][1:]
    df_se['days_irri'] = days_l
    df_se['iwu_mean'] = [iwu_mean] * len(df_se.index)
    
    return(s2, df_se, f)

        
def run_sd_calculation_parallel(
    catch_id_list=list,
    pep_dir_list=list,
    q_dir_list=list,
    out_dir_list=list,
    snow_id_list=list,
    snow_dir_list=list,
    work_dir_list=list,
    ir_case_list=list,
    # threads=None
    threads=200
):
    """
    Runs function area_weighted_shapefile_rasterstats in parallel.

    catch_list:  str, list, list of catchment ids
    pep_dir_list:     str, list, list of input folders for pep forcing data
    q_dir_list:   str, list, list of folder with q timeseries
    output_dir_list: str, list, list of output directories
    snow_id_list: str,list, list of catchments with snow
    snow_dir_list: str, list, list of snow timeseries directory
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
        snow_id_list,
        snow_dir_list,
        work_dir_list,
        ir_case_list
    )
                   
            
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

    # calculate annual max-min Sd for hydro years
    Sd_max=[]
    Sd_maxmin = []
    if (str(date_start)=='1-1'):
        for i in range(0,total_years+1,1):
            sd_max_i = max(Sd.loc[str(years[i])+'-'+str(date_start):str(years[i])+'-'+str(date_end)]) #max value
            Sd_max.append(sd_max_i) #append max deficit per year
            # print(str(years[i])+'-'+str(date_start), str(years[i])+'-'+str(date_end))

            sd_max_ix = Sd.loc[str(years[i])+'-'+str(date_start):str(years[i])+'-'+str(date_end)].idxmax() #find index of max value
            sd_hystart_maxvalue = Sd.loc[str(years[i])+'-'+str(date_start):sd_max_ix] #timeseries from start hydroyear to index of max value
            min_value = min(sd_hystart_maxvalue) #find min value in timeseries before max value
            Sd_maxmin.append(sd_max_i-min_value) #append max-min sd per year
    else:
        for i in range(0,total_years,1):
            sd_max_i = max(Sd.loc[str(years[i])+'-'+str(date_start):str(years[i+1])+'-'+str(date_end)]) #max value
            Sd_max.append(sd_max_i) #append max deficit per year
            # print(str(years[i])+'-'+str(date_start), str(years[i+1])+'-'+str(date_end))

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
    
    if ((start_rz_year) < (Sd.index[0].month)):
        year_start = year_start+1
    total_years = year_end - year_start
    years = range(year_start,year_end+1,1)

    # calculate annual max Sd - without iterations for rootzone years 
    Sd_max_rz_year = []
    Sd_maxmin_rz_year = []
    if (str(date_start_rz_year)=='1-1'):
        for i in range(0,total_years+1,1):
            sd_max_i = max(Sd.loc[str(years[i])+'-'+str(date_start_rz_year):str(years[i])+'-'+str(date_end_rz_year)])
            Sd_max_rz_year.append(sd_max_i) #append max deficit per year
            # print(str(years[i])+'-'+str(date_start_rz_year), str(years[i])+'-'+str(date_end_rz_year))

            sd_max_ix = Sd.loc[str(years[i])+'-'+str(date_start_rz_year):str(years[i])+'-'+str(date_end_rz_year)].idxmax() #find index of max value
            sd_hystart_maxvalue = Sd.loc[str(years[i])+'-'+str(date_start_rz_year):sd_max_ix] #timeseries from start rzyear to index of max value
            min_value = min(sd_hystart_maxvalue) #find min value in timeseries before max value
            Sd_maxmin_rz_year.append(sd_max_i-min_value) #append max-min sd per year
    else:
        for i in range(0,total_years,1):
            sd_max_i = max(Sd.loc[str(years[i])+'-'+str(date_start_rz_year):str(years[i+1])+'-'+str(date_end_rz_year)])
            Sd_max_rz_year.append(sd_max_i) #append max deficit per year
            # print(str(years[i])+'-'+str(date_start_rz_year), str(years[i+1])+'-'+str(date_end_rz_year))

            sd_max_ix = Sd.loc[str(years[i])+'-'+str(date_start_rz_year):str(years[i+1])+'-'+str(date_end_rz_year)].idxmax() #find index of max value
            sd_hystart_maxvalue = Sd.loc[str(years[i])+'-'+str(date_start_rz_year):sd_max_ix] #timeseries from start rzyear to index of max value
            min_value = min(sd_hystart_maxvalue) #find min value in timeseries before max value
            Sd_maxmin_rz_year.append(sd_max_i-min_value) #append max-min sd per year        

    return(Sd_maxmin_rz_year)
    # return(Sd_maxmin)



def gumbel(Sd_maxmin):
    m = len(Sd_maxmin)
    df = pd.DataFrame(index=np.arange(1,len(Sd_maxmin)+1,1),columns=['sd','i','p','q','y'])
    df['sd'] = Sd_maxmin
    df = df.sort_values('sd',ascending=False)
    df['i'] = np.arange(1,len(Sd_maxmin)+1,1)
    df['p'] = df['i']/(m+1)
    df['q'] = 1-df['p']
    df['y'] = -np.log(-np.log(df['q']))

    # exercise: compute Gumbel parameters (name them sigma and mu)
    s_R = df['sd'].std()
    s_y = df['y'].std()
    y_gem = df['y'].mean()
    R_max_gem = df['sd'].mean()
    sigma = s_R / s_y
    mu = R_max_gem - s_R * (y_gem / s_y)

    # Here we define the return periods and we inspect the difference between the two return periods
    T_interest = np.asarray([1.5,2,3,5,10,20,30,40,50,60,70,80])
    T_a_interest = 1 / (1-np.exp(-1/T_interest))

    # real return period for the observed annual maxima
    T_a = 1 / df.p
    df.loc[:,'T_a'] = T_a
    T = -1 / np.log(1 - 1/df['T_a'].values)
    df.loc[:,'T'] = T

    # exercise, Gumbel estimate for return period array T_interest
    gumbel_estimate = sigma * (-np.log(-np.log(1-(1/T_interest)))) + mu
    
    return(df,T_interest,gumbel_estimate)

## 6
def run_sr_calculation(catch_id, rp_array, sd_dir, out_dir,ir_case):
    """
    run sr calculation
    
    catch_id:   str, catchment id
    rp_array:   int, array, array of return periods
    sd_dir:     str, dir, directory with sd dataframes
    out_dir:    str, dir, output directory
    ir_case:    str, irrigation case 'ni', 'iaf', or 'iwu'
    
    returns:    sr_df, dataframe with sr for catchment, stored as csv
    
    """
    if (ir_case=='ni'):
        # check if sd exists for catchment id
        if(os.path.exists(f'{sd_dir}/no_irri/sd/{catch_id}.csv')==True):  

            # read storage deficit table
            sd_table = pd.read_csv(f'{sd_dir}/no_irri/sd/{catch_id}.csv',index_col=0)
            sd_table.index = pd.to_datetime(sd_table.index)

            # get sd, start and end year and date from sd_table
            if 'sd2' in sd_table.columns:
                Sd = sd_table.sd2
            else:
                Sd = sd_table.Sd

            year_start = sd_table.index[0].year
            year_end = sd_table.index[-1].year
            date_start = str(sd_table.index[0].month)+'-'+str(sd_table.index[0].day)
            date_end = str(sd_table.index[-1].month)+'-'+str(sd_table.index[-1].day)
            if(date_end=='2-29'):
                date_end='2-28'

            if ((year_end-year_start)>10) and (sd_table.Et.max()>0):#only if our timeseries is longer than 10years and Et is not nan
                # calculate sr for different return periods using (4)
                sd_maxmin = sr_return_periods_minmax_rzyear(rp_array, Sd, year_start, year_end, date_start, date_end)

                # get observed extremes as points
                df = gumbel(sd_maxmin)[0]
                df.to_csv(f'{sd_dir}/no_irri/sr/{catch_id}_points.csv')

                # get gumbel fit for different T
                T_interest = gumbel(sd_maxmin)[1]
                gumbel_estimate = gumbel(sd_maxmin)[2]
                sr_df = pd.DataFrame(index=[catch_id], columns=T_interest)
                sr_df.loc[catch_id]=gumbel_estimate
                sr_df.to_csv(f'{sd_dir}/no_irri/sr/{catch_id}_gumbelfit.csv')
                return(sr_df)
    
    else:
        if (ir_case=='iwu'):
            f='iwu2'
        if (ir_case=='iaf'):
            f='0.9ia'
        # check if sd exists for catchment id
        if(os.path.exists(f'{sd_dir}/irri/f{f}/sd/{catch_id}_f{f}.csv')==True):  

            # read storage deficit table
            sd_table = pd.read_csv(f'{sd_dir}/irri/f{f}/sd/{catch_id}_f{f}.csv',index_col=0)
            sd_table.index = pd.to_datetime(sd_table.index)

            # get sd, start and end year and date from sd_table
            if 'sd2' in sd_table.columns:
                Sd = sd_table.sd2
            else:
                Sd = sd_table.Sd

            year_start = sd_table.index[0].year
            year_end = sd_table.index[-1].year
            date_start = str(sd_table.index[0].month)+'-'+str(sd_table.index[0].day)
            date_end = str(sd_table.index[-1].month)+'-'+str(sd_table.index[-1].day)
            if(date_end=='2-29'):
                date_end='2-28'

            if ((year_end-year_start)>10) and (sd_table.Et.max()>0):#only if our timeseries is longer than 10years and Et is not nan
                # calculate sr for different return periods using (4)
                sd_maxmin = sr_return_periods_minmax_rzyear(rp_array, Sd, year_start, year_end, date_start, date_end)

                # get oberved extremes as points
                df = gumbel(sd_maxmin)[0]
                df.to_csv(f'{sd_dir}/irri/f{f}/sr/{catch_id}_f{f}_points.csv')

                # get gumbel fit for different T
                T_interest = gumbel(sd_maxmin)[1]
                gumbel_estimate = gumbel(sd_maxmin)[2]
                sr_df = pd.DataFrame(index=[catch_id], columns=T_interest)
                sr_df.loc[catch_id]=gumbel_estimate
                sr_df.to_csv(f'{sd_dir}/irri/f{f}/sr/{catch_id}_f{f}_gumbelfit.csv')

                return(sr_df)

            
## 6
def run_sr_calculation2(catch_id, rp_array, sd_dir, out_dir):
    """
    run sr calculation
    
    catch_id:   str, catchment id
    rp_array:   int, array, array of return periods
    sd_dir:     str, dir, directory with sd dataframes
    out_dir:    str, dir, output directory   
    returns:    sr_df, dataframe with sr for catchment, stored as csv
    
    """
    if(os.path.exists(f'{sd_dir}/sd/{catch_id}.csv')==True):  

        # read storage deficit table
        sd_table = pd.read_csv(f'{sd_dir}/sd/{catch_id}.csv',index_col=0)
        sd_table.index = pd.to_datetime(sd_table.index)

        # get sd, start and end year and date from sd_table
        if 'sd2' in sd_table.columns:
            Sd = sd_table.sd2
        else:
            Sd = sd_table.Sd

        year_start = sd_table.index[0].year
        year_end = sd_table.index[-1].year
        date_start = str(sd_table.index[0].month)+'-'+str(sd_table.index[0].day)
        date_end = str(sd_table.index[-1].month)+'-'+str(sd_table.index[-1].day)
        if(date_end=='2-29'):
            date_end='2-28'

        if ((year_end-year_start)>10) and (sd_table.Et.max()>0):#only if our timeseries is longer than 10years and Et is not nan
            # calculate sr for different return periods using (4)
            sd_maxmin = sr_return_periods_minmax_rzyear(rp_array, Sd, year_start, year_end, date_start, date_end)

            # get observed extremes as points
            df = gumbel(sd_maxmin)[0]
            df.to_csv(f'{sd_dir}/sr/{catch_id}_points.csv')

            # get gumbel fit for different T
            T_interest = gumbel(sd_maxmin)[1]
            gumbel_estimate = gumbel(sd_maxmin)[2]
            sr_df = pd.DataFrame(index=[catch_id], columns=T_interest)
            sr_df.loc[catch_id]=gumbel_estimate
            sr_df.to_csv(f'{sd_dir}/sr/{catch_id}_gumbelfit.csv')
            return(sr_df)
        
def run_sr_calculation_parallel(
    catch_id_list=list,
    rp_array_list=list,
    sd_dir_list=list,
    out_dir_list=list,
    ir_case_list=list,
    # threads=None
    threads=100
):
    """
    Runs function area_weighted_shapefile_rasterstats in parallel.

    catch_list:  str, list, list of catchment ids
    rp_array_list:     str, list, list of return periods
    sd_dir_list:   str, list, list of folder with sd
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
        run_sr_calculation,
        catch_id_list,
        rp_array_list,
        sd_dir_list,
        out_dir_list,
        ir_case_list,
    )
    
def run_sr_calculation_parallel2(
    catch_id_list=list,
    rp_array_list=list,
    sd_dir_list=list,
    out_dir_list=list,
    # threads=None
    threads=100
):
    """
    Runs function area_weighted_shapefile_rasterstats in parallel.

    catch_list:  str, list, list of catchment ids
    rp_array_list:     str, list, list of return periods
    sd_dir_list:   str, list, list of folder with sd
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
        run_sr_calculation2,
        catch_id_list,
        rp_array_list,
        sd_dir_list,
        out_dir_list,
    )

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
def merge_sr_catchments_maxmin(sr_dir,out_dir):
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
    frame.to_csv(f'{out_dir}/sr_all_catchments_maxmin.csv')
    
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
