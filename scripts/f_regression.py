#%% LOAD PACKAGES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime, timedelta
from scipy.optimize import least_squares
import calendar
import sklearn
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

def regression_input(cc_df, sr_df, dpar, rp, tc_th, ntc_th, nonveg_th):
    # rename sr columns
    sr_df = sr_df.rename(columns={'2':'sr_2', '3':'sr_3', '5':'sr_5', '10':'sr_10', '20':'sr_20', '30':'sr_30', '40':'sr_40', '50':'sr_50', '60':'sr_60'})
    
    # concat catchment characteristics with sr dataframe
    df = pd.concat([cc_df,sr_df],axis=1)

    # define columns for regression (dpar and sr column)
    sr_rp = f'sr_{rp}'
    col = dpar + [sr_rp]
    df = df[col]
    
    # treecover
    df_tc = cc_df[['tc', 'ntc', 'nonveg']]

    # select catchments based on treecover threshold
    df = df[df_tc['tc']>tc_th]
    df = df[df_tc['ntc']>ntc_th]
    df = df[df_tc['nonveg']>nonveg_th]

    # exclude nan catchments
    df = df.dropna()

    # exclude sr=zero catchments
    df = df[df[sr_rp]>0]

    # standardize values
    df_st = pd.DataFrame(index=df.index, columns=[col])
    df_st = (df - df.mean())/df.std()
    
    return df, df_st

def regression(df, df_st, dpar, rp):
    sr_rp = f'sr_{rp}' 
    x,sr = df_st[dpar],df_st[sr_rp]
    model = sm.OLS(sr,x)
    results = model.fit()
    par = np.round(results.params,3)
    pred_st = model.predict(par)
    r_sq = np.round(results.rsquared,3)
    r_sq_adj = np.round(results.rsquared_adj,3)
    aic = np.round(results.aic,3)
    pval = np.round(results.pvalues,3)
    nobs = results.nobs

    pred = pred_st * np.std(df[sr_rp]) + np.mean(df[sr_rp])
    x = df[sr_rp]
    y = pred
    
    return x,y,par,r_sq,r_sq_adj,aic,pval,nobs

def regression_table(dpar,par,r_sq,r_sq_adj,aic,pval,nobs, tc_th, ntc_th, nonveg_th):
    df = pd.DataFrame(index=[0], columns=[dpar])
    df[dpar] = par
    df['tc_th']=tc_th
    df['ntc_th']=ntc_th
    df['nonveg_th']=nonveg_th
    df['r_sq'] = r_sq
    df['r_sq_adj'] = r_sq_adj
    df['aic'] = aic
    df['pval'] = pval
    df['nobs'] = nobs
    return df


def regression_plot(sr,sr_pred,dpar,par,r_sq, tc_th, ntc_th, nonveg_th):    
    # Calculate the point density -> this only works if we have more points
    # xy = np.vstack([sr,sr_pred])
    # z = gaussian_kde(xy)(xy)
    x,y = sr, sr_pred
    #plot
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.plot(x,y,'bo')
    # ax.scatter(x, y, c=z)
    ax.plot([0,1200],[0,1200],'r--')
    ax.set_xlabel('Sr water balance (mm)')
    ax.set_ylabel('Sr predicted (mm)')
    ax.set_xlim(0,800)
    ax.set_ylim(0,800)
    ax.text(0,-150,'Dpar: '+str(dpar))
    ax.text(0,-190,'Coeff: '+str(np.round(par.values,3)))
    ax.set_title(f'veg thresholds: tc:{tc_th}, ntc:{ntc_th}, nonveg:{nonveg_th},  R2={np.round(r_sq,3)}')
    
    
def run_regression(cc_df, sr_df, dpar, rp, tc_th, ntc_th, nonveg_th):
    # prepare input
    df = regression_input(cc_df, sr_df, dpar, rp, tc_th, ntc_th, nonveg_th)[0]
    df_st = regression_input(cc_df, sr_df, dpar, rp, tc_th, ntc_th, nonveg_th)[1]
    
    # regression
    rp = 20
    r = regression(df, df_st, dpar, rp)
    sr, sr_pred = r[0], r[1]
    par,r_sq,r_sq_adj,aic,pval,nobs = r[2],r[3],r[4],r[5],r[6],r[7]
    
    # plot results
    regression_plot(sr,sr_pred,dpar,par,r_sq, tc_th, ntc_th, nonveg_th)
    
    # make table
    df = regression_table(dpar,par,r_sq,r_sq_adj,aic,pval,nobs, tc_th, ntc_th, nonveg_th)
    
    return df