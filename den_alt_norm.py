# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:03:23 2024

@author: krmurph1
"""

# something to create the data set
# something to plot the delta latitude and longitude
# something to calculate the msis profiles where the satellites are close
# something to normalize the data for those dates
# something to compare the normalized data with the GOCE and CHAMP data

import os, sys

 #add read_io module to current path ()
file_path = 'D:\\GitHub\\DataIO\\'
sys.path.append(os.path.dirname(file_path))

import data_io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pymsis import msis
from scipy.optimize import curve_fit
from sklearn import metrics 


def gen_sat_data(sat: str='ch',
                 sdate='2002-01-01',
                 edate='2014-01-01',
                 t_tol='5 second',
                 t_back= 40,
                 t_tot=None,
                 data_dir='D:\\data\\SatDensities\\',
                 cols=['DateTime', 'alt','lat','lon','dens_x'], 
                 gen=False):
    
    tol = pd.Timedelta(t_tol)
    if not t_tot:
        t_tot = 2*t_back
    
    # check if the satellite data has been processed already
    sat_file = data_dir+f'{sat}_den_profile.hdf'
    sat_path = os.path.exists(sat_file)
    
    # process satellite data
    if not sat_path or gen:
        # read in grace b data
        gr, gr_u, gr_m = \
            data_io.toleos_den.load_toleos(sat='gb',sdate=sdate,edate=edate)
        sat_dat, sat_u, sat_m = \
            data_io.toleos_den.load_toleos(sat=sat,sdate=sdate,edate=edate)
    
        if sat == 'ch':
            sat_dat = \
                sat_dat.rename(
                    columns={'rho_x':'dens_x', 'rho_mean':'dens_mean'})
    
        #drop some columns
        gr = gr[cols]
        sat_dat = sat_dat[cols]
    
        # create a database of champ and grace time matched observations
        sat_dat = sat_dat.rename(columns={'DateTime':f'DateTime_{sat}'})
        sat_dat.index = sat_dat[f'DateTime_{sat}']
    
        gr = gr.rename(columns={'DateTime':'DateTime_gr'})
        gr.index = gr['DateTime_gr']
    
        # combine dataframes to match DateTime indices
        sat_gr = pd.merge_asof(left=sat_dat,right=gr,
                               right_index=True,left_index=True,
                               direction='nearest',tolerance=tol,
                               suffixes=[f'_{sat}','_gr'])
    
        sat_gr['del_lon'] = sat_gr[f'lon_{sat}']-sat_gr['lon_gr']
        sat_gr['del_lat'] = sat_gr[f'lat_{sat}']-sat_gr['lat_gr']
    
        #free up space
        del sat_dat
        del gr
    
        #find nearest neigbhour latitude position 
        # for each orbit
        ch_lat = sat_gr[f'lat_{sat}'].to_numpy()
        gr_lat = sat_gr['lat_gr'].to_numpy()
        gr_lon = sat_gr['lon_gr'].to_numpy()
    
        res = (pd.Series(sat_gr.index[1:]) -
                   pd.Series(sat_gr.index[:-1])).value_counts()
        res = res.index[0]
    
        t_back = t_back*60/res.seconds
        t_tot = t_tot*60/res.seconds
    
        lat_pos = np.empty(ch_lat.shape,dtype=np.int32)
        del_lat = np.empty(ch_lat.shape[0])
        val_lat = np.empty(ch_lat.shape[0])
        val_lon = np.empty(ch_lat.shape[0])
    
        for i in np.arange(ch_lat.shape[0]):
            st = int(i-t_back)
            if st < 0: 
                st = 0
            
            en = int(st+t_tot)
            lat_pos[i] = np.abs(gr_lat[st:en]-ch_lat[i]).argmin()+st
            del_lat[i] = gr_lat[lat_pos[i]]-ch_lat[i]
            val_lat[i] = gr_lat[lat_pos[i]]
            val_lon[i] = gr_lon[lat_pos[i]]
    
        sat_gr['del_lat_near'] = del_lat
        sat_gr['lat_gr_near'] = val_lat
        sat_gr['lon_gr_near'] = val_lon
        sat_gr['pos'] = lat_pos 
    
        sat_gr.to_hdf(sat_file,key='sat_df',complevel=9, mode='w')
    else:
        print(f'Loading HDF file {sat_file}')
        sat_gr = pd.read_hdf(sat_file)
    
    
    return sat_gr
    
    
    
def gen_msis_profiles(sat: str='ch',
                 del_lon=5,
                 del_lat_near=1,
                 del_lon_near=5,
                 alt_min=250,
                 alt_max=750,
                 alt_res=1,
                 gen=False, 
                 data_dir='D:\\data\\SatDensities\\'):
    
    
    # load the satellite density data
    sat_dat = gen_sat_data(sat=sat, gen=gen, data_dir=data_dir)
    
    # calculate the delta longitude at the nearest 
    # neighbour latitude value
    sat_dat['del_lon_near'] = sat_dat['lon_gr_near']-sat_dat[f'lon_{sat}']
    

    # create a new data set where the satellites are conjugate in space
    sat_dat = sat_dat.drop(columns=f'DateTime_{sat}').reset_index()
    
    # find the points when the two satellites are close together
    gd_lon = sat_dat['del_lon'].abs() < del_lon
    gd_lat_nr = sat_dat['del_lat_near'].abs() < del_lat_near
    gd_lon_nr = sat_dat['del_lon_near'].abs() < del_lon_near
    
    conj_data = sat_dat[gd_lon & gd_lat_nr & gd_lon_nr].copy()
    
    conj_date = sat_dat.loc[conj_data['pos'],'DateTime_gr']. \
        reset_index(drop=True).copy()
    conj_dens = sat_dat.loc[conj_data['pos'],'dens_x_gr']. \
        reset_index(drop=True).copy()
    conj_alts = sat_dat.loc[conj_data['pos'],'alt_gr']. \
        reset_index(drop=True).copy()
    conj_lat = conj_data['lat_gr_near'].reset_index(drop=True).copy()
    conj_lon = conj_data['lon_gr_near'].reset_index(drop=True).copy()
    
    sat_alts = conj_data[f'alt_{sat}']
    
    del sat_dat
    
    # create altitudes for MSIS data
    if alt_min > alt_max:
        alt_min=250
        alt_max=500
        
    alts = np.linspace(alt_min,alt_max,alt_max-alt_min+1)
    
    # check if the msis profiles have already been generated
    msis_file = data_dir+f'msis_profiles_{sat}.npy'
    msis_path = os.path.exists(msis_file)
    
    # derive profiles and save or open
    if not msis_path or gen:
        # derive profiles
        # all profiles have the same altitudes
        # we only want total density which is element 0 of the
        # last dimension
        
        # for simplicity and so we have all data, do a normal run and a 
        # storm run, geomagnetic_storm=-1
        
        # add the altitude of grace and the second satelite at the end
        # of the altitude arrays
        msis_dat = [
            msis.run(date,lon,lat, \
            np.append(alts,[c_alt/1000.,s_alt/1000.]))[0,0,0,:,0] \
            for date, lat, lon, c_alt, s_alt, \
            in zip(conj_date, conj_lat, conj_lon, conj_alts, sat_alts)
                    ]
            
        msis_s = [
            msis.run(date,lon,lat, \
            np.append(alts,[c_alt/1000.,s_alt/1000.]), \
            geomagnetic_activity=-1)[0,0,0,:,0] \
            for date, lat, lon, c_alt, s_alt, \
            in zip(conj_date, conj_lat, conj_lon, conj_alts, sat_alts)
                    ]     
        
        msis_dat = np.array([msis_dat,msis_s])
        np.save(msis_file,msis_dat)
    else:
        print(f'Loading {msis_file}')
        msis_dat = np.load(msis_file)
    
    return msis_dat, conj_data.reset_index(), conj_date, \
        conj_dens, conj_alts, sat_alts, alts


def exp_fit(x,j,h) -> float:
    """
    
    Returns
    -------
    Function for exponential fit.
    
    """
    return j*np.exp(-x/h)

def lin_fit(x,a,b):
    """

    Returns
    -------
    Function for linear fit

    """
    return a*x+b

    
def den_norm(sat: str='ch',
             storm_times=True, 
             storm_txt=
             'D:\\GitHub\\SatDrag\\data\\storms_drag_epochs_no_overlap.txt',
             gen=False,
             data_dir='D:\\data\\SatDensities\\'):
    
    # get the data for fitting
    msis_arr, conj_sdat, conj_date, conj_dens, conj_alts, sat_alts, alts = \
        gen_msis_profiles(sat=sat)
          
        
    # add storm times to the data
    # read in storm start and end times
    storm_val = np.empty(conj_date.shape[0],dtype=int)
    storm_val[:] = 1 
    if storm_times:
        print('Running storm times')
        print('Reading Storm Times')
        storm_time = pd.read_csv(storm_txt, header=None, skiprows=1, 
                         sep='\s+', names = ['t_st','t_dst','t_en'], 
                         parse_dates=[0, 1, 2])
                
        for index, row in storm_time.iterrows():
            stp = (conj_date>=row['t_st']) & (conj_date<row['t_en'])
            storm_val[stp] = -1
        
        stp = storm_val==-1
        msis_d = msis_arr[0,:,:]
        msis_d[stp,:] = msis_arr[1,stp,:]
        
    else:
        print('Running all normal times')
        msis_d = msis_arr[0,:,:]
    
    
    st_t = 'storm' if storm_times else 'nostorm'
    # check if the fits to profiles have already been generated
    fit_file = data_dir+f'msis_profile_fits_{sat}_{st_t}.npy'
    fit_path = os.path.exists(fit_file)
    
    # derive profiles and save or open
    if not fit_path or gen:
        print(f'Running fits and saving fit file {fit_file}')
        # fit a line to the msis density profiles so the
        # exponential fit can be seeded
        lin_d = [curve_fit(lin_fit,alts,np.log(den[:-2]))[0] for den in msis_d]
        
        # fit an exponential to msis density profiles
        exp_d = [
                curve_fit(exp_fit,alts.astype('float64'),
                          den[0:-2].astype('float64'),
                          p0=[np.exp(lf[1]),-1/lf[0]])[0]
                for den, lf in zip(msis_d, lin_d)
                ]
        
        exp_d = np.array(exp_d)
        np.save(fit_file,exp_d)
    else:
        print(f'Loading {fit_file}')
        exp_d = np.load(fit_file)
        
    
    # calculate density using the scale height
    rho0 = conj_dens/np.exp(-conj_alts/1000./exp_d[:,1])
    mod_den1 = rho0*np.exp(-conj_sdat[f'alt_{sat}'].to_numpy()/1000./exp_d[:,1])
    
    # calculate density by shifting the density profile to match grace
    den_ratio = conj_dens/msis_d[:,-2]
    mod_den2 = den_ratio*msis_d[:,-1]
    
    # calculte the new profiles as well
    mod_profile = [prof*ratio for prof, ratio in zip(msis_d, den_ratio)]
    mod_profile = np.array(mod_profile)
    
    obs_den = conj_sdat[f'dens_x_{sat}']
    
    return conj_dens, conj_alts, obs_den, \
        mod_den1, mod_den2, mod_profile, msis_d, alts, \
        storm_times, storm_val, conj_sdat 

    
def mod_residuals(sat: str='ch',
                  storm_times=True,
                  density=True,
                  lq=0.05,
                  uq=0.99):
 
    if sat == 'ch':
        sat_t='CHAMP'
    elif sat == 'go':
        sat_t='GOCE'
    else:
        sat_t='CHAMP'
        sat='ch'
    
    if storm_times:
        geo_t = 'Quiet & Storm Times'
    else:
        geo_t = 'All Quiet'
     
    conj_dens, conj_alts, obs_den, \
    mod_den1, mod_den2, \
    mod_profile, msis_d, \
    alts, storm_val, storm_val, \
    conj_sdat = den_norm(sat=sat, storm_times=storm_times)
    
    
    resid_1 = (obs_den-mod_den1)#*1E12
    resid_2 = (obs_den-mod_den2)#*1E12
    
    res_min = np.min([resid_1.quantile(lq),resid_2.quantile(lq)])
    res_max = np.max([resid_1.quantile(uq),resid_2.quantile(uq)])
    
    res_b1 = np.histogram_bin_edges(resid_1,bins='fd', range=[res_min,res_max])
    res_b2 = np.histogram_bin_edges(resid_2,bins='fd', range=[res_min,res_max])
    
    ratio_1 = obs_den/mod_den1
    ratio_2 = obs_den/mod_den2
    
    rat_max = np.max([ratio_1.quantile(uq),ratio_2.quantile(uq)])
    
    rat_b1 = np.histogram_bin_edges(ratio_1,bins='fd', range=[0,rat_max])
    rat_b2 = np.histogram_bin_edges(ratio_2,bins='fd', range=[0,rat_max])
        
    plt.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots(1,2, figsize=(6,3))
    
    density=True
    
    ax[0].hist(resid_1,bins=res_b1, alpha = 1, \
                 label='Scale Height', density=density, color='steelblue') 
    ax[0].hist(resid_2,bins=res_b2, alpha = 0.5, \
                 label='Profile Ratio', density=density, color='red')
    ax[0].legend(fontsize=8)
    ax[0].set(title='Residuals: Obs-Mod', xlabel='Residuals', 
              ylabel='Probability')
    
    ax[1].hist(ratio_1,bins=rat_b1, alpha = 1, \
                 label='Scale Height', density=density, color='steelblue') 
    ax[1].hist(ratio_2,bins=rat_b2, alpha = 0.5, \
                 label='Profile Ratio', density=density, color='red')
    ax[1].legend(fontsize=8) 
    ax[1].set(title='Ratio: Obs/Mod', xlabel='Ratio', 
          ylabel='Probability')

    fig.suptitle(f'{geo_t} - {sat_t}', fontsize=10)       


    
def pro_residuals(sat: str='ch',
                  storm_times=True,
                  density=True,
                  lq=0.02,
                  uq=0.99):
 
    if sat == 'ch':
        sat_t='CHAMP'
    elif sat == 'go':
        sat_t='GOCE'
    else:
        sat_t='CHAMP'
        sat='ch'
    
    if storm_times:
        geo_t = 'Quiet & Storm Times'
    else:
        geo_t = 'All Quiet'
     
    conj_dens, conj_alts, obs_den, \
    mod_den1, mod_den2, \
    mod_profile, msis_d, \
    alts, storm_val, storm_val, \
    conj_sdat = den_norm(sat=sat, storm_times=storm_times)
    
    diff_pro = obs_den-mod_den2
    diff_act = obs_den-msis_d[:,-1]
    diff_b = np.histogram_bin_edges(diff_pro,bins='fd',
                range=[diff_pro.quantile(lq),diff_pro.quantile(uq)])
    diff_ba = np.histogram_bin_edges(diff_act,bins='fd',
                range=[diff_act.quantile(lq),diff_act.quantile(uq)])
    
    ratio_pro = obs_den/mod_den2
    ratio_act = obs_den/msis_d[:,-1]
    ratio_b = np.histogram_bin_edges(ratio_pro,bins='fd',
                range=[ratio_pro.quantile(lq),ratio_pro.quantile(uq)])
    ratio_ba = np.histogram_bin_edges(ratio_act,bins='fd',
                range=[ratio_act.quantile(lq),ratio_act.quantile(uq)])
    
    
    
    del_lat = conj_sdat['del_lat_near']
    del_lon = conj_sdat['del_lon_near']
    
    r_ran = [ratio_b.min(),ratio_b.max()]
    d_ran = [diff_b.min(),diff_b.max()]
    
    plt.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots(3,2, figsize=(6,9))
    
    ax[0,0].hist(diff_act,bins=diff_ba,density=density, \
                 color='steelblue', label='MSIS Actual')
    ax[0,0].hist(diff_pro,bins=diff_b,density=density, color='r', \
                 alpha=0.5, label='MSIS Norm')
    ylim = ax[0,0].get_ylim()
    ax[0,0].plot([0,0],ylim,'k--')
    ax[0,0].set_ylim(ylim)
    ax[0,0].legend(fontsize=8)
    ax[0,0].set(title='Residuals: Obs-Mod')
    
    ax[1,0].hist2d(diff_pro,del_lat,bins=50,range=[d_ran,[-0.25,0.25]])
    ylim = ax[1,0].get_ylim()
    ax[1,0].plot([0,0],ylim,'k--')
    ax[1,0].set_ylim(ylim)
    
    ax[2,0].hist2d(diff_pro,del_lon,bins=50,range=[d_ran,[-5,5]])
    ylim = ax[2,0].get_ylim()
    ax[2,0].plot([0,0],ylim,'k--')
    ax[2,0].set_ylim(ylim)
    
    # ratios
    ax[0,1].hist(ratio_act,bins=ratio_ba,density=density, \
                 color='steelblue', label='MSIS Actual')
    ax[0,1].hist(ratio_pro,bins=ratio_b,density=density, color='r', \
                 alpha=0.5, label='MSIS Norm')
    ylim = ax[0,1].get_ylim()
    ax[0,1].plot([1,1],ylim,'k--')
    ax[0,1].set_ylim(ylim)
    ax[0,1].legend(fontsize=8)
    ax[0,1].set(title='Ratio: Obs/Mod')
    
    ax[1,1].hist2d(ratio_pro,del_lat,bins=50,range=[r_ran,[-0.25,0.25]])
    ylim = ax[1,1].get_ylim()
    ax[1,1].plot([1,1],ylim,'k--')
    ax[1,1].set_ylim(ylim)
    
    ax[2,1].hist2d(ratio_pro,del_lon,bins=50,range=[r_ran,[-5,5]])
    ylim = ax[2,1].get_ylim()
    ax[2,1].plot([1,1],ylim,'k--')
    ax[2,1].set_ylim(ylim)
    
    fig.suptitle(f'{geo_t} - {sat_t}', fontsize=10)   
    plt.tight_layout()
    
    # calculate some metrics
    nprof_mea = metrics.median_absolute_error(obs_den, mod_den2)
    nmsis_mea = metrics.median_absolute_error(obs_den, msis_d[:,-1]) 
    
    # rms
    nprof_rms = metrics.mean_squared_error(obs_den, mod_den2)
    nmsis_rms = metrics.mean_squared_error(obs_den, msis_d[:,-1]) 
    
    # mape
    nprof_mape = metrics.mean_absolute_percentage_error(obs_den, mod_den2)
    nmsis_mape = metrics.mean_absolute_percentage_error(obs_den, msis_d[:,-1]) 
    
    # best is 1
    nprof_ev = metrics.explained_variance_score(obs_den, mod_den2, multioutput='uniform_average')
    nmsis_ev = metrics.explained_variance_score(obs_den, msis_d[:,-1], multioutput='uniform_average') 
    
    # r2 - best is 1
    nprof_r2 = metrics.r2_score(obs_den, mod_den2)
    nmsis_r2 = metrics.r2_score(obs_den, msis_d[:,-1]) 
    
    print(f'MEA: {nprof_mea}, {nmsis_mea}')
    print(f'RMS: {nprof_rms}, {nmsis_rms}')
    print(f'MAPE: {nprof_mape}, {nmsis_mape}')
    print(f'EV: {nprof_ev}, {nmsis_ev}')
    print(f'R2: {nprof_r2}, {nmsis_r2}')
    
    
    print('fin')
    
    return 0 
    

def fitting_ex():
    
    conj_dens, conj_alts, obs_den, \
    mod_den1, mod_den2, \
    mod_profile, msis_d, \
    alts, storm_val, storm_val, \
    conj_sdat = den_norm()

    #TODO
    # show when method 1 works best
    # show when method 2 works best
    # show when they are both the worst

    p_err = np.array([metrics.mean_squared_error(x[0:-2], y[0:-2])
             for x, y in zip(mod_profile,msis_d)])
    
    
    #best and worst method 1
    p1_w = np.abs(np.array(
        mod_den1[0:50000]-conj_sdat.loc[0:50000-1,'dens_x_ch'])).argmax()
    p1_b = np.abs(np.array(
        mod_den1[0:50000]-conj_sdat.loc[0:50000-1,'dens_x_ch'])).argmin() 
    
    
    #best and worst method 2
    p2_w = np.abs(np.array(
        mod_den2[0:50000]-conj_sdat.loc[0:50000-1,'dens_x_ch'])).argmax()
    p2_b = np.abs(np.array(
        mod_den2[0:50000]-conj_sdat.loc[0:50000-1,'dens_x_ch'])).argmin()


    p_pos = np.abs(np.array(
        msis_d[0:50000,-1]-conj_sdat.loc[0:50000-1,'dens_x_ch'])).argmax()
    
    
    # calculate density using the scale height
    fit_file = 'D:\data\SatDensities\msis_profile_fits_ch_storm.npy'
    exp_d = np.load(fit_file)
    rho0 = conj_dens/np.exp(-conj_alts/1000./exp_d[:,1])
    
    p_pos = [p1_b,p1_w,p2_b,p2_w]
    p_tit = ['Fit - Best', 'Fit - Worst', 'Shift - Best', 'Shift - Worst']
    
    fig, ax = plt.subplots(2,2, figsize=(8,6))
    
    for ap, pp, pt in zip(ax.flatten(),p_pos,p_tit):
        # profile of the fit to MSIS data
        mfit_profile = exp_fit(alts,exp_d[pp,0],exp_d[pp,1])
        # profile of the fit shifted to satellite data
        ofit_profile = exp_fit(alts,rho0[pp],exp_d[pp,1])
    
    
        ap.plot(msis_d[pp,0:-2], alts, 
                label="MSIS Profile", color='steelblue')
        ap.plot(mod_profile[pp,0:-2], alts,
                label="Shift Profile to GRACE", color='crimson')
        ap.plot(conj_dens[pp], 
                conj_alts[pp]/1000.,
                marker='D', label="GRACE", color='darkorange')
        ap.plot(conj_sdat.loc[pp,'dens_x_ch'],
                conj_sdat.loc[pp,'alt_ch']/1000.,
                marker='X', label='CHAMP', color='darkorange')
        ap.plot(mfit_profile,alts,color='steelblue',
                label='Fit to MSIS',linestyle='dashed')
        ap.plot(ofit_profile,alts, color='crimson',
                label='Fit Shifted to GRACE',linestyle='dashed')
        ap.legend()
        ap.set_title(pt)
        ap.set_ylabel('Altitude - km')
        ap.set_xlabel('Density - kg/m$\mathregular{^3}$')
        


    
    return fig, ax


# running stuff  
#fig, ax = fitting_ex()    

#a = den_norm(sat='go')

#a = den_norm(storm_times=False)
#a = den_norm(sat='go',storm_times=False)

pro_residuals(sat='ch')
    
    
    
    