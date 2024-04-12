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
        gr, gr_u, gr_m = data_io.toleos_den.load_toleos(sat='gb',sdate=sdate,edate=edate)
        sat_dat, sat_u, sat_m = data_io.toleos_den.load_toleos(sat=sat,sdate=sdate,edate=edate)
    
        if sat == 'ch':
            sat_dat = sat_dat.rename(columns={'rho_x':'dens_x', 'rho_mean':'dens_mean'})
    
        #drop some columns
        gr = gr[cols]
        sat_dat = sat_dat[cols]
    
        # create a database of champ and grace time matched observations
        sat_dat = sat_dat.rename(columns={'DateTime':'DateTime_ch'})
        sat_dat.index = sat_dat['DateTime_ch']
    
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
        # all profiles have the altitudes
        # we only want total density which is element 0 of the
        # last dimension
        msis_dat = [msis.run(date,lat,lon,alts)[0,0,0,:,0] \
                    for date, lat, lon in zip(conj_date, conj_lat, conj_lon)]
        
        msis_dat = np.array(msis_dat)
        np.save(msis_file,msis_dat)
    else:
        print(f'Loading {msis_file}')
        msis_dat = np.load(msis_file)
        
    
    
    return msis_dat, conj_data.reset_index(), conj_date, conj_dens, conj_alts, alts


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

    
def den_norm(sat: str='ch',):
    
    # get the data for fitting
    msis_d, conj_sdat, conj_date, conj_dens, conj_alts, alts = \
        gen_msis_profiles(sat=sat)
        
        
    
    # fit a line to the msis density profiles so the
    # exponential fit can be seeded
    lin_d = [curve_fit(lin_fit,alts,np.log(den))[0] for den in msis_d]
    
    # fit an exponential to msis density profiles
    exp_d = [
            curve_fit(exp_fit,alts.astype('float64'),den.astype('float64'),
            p0=[np.exp(lf[1]),-1/lf[0]])[0]
            for den, lf in zip(msis_d, lin_d)
            ]
    
    exp_d = np.array(exp_d)
    
    rho0 = conj_dens/np.exp(-conj_alts/1000./exp_d[:,1])
    sat_den = rho0*np.exp(-conj_sdat[f'alt_{sat}'].to_numpy()/1000./exp_d[:,1])
    
    #TODO test everything below this
    
    conj_pos = [np.abs(alts-sat_alt/1000.).argmin() for sat_alt in conj_alts]
    sat_pos = [np.abs(alts-sat_alt/1000.).argmin
               for sat_alt in conj_sdat['alt_ch'].to_numpy()]
    
    den_ratio = conj_dens/msis_d[:,conj_pos]
    sat_dens2 = den_ratio*msis_d[:,sat_pos]
    
    
    return sat_den, conj_sdat.reset_index()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    