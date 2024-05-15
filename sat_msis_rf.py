# -*- coding: utf-8 -*-
"""
Created on Wed May 15 08:39:37 2024

@author: krmurph1
"""

import os, sys

#add read_io module to current path ()
file_path = 'D:\\GitHub\\DataIO\\'
sys.path.append(os.path.dirname(file_path))

import data_io
import pandas as pd
import numpy as np


from pymsis import msis
from scipy.optimize import curve_fit
from sklearn import metrics

def msis_prof(fn='D:/data/SatDensities/satdrag_database_grace_b_v3.hdf5',
              alt_min=250,
              alt_max=1000,
              small_batch=False):
    
    # read in satellite data
    gr_d = pd.read_hdf(fn)
    
    if alt_max < alt_min:
        alt_min=250
        alt_max=1000
    
    alts = np.arange(alt_min,alt_max+1,1)  
    
    if small_batch:
        gr_d = gr_d[0:1000].copy()
    
    # get msis profiles for all grace locations    
    msis_dat = [
        msis.run(date,lat,lon,
                 np.append(alts,galt/1000.))[0,0,0,:,0] \
        for date, lat, lon, galt \
        in zip(gr_d['DateTime_gr'], gr_d['lat'], gr_d['lon'],gr_d['alt'])
           ]
    msis_dat = np.array(msis_dat)
    # shift the profile to match grace data
    den_ratio = gr_d['dens_x']/msis_dat[:,-1]
    den_profile = [prof*ratio for prof, ratio in zip(msis_dat, den_ratio)]
    den_profile = np.array(den_profile)
    
    den_profile = den_profile[:,:-2].copy()
    
    # save the density profiles
    sv_fn = os.path.splitext(fn)[0]+'_msis_profile'
    np.savez_compressed(sv_fn, alts=alts, 
                        den_profile=den_profile, msis_profile=msis_dat)
    
    return gr_d, msis_dat, den_profile



gr_d, msis_dat, mod_p = msis_prof( )