# -*- coding: utf-8 -*-
"""
Created on Wed May 15 08:39:37 2024

@author: krmurph1
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


from pymsis import msis

def msis_prof(fn='D:/data/SatDensities/satdrag_database_grace_B_v3.hdf5',
              fo=False,
              alt_min=250,
              alt_max=1000,
              columns=['DateTime_gr','lat','lon','alt','storm','dens_x'], 
              chunk_size=50000,
              small_batch=False):
    
    
    # get altitudes for generating profiles
    if alt_max < alt_min:
        alt_min=250
        alt_max=1000
    alts = np.arange(alt_min,alt_max+1,1)  
    
    if not fo:
        fo, f_ext = os.path.splitext(fn)
        fo = fo+'_MSIS'+f_ext
    elif small_batch:
        fo = 'D:/data/SatDensities/small_batch_test.hdf5'
        
    
    # get itterator for reading
    # in the data 
    gr_it = pd.read_hdf(fn,columns=columns, iterator=True, chunksize=chunk_size)
    
    lp = 0
    for df in gr_it:
        tqdm.pandas(desc='MSIS Profiles')
        #storm times in the gr dataframe are 1
        #to run storm times in msis the geomagnetic activity
        #flag should be set to -1
        
        pro_m = df.progress_apply(lambda x : msis.run(x.DateTime_gr,
                              x.lon, 
                              x.lat,
                              np.append(alts,x.alt/1000.),
                              geomagnetic_activity=-1*x.storm)[0,0,0,:,0],
                              axis=1, raw=True)
        p_df = pd.DataFrame()
        p_df['DateTime'] = df['DateTime_gr']
        p_df['MSIS'] = np.array(pro_m)
        p_df['alts'] = df['alt']/1000.
        
        
        # shift the profile to match grace data
        pro_m = np.array(pro_m.to_list())
        den_ratio = df['dens_x']/pro_m[:,-1]
        den_profile = [prof*ratio for prof, ratio in zip(pro_m, den_ratio)]
        
        # add the shifted profile to the DataFrame
        p_df['MSIS_mod'] = den_profile
        
        # flatten the msis array
        m_df = pd.concat(
            pd.DataFrame({'MSIS':row.MSIS,'MSIS_mod':row.MSIS_mod,
                          'alts':np.append(alts,row.alts),
                          'DateTime':row.DateTime}) 
            for ind, row in p_df.iterrows())
        
        # print to file
        m_df.reset_index(drop=True).to_hdf(
                fo,
                key='profiles', append=True, format='table', 
                complevel=9)
        
        lp = lp+1
        #del m_df, pro_m
        if small_batch and lp > 2:
            break
        

    

    
    return p_df, m_df



#p_df, m_df = msis_prof(
#    fn='D:/data/SatDensities/satdrag_database_grace_B_v3.hdf5',
#    chunk_size=1000,
#    small_batch=True)

p_df, m_df = msis_prof(
    fn='D:/data/SatDensities/satdrag_database_grace_B_v3.hdf5')