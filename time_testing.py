# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:44:33 2024

@author: krmurph1
"""

import numpy as np
import pandas as pd
import swifter
import tqdm

from pymsis import msis

tqdm.pandas()

alt_min=250
alt_max=1000

alts = np.arange(alt_min,alt_max+1,1)  

test_value=10000
fn='D:/data/SatDensities/satdrag_database_grace_b_v3.hdf5'
gr_t = pd.read_hdf(fn, stop=test_value)

#method 1
# pandas apply
#CPU times: total: 25.5 s
#Wall time: 6min 15s
%%time
dd = gr_t.progress_apply(lambda x : msis.run(x.DateTime_gr,
                      x.lat, 
                      x.lon,
                      np.append(alts,x.alt/1000.),
                      geomagnetic_activity=-1*x.storm)[0,0,0,:,0],axis=1)

#method 1a
# use pandas apply with swifter
# can't allocate the required memory
%%time
dd = gr_t.swifter.apply(lambda x : msis.run(x.DateTime_gr,
                      x.lat, 
                      x.lon,
                      np.append(alts,x.alt/1000.),
                      geomagnetic_activity=-1*x.storm)[0,0,0,:,0],axis=1)

#method 2
# list comprehension
#CPU times: total: 1min 9s
#Wall time: 13min 28s
%%time
dd = [
    msis.run(date,lat,lon,
             np.append(alts,galt/1000.))[0,0,0,:,0] \
    for date, lat, lon, galt \
    in zip(gr_t['DateTime_gr'], gr_t['lat'], gr_t['lon'],gr_t['alt'])
       ]




#method 3
# for loop
# cpu 1.41 s
# wall 29.1 s
%%time
for index,row in gr_t.iterrows():
   #storm times in the gr dataframe are 1
   #to run storm times in msis the geomagnetic activity
   #flag should be set to -1
   msis_dat.append(msis.run(row['DateTime_gr'],
                         row['lat'], 
                         row['lon'],
                         np.append(alts,row['alt']/1000.),
                         geomagnetic_activity=-1*row['storm'])[0,0,0,:,0])