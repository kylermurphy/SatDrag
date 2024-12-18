# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 09:41:14 2024

@author: krmurph1
"""

import matplotlib.pyplot as plt
import os, sys
import pandas as pd
import numpy as np
import xarray as xr
import glob

# for converting to
# geomagnetic coord
import aacgmv2

# add read_io module to current path ()
# and import
file_path = 'D:\\GitHub\\DataIO\\'
sys.path.append(os.path.dirname(file_path))
import data_io as dio

# dates to read in
sdate = '2000-01-01'
edate = '2024-01-01'


# load grace-fo data
gr_c, gr_u, gr_m = dio.toleos_den.load_toleos(sat='GC', sdate='2018-01-01',
                                              edate='2024-01-01')
# load grace data
gr_b, _, _ = dio.toleos_den.load_toleos(sat='GB',sdate='2002-01-01',
                                              edate='2018-01-01')
# load champ
ch, _, _ = dio.toleos_den.load_toleos(sat='CH',sdate='2002-01-01',
                                              edate='2018-01-01')
# goce
go, _, _ = dio.toleos_den.load_toleos(sat='GO',sdate='2009-01-01',
                                              edate='2014-01-01')
# swarm
sw, _, _ = dio.toleos_den.load_toleos(sat='SC',sdate='2014-01-01',
                                              edate='2024-01-01')

fig1, ax1 = plt.subplots(1,1,figsize=(8,3),sharex=True, layout='constrained')

ax1.plot(gr_c['DateTime'], gr_c['alt']/1000., label='GRACE-FO')
ax1.plot(gr_b['DateTime'], gr_b['alt']/1000., label='GRACE')
ax1.plot(ch['DateTime'], ch['alt']/1000., label='CHAMP')
ax1.plot(sw['DateTime'], sw['alt']/1000., label='Swarm')
ax1.plot(go['DateTime'], go['alt']/1000., label='GOCE')


ax1.set_xlim(pd.to_datetime(sdate),pd.to_datetime(edate))
ax1.set_ylim(200,600)
ax1.set_xlabel('Date')
ax1.set_ylabel('Alt - km')
ax1.legend(loc='lower right')

plt.show()

