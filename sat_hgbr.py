# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:27:46 2024

@author: krmurph1
"""


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import gc
import os

import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV

import sat_random_forest as srf

rnd=17

target_dat = "C:\data\SatDensities\satdrag_database_grace_B_reduced_feature_v3.hdf5"
col = ['2500_03', '43000_09', '85550_13','irr_1216',
       'B', 'AE', 'SYM_H index', 'ASY_D index', 'ASY_H index',
       'alt', 'lat']
lt_col = ['lon']
y_col = 'dens_x'
t_col = 'DateTime'
log_col = False
s_sz = 400000

grid_space = dict(
    learning_rate=[0.05, 0.1, 0.2],
    max_depth=[2, 5, 10, None],
    max_iter=[100,300,500,750,1000],
    min_samples_leaf=[1, 5, 10, 20]
)

# create data sets
kcol = [col,[y_col],[t_col],lt_col]
kflt = [item for sublist in kcol for item in sublist]
# read in target data
df = pd.read_hdf(target_dat)
#take a random sample of the target data
# to help reduce number of points
df = df[kflt].dropna().sample(s_sz)

reg_x, reg_y = srf.dat_create(dat=df,col=col,log_col=log_col,lt_col=lt_col,
                          y_col=y_col,t_col=t_col)
reg_y = reg_y*(10**12)

del df
gc.collect

# create train test splits
train_x, test_x, train_y, test_y = train_test_split(reg_x, reg_y, 
                                                    test_size=0.7, 
                                                    random_state=rnd)

# get and drop DateTime column
train_t = train_x[t_col].copy()
test_t = test_x[t_col].copy()

train_x = train_x.drop(columns=t_col)
test_x = test_x.drop(columns=t_col)
    
t0 = time.time()
gbr_ls = HistGradientBoostingRegressor(loss="squared_error", random_state=rnd)

print('Starting Grid Search')
grid = GridSearchCV(gbr_ls,param_grid=grid_space,cv=3, verbose=4,
            scoring=['neg_mean_absolute_error','r2'], n_jobs=6, 
            return_train_score=True,
            refit=False)

model_grid = grid.fit(train_x,train_y)


