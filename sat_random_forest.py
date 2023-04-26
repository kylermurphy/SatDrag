import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import gc
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV

rnd=17
rf_params = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf":5,
    "warm_start":False,
    "oob_score":True,
    "random_state": rnd,
    "max_features":0.5,
    "n_jobs":10
    }

grid_space = {
    "n_estimators": [300,500,750,1000],
    "max_depth": [None],
    "min_samples_split": [2],
    "min_samples_leaf":[2,5,10,20],
    "max_features":[0.5,1,2,4]
    }

def dat_create(dat, col, log_col, lt_col, y_col, t_col):

    x_dat = dat[col+t_col+[y_col]].dropna().copy()
    

    if log_col:
       for i in log_col:
            try:
                x_dat[i] = np.log10(x_dat[i])
            except:
                print(f'Could not log column {i}')
    
    if lt_col:
        for i in lt_col:
            try:
                x_dat[f'cos_{i}'] = np.cos(dat[i]*2*np.pi/24.)
                x_dat[f'sin_{i}'] = np.sin(dat[i]*2*np.pi/24.)
            except:
                print(f'Could not add {i} as a cos/sin time column')
    
    x_dat = x_dat[~x_dat.isin([np.nan, np.inf, -np.inf]).any(axis=1)].dropna()
    y_dat = x_dat[y_col].copy()
    x_dat = x_dat.drop(columns=y_col)    
    
    return x_dat, y_dat

def get_permimp(rf_model, rf_x, rf_y, 
        random_state=17, n_repeats=10, n_jobs=2):
    
    t_te = permutation_importance(rf_model, rf_x, rf_y,
                n_repeats=n_repeats, random_state=random_state, 
                n_jobs=n_jobs,scoring='r2')
    
    sorted_importances_idx = t_te.importances_mean.argsort()
    importances_te = pd.DataFrame(
        t_te.importances[sorted_importances_idx].T,
        columns=rf_x.columns[sorted_importances_idx],
    )

    return importances_te


def rf_model(col=['1300_02', 'SYM_H index','SatLat'], 
             y_col='400kmDensity', 
             t_col=['DateTime'], 
             log_col=['1300_02'], 
             lt_col=['SatMagLT'], 
             rf_params=rf_params, 
             target_dat='D:\\data\\SatDensities\\satdrag_database_grace_B.hdf5', 
             oos_dat='D:\\data\\SatDensities\\satdrag_database_grace_A.hdf5',
             oos_dat2='D:\\data\\SatDensities\\satdrag_database_grace_CHAMP_SI_int.hdf5',
             n_repeats=10):
    
    
    rnd = rf_params['random_state']
    
    kcol = [col,[y_col],t_col,lt_col]
    kflt = [item for sublist in kcol for item in sublist]
    df = pd.read_hdf(target_dat)
    df = df[kflt].dropna()

    reg_x, reg_y = dat_create(dat=df,col=col,log_col=log_col,lt_col=lt_col,
                              y_col=y_col,t_col=t_col)
    reg_y = reg_y*(10**12)
    

    # create data set from out of sample data
    df_oos = pd.read_hdf(oos_dat)
    oos_x, oos_y = dat_create(dat=df_oos,col=col,log_col=log_col,lt_col=lt_col,
                              y_col=y_col,t_col=t_col)
    oos_y = oos_y*(10**12)
    oos_t = oos_x[t_col]
    oos_x = oos_x.drop(columns=t_col)
    
    if oos_dat2:
        df_oos2 = pd.read_hdf(oos_dat2)
        oos_x2, oos_y2 = dat_create(dat=df_oos2,col=col,log_col=log_col,lt_col=lt_col,
                                  y_col=y_col,t_col=t_col)
        oos_y2 = oos_y2*(10**12)
        oos_t2 = oos_x2[t_col]
        oos_x2 = oos_x2.drop(columns=t_col)


    del df
    del df_oos
    del df_oos2
    gc.collect
    
    # create train test splits
    train_x, test_x, train_y, test_y = train_test_split(reg_x, reg_y, 
                                                        test_size=0.3, 
                                                        random_state=rnd)

    # get and drop DateTime column
    train_t = train_x[t_col].copy()
    test_t = test_x[t_col].copy()

    train_x = train_x.drop(columns=t_col)
    test_x = test_x.drop(columns=t_col)

    print('Train and fit model')

    start = time.time()
    print("Time elapsed working on RandomForest")

    rfr = RandomForestRegressor(**rf_params)
    rfr.fit(train_x, train_y)

    end = time.time()
    print("Time consumed in working: ",end - start)

    #Make predictions and calculate error
    predictions = rfr.predict(test_x)
    pre_oos = rfr.predict(oos_x)
    pre_tr = rfr.predict(train_x)

    #the mean absolute error
    mea = mean_absolute_error(test_y, predictions)
    mae_oss = mean_absolute_error(oos_y,pre_oos)
    mae_tr = mean_absolute_error(train_y,pre_tr)

    #mean absolute percentage error
    mape = mean_absolute_percentage_error(test_y, predictions)
    mape_oss = mean_absolute_percentage_error(oos_y,pre_oos)
    mape_tr = mean_absolute_percentage_error(train_y,pre_tr)

    med = median_absolute_error(test_y, predictions)
    med_oss = median_absolute_error(oos_y,pre_oos)
    med_tr = median_absolute_error(train_y,pre_tr)

    #Print r-squared score of model
    r2 = r2_score(test_y, predictions)
    r2_oos = r2_score(oos_y, pre_oos)
    r2_tr = r2_score(train_y,pre_tr)
    
    metr_df = pd.DataFrame({'r2':[r2_tr,r2,r2_oos],
                            'MedAE':[med_tr,med,med_oss],
                            'MAE':[mae_tr,mea,mae_oss],
                            'MAPE':[mape_tr,mape,mape_oss]},
                           index=['train','test','oos'])
    
    if oos_dat2:
        pre_oos2 = rfr.predict(oos_x2)
        mae_oss2 = mean_absolute_error(oos_y2,pre_oos2)
        mape_oss2 = mean_absolute_percentage_error(oos_y2,pre_oos2)
        med_oss2 = median_absolute_error(oos_y2,pre_oos2)
        r2_oos2 = r2_score(oos_y2, pre_oos2)

        print(f"Score train/test/oos/oos2: {r2_tr:.3}/{r2:.3}/{r2_oos:.3}/{r2_oos2:.3}")    
        print(f"MedAE train/test/oos/oos2: {med_tr:.3}/{med:.3}/{med_oss:.3}/{med_oss2:.3}")
        print(f"MAE train/test/oos/oos2: {mae_tr:.3}/{mea:.3}/{mae_oss:.3}/{mae_oss2:.3}")
        print(f"MAPE train/test/oos/oos2: {mape_tr:.3}/{mape:.3}/{mape_oss:.3}/{mape_oss2:.3}")
        
        metr_oos2 = pd.DataFrame({'r2':[r2_oos2],
                                'MedAE':[med_oss2],
                                'MAE':[mae_oss2],
                                'MAPE':[mape_oss2]},
                               index=['oos2'])
        metr_df = pd.concat([metr_df,metr_oos2])
        
    else:
        print(f"Score train/test/oos: {r2_tr:.3}/{r2:.3}/{r2_oos:.3}")
        print(f"MedAE train/test/oos: {med_tr:.3}/{med:.3}/{med_oss:.3}")
        print(f"MAE train/test/oos: {mae_tr:.3}/{mea:.3}/{mae_oss:.3}")
        print(f"MAPE train/test/oos: {mape_tr:.3}/{mape:.3}/{mape_oss:.3}")


    #Examine feature importances
    feature_names = rfr.feature_names_in_
    mdi_importances = pd.Series(rfr[-1].feature_importances_, 
                                index=feature_names).sort_values(ascending=True)


    #Examin importances with test set
    start = time.time()
    print("Time elapsed working on Permutation Importance")

    print('Test Importance')
    importances_te = get_permimp(rfr, test_x, test_y, 
                n_repeats=n_repeats, random_state=rnd, 
                n_jobs=1)

    end = time.time()
    print("Time consumed in working: ",end - start)

    #train
    print('Train Importance')
    importances_tr = get_permimp(rfr, train_x, train_y, 
                n_repeats=n_repeats, random_state=rnd,
                n_jobs=1)

    #oss
    print('Out of sample Importance')
    importances_oos  = get_permimp(rfr, oos_x, oos_y, 
                n_repeats=n_repeats, random_state=rnd, 
                n_jobs=1)
    
    #oos2
    print('Out of sample 2 Importances')
    importances_oos2  = get_permimp(rfr, oos_x2, oos_y2, 
                n_repeats=n_repeats, random_state=rnd, 
                n_jobs=1)

    #plot importances
    fig, ax = plt.subplots(1,5, figsize=(14,5),gridspec_kw={'bottom':0.15})
    plt.subplots_adjust(wspace=0.5)

    ax[0] = mdi_importances.plot.barh(ax=ax[0])
    ax[0].set_title("Feature Importances (MDI)")


    ax[1] = importances_te.plot.box(vert=False, ax=ax[1])
    ax[1].set_title("Permutation Importances \n test set")
    ax[1].axvline(x=0, color="k", linestyle="--")
    ax[1].set_xlabel("Decrease in accuracy score")

    ax[2] = importances_tr.plot.box(vert=False, ax=ax[2])
    ax[2].set_title("Permutation Importances \n train set")
    ax[2].axvline(x=0, color="k", linestyle="--")
    ax[2].set_xlabel("Decrease in accuracy score")

    ax[3] = importances_oos.plot.box(vert=False,  ax=ax[3])
    ax[3].set_title("Permutation Importances \n out of sample")
    ax[3].axvline(x=0, color="k", linestyle="--")
    ax[3].set_xlabel("Decrease in accuracy score")
    
    ax[4] = importances_oos2.plot.box(vert=False, ax=ax[4])
    ax[4].set_title("Permutation Importances \n out of sample 2")
    ax[4].axvline(x=0, color="k", linestyle="--")
    ax[4].set_xlabel("Decrease in accuracy score")

    fig.show()
    
    metr_df.index.name = 'Metric'
    
    # combine data sets into single dataframes
    train_d = train_x.join([train_y,train_t], how='left')
    test_d = test_x.join([test_y,test_t], how='left')
    oos_d = oos_x.join([oos_y,oos_t], how='left')
    oos2_d = oos_x2.join([oos_y2,oos_t2], how='left')
    
    # add predictions to the dataframes
    train_d[y_col+'_pred'] = pre_tr
    test_d[y_col+'_pred'] = predictions
    oos_d[y_col+'_pred'] = pre_oos
    oos2_d[y_col+'_pred'] = pre_oos2
    
    imp = {'mdi':mdi_importances, 
           'imp_te': importances_te, 
           'imp_tr': importances_tr,
           'imp_oos': importances_oos,
           'imp_oos2': importances_oos2}
    
    
    return metr_df, imp, train_d, test_d, oos_d, oos2_d

def rf_tune(col=['1300_02', 'SYM_H index','SatLat'], 
             y_col='400kmDensity', 
             t_col='DateTime', 
             log_col=['1300_02'], 
             lt_col=['SatMagLT'], 
             grid_space=grid_space, 
             target_dat='D:\\data\\SatDensities\\satdrag_database_grace_B.hdf5', 
             oos_dat='D:\\data\\SatDensities\\satdrag_database_grace_A.hdf5',
             n_repeats=4,
             s_sz=100000,
             scoring='neg_mean_absolute_error',
             refit=True):
    
    # create data sets
    kcol = [col,[y_col],[t_col],lt_col]
    kflt = [item for sublist in kcol for item in sublist]
    df = pd.read_hdf(target_dat)
    df = df[kflt].dropna().sample(s_sz)

    reg_x, reg_y = dat_create(dat=df,col=col,log_col=log_col,lt_col=lt_col,
                              y_col=y_col,t_col=t_col)
    reg_y = reg_y*(10**12)

    del df
    gc.collect

    # create train test splits
    train_x, test_x, train_y, test_y = train_test_split(reg_x, reg_y, 
                                                        test_size=0.7, 
                                                        random_state=rnd)
    # drop time column
    train_x = train_x.drop(columns=t_col)
    test_x = test_x.drop(columns=t_col)

    # create regressor
    rfr = RandomForestRegressor(random_state=17)
    print('Starting Grid Search')
    grid = GridSearchCV(rfr,param_grid=grid_space,cv=3, verbose=4,
                scoring=scoring, n_jobs=6, return_train_score=True,
                refit=refit)
    model_grid = grid.fit(train_x,train_y)

    return model_grid


def rf_run():
    """
    Run a set of random forest models 

    Returns
    -------
    None.
    
    Saves data frames to file for subsequent analysis

    """
    
    # out_dir 
    o_dir = 'D:\\data\\SatDensities\\'
    
    # repeats for permutation importance
    n_repeats = 5
    # columns that are not used in the model but are returned
    # to make subsequent analysis easier
    t_col = ['DateTime','storm','storm phase']
    
    # columns to log for fism and geo datasets
    fi_log = ['1300_02', '43000_09', '85550_13', '94400_18']
    
    # solar indice columns
    si_col = ['F10', 'F81', 'S10', 'S81c', 'M10', 'M81c', 'Y10', 'Y81c', 'SatLat']
            
    # fism2 columns
    fi_col = ['1300_02', '43000_09', '85550_13', '94400_18', 'SatLat']

    
    # fism2 and geo columns
    fgeo_col = ['1300_02', '43000_09', '85550_13', '94400_18', 'SYM_H index', 'AE', 'SatLat']

    # labels
    data_labels = ['SI','FI','FI_GEO']
    data_sets = [si_col, fi_col, fgeo_col]
    
    for col, d_in in zip(data_sets,data_labels):
        
        rf_dat = rf_model(col=col, t_col=t_col, log_col=fi_log, n_repeats=n_repeats)
        
        with open(o_dir+d_in+'_RFdat.pkl', 'wb') as f:
            pickle.dump(rf_dat, f)
            
        del rf_dat
        gc.collect
    
    
    
def metric_plot( ):
    """
    Function to return a pandas DataFrame which contains
    the metrics for each of the models ran from rf_run( )
    
    NOTE: If we introduce more models we can add parameters to the function
    to read in specific files

    Parameters
    ----------
    
    None
    
    : TYPE
        DESCRIPTION.

    Returns
    -------
    metr : Pandas DataFrame
        DataFrame with the metrics for each model.

    """
    
    d_dir = 'D:/data/SatDensities/'  
    d_mod = ['SI','FI','FI_GEO']
    d_leg = ['Solar', 'FISM', 'FSIM/Geo']
    
    metr = pd.DataFrame()
    
    for mod,leg in zip(d_mod,d_leg):
        
        with open(d_dir+f'{mod}_RFdat.pkl','rb') as f:
            print(f)
            t = pickle.load(f)[0].reset_index()
            t['Model'] = leg
    
        metr = pd.concat([metr,t])
        
    return metr    
    
    
    
    
    
    
    
    
    