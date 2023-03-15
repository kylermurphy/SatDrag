import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

n_repeats=10
rnd = 17

params = {
    "n_estimators": 150,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf":20,
    "warm_start":False,
    "oob_score":True,
    "random_state": rnd,
    "max_features":4,
    "n_jobs":4
}

col = ['DateTime','1300', '2500', '5100', '11250',
        '18950', '43000', '85550', '94400',
        '103850', 'Bz_GSM', 'Vsw', 'Esw',
        'AE', 'AL', 'AU', 'SYM_H index',
        'storm', 'storm phase', 
        'SatLat', '400kmDensity']

col = ['DateTime','1300', '2500','Bz_GSM', 'Vsw', 'Esw',
        'AE', 'AL', 'AU', 'SYM_H index',
        'SatLat', '400kmDensity']

log_col = ['1300', '2500']

y_col = '400kmDensity'
t_col = 'DateTime'  

# create data sets
fn = 'D:\\data\\SatDensities\\satdrag_database_grace_B.hdf5'
df = pd.read_hdf(fn)
reg_x = df[col].dropna().copy()
# map local time and magnetic local time
# to 0-1 using cos and sin so that it is continous
# across midnight 
reg_x['cos_mlt'] = np.cos(df['SatMagLT']*2*np.pi/24.)
reg_x['sin_mlt'] = np.sin(df['SatMagLT']*2*np.pi/24.)
reg_x['1300'] = np.log10(reg_x['1300'])
reg_x['2500'] = np.log10(reg_x['2500'])

# create the y data and drop it from the x data
reg_y = (reg_x[y_col].copy())*(10**12)
reg_x = reg_x.drop(columns=y_col)

# create data set from Grace A
fn = 'D:\\data\\SatDensities\\satdrag_database_grace_A.hdf5'
df_oos = pd.read_hdf(fn)
oos_x = df_oos[col].dropna().copy()
oos_x['cos_mlt'] = np.cos(df_oos['SatMagLT']*2*np.pi/24.)
oos_x['sin_mlt'] = np.sin(df_oos['SatMagLT']*2*np.pi/24.)
oos_x['1300'] = np.log10(oos_x['1300'])
oos_x['2500'] = np.log10(oos_x['2500'])

oos_y = (oos_x[y_col].copy())*(10**12)
oos_t = oos_x[t_col].copy()
oos_x = oos_x.drop(columns=[y_col,t_col])

# create train test splits
train_x, test_x, train_y, test_y = train_test_split(reg_x, reg_y, 
                                                    test_size=0.7, 
                                                    random_state=rnd)

# get and drop DateTime column
train_t = train_x[t_col].copy()
test_t = test_x[t_col].copy()

train_x = train_x.drop(columns=t_col)
test_x = test_x.drop(columns=t_col)

print('Train and fit model')

rfr = RandomForestRegressor(**params)
rfr.fit(train_x, train_y)

#Make predictions and calculate error
predictions = rfr.predict(test_x)
pre_oos = rfr.predict(oos_x)

#the mean absolute error
mea = mean_absolute_error(test_y, predictions)
mae_oss = mean_absolute_error(oos_y,pre_oos)

#mean absolute percentage error
mape = mean_absolute_percentage_error(test_y, predictions)
mape_oss = mean_absolute_percentage_error(oos_y,pre_oos)



#Print r-squared score of model
r2 = r2_score(test_y, predictions)
r2_oos = r2_score(oos_y, pre_oos)

print(f"MAE test/oos: {mea}/{mae_oss}")
print(f"MAPE test/oos: {mape}/{mape_oss}")
print(f"Score test/oos: {r2}/{r2_oos}")

#Examine feature importances
feature_names = rfr.feature_names_in_
mdi_importances = pd.Series(rfr[-1].feature_importances_, 
                            index=feature_names).sort_values(ascending=True)


#Examin importances with test set
print('Test Importance')
t_te = permutation_importance(
    rfr, test_x, test_y, n_repeats=n_repeats, random_state=rnd, n_jobs=4
)

sorted_importances_idx = t_te.importances_mean.argsort()
importances_te = pd.DataFrame(
    t_te.importances[sorted_importances_idx].T,
    columns=test_x.columns[sorted_importances_idx],
)

#train
print('Train Importance')
t_tr = permutation_importance(
    rfr, train_x, train_y, n_repeats=n_repeats, random_state=rnd, n_jobs=4
)

sorted_importances_idx = t_tr.importances_mean.argsort()
importances_tr = pd.DataFrame(
    t_tr.importances[sorted_importances_idx].T,
    columns=test_x.columns[sorted_importances_idx],
)

#oss
print('Out of sample Importance')
t_oos = permutation_importance(
    rfr, oos_x, oos_y, n_repeats=n_repeats, random_state=rnd, n_jobs=4
)

sorted_importances_idx = t_oos.importances_mean.argsort()
importances_oos = pd.DataFrame(
    t_oos.importances[sorted_importances_idx].T,
    columns=oos_x.columns[sorted_importances_idx],
)

#plot importances
fig, ax = plt.subplots(1,3, figsize=(11,5),gridspec_kw={'bottom':0.15})
plt.subplots_adjust(wspace=0.5)

ax[0] = mdi_importances.plot.barh(ax=ax[0])
ax[0].set_title("Feature Importances (MDI)")


ax[1] = importances_te.plot.box(vert=False, whis=10, ax=ax[1])
ax[1].set_title("Permutation Importances \n test set")
ax[1].axvline(x=0, color="k", linestyle="--")
ax[1].set_xlabel("Decrease in accuracy score")

ax[2] = importances_tr.plot.box(vert=False, whis=10, ax=ax[2])
ax[2].set_title("Permutation Importances \n train set")
ax[2].axvline(x=0, color="k", linestyle="--")
ax[2].set_xlabel("Decrease in accuracy score")

ax[2] = importances_oos.plot.box(vert=False, whis=10, ax=ax[2])
ax[2].set_title("Permutation Importances \n out of sample")
ax[2].axvline(x=0, color="k", linestyle="--")
ax[2].set_xlabel("Decrease in accuracy score")

fig.show()