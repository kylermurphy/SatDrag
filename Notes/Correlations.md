# Feature Correlations for New Model 

Identify features that can be used in the new Random Forest model which accounts for altitude variations in atmonspheric density.

## Datasets

- FISM 2
    - High cadence Stan Bands
    - Note that the dataset changes around 2010 and the correlations get better as more data is introduced to the data set
- GOES
    - Lyman alpha solar irridiance
    - 2010-2020
- Omni
    - Geomagnetic Conditions
- Grace B
    - 2002-2018
    
If we want to use GOES then we have  ~six years of data to train on. This is still the longest single satellite dataset.

## FISM 2 Correlations

Including data pre 2010 the correlations peak for specific bands. Post 2010 all bands, except the shortest, have good correlations.

All data             |  Post 2010
:-------------------------:|:-------------------------:
![](Figures\FISM_cor.png)  |  ![](Figures\FISM2010_cor.png) 


## GOES correlations

The goes x-ray bands are poorly correlated while the 1216 nm band (Lyman alpha is very well correlated)

|               |      All |     Quiet |     Storm |       Main |   Recovery |
|:--------------|---------:|----------:|----------:|-----------:|-----------:|
| xrsa_flux_g15 | 0.022038 | 0.0213592 | 0.0248082 | 0.00113827 |  0.0359312 |
| xrsb_flux_g15 | 0.067905 | 0.0852563 | 0.0550155 | 0.0131563  |  0.0730478 |
| irr_1216      | 0.582378 | 0.603784  | 0.550492  | 0.50193    |  0.564081  |

## OMNI Correlations

Several solar wind and geomagnetic indices have good correlations. They generally peak during the mainphase of geomagentic storms. 

![](Figures\OMNI_cor.png)


## Lagged Correlations

Lagged Correlations show a similar features with FISM and GOES correlations having a second peak around 27 days. Correlations for all variables remain high for several __lagged__ hours which gives the possibility of some forecasting. 

| Heatmap | Time Series to 45 days | Time series to several hours | 
| ------- | ---------------------- | ----------------------- |
| ![](Figures\FISM_lagged_cor.png) | ![](Figures\FISM_lagged_cor_days.png) |![](Figures\FISM_lagged_cor_hours.png)
| ![](Figures\OMNI_lagged_cor.png) | ![](Figures\OMNI_lagged_cor_days.png) |![](Figures\OMNI_lagged_cor_hours.png)

## Feature Colinearity

Take a subset of features with good correlations to look at the colinearity and whether features can possibly be dropped.

![](Figures\Colinearity.png)

## Feature Selection 

