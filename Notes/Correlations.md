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

