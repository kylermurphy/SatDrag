# SatDrag
A set of machine learning models to explore and understand satellite drag 


# Grace

Grace A and B data. Contains:
- Time stamp (second of day, ~45s cadence)
- Satelite position
- Neutral density
- Neutral density normalized to 400 and 410 km and at Satellite height
- Number of data points in averaging bin
- Number of averaging points affected by thrusters
- Average coefficient of drag used in averaging bin

# Omni

Omni solar wind data and geomagnetic data. 

# FISM

[FISM2](https://lasp.colorado.edu/lisird/data/fism_daily_hr/) is an empirical model of the Solar Spectral Irradiance from 0.01-190nm at 0.1 nm spectral bins. This is the daily average product with one spectrum for each day, while the flare product is also available at 60 sec cadence. FISM2 daily is based on SORCE XPS L4, SDO EVE, and SORCE SOLSTICE data, and these base datasets are related to proxies such as F10.7, Mg II c/w, Lyman Alpha, and SDO/EVE/ESP Quad Diode (0-7nm).

A hiresolution data set also exists [here](https://lasp.colorado.edu/lisird/data/fism_flare_hr/)

# MSIS

[Mass Spectrometer and Incoherent Scatter model](https://kauai.ccmc.gsfc.nasa.gov/CMR/view/model/SimulationModel?resourceID=spase://CCMC/SimulationModel/NRLMSIS/v2.0). Empirical atmospheric model that extends from the ground to the exobase and describes the average observed behavior of temperature, 8 species densities, and mass density via a parametric analytic formulation.
