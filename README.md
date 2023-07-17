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

# JBM2008

The JB2008 model provides neutral density in the thermosphere using global exospheric temperatureequations driven by four solar indices/proxies to represent different solar heating sources (Bowman et al., 2008; Tobiska et al., 2008).


# Notes:
- Solar wind variables are daily variables and so we might have to a typical superposed epoch analysis if the storm phases are short as opposed to a time-normalized analysis.
- Solar Variables used in models
    - F10.7, S10, M10, Y10, F30, F81
- Geomagnetic Variables used in models
    - Kp, ap, Dst, n_sw, v_sw, IMF
- DTM2020 has to models, an operational and research model
    - Operational uses lower cadence F10.7 and KP indices as inputs
    - Research model used higher cadence F30 and Hpo indices as inputs


Description of *newish* data (**bolded** has be retrieved, *italics* not retreived): 
- **S10**, the S10.7 index is an activity indicator of the integrated 26–34 nm solar irradiance measured by the Solar Extreme-ultraviolet Monitor (SEM) instrument on the NASA/ESA Solar and Heliospheric Observatory (SOHO) satellite
- **M10**, the M10.7 index is derived from the Mg II core-to-wing ratio that originated from the NOAA series operational satellites, e.g., NOAA-16,-17,-18, which host the Solar Backscatter Ultraviolet (SBUV) spectrometer.
- **Y10**, a composite solar index of the Xb10 index, Lyman-α emission and 81-day centered smoothed F10.7. Xb10 index and is used to represent the daily energy that is deposited into the mesosphere and lower thermosphere.
- *F30*, the F30 is the Solar Flux measured by the Nobeyama Radio Observatory, which performs daily measurements of the 30 cm radio flux on an operational 7/365 basis.
- *F81*
- *Hpo* 

[JB2008 indicies](https://sol.spacenvironment.net/jb2008/indices.html)
ESA Space Weather Service Network, [Space Surveillance and Tracking (Archive of geomagnetic and solar indices for drag calculation)](https://swe.ssa.esa.int/sst_arv)
