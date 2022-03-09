# global_sr_module

# Introduction
This module can be used to calculate root zone storage capacities globally.  

# Installation
For this module installation of anaconda (https://docs.anaconda.com/anaconda/install/index.html) or miniconda (https://docs.conda.io/en/latest/miniconda.html) is required. Different python packages and environments are used. Details on installation and use of those are provided in the jupyter notebooks.

# Data
The following input data is needed:
- global gridded daily precipitation data (netcdf)
- global gridded daily potential evaporation data (netcdf)
- global gridded daily temperature data (netcdf)
- GSIM catchment yearly discharge timeseries (https://essd.copernicus.org/articles/10/787/2018/)
- GSIM catchment shapes (shapefile) (https://essd.copernicus.org/articles/10/765/2018/)

# Codes
The module consists of 3 run-scripts (jupyter notebooks) and several function scripts (.py)

**run scripts**
1. *run_script_main* is the main run script that guides you through the entire calculation procedure.
2. *run_script_grid_to_catchments* is used to extract catchment timeseries from the global gridded data. *run_script_main* tells you when to switch to this script.
3. *run_script_earth_engine* is used to extracth catchment timeseries from satellite products using google earth engine. *run_script_main* tells you when to switch to this script.

**function scripts**
1. *f_*
2. *f_*
3. *f_*



