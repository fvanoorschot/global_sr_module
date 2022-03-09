# global_sr_module

# introduction
This module can be used to calculate root zone storage capacities globally.  

# installation
For this module installation of anaconda/miniconda is required (https://docs.conda.io/en/latest/miniconda.html) 
This module uses the 'ewatercycle' python environment. Details on installation of this environment can be found here: https://ewatercycle.readthedocs.io/en/latest/system_setup.html#conda-environment.

# data
The following input data is needed:
- global gridded daily precipitation data (netcdf)
- global gridded daily potential evaporation data (netcdf)
- global gridded daily temperature data (netcdf)
- GSIM catchment yearly discharge timeseries (https://essd.copernicus.org/articles/10/787/2018/)
- GSIM catchment shapes (shapefile) (https://essd.copernicus.org/articles/10/765/2018/)

# usage
The module calculates global Sr step by step:

1. 



