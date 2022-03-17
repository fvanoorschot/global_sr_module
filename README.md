# global_sr_module

# Introduction
This module can be used to calculate root zone storage capacities globally.  

# Installation
This module requires a linux environment because it uses python packages that are only compatible with linux. When you are working on windows10, you could use WSL. Detailed steps on how to use this module in WSL are provided here: https://docs.google.com/document/d/1-NzAk0YgFRNr7qcqgXqP1tij6xgiNaLK2S2uuHJIlV4/edit?usp=sharing 

The basic steps to use this module in miniconda are as follows:
1. Install miniconda in your home directory and activate (https://docs.conda.io/en/latest/miniconda.html)
- *wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh*
- *bash ~/Miniconda3-latest-Linux-x86_64.sh*
- *source $HOME/miniconda3/bin/activate*
2. Clone the files in this git repository to your home directory
- *git clone https://github.com/fvanoorschot/global_sr_module.git*
3. Create your conda environment
- The git repository contains a sr_environment.yml file. This is the conda environment with all the required packages. 
- Install the sr_environment: 
   *conda env create --file sr_environment.yml*
- Activate your environment:
   *conda activate sr_env*

# Data
The following input data is needed:
- global gridded daily precipitation data (netcdf)
- global gridded daily potential evaporation data (netcdf)
- global gridded daily temperature data (netcdf)
- GSIM catchment yearly discharge timeseries (https://essd.copernicus.org/articles/10/787/2018/)
- GSIM catchment shapes (shapefile) (https://essd.copernicus.org/articles/10/765/2018/)

An example dataset can be downloaded from: https://surfdrive.surf.nl/files/index.php/s/LJIf99kABsHszdq

# Running
The module consists of three run-scripts (jupyter notebooks) and five function scripts (.py)

**run scripts**
1. *run_script_main* is the main run script that guides you through the entire calculation procedure.
2. *run_script_grid_to_catchments* is used to extract catchment timeseries from the global gridded data. *run_script_main* tells you when to use this script.
3. *run_script_earth_engine* is used to extract catchment timeseries from satellite products using google earth engine. *run_script_main* tells you when to use this script.

**function scripts**
1. *f_preprocess_discharge* -> preprocessing GSIM discharge data
2. *f_grid_to_catchments* -> extract catchment timeseries from gridded products
3. *f_earth_engine* -> extract catchment average values from satellite products using Google Earth Engine
4. *f_sr_calculation* -> calculate catchment root zone storage capacity based on catchment water balances
5. *f_catch_characteristics* -> calculate catchment climate and landscape characteristics and store in table

# Examples
*add this*


