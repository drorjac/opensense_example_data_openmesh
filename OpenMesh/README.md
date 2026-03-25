# OpenMesh

OpenMesh is a dataset of  microwave link and personal weather station (PWS)
measurements over New York City, covering January 2024. It also includes ASOS reference station
data for validation. The raw data is sourced from two Zenodo records (jacoby_2025_OpenMesh).


## Example files

The root folder contains ready-to-use NetCDF sample files at three time windows.
File names follow the convention: openmesh_{sensor}_{duration}.nc

CML files (75 links, received signal level in dBm):
  openmesh_cml_1d.nc       one day
  openmesh_cml_1w.nc       one week
  openmesh_cml_20d.nc      20 days

PWS files (35 Weather Underground stations, rainfall, temperature, humidity, wind, pressure):
  openmesh_wu_pws_1d.nc    one day
  openmesh_wu_pws_1w.nc    one week
  openmesh_wu_pws_20d.nc   20 days

ASOS reference station files (airport weather stations, used for comparison and validation):
  openmesh_asos_ws_20d.nc  20 days


## Notebooks

create_OpenMesh_data.ipynb
  Downloads the raw CML and PWS data from Zenodo into notebooks/data/jacoby_2025_OpenMesh.
  Generates sample NetCDF files for any time window using the functions in openmesh_data_io.py.
  Also produces the root example files above. 

analyze_OpenMesh_data.ipynb
  Loads CML and PWS datasets either from the root example files or by creating them on the fly
  from the raw data for a custom time window. Provides an overview of the dataset dimensions,
  link/station metadata, and basic visualizations of signal level and weather variables.

ASOS_opensense.ipynb
  Downloads and processes ASOS airport station data (JFK, EWR, LGA) for the same time window
  as the CML and PWS data. Converts the data to the opensense NetCDF format and saves it to
  the samples folder. Used to generate the reference files for validation.


## Helper modules

openmesh_data_io.py
  Central module for all data operations. Key functions:
  - download_openmesh() and download_pws_wu(): download and extract Zenodo archives
  - create_openmesh_sample() and create_pws_wu_sample(): slice a time window from raw data
  - create_asos_sample(): fetch and convert ASOS data for a given time window
  - create_all_datasets(): run all three in one call, with save and resample options
  - load_cml() and load_pws_wu(): load the full raw source files into memory

ws_opensense_conversion.py
  Handles unit conversions and variable transformations for PWS data, converting raw
  Weather Underground fields to the opensense standard format.


## Data folder

The raw downloaded data is stored in notebooks/data/jacoby_2025_OpenMesh and is not tracked
by git. Running create_OpenMesh_data.ipynb will populate it automatically. The layout is:

  raw/        full source NetCDF files downloaded from Zenodo
  meta/       station and link metadata CSVs, and HTML map visualizations
  archived/   original ZIP archives from Zenodo
  examples/   example notebooks included in the Zenodo release
  docs/       README from the Zenodo release
