# OpenSense Example Data

This is a collection of small example datasets, following OpenSense naming convetion, derived from larger open datasets, to be used in example notebooks of `poligrain`, `pypwsqc` and `mergeplg`.

The data will be downloaded by specific functions in `poligrain` (still to be added), similar to how it is done in [`xarray.tutorial`](https://github.com/pydata/xarray/blob/b9780e7a32b701736ebcf33d9cb0b380e92c91d5/xarray/tutorial.py) from the [xarray-data github repo](https://github.com/pydata/xarray-data).

## Guidlines for adding data

1. All data has to conform to the OpenSense data format convenctions and must be stored as NetCDF (maybe we later add some CSV examples, but this is not a priority).
2. There must be a clear reference to the original data source either in a README next to the files or in the files e.g. in the NetCDF attributes.
3. We create a directory for each original data source, e.g. `OpenMRG` when using the OpenMRG dataset. All files for different sensors and different covered periods should be placed there.
4. We create individual files for the individual sensors.
5. We provide different sizes of the same dataset by cropping to approx. 1 hour, 1 day and 1 week indicated by the file name ending e.g. `_1d.nc`.
6. File size should be as small as possible by using NetCDF compression techniques.
7. We store the notebook used for subsetting, cropping and/or processing the original data in a subdirectory called `notebooks` in the directoty of the indivdual datasets. These should of course be as reproducible as possible, but priority is to just document what was done with the data
8. The data should not change very often, ideally not at all. Otherwise we might have to come up with some kind of versioning.

## Available data

### OpenMRG
CML and gauge data from Gothenburg, Sweden, covering a summer 2015 event. Includes received
signal level from commercial microwave links, city rain gauges, SMHI gauges, and weather radar.
Full dataset: Andersson et al. 2022, Zenodo.

### OpenRainER
CML and gauge data from a European network, covering an 8-day period. Includes received signal
level from commercial microwave links and rain gauge reference measurements.
Full dataset: Zenodo (10.5281/zenodo.10593848).

### AMS PWS
Personal weather station data from Amsterdam, Netherlands. Includes rainfall and meteorological
measurements from citizen weather stations alongside reference gauge data.

### OpenMesh
CML and PWS data from New York City, covering January 2024. Includes received signal level from
75 commercial microwave links (NYC Mesh network) and rainfall, temperature, humidity, wind, and
pressure from 35 Weather Underground personal weather stations. Also includes ASOS airport
reference station data (JFK, EWR, LGA) for validation.
The full repository with detailed fetching functions and examples is available at
https://github.com/drorjac/OpenMesh
