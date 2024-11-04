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

...to be added
