# gA chiral-continuum extrapolation scripts

[![DOI](https://zenodo.org/badge/85146210.svg)](https://zenodo.org/badge/latestdoi/85146210)

This repository contains Python scripts that are released with the publication of [An accurate calculation of the nucleon axial charge with lattice QCD](https://arxiv.org/abs/1704.01114).  They have been tested extensively with Python 2.7.

We release the scripts used to perform the chiral-continuum extrapolation along with bootstrapped correlator results necessary for the extrapolation in `hdf5` format.

- The analysis is performed by running `python c51_gA_analysis.py` where user selectable options may be viewed in the help section by assigning the flag `--h`.
- User defined constants for the extrapolation, and plotting parameters for `matplotlib` are assigned in `data_params.py`.
- Various fit ansatzes and their corresponding chi^2 statistic is defined in `ga_fit_funcs.py`.
- Plotting scripts for the extrapolation curves as a function of mpi, a, and mL are defined in `utils.py`.
- `c51_gA_analysis.py` may be used to bootstrap the extrapolation results. If this is done, the bootstrapped results will be written to `c51_ga.sqlite` in `sqlite3` format. This file may be read by running `sqlite3 c51_ga.sqlite` from the terminal.
- `sqlite_store.py` contains the script which creates and writes to the `c51_ga.sqlite` database.
- `c51_gA_histogram.py` generates histograms from bootstrapped extrapolation results. Options for this scripts are accessible under the function `fit_list()` inside the file. The user may choose the set of models considered in the AIC weighted average. The script will plot histograms of the individual extrapolations, as well as the AIC averaged histogram.

# Dependencies

We rely on the `numpy`, `scipy`, `matplotlib`, `iminuit`, `sqlite3`, `tqdm`, and `tables` Python modules.  They may be installed via the Python package utility `pip` via

```
pip install numpy scipy matplotlib iminuit tqdm tables
```

The `tables` module relies on [HDF5][hdf5], which is commonly installed in many supercomputing environments and can easily be installed on local machines.


[arxiv]:    http://www.arxiv.org/link/to/paper
[hdf5]:     https://www.hdfgroup.org/hdf5/
