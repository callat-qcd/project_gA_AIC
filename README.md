# gA chiral-continuum extrapolation scripts

This repository contains Python scripts that are released with the publication of "Towards a precise calculation of the nucleon axial charge with lattice QCD" <link to arxiv page>.

We release the scripts used to perform the chiral-continuum extrapolation along with bootstrapped correlator results necessary for the extrapolation in `hdf5` format.

- The analysis is performed by running `python c51_gA_analysis.py` where user selectable options may be viewed in the help section by assigning the flag `--h`.
- User defined constants for the extrapolation, and plotting parameters for `matplotlib` are assigned in `data_params.py`.
- Various fit ansatzes and their corresponding chi^2 statistic is defined in `ga_fit_funcs.py`.
- Plotting scripts for the extrapolation curves as a function of mpi, a, and mL are defined in `utils.py`.
- `c51_gA_analysis.py` may be used to bootstrap the extrapolation results. If this is done, the bootstrapped results will be written to `c51_ga.sqlite` in `sqlite3` format. This file may be read by running `sqlite3 c51_ga.sqlite` from the terminal.
- `sqlite_store.py` contains the script which creates and writes to the `c51_ga.sqlite` database.
- `c51_gA_histogram.py` generates histograms from bootstrapped extrapolation results. Options for this scripts are accessible under the function `fit_list()` inside the file. The user may choose the set of models considered in the AIC weighted average. The script will plot histograms of the individual extrapolations, as well as the AIC averaged histogram.

```
Here's a rabbit
 ()()
(_ _)
(u u)o
Cheers!
```
