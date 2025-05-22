# DMM_param_tuning-PT
This repository includes all the necessary code to reproduce the results of [Sipling et al. 2025](**ADD LINK TO PAPER**). In particular, this software:
1. Simulates the evolution of Digital Memcomputing Machines (DMMs)
2. Optimizes DMM parameters using Parallel Tempering (PT)
3. Visualizes some of the paper's key results (scalabilities, avalanche distributions, time-to-solution distributions, and number of anti-instantons)


## Runnable files

There are 4 main runnable files, listed below. All of their free parameters are listed and described at the top of each file. They should be run in the following order:

### *dataset.py*
Produces many unique [Barthel](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.88.188701) 3-SATisfiability instances at a variety of sizes $N$ (number of logical variables).

If other SAT instance are to be simulated, they will need to be uploaded to this directory from elsewhere.

### *main_PT.py*
Extracts relevant SAT instances, then simulates a DMM for a fixed amount of time. Statistics are then extracted from that run (in particular, how many timesteps did it take to solve each instance in a batch) to evaluate the quality of that DMM with some metric (can be changed!). Parameters are then updated using Parallel Tempering (PT; [Earl and Deem 2005)](https://pubs.rsc.org/en/content/articlehtml/2005/cp/b509983h).

One can also produce an energy profile $<E(T)>$ based on simulated annealing over many individual replicas (with no mixing). This establishes the relevant lower and upper temperature bounds to be used in the main PT process.

### *benchmark.py*
Simulates a single DMM with optimal parameters over many iterations, as determined via the PT procedure. Plotting additional data/variable trajectories is optional.


## Additional functionalities

1. Further data analyses can be performed upon completing the PT procedure by uncommenting lines 63-181 in *data_analysis.py* and lines 258-278 and 290-301 in *solver_PT.py* simultaneously.
2. The functional form of the DMM equations can be varied in *operators.py*. In particular, equations exist there which are better for XORSAT (in particular, 3R3X instances) than ORSAT.
3. Lines 162-191 in *benchmark.py* feature additional functionalities which can be toggled, including extracting/plotting avalanches anti-instantons, and time-to-solution (TTS) distributions. **When toggling anti-instanton extraction ON, lines 139 and 158-174 of dmm_utils.py must be UNCOMMENTED, and lines 141 and 176-182 of dmm_utils.py must be COMMENTED (and vice-versa, when OFF).**
4. Lines 85-94 in *dmm_utils.py* can be uncommented at any time during DMM simulation to write any found SAT solutions to their own Solution/ directory.


## Visualization

Both *visualization.py* and *visualization_wide_param_search.py* were used in [Sipling et al. 2025](**ADD LINK TO PAPER**) to produce all its figures. The relevant simulations (over all parameters, with identical file tags) would need to first be run to reproduce such figures.


## Contact

Email: csipling@ucsd.edu