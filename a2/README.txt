# Project 2 README
This section of code contains a lot assets which were used to generate the results which are displayed on the paper. The ones that are required to reproduce the graphs are:

**hyper_parameter_tuning.py** - When run using `python3 ./hyper_parameter_tuning.py` will generate the best set of parameters for each algorithm for each problem for each problem length.

**evaluation.py** - When run using `python3 ./evaluation.py` will use the tuned parameters to evaluate them on a bunch of different heuristics, including iterations, fevals, etc.

**randomized_optimization.ipynb** - When run will use the output of the evaluation script and create the required graphs.

**use_for_nn_2.ipynb** - You must insert the best parameters for each model into the second cell, but then it will evaluate the performance against the default gradient descent method.