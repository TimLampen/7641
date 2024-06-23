import six
import sys
import mlrose_hiive as mlrose
import multiprocessing
from datetime import datetime
import pandas as pd
import random
import time
from sklearn.model_selection import ParameterGrid
import numpy as np
from functools import reduce

pd.set_option('display.max_colwidth', 1000)

RANDOM_SEED = 42
NUM_TUNING_REPEATS = 10
NUM_EVAL_REPEATS = 10
random.seed(RANDOM_SEED)

genetic_param_grid = {
    'pop_size': [100, 200, 400],
    'mutation_prob': [0.1, 0.2, 0.4],
}

random_hill_param_grid = {
    'max_attempts': [10,20],
}

simulated_annealing_param_grid = {
    'schedule': [mlrose.ExpDecay(init_temp,exp_cost,min_temp) for init_temp in np.arange(0.5,1.1,.2) for exp_cost in np.arange(0.001, 0.011, 0.001) for min_temp in np.arange(0.001, 0.011, 0.02)]
    + [mlrose.GeomDecay(init_temp,decay,min_temp) for init_temp in np.arange(0.5,1.1,.2) for decay in [0.90, 0.92, 0.95, 0.97, 0.99] for min_temp in np.arange(0.001, 0.011, 0.02)]
    + [mlrose.ArithDecay(init_temp,decay,min_temp) for init_temp in np.arange(0.5,1.1,.2) for decay in np.arange(0.001, 0.011, 0.002) for min_temp in np.arange(0.001, 0.011, 0.02)]
}

# Function to calculate Euclidean distance
def euclidean_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

# Define the fitness function using the inverted distances
def max_tsp_fitness(state,coords):
    total_distance = 0
    for i in range(len(state) - 1):
        dist = euclidean_distance(coords[state[i]], coords[state[i + 1]])
        total_distance += dist
    total_distance += euclidean_distance(coords[state[-1]], coords[state[0]])  # Complete the loop

    # Invert the distance to convert it into a maximization problem
    max_possible_distance = len(state) * max([euclidean_distance(c1, c2) for c1 in coords for c2 in coords])
    return max_possible_distance - total_distance

def shift(a, num, fill_value=np.nan):
    result = np.empty(a.shape)
    if num > 0:
        result[:num] = fill_value
        result[num:] = a[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = a[-num:]
    else:
        result[:] = a
    return result

def queens_fitness(state):
    """Evaluate the fitness of a state vector.

    Parameters
    ----------
    state: array
        State array for evaluation.

    Returns
    -------
    fitness: float
        Value of fitness function.
    """

    # check for horizontal matches.
    f_h = (np.unique(state, return_counts=True)[1]-1).sum()

    # check for diagonal matches.
    # look at the state_shifts to figure out how this works. (I'm quite pleased with it)
    ls = state.size
    # rows 0-3:   checking up left.
    # rows 4-7:   checking down right.
    # rows 8-11:  checking up right
    # rows 12-15: checking down left
    state_shifts = np.array([shift(state, i)+i for i in np.arange(1-ls, ls) if i != 0] +
                            [shift(state, -i)+i for i in np.arange(1-ls, ls) if i != 0])
    # state_shifts[(state_shifts < 0)] = np.NaN
    # state_shifts[(state_shifts >= ls)] = np.NaN

    f_d = np.sum(state_shifts == state) // 2  # each diagonal piece is counted twice
    fitness = f_h + f_d
    return float(-fitness)

def tune_algorithm(problem_name, ts_length, alg_name, alg_ptr, param_grid):
    output = []
    for params, _ in [(params, repeat) for params in ParameterGrid(param_grid) for repeat in range(0, NUM_TUNING_REPEATS)]:        
        if problem_name == "four_peaks":
            fitness = mlrose.FourPeaks(t_pct=0.15)
            problem_fit = mlrose.DiscreteOpt(length=ts_length, fitness_fn=fitness, maximize=True)
        elif problem_name == "tsp":
            coord_list = [(random.randint(0, 1000),random.randint(0,1000)) for _ in range(0, ts_length)]
            fitnesss = mlrose.CustomFitness(max_tsp_fitness, problem_type="tsp",coords=coord_list)
            problem_fit = mlrose.TSPOpt(length=len(coord_list), fitness_fn=fitnesss, maximize=True)
        elif problem_name == "queens":
            fitnesss = mlrose.CustomFitness(queens_fitness, problem_type="discrete")
            problem_fit = mlrose.DiscreteOpt(length=ts_length, fitness_fn=fitnesss, maximize=True)
       
        fitness = alg_ptr(problem_fit, **params)

        if alg_name != "simulated_annealing":
            output.append([alg_name, ts_length, params, fitness[1]])
            continue

        if isinstance(params["schedule"], mlrose.ExpDecay):
            output.append([alg_name, ts_length, params | {"init_temp": float(params["schedule"].init_temp), "exp_cost": float(params["schedule"].exp_const), "min_temp": float(params["schedule"].min_temp)}, fitness[1]])
        elif isinstance(params["schedule"], mlrose.GeomDecay):
            output.append([alg_name, ts_length, params | {"init_temp": float(params["schedule"].init_temp), "decay": float(params["schedule"].decay), "min_temp": float(params["schedule"].min_temp)},fitness[1]])
        elif isinstance(params["schedule"], mlrose.ArithDecay):
            output.append([alg_name, ts_length, params | {"init_temp": float(params["schedule"].init_temp), "decay": float(params["schedule"].decay), "min_temp": float(params["schedule"].min_temp)}, fitness[1]])

    print(f"{datetime.now()} {ts_length, alg_name} done")
    return output

def multi_process_tune(problem_name):
    print(f"Tuning for {problem_name}")
    with multiprocessing.Pool(processes=8) as pool:
        results = [
            pool.apply_async(tune_algorithm,(problem_name, ts_length, alg_name, alg_ptr, param_grid))
            for ts_length in ([5, 10, 25, 50, 75, 100] if not problem_name=="four_peaks" else [5, 10, 25, 50, 75, 100, 250, 500])
            for alg_name, alg_ptr, param_grid in [
                    ("genetic_alg", mlrose.genetic_alg, genetic_param_grid),
                    ("random_hill_climbing", mlrose.random_hill_climb, random_hill_param_grid),
                    ("simulated_annealing", mlrose.simulated_annealing, simulated_annealing_param_grid)
                ]
        ]
        print(f"Distributing work for {len(results)} items")

        pool.close()
        pool.join()
        list_results = map(lambda x: x.get(), results)
        output = reduce(lambda x, y: x + y, list_results)
        output_df = pd.DataFrame(output, columns=["alg_name", "ts_length", "params", "fitness"])
        output_df.to_csv(f"{problem_name}_parameter_tuning_{time.time()}.csv")

    print("done")

if __name__ == "__main__":  
    multi_process_tune("queens")
    multi_process_tune("four_peaks")
    # multi_process_tune("tsp")
