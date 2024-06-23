import mlrose_hiive as mlrose
import time
import glob
import pandas as pd
import ast
import multiprocessing
import numpy as np
NUM_EVAL_REPEATS = 30
output = []



def find_best_parameters(problem_name):
    # Get the list of files
    files = glob.glob(f'{problem_name}_parameter_tuning_*.csv')
    
    # Find the file with the highest number
    latest_file = max(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # Load the latest file into a pandas dataframe
    df = pd.read_csv(latest_file)
    
    # Group by the specified columns and calculate the mean of the fitness column
    grouped = df.groupby(['alg_name', 'ts_length', 'params']).fitness.mean().reset_index()
    
    # Find the params with the highest mean fitness score for each alg_name and ts_length
    return grouped.loc[grouped.groupby(['alg_name', 'ts_length']).fitness.idxmax()].reset_index(drop=True)

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

def worker(problem_name, row):
    if problem_name == "flip_flop":
        problem = mlrose.FlipFlop()
    elif problem_name=="queens":
        problem = mlrose.CustomFitness(queens_fitness, problem_type="discrete")
    elif problem_name=="four_peaks":
        problem = mlrose.FourPeaks()
    else: raise ValueError()

    problem_fit = mlrose.DiscreteOpt(length=row["ts_length"], fitness_fn=problem, maximize=True)
    if row["alg_name"]=="genetic_alg":
        start_time = time.time()
        _, fitness, curve =  mlrose.genetic_alg(problem_fit, **ast.literal_eval(row["params"]),curve=True)
    elif row["alg_name"]=="random_hill_climbing":
        start_time = time.time()
        _, fitness, curve =  mlrose.random_hill_climb(problem_fit, **ast.literal_eval(row["params"]),curve=True)
    elif row["alg_name"]=="simulated_annealing":
        params = ast.literal_eval(row["params"].replace("'schedule': ", "'schedule': '").replace(")", ")'"))
        start_time = time.time()
        if "ExpDecay" in params["schedule"]:
            schedule = mlrose.ExpDecay(params["init_temp"], params["exp_cost"], params["min_temp"])
        elif "ArithDecay" in params["schedule"]:
            schedule = mlrose.ArithDecay(params["init_temp"], params["decay"], params["min_temp"])
        elif "GeomDecay" in params["schedule"]:
            schedule = mlrose.GeomDecay(params["init_temp"], params["decay"], params["min_temp"])
        else: raise ValueError()

        _, fitness, curve =  mlrose.simulated_annealing(problem_fit, schedule=schedule, curve=True)
    else: raise ValueError()

    duration = time.time() - start_time
    print(f'{time.time()}: {row["alg_name"], row["ts_length"]}')
    return [problem_name, row["alg_name"], row["ts_length"], row["params"], fitness, curve.tolist(), duration]

def multi_process_evaluate():
    with multiprocessing.Pool(processes=8) as pool:
        results = [
            pool.apply_async(worker,(problem_name, row))
            for problem_name in {"queens", "four_peaks"}
            for (_, row) in find_best_parameters(problem_name).iterrows()
            for _ in range(0,NUM_EVAL_REPEATS)
        ]
        print(f"Distributing work for {len(results)} items")

        pool.close()
        pool.join()
        output = map(lambda x: x.get(), results)
        output_df = pd.DataFrame(output, columns=["problem_name", "alg_name", "ts_length", "params", "fitness", "curve", "duration"])
        output_df.to_csv(f"combined_evaluation_{time.time()}.csv")

    print("done")

if __name__ == "__main__":  
    multi_process_evaluate()