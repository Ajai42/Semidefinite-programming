import sys
import os
import pandas as pd
import numpy as np
import argparse
import time
import datetime
import pickle

# Add the project root to the search path
sys.path.append(os.path.abspath(".."))

# Import from src
from src import *

# Function to process a single instance
def process_instance(idx, df_pickle, N):
    df = pd.read_pickle(df_pickle)
    instance = df["cnf"][idx]
    instance = [tuple(arr.tolist()) for arr in instance]
    sol, sol_nvc = rc2_solver(N, instance)
    return sol, sol_nvc

# Main function
def main(idx, N, output_file):
    # Load DataFrame
    df_pickle = f"../data/Max_2_SAT_N_{N}_λstd_0_T_1024.0_reps_50.pkl"
    df = pd.read_pickle(df_pickle)

    # Get the alpha value for the current index
    alpha = df.loc[idx, 'alpha']

    # Initialize arrays to store results
    SOL = []
    SOL_NVC = []

    start_time = time.time()  # Record the start time for the index
    sol, sol_nvc = process_instance(idx, df_pickle, N)
    SOL.append(sol)
    SOL_NVC.append(sol_nvc)
    end_time = time.time()  # Record the end time for the index
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))  # Format the elapsed time
    print(f"Index {idx} with alpha {alpha} execution time: {elapsed_time_str}")

    # Save the results to a file
    with open(output_file, 'wb') as f:
        pickle.dump((SOL, SOL_NVC), f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallelize the RC2 solver")
    parser.add_argument("--idx", type=int, required=True, help="Index for processing")
    parser.add_argument("--N", type=int, required=True, help="Value of N")
    parser.add_argument("--output_file", type=str, required=True, help="Output file to save results")
    args = parser.parse_args()
    main(args.idx, args.N, args.output_file)
