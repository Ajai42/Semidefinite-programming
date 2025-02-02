import pandas as pd
import sys
import os
from tqdm import tqdm
import pickle
import argparse

# Add the project root to the search path
sys.path.append(os.path.abspath(".."))

# Import from src
from src import *

def main(N):
    # Load DataFrame
    df = pd.read_pickle(f"../data/Max_2_SAT_N_{N}_λstd_0_T_1024.0_reps_50_with_solution_merged.pkl")

    # Create directory to save Y matrices
    output_dir = f"../data/SDP_{N}/"
    os.makedirs(output_dir, exist_ok=True)

    # Using SDP on the instance
    for idx in tqdm(range(len(df))):
        instance = df["cnf"][idx]
        instance = [tuple(arr.tolist()) for arr in instance]
        
        Y = SDP_max_sat(N, instance, "Mosek")
        
        # Save the Y matrix to a separate file
        output_file = os.path.join(output_dir, f"{idx}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(Y, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and save Y matrices using SDP")
    parser.add_argument("--N", type=int, required=True, help="Value of N")
    args = parser.parse_args()
    main(args.N)