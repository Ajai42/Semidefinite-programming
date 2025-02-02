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

    # Directory containing Y matrices
    input_dir = f"../data/SDP_{N}/"

    # Using rounding on the Y matrices
    SDP_nvc = []
    for idx in tqdm(range(len(df))):
        # Load the Y matrix from the file
        input_file = os.path.join(input_dir, f"{idx}.pkl")
        with open(input_file, 'rb') as f:
            Y = pickle.load(f)
        
        instance = df["cnf"][idx]
        instance = [tuple(arr.tolist()) for arr in instance]
        
        NVC = rounding_87(Y, N, instance)
        SDP_nvc.append(NVC)
        
    # Add the SDP_nvc results to the DataFrame
    df["SDP_87_nvc"] = SDP_nvc

    # Save the updated DataFrame
    df.to_pickle(f"../data/Max_2_SAT_N_{N}_λstd_0_T_1024.0_reps_50_with_solution_and_SDP_87.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform rounding on Y matrices and update DataFrame")
    parser.add_argument("--N", type=int, required=True, help="Value of N")
    args = parser.parse_args()
    main(args.N)