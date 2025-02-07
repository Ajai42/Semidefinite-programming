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

def main(N, h_planes):
    # Load DataFrame
    df = pd.read_pickle(f"../data/Max_2_SAT_N_{N}_λstd_0_T_1024.0_reps_50_with_solution_merged.pkl")

    # Directory containing Y matrices
    input_dir = f"../data/SDP_{N}/"

    # Using rounding on the Y matrices
    SDP_87_nvc = []
    SDP_gw_nvc = []
    SDP_rot_nvc = []
    SDP_skew_nvc = []
    SDP_rot_skew_nvc = []

    for idx in tqdm(range(len(df))):
        # Load the Y matrix from the file
        input_file = os.path.join(input_dir, f"{idx}.pkl")
        with open(input_file, 'rb') as f:
            Y = pickle.load(f)
        
        instance = df["cnf"][idx]
        instance = [tuple(arr.tolist()) for arr in instance]
        
        # Perform all rounding techniques
        # NVC_87 = rounding_87(Y, N, instance, h_planes)
        # NVC_gw = gw_rounding(Y, N, instance, h_planes)
        NVC_rot = rot_rounding(Y, N, instance, h_planes)
        # NVC_skew = skew_rounding(Y, N, instance, h_planes)
        NVC_rot_skew = rot_skew_rounding(Y, N, instance, h_planes)
        
        # SDP_87_nvc.append(NVC_87)
        # SDP_gw_nvc.append(NVC_gw)
        SDP_rot_nvc.append(NVC_rot)
        # SDP_skew_nvc.append(NVC_skew)
        SDP_rot_skew_nvc.append(NVC_rot_skew)
        
    # Add the SDP_nvc results to the DataFrame
    # df["SDP_87_nvc"] = SDP_87_nvc
    # df["SDP_gw_nvc"] = SDP_gw_nvc
    df["SDP_rot_nvc"] = SDP_rot_nvc
    # df["SDP_skew_nvc"] = SDP_skew_nvc
    df["SDP_rot_skew_nvc"] = SDP_rot_skew_nvc

    # Save the updated DataFrame
    df.to_pickle(f"../data/Max_2_SAT_N_{N}_λstd_0_T_1024.0_reps_50_with_solution_and_SDP.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform rounding on Y matrices and update DataFrame")
    parser.add_argument("--N", type=int, required=True, help="Value of N")
    parser.add_argument("--h_planes", type=int, default=1000, help="Number of random hyperplanes to use for rounding")
    args = parser.parse_args()
    main(args.N, args.h_planes)