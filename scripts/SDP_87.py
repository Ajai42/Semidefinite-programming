import pandas as pd
import sys
import numpy as np
import os
from tqdm import tqdm

# Add the project root to the search path
sys.path.append(os.path.abspath(".."))

# Import from src
from src import *

N = 50


# Load DataFrame
# df = pd.read_pickle("../data/Max_2_SAT_N_"+str(N)+"_λstd_0_T_1024.0_reps_50_with_solution.pkl")
df = pd.read_pickle("../data/Max_2_SAT_N_"+str(N)+"_λstd_0_T_1024.0_reps_50_with_solution_merged.pkl")

# using SDP on the instance

SDP_nvc = []
for idx in tqdm(range(len(df))):
    
    instance = df["cnf"][idx]
    instance = [tuple(arr.tolist()) for arr in instance]
    
    Y   = SDP_max_sat(N, instance, "Mosek")
    NVC = rounding_87(Y, N, instance)
    SDP_nvc.append(NVC)
    
df["SDP_nvc"] = SDP_nvc

df.to_pickle("../data/Max_2_SAT_N_"+str(N)+"_λstd_0_T_1024.0_reps_50_with_solution_and_SDP_test.pkl") 
