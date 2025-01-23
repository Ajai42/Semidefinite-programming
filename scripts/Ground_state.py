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
df = pd.read_pickle("../data/Max_2_SAT_N_"+str(N)+"_λstd_0_T_1024.0_reps_50.pkl")

# Get the exact solution using rc2 solver

SOL = []
SOL_NVC = []
for idx in tqdm(range(len(df)), desc="N="+str(N)):

    instance = df["cnf"][idx]
    instance = [tuple(arr.tolist()) for arr in instance]

    sol, sol_nvc = rc2_solver(N, instance)
    SOL.append(sol)
    SOL_NVC.append(sol_nvc)
    
df["Solution"] = SOL
df["Solution_nvc"] = SOL_NVC

df.to_pickle("../data/Max_2_SAT_N_"+str(N)+"_λstd_0_T_1024.0_reps_50_with_solution.pkl") 
