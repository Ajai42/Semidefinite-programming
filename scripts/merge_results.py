import numpy as np
import pandas as pd
import os
import pickle
import argparse

def merge_results(output_dir, df_pickle, output_file):
    output_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.pkl')]

    results = []

    for file in output_files:
        with open(file, 'rb') as f:
            sol, sol_nvc = pickle.load(f)
            idx = int(os.path.basename(file).split('_')[1].split('.')[0])  # Extract index from filename
            results.append((idx, sol, sol_nvc))

    # Sort results by index
    results.sort(key=lambda x: x[0])

    # Separate SOL and SOL_NVC
    SOL = [sol for idx, sol, sol_nvc in results]
    SOL_NVC = [sol_nvc for idx, sol, sol_nvc in results]

    # Convert lists to numpy arrays
    SOL = np.array(SOL)
    SOL_NVC = np.array(SOL_NVC)

    # Load the original DataFrame
    df = pd.read_pickle(df_pickle)

    # Add the results to the DataFrame
    df["Solution"] = list(SOL)
    df["Solution_nvc"] = SOL_NVC

    # Save the merged DataFrame
    df.to_pickle(output_file)
    print(f"Results merged into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge results from multiple instances")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing result files")
    parser.add_argument("--df_pickle", type=str, required=True, help="Original DataFrame pickle file")
    parser.add_argument("--output_file", type=str, required=True, help="Output file to save merged results")
    args = parser.parse_args()
    merge_results(args.output_dir, args.df_pickle, args.output_file)