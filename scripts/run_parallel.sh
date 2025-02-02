#!/bin/bash

# filepath: /home/ubuntu/Semidefinite-programming/scripts/run_parallel.sh

# Check if N value is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <N_value>"
    exit 1
fi

# N value
N=$1

# Check if the merge-only flag is provided
if [ "$2" == "--merge-only" ]; then
    OUTPUT_DIR="./results/N_$N"
    python3 /home/ubuntu/Semidefinite-programming/scripts/merge_results.py --output_dir $OUTPUT_DIR --df_pickle "../data/Max_2_SAT_N_${N}_λstd_0_T_1024.0_reps_50.pkl" --output_file "../data/Max_2_SAT_N_${N}_λstd_0_T_1024.0_reps_50_with_solution_merged.pkl"
    exit 0
fi

# Number of instances to run in parallel
NUM_INSTANCES=125

# Total number of indices to process
TOTAL_INDICES=1350

# Directory to save intermediate results
OUTPUT_DIR="./results/N_$N"
mkdir -p $OUTPUT_DIR

# Record the start time of the script
script_start_time=$(date +%s)

# Run multiple instances of the Python script in batches
for ((start_idx=0; start_idx<TOTAL_INDICES; start_idx+=NUM_INSTANCES)); do
    end_idx=$((start_idx + NUM_INSTANCES))
    if [ $end_idx -gt $TOTAL_INDICES ]; then
        end_idx=$TOTAL_INDICES
    fi

    for ((i=start_idx; i<end_idx; i++)); do
        output_file="$OUTPUT_DIR/result_$i.pkl"
        python3 /home/ubuntu/Semidefinite-programming/scripts/Ground_state_parallel.py --idx $i --N $N --output_file $output_file &
    done

    # Wait for all instances in this batch to complete
    wait
done

echo "All instances completed."

# Merge all results into a single DataFrame
python3 /home/ubuntu/Semidefinite-programming/scripts/merge_results.py --output_dir $OUTPUT_DIR --df_pickle "../data/Max_2_SAT_N_${N}_λstd_0_T_1024.0_reps_50.pkl" --output_file "../data/Max_2_SAT_N_${N}_λstd_0_T_1024.0_reps_50_with_solution_merged.pkl"

echo "Results merged into ../data/Max_2_SAT_N_${N}_λstd_0_T_1024.0_reps_50_with_solution_merged.pkl"

# Record the end time of the script
script_end_time=$(date +%s)
script_elapsed_time=$((script_end_time - script_start_time))
script_elapsed_time_str=$(date -u -d @${script_elapsed_time} +"%T")

echo "Total script execution time: ${script_elapsed_time_str}"