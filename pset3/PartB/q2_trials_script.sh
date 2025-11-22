#!/bin/bash

echo "Running Part-B Q2 with different K values and scenarios (multiple runs per K)"
echo "============================================================================="
# still doing trials, each with multiple runs for averaging
# total number of runs: 5x3x5 = 75
K_values=(1 5 10 50 100) # in millions
# 1: using one block, 
# 2: multiple blocks with fewer threads
# 3: multiple blocks with more threads
SCENARIOS=(1 2 3) 
N_RUNS=5

make q2 # make if not already made

# make sure the results directory exists
mkdir -p results

for K in "${K_values[@]}"; do
    for scenario in "${SCENARIOS[@]}"; do
        OUTPUT_FILE="results/q2_scenario${scenario}_K${K}.txt"
        
        echo "----------------------------------------"
        echo "testing K = $K million, Scenario $scenario ($N_RUNS runs)"
        echo "output: $OUTPUT_FILE"
        echo "----------------------------------------"
    
        
        for ((run=1; run<=N_RUNS; run++)); do
            echo "run $run/$N_RUNS:" | tee -a "$OUTPUT_FILE"
            ./q2 $K $scenario | tee -a "$OUTPUT_FILE"
            echo "" | tee -a "$OUTPUT_FILE"
        done
    done
    echo ""
done

echo "============================================================================="
echo "generated files:"
ls -l results/q2_scenario*.txt