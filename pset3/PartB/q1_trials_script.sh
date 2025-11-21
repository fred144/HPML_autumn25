#!/bin/bash

echo "Running Part-B Q1 with different K values (multiple runs per K)"
echo "================================================================"

K_values=(1 5 10 50 100)
N_RUNS=5

# create results directory if it doesn't exist
mkdir -p results

for K in "${K_values[@]}"; do
    OUTPUT_FILE="results/q1_runs_K${K}.txt"
    
    echo "----------------------------------------"
    echo "Testing K = $K million ($N_RUNS runs)"
    echo "Output: $OUTPUT_FILE"
    echo "----------------------------------------"
    
    echo "Q1 results for K=$K million ($N_RUNS runs)" > "$OUTPUT_FILE"
    echo "==========================================" >> "$OUTPUT_FILE"
    
    for ((run=1; run<=N_RUNS; run++)); do
        echo "Run $run/$N_RUNS:" | tee -a "$OUTPUT_FILE"
        ./q1 $K | tee -a "$OUTPUT_FILE"
        echo "" | tee -a "$OUTPUT_FILE"
    done
    echo ""
done

echo "================================================================"
echo "Generated files:"
ls -l results/q1_runs_K*.txt