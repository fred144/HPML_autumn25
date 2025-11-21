#!/bin/bash

echo "Running Part-B Q2 with different K values and scenarios (multiple runs per K)"
echo "============================================================================="

K_values=(1 5 10 50 100)
SCENARIOS=(1 2 3)
N_RUNS=5

make q2

# Create results directory if it doesn't exist
mkdir -p results

for K in "${K_values[@]}"; do
    for scenario in "${SCENARIOS[@]}"; do
        OUTPUT_FILE="results/q2_scenario${scenario}_K${K}.txt"
        
        echo "----------------------------------------"
        echo "Testing K = $K million, Scenario $scenario ($N_RUNS runs)"
        echo "Output: $OUTPUT_FILE"
        echo "----------------------------------------"
        
        # Get scenario description
        case $scenario in
            1) desc="One block with 1 thread" ;;
            2) desc="One block with 256 threads" ;;
            3) desc="Multiple blocks with 256 threads per block" ;;
        esac
        
        echo "Q2 results for K=$K million, Scenario $scenario: $desc ($N_RUNS runs)" > "$OUTPUT_FILE"
        echo "==========================================" >> "$OUTPUT_FILE"
        
        for ((run=1; run<=N_RUNS; run++)); do
            echo "Run $run/$N_RUNS:" | tee -a "$OUTPUT_FILE"
            ./q2 $K $scenario | tee -a "$OUTPUT_FILE"
            echo "" | tee -a "$OUTPUT_FILE"
        done
    done
    echo ""
done

echo "============================================================================="
echo "Generated files:"
ls -l results/q2_scenario*.txt