#!/bin/bash

NUM_TRIALS=5
VALUES_PER_THREAD="500 1000 2000"

echo "running $NUM_TRIALS trials for each configuration..."

# run Q1 trials (vecadd00 - non-coalesced)
echo "running Q1 trials (non-coalesced)..."
for vpt in $VALUES_PER_THREAD; do
    echo "  ValuesPerThread: $vpt"
    for ((i=1; i<=NUM_TRIALS; i++)); do
        ./vecadd00 $vpt | tee -a "./results/q1_${vpt}_trials.txt"
        echo "    trial $i completed"
        echo ""  # add a newline for better readability
    done
done

# run Q2 trials (vecadd01 - coalesced)
echo "running Q2 trials (coalesced)..."
for vpt in $VALUES_PER_THREAD; do
    echo "  ValuesPerThread: $vpt"
    for ((i=1; i<=NUM_TRIALS; i++)); do
        ./vecadd01 $vpt | tee -a "./results/q2_${vpt}_trials.txt"
        echo "    trial $i completed"
        echo ""  # add a newline for better readability
    done
done

echo "### all trials completed!"
echo "running analysis... output saved to ./results/STDOUT_analysis_q1q2"
python3 analyze_trials_P1_q1q2.py | tee -a ./results/STDOUT_analysis_q1q2