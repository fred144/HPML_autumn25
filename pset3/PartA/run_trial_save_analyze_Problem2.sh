#!/bin/bash

NUM_TRIALS=8

echo "Testing $NUM_TRIALS trials for each required matrix size"
echo ""

# test matmult00 (Q3) - ORIGINAL
echo "=== TESTING matmult00 (Q3 - Original) ==="

echo "256x256 matrices: ./matmult00 16"
for ((i=1; i<=NUM_TRIALS; i++)); do
    echo "  Trial $i..."
    ./matmult00 16 | tee -a "./results/matmult00_256x256_mat_16_in.txt"
    echo ""
done

echo "512x512 matrices: ./matmult00 32"
for ((i=1; i<=NUM_TRIALS; i++)); do
    echo "  Trial $i..."
    ./matmult00 32 | tee -a "./results/matmult00_512x512_mat_32_in.txt"
    echo ""
done

echo "1024x1024 matrices: ./matmult00 64"
for ((i=1; i<=NUM_TRIALS; i++)); do
    echo "  Trial $i..."
    ./matmult00 64 | tee -a "./results/matmult00_1024x1024_mat_64_in.txt"
    echo ""
done

# test matmult01 (Q4) - OPTIMIZED
echo "=== TESTING matmult01 (Q4 - Optimized) ==="

echo "256x256 matrices: ./matmult01 8"
for ((i=1; i<=NUM_TRIALS; i++)); do
    echo "  Trial $i..."
    ./matmult01 8 | tee -a "./results/matmult01_256x256_mat_8_in.txt"
    echo ""
done

echo "512x512 matrices: ./matmult01 16"
for ((i=1; i<=NUM_TRIALS; i++)); do
    echo "  Trial $i..."
    ./matmult01 16 | tee -a "./results/matmult01_512x512_mat_16_in.txt"
    echo ""
done

echo "1024x1024 matrices: ./matmult01 32"
for ((i=1; i<=NUM_TRIALS; i++)); do
    echo "  Trial $i..."
    ./matmult01 32 | tee -a "./results/matmult01_1024x1024_mat_32_in.txt"
    echo ""
done

echo "### all trials completed!"
echo "running analysis...output saved to ./results/STDOUT_analysis_q3q4"

python3 analyze_trials_P1_q3q4.py | tee -a ./results/STDOUT_analysis_q3q4