#!/bin/bash

echo "Running Part-B Q3 with different K values and scenarios (multiple runs per K)"
echo "============================================================================="
# still doing trials, each with multiple runs for averaging
# total number of runs: 5x3x5 = 75
K_values=(1 5 10 50 100) # in millions
# 1: using one block, 1 thread
# 2: one block with 256 threads  
# 3: multiple blocks with 256 threads per block
SCENARIOS=(1 2 3) 
N_RUNS=5

make q3 # make if not already made

# make sure the results directory exists
mkdir -p results

for K in "${K_values[@]}"; do
    for scenario in "${SCENARIOS[@]}"; do
        OUTPUT_FILE="results/q3_scenario${scenario}_K${K}.txt"
        
        echo "----------------------------------------"
        echo "testing K = $K million, Scenario $scenario ($N_RUNS runs)"
        echo "output: $OUTPUT_FILE"
        echo "----------------------------------------"
        
        # clear the file first
        > "$OUTPUT_FILE"
        
        for ((run=1; run<=N_RUNS; run++)); do
            echo "run $run/$N_RUNS:" | tee -a "$OUTPUT_FILE"
            ./q3 $K $scenario | tee -a "$OUTPUT_FILE"
            echo "" | tee -a "$OUTPUT_FILE"
        done
    done
    echo ""
done#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include "timer.h"

// Relevant dimensions
const int H = 1024;
const int W = 1024;
const int C = 3;
const int FH = 3;
const int FW = 3;
const int K = 64;
const int P = 1;

// Padded dimensions
const int H_padded = H + 2 * P;
const int W_padded = W + 2 * P;

// Use the same block size as C1 for compatibility
const int BLOCK_W = 8;
const int BLOCK_H = 8;
const int BLOCK_K = 4;

/* Minimal tiled convolution - add shared memory to working C1 */
__global__ void convolution_tiled_simple(const double *I_padded, const double *F, double *O)
{
    // Small shared memory tile for current block
    __shared__ double I_tile[BLOCK_H + 2][BLOCK_W + 2];
    
    // Thread indices (same as C1)
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    
    // Output coordinates (same as C1)
    int x = blockIdx.x * BLOCK_W + tx;
    int y = blockIdx.y * BLOCK_H + ty;
    int k = blockIdx.z * BLOCK_K + tz;

    if (x < W && y < H && k < K) {
        double sum = 0.0;
        
        for (int c = 0; c < C; c++) {
            // Load shared memory for current channel
            // Each thread loads the element it needs for convolution
            for (int fj = 0; fj < FH; fj++) {
                for (int fi = 0; fi < FW; fi++) {
                    int load_x = tx + fi;
                    int load_y = ty + fj;
                    
                    if (load_x < BLOCK_W + 2 && load_y < BLOCK_H + 2) {
                        int input_x = blockIdx.x * BLOCK_W + load_x - 1;
                        int input_y = blockIdx.y * BLOCK_H + load_y - 1;
                        
                        if (input_x >= 0 && input_x < W_padded && input_y >= 0 && input_y < H_padded) {
                            int input_idx = c * H_padded * W_padded + input_y * W_padded + input_x;
                            I_tile[load_y][load_x] = I_padded[input_idx];
                        } else {
                            I_tile[load_y][load_x] = 0.0;
                        }
                    }
                }
            }
            
            __syncthreads();
            
            // Perform convolution using shared memory
            for (int fj = 0; fj < FH; fj++) {
                for (int fi = 0; fi < FW; fi++) {
                    int input_x = x + fi;
                    int input_y = y + fj;
                    
                    if (input_x >= 0 && input_x < W_padded && input_y >= 0 && input_y < H_padded) {
                        int flipped_fi = FW - 1 - fi;
                        int flipped_fj = FH - 1 - fj;
                        int filter_idx = k * C * FH * FW + c * FH * FW + flipped_fj * FW + flipped_fi;
                        
                        // Use shared memory instead of global memory
                        double input_val = I_tile[ty + fj][tx + fi];
                        sum += F[filter_idx] * input_val;
                    }
                }
            }
            
            __syncthreads();
        }
        
        int output_idx = k * H * W + y * W + x;
        O[output_idx] = sum;
        
        if (k == 0 && y == 0 && x < 3) {
            printf("TILED_SIMPLE: O[%d,%d,%d] = %f\n", k, y, x, sum);
        }
    }
}

int main() {
    printf("> Minimal Tiled Convolution (C1 + Shared Memory):\n");

    // Calculate sizes
    size_t input_size = C * H_padded * W_padded * sizeof(double);
    size_t filter_size = K * C * FH * FW * sizeof(double);
    size_t output_size = K * H * W * sizeof(double);

    // Allocate host memory
    double *h_I_padded = (double *)malloc(input_size);
    double *h_F = (double *)malloc(filter_size);
    double *h_O = (double *)malloc(output_size);

    // Initialize data
    printf("Initializing data...\n");
    for (int c = 0; c < C; c++) {
        for (int y = 0; y < H_padded; y++) {
            for (int x = 0; x < W_padded; x++) {
                int idx = c * H_padded * W_padded + y * W_padded + x;
                if (x >= P && x < W_padded - P && y >= P && y < H_padded - P) {
                    h_I_padded[idx] = c * ((x - P) + (y - P));
                } else {
                    h_I_padded[idx] = 0.0;
                }
            }
        }
    }

    for (int k = 0; k < K; k++) {
        for (int c = 0; c < C; c++) {
            for (int fj = 0; fj < FH; fj++) {
                for (int fi = 0; fi < FW; fi++) {
                    int idx = k * C * FH * FW + c * FH * FW + fj * FW + fi;
                    h_F[idx] = (c + k) * (fi + fj);
                }
            }
        }
    }

    // Allocate device memory
    double *d_I_padded, *d_F, *d_O;
    cudaMalloc(&d_I_padded, input_size);
    cudaMalloc(&d_F, filter_size);
    cudaMalloc(&d_O, output_size);

    // Copy data to device
    cudaMemcpy(d_I_padded, h_I_padded, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, h_F, filter_size, cudaMemcpyHostToDevice);

    // Use EXACT same configuration as C1
    dim3 blockDim(BLOCK_W, BLOCK_H, BLOCK_K);
    dim3 gridDim((W + BLOCK_W - 1) / BLOCK_W,
                 (H + BLOCK_H - 1) / BLOCK_H,
                 (K + BLOCK_K - 1) / BLOCK_K);

    printf("Kernel configuration: grid(%d, %d, %d), block(%d, %d, %d)\n",
           gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);

    size_t shared_mem = (BLOCK_H + 2) * (BLOCK_W + 2) * sizeof(double);
    printf("Shared memory: %.2f KB\n", shared_mem / 1024.0);

    // Warm-up
    convolution_tiled_simple<<<gridDim, blockDim, shared_mem>>>(d_I_padded, d_F, d_O);
    cudaDeviceSynchronize();

    printf("> Starting timer...\n");
    initialize_timer();
    start_timer();

    convolution_tiled_simple<<<gridDim, blockDim, shared_mem>>>(d_I_padded, d_F, d_O);

    cudaDeviceSynchronize();
    stop_timer();
    double execution_time_ms = elapsed_time() * 1000.0;

    // Copy result back
    cudaMemcpy(h_O, d_O, output_size, cudaMemcpyDeviceToHost);

    // Calculate checksum
    double checksum = 0.0;
    for (int i = 0; i < K * H * W; i++) {
        checksum += h_O[i];
    }

    printf("C2_checksum: %.6f\n", checksum);
    printf("C2_execution_time [ms]: %.3f\n", execution_time_ms);

    // Cleanup
    cudaFree(d_I_padded);
    cudaFree(d_F);
    cudaFree(d_O);
    free(h_I_padded);
    free(h_F);
    free(h_O);

    printf("C2 completed.\n");
    return 0;
}

echo "============================================================================="
echo "generated files:"
ls -l results/q3_scenario*.txt