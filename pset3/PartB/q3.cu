///
/// q3.cu
/// Q3: CUDA Unified Memory vector addition with K million elements
///

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include "timer.h"

// CUDA kernel for vector addition
__global__ void vectorAdd(const double *a, const double *b, double *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Function to run and time a specific scenario with unified memory
double run_scenario_unified(int scenario, double *A, double *B, double *C, int N) {
    // Initialize and start timer
    initialize_timer();
    start_timer();
    
    // Launch kernel based on scenario
    switch(scenario) {
        case 1: // One block with 1 thread
            vectorAdd<<<1, 1>>>(A, B, C, N);
            break;
        case 2: // One block with 256 threads
            vectorAdd<<<1, 256>>>(A, B, C, N);
            break;
        case 3: // Multiple blocks with 256 threads per block
            {
                int threadsPerBlock = 256;
                int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
                vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
            }
            break;
    }
    
    cudaDeviceSynchronize(); // Wait for kernel to complete
    stop_timer();
    
    return elapsed_time();
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s K\n", argv[0]);
        printf("where K is the number of millions of elements (e.g., 1 = 1 million elements)\n");
        return 1;
    }
    int K = atoi(argv[1]);
    if (K <= 0) {
        printf("error: K must be a positive integer\n");
        return 1;
    }

    // Calculate total number of elements
    int N = K * 1000000;  // K million elements
    printf("Vector size: %d elements (%d million)\n", N, K);

    // Allocate Unified Memory - accessible from CPU and GPU
    double *A, *B, *C;
    cudaError_t err = cudaMallocManaged(&A, N * sizeof(double));
    if (err != cudaSuccess) {
        printf("cudaMallocManaged failed for A: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMallocManaged(&B, N * sizeof(double));
    if (err != cudaSuccess) {
        printf("cudaMallocManaged failed for B: %s\n", cudaGetErrorString(err));
        cudaFree(A);
        return 1;
    }
    err = cudaMallocManaged(&C, N * sizeof(double));
    if (err != cudaSuccess) {
        printf("cudaMallocManaged failed for C: %s\n", cudaGetErrorString(err));
        cudaFree(A);
        cudaFree(B);
        return 1;
    }

    /*
    Initialize vectors A and B with values:
    A[i] = i and B[i] = N - i for i = 0 to N-1
    so that C[i] = A[i] + B[i] = i + (N - i) = N
    */
    printf("initializing vectors...\n");
    for (int i = 0; i < N; i++) {
        A[i] = static_cast<double>(i);          // A[i] = i
        B[i] = static_cast<double>(N - i);      // B[i] = N - i
    }

    printf("performing vector addition on GPU with Unified Memory...\n");
    
    // Test all three scenarios
    for (int scenario = 1; scenario <= 3; scenario++) {
        double execution_time = run_scenario_unified(scenario, A, B, C, N);
        
        // Calculate performance metrics
        double data_size_bytes = 3.0 * N * sizeof(double); // read A, read B, write C
        double data_size_gb = data_size_bytes / (1024.0 * 1024.0 * 1024.0);
        double bandwidth_gb_per_sec = data_size_gb / execution_time;
        double flops = N; // 1 FLOP per element (a + b)
        double gflops_per_sec = (flops / execution_time) / 1e9;
        
        // Verify result
        bool correct = true;
        int check_count = (N < 10) ? N : 10;
        for (int i = 0; i < check_count; i++) {
            double expected = static_cast<double>(N);
            if (std::abs(C[i] - expected) > 1e-10) {
                correct = false;
                break;
            }
        }
        
        // Print results with scenario description
        const char* scenario_desc[] = {
            "", // 0-indexed
            "1 block, 1 thread",
            "1 block, 256 threads", 
            "multiple blocks, 256 threads/block"
        };
        
        printf("Scenario %d (%s):\n", scenario, scenario_desc[scenario]);
        printf("N: %d <T>: %.12f sec  <B>: %.12f GB/sec  <F>: %.12f GFLOP/sec\n", 
               N, execution_time, bandwidth_gb_per_sec, gflops_per_sec);
        printf("Result verification: %s\n", correct ? "PASSED" : "FAILED");
        printf("\n");
    }
    
    // Free unified memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}