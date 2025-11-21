/*
q3.cu
Q3: CUDA Unified Memory vector addition with K million elements
*/

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include "timer.h"

/*
CUDA kernel for vector addition with coalesced memory access
adapted from Part A
*/
__global__ void AddVectors(const double* A, const double* B, double* C, int elementsPerThread, int totalElements)
{
    // total num of threads in the grid
    int totalThreads = gridDim.x * blockDim.x;
    // thread unique ID
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    
    // per thread, processes elementsPerThread elements with coalesced memory access
    for (int i = 0; i < elementsPerThread; i++) {
        int index = threadId + i * totalThreads;
        // bounds check to avoid accessing beyond array
        if (index < totalElements) {
            C[index] = A[index] + B[index];
        }
    }
}

/*function to run and time a specific scenario*/
double run_scenario_unified(int scenario, double *A, double *B, double *C, int N) {
    // reset and start timer
    reset_timer();
    start_timer();
    
    // launch kernel based on scenario
    int blocks, threads;
    switch(scenario) {
        case 1: // one block with 1 thread
            blocks = 1;
            threads = 1;
            break;
        case 2: // one block with 256 threads
            blocks = 1;
            threads = 256;
            break;
        case 3: // multiple blocks with 256 threads per block
            threads = 256;
            blocks = (N + threads - 1) / threads;
            break;
    }
    
    // calculate how many elements each thread needs to process
    int totalThreads = blocks * threads;
    int elementsPerThread = (N + totalThreads - 1) / totalThreads;
    
    printf("launching kernel: %d blocks, %d threads, %d total threads, %d elements per thread\n", 
           blocks, threads, totalThreads, elementsPerThread);
    
    AddVectors<<<blocks, threads>>>(A, B, C, elementsPerThread, N);
    
    // check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1.0;
    }
    
    // wait for kernel to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return -1.0;
    }
    
    stop_timer();
    return elapsed_time();
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s K scenario\n", argv[0]);
        printf("where K is the number of millions of elements (e.g., 1 = 1 million elements)\n");
        printf("scenario: 1, 2, or 3\n");
        printf("  1: one block with 1 thread\n");
        printf("  2: one block with 256 threads\n");
        printf("  3: multiple blocks with 256 threads per block\n");
        return 1;
    }
    int K = atoi(argv[1]);
    int scenario = atoi(argv[2]);
    
    if (K <= 0) {
        printf("error: K must be a positive integer\n");
        return 1;
    }
    if (scenario < 1 || scenario > 3) {
        printf("error: scenario must be 1, 2, or 3\n");
        return 1;
    }

    int N = K * 1000000;  // K million elements
    printf("vector size: %d elements (%d million)\n", N, K);

    /*
    ********** allocate unified memory - accessible from both CPU and GPU
    */ 
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
    initialize vectors A and B with values:
    A[i] = i and B[i] = N - i for i = 0 to N-1
    so that C[i] = A[i] + B[i] = i + (N - i) = N
    */
    printf("initializing vectors...\n");
    for (int i = 0; i < N; i++) {
        A[i] = static_cast<double>(i);          // A[i] = i
        B[i] = static_cast<double>(N - i);      // B[i] = N - i
    }

    // Scenario descriptions
    const char* scenario_desc[] = {
        "", // 0-indexed
        "1 block, 1 thread",
        "1 block, 256 threads", 
        "multiple blocks, 256 threads/block"
    };

    printf("performing vector addition on GPU with Unified Memory (Scenario %d: %s)...\n", scenario, scenario_desc[scenario]);
    
    double execution_time = run_scenario_unified(scenario, A, B, C, N);
    
    if (execution_time < 0) {
        printf("error occurred during execution\n");
        cudaFree(A); cudaFree(B); cudaFree(C);
        return 1;
    }
    
    // Verify result
    bool correct = true;
    int check_count = (N < 10) ? N : 10;
    for (int i = 0; i < check_count; i++) {
        double expected = static_cast<double>(N);
        if (std::abs(C[i] - expected) > 1e-5) {
            printf("verification failed at index %d: C[%d]=%.1f, expected=%.1f\n", i, i, C[i], expected);
            correct = false;
            break;
        }
    }
    
    // calculate performance metrics
    double data_size_bytes = 3.0 * N * sizeof(double); // read A, read B, write C
    double data_size_gb = data_size_bytes / (1024.0 * 1024.0 * 1024.0);
    double bandwidth_gb_per_sec = data_size_gb / execution_time;
    double flops = N; // 1 FLOP per element (a + b)
    double gflops_per_sec = (flops / execution_time) / 1e9;
    
    // Print results
    printf("Scenario %d (%s):\n", scenario, scenario_desc[scenario]);
    printf("N: %d <T>: %.12f sec  <B>: %.12f GB/sec  <F>: %.12f GFLOP/sec\n", 
           N, execution_time, bandwidth_gb_per_sec, gflops_per_sec);
    printf("Result verification: %s\n", correct ? "PASSED" : "FAILED");
    
    // free unified memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}