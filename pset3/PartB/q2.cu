///
/// q2.cu - FIXED VERSION
/// Q2: CUDA vector addition with explicit memory management
///

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include "timer.h"

// CUDA kernel for vector addition - FIXED with loop
__global__ void vectorAdd(const double *a, const double *b, double *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Each thread processes multiple elements
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// Function to run and time a specific scenario
double run_scenario(int scenario, double *h_A, double *h_B, double *h_C, int N) {
    double *d_A, *d_B, *d_C;
    cudaError_t err;
    
    // Allocate device memory
    err = cudaMalloc(&d_A, N * sizeof(double));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed for d_A: %s\n", cudaGetErrorString(err));
        return -1.0;
    }
    err = cudaMalloc(&d_B, N * sizeof(double));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed for d_B: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        return -1.0;
    }
    err = cudaMalloc(&d_C, N * sizeof(double));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed for d_C: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return -1.0;
    }
    
    // Copy input data from host to device
    err = cudaMemcpy(d_A, h_A, N * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("cudaMemcpy failed for d_A: %s\n", cudaGetErrorString(err));
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return -1.0;
    }
    err = cudaMemcpy(d_B, h_B, N * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("cudaMemcpy failed for d_B: %s\n", cudaGetErrorString(err));
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return -1.0;
    }
    
    // Initialize device memory for C to catch uncomputed elements
    err = cudaMemset(d_C, 0, N * sizeof(double));
    if (err != cudaSuccess) {
        printf("cudaMemset failed for d_C: %s\n", cudaGetErrorString(err));
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return -1.0;
    }
    
    // Reset and start timer
    reset_timer();
    start_timer();
    
    // Launch kernel based on scenario
    int blocks, threads;
    switch(scenario) {
        case 1: // One block with 1 thread
            blocks = 1;
            threads = 1;
            break;
        case 2: // One block with 256 threads
            blocks = 1;
            threads = 256;
            break;
        case 3: // Multiple blocks with 256 threads per block
            threads = 256;
            blocks = (N + threads - 1) / threads;
            break;
    }
    
    printf("Launching kernel: %d blocks, %d threads, %d elements\n", blocks, threads, N);
    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return -1.0;
    }
    
    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return -1.0;
    }
    
    stop_timer();
    double execution_time = elapsed_time();
    
    // Copy result back from device to host
    err = cudaMemcpy(h_C, d_C, N * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy failed for h_C: %s\n", cudaGetErrorString(err));
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return -1.0;
    }
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return execution_time;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s K scenario\n", argv[0]);
        printf("where K is the number of millions of elements (e.g., 1 = 1 million elements)\n");
        printf("scenario: 1, 2, or 3\n");
        printf("  1: One block with 1 thread\n");
        printf("  2: One block with 256 threads\n");
        printf("  3: Multiple blocks with 256 threads per block\n");
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

    // Calculate total number of elements
    int N = K * 1000000;  // K million elements
    printf("Vector size: %d elements (%d million)\n", N, K);

    // Allocate host memory
    double *A = new double[N];
    double *B = new double[N];
    double *C = new double[N];

    if (!A || !B || !C) {
        printf("error: Memory allocation failed\n");
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

    // Scenario descriptions
    const char* scenario_desc[] = {
        "", // 0-indexed
        "1 block, 1 thread",
        "1 block, 256 threads", 
        "multiple blocks, 256 threads/block"
    };

    printf("performing vector addition on GPU (Scenario %d: %s)...\n", scenario, scenario_desc[scenario]);
    
    double execution_time = run_scenario(scenario, A, B, C, N);
    
    if (execution_time < 0) {
        printf("Error occurred during execution\n");
        delete[] A; delete[] B; delete[] C;
        return 1;
    }
    
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
        if (std::abs(C[i] - expected) > 1e-5) {
            printf("Verification failed at index %d: C[%d]=%.1f, expected=%.1f\n", i, i, C[i], expected);
            correct = false;
            break;
        }
    }
    
    // Print results
    printf("Scenario %d (%s):\n", scenario, scenario_desc[scenario]);
    printf("N: %d <T>: %.12f sec  <B>: %.12f GB/sec  <F>: %.12f GFLOP/sec\n", 
           N, execution_time, bandwidth_gb_per_sec, gflops_per_sec);
    printf("Result verification: %s\n", correct ? "PASSED" : "FAILED");
    
    // Free host memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}