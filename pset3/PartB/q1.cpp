///
/// q1.cpp
/// Q1: CPU-only vector addition with K million elements
///

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include "timer.h"

int main(int argc, char** argv) {
    // std in  and assign K
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

    // Allocate memory for vectors
    float* A = new float[N];
    float* B = new float[N];
    float* C = new float[N];

    if (!A || !B || !C) {
        printf("error: Memory allocation failed\n");
        return 1;
    }

    /*
    we initialize vectors A and B with some values,
    where we set A[i] = i and B[i] = N - i for i = 0 to N-1
    so that C[i] = A[i] + B[i] = i + (N - i) = N
    */
    printf("initializing vectors...\n");
    for (int i = 0; i < N; i++) {
        A[i] = static_cast<float>(i);          // A[i] = i
        B[i] = static_cast<float>(N - i);      // B[i] = N - i
    }

    printf("performing vector addition...\n");
    
    // Initialize and start timer
    initialize_timer();
    start_timer();
    
    // Vector addition kernel (CPU version)
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
    
    stop_timer();
    double execution_time = elapsed_time();
    
    // Calculate performance metrics
    double data_size_bytes = 3.0 * N * sizeof(float); // 3 arrays: read A, read B, write C
    double data_size_gb = data_size_bytes / (1024.0 * 1024.0 * 1024.0);
    double bandwidth_gb_per_sec = data_size_gb / execution_time;
    double flops = N; // 1 FLOP per element (a + b)
    double gflops_per_sec = (flops / execution_time) / 1e9;

    // verify result (check a few elements), not timed
    bool correct = true;
    int check_count = (N < 10) ? N : 10;
    for (int i = 0; i < check_count; i++) {
        float expected = static_cast<float>(N); // Should be N for all elements
        if (std::abs(C[i] - expected) > 1e-5) {
            printf("Error at index %d: C[%d] = %f, expected = %f\n", i, i, C[i], expected);
            correct = false;
            break;
        }
    }

    // print results
    printf("N: %d <T>: %.12f sec  <B>: %.12f GB/sec  <F>: %.12f GFLOP/sec\n", 
           N, execution_time, bandwidth_gb_per_sec, gflops_per_sec);
    printf("Result verification: %s\n", correct ? "PASSED" : "FAILED");
    
    // Free memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}