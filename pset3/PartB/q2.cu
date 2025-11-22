/*
q2.cu - DEBUG VERSION WITH FIXED KERNEL
Q2: CUDA vector addition with explicit memory management
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
__global__ void AddVectors(const double *A, const double *B, double *C, int elementsPerThread, int totalElements)
{
    // total num of threads in the grid
    int totalThreads = gridDim.x * blockDim.x;
    // thread unique ID
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    // per thread, processes elementsPerThread elements with coalesced memory access
    // consecutive threads access consecutive memory locations
    for (int i = 0; i < elementsPerThread; i++)
    {
        int index = threadId + i * totalThreads;
        if (index < totalElements)
        { // bounds check to avoid accessing beyond array
            C[index] = A[index] + B[index];
        }
    }
}

/*function to run and time a specific scenario*/
double run_scenario(int scenario, double *h_A, double *h_B, double *h_C, int N)
{
    double *d_A, *d_B, *d_C;
    cudaError_t err;

    // memory allocation, do some checks to allow easier debugging
    err = cudaMalloc(&d_A, N * sizeof(double));
    if (err != cudaSuccess)
    {
        printf("cudaMalloc failed for d_A: %s\n", cudaGetErrorString(err));
        return -1.0;
    }
    err = cudaMalloc(&d_B, N * sizeof(double));
    if (err != cudaSuccess)
    {
        printf("cudaMalloc failed for d_B: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        return -1.0;
    }
    err = cudaMalloc(&d_C, N * sizeof(double));
    if (err != cudaSuccess)
    {
        printf("cudaMalloc failed for d_C: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return -1.0;
    }

    // copy input data from host to device CPU/RAM -> GPU/VRAM
    err = cudaMemcpy(d_A, h_A, N * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("cudaMemcpy failed for d_A: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1.0;
    }
    err = cudaMemcpy(d_B, h_B, N * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("cudaMemcpy failed for d_B: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1.0;
    }

    // reset and start timer
    reset_timer();
    start_timer();

    // launch kernel based on scenario
    int blocks, threads;
    switch (scenario)
    {
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

    printf("Launching kernel: %d blocks, %d threads, %d total threads, %d elements per thread\n", blocks, threads, totalThreads, elementsPerThread);

    // kernel launch config <<< number of blocks, threads per block>>>
    AddVectors<<<blocks, threads>>>(d_A, d_B, d_C, elementsPerThread, N);

    // check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1.0;
    }

    // wait for kernel to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1.0;
    }

    stop_timer();
    double execution_time = elapsed_time();

    // copy result back from device to host
    err = cudaMemcpy(h_C, d_C, N * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("cudaMemcpy failed for h_C: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1.0;
    }

    // free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return execution_time;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("usage: %s K scenario\n", argv[0]);
        printf("where K is the number of millions of elements (e.g., 1 = 1 million elements)\n");
        printf("scenario: 1, 2, or 3\n");
        printf("  1: One block with 1 thread\n");
        printf("  2: One block with 256 threads\n");
        printf("  3: Multiple blocks with 256 threads per block\n");
        return 1;
    }
    int K = atoi(argv[1]);        // how many million
    int scenario = atoi(argv[2]); // scenario

    if (K <= 0)
    {
        printf("error: K must be a positive integer\n");
        return 1;
    }
    if (scenario < 1 || scenario > 3)
    {
        printf("error: scenario must be 1, 2, or 3\n");
        return 1;
    }

    // calculate total number of elements
    int N = K * 1000000; // K million elements
    printf("Vector size: %d elements (%d million)\n", N, K);

    // allocate host memory
    double *A = new double[N];
    double *B = new double[N];
    double *C = new double[N];

    if (!A || !B || !C)
    {
        printf("error: Memory allocation failed\n"); // in case it fails
        return 1;
    }

    /*
    initialize vectors A and B with values:
    A[i] = i and B[i] = N - i for i = 0 to N-1
    so that C[i] = A[i] + B[i] = i + (N - i) = N
    this makes verification straightforward
    */
    printf("initializing vectors...\n");
    for (int i = 0; i < N; i++)
    {
        A[i] = static_cast<double>(i);     // A[i] = i
        B[i] = static_cast<double>(N - i); // B[i] = N - i
    }

    // scenario descriptions
    const char *scenario_desc[] = {
        "", // 0-indexed placeholder-- not an option
        "1 block, 1 thread",
        "1 block, 256 threads",
        "multiple blocks, 256 threads/block"};

    printf("performing vector addition on GPU (Scenario %d: %s)...\n", scenario, scenario_desc[scenario]);

    // body of the work
    double execution_time = run_scenario(scenario, A, B, C, N);

    if (execution_time < 0)
    {
        printf("Error occurred during execution\n");
        delete[] A;
        delete[] B;
        delete[] C;
        return 1;
    }

    // calculate performance metrics similar to Pset1, again
    double data_size_bytes = 3.0 * N * sizeof(double); // read A, read B, write C
    double data_size_gb = data_size_bytes / (1024.0 * 1024.0 * 1024.0);
    double bandwidth_gb_per_sec = data_size_gb / execution_time;
    double flops = N; // 1 FLOP per element (a + b)
    double gflops_per_sec = (flops / execution_time) / 1e9;

    // verify result by looking at the beginning of the array
    bool correct = true;
    int check_count = (N < 10) ? N : 10;
    for (int i = 0; i < check_count; i++)
    {
        double expected = static_cast<double>(N);
        if (std::abs(C[i] - expected) > 1e-5)
        {
            printf("Verification failed at index %d: C[%d]=%.1f, expected=%.1f\n", i, i, C[i], expected);
            correct = false;
            break;
        }
    }

    // print results, to save and calculate stats later
    printf("Scenario %d (%s):\n", scenario, scenario_desc[scenario]);
    printf("N: %d <T>: %.12f sec  <B>: %.12f GB/sec  <F>: %.12f GFLOP/sec\n",
           N, execution_time, bandwidth_gb_per_sec, gflops_per_sec);
    printf("Result verification: %s\n", correct ? "PASSED" : "FAILED");

    // free host memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}