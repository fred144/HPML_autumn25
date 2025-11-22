/*
brief outline of the
we are told that we are implemneting a 2D convolution with multiple input channels
breif recall

input tensor I with dimension:
- C = 3
- H = 1024
- W = 1024
so the total element is: 3 x 1024 x 1024 = 3145728

a set of filters:
F[K, C, FH, FW]
- K = 64    filters
- C = 3     channels
- FH = 3    filter height
- FW = 3    filter widht
total elements: 64 × 3 × 3 × 3 = 1728

the outputL
O[K, H, W]
K = 64      output channels
H = 1024    height
W = 1024    widht

the pseudocode is something  like
O[k,x,y] = sum over c=0..C-1
           sum over j=0..FH-1
           sum over i=0..FW-1
           F[k,c,FW-1-i,FH-1-j] * I0[c,x+i,y+j]

where I0 is the padded input tensor, extending it to
I0[C, H+2*P, W+2*P]
P = 1 is the padding size
so that the output tensor has the same height and width as the input tensor
*/

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include "timer.h"

// key vars
const int H = 1024;
const int W = 1024;
const int C = 3;
const int FH = 3;
const int FW = 3;
const int K = 64;
const int P = 1;

/*
 simple convolution kernel without tiling or shared memory
 each thread computes one output element O[k,x,y]
 */
__global__ void convolution(const double *I, const double *F, double *O)
{
    // calculate output coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // bounds check
    if (x >= W || y >= H || k >= K)
        return;

    double sum = 0.0;

    // loop over input channels
    for (int c = 0; c < C; c++)
    {
        // loop over filter height and width
        for (int j = 0; j < FH; j++)
        {
            for (int i = 0; i < FW; i++)
            {
                // calculate padded input coordinates
                // in padded space: x+i, y+j map to original coordinates
                int ix = x + i; // ranges from 0 to W+1
                int iy = y + j; // ranges from 0 to H+1

                // index into padded input I'[c, iy, ix]
                // Padded dimensions: [C, H+2*P, W+2*P]
                int input_idx = c * (H + 2 * P) * (W + 2 * P) + iy * (W + 2 * P) + ix;

                // flipped filter index for convolution (not cross-correlation)
                // F[k, c, FH-1-j, FW-1-i]
                int fi = FW - 1 - i;
                int fj = FH - 1 - j;
                int filter_idx = k * C * FH * FW + c * FH * FW + fj * FW + fi;

                sum += F[filter_idx] * I[input_idx];
            }
        }
    }
    int output_idx = k * H * W + y * W + x;
    O[output_idx] = sum;
}

int main()
{
    // calculate memory sizes needed
    size_t input_size = C * (H + 2 * P) * (W + 2 * P) * sizeof(double);
    size_t filter_size = K * C * FH * FW * sizeof(double);
    size_t output_size = K * H * W * sizeof(double);

    printf("convolution Configuration:\n");
    printf("  I:  [C=%d, H=%d, W=%d] -> Padded: [%d, %d, %d]\n",
           C, H, W, C, H + 2 * P, W + 2 * P);
    printf("  F: [K=%d, C=%d, FH=%d, FW=%d]\n", K, C, FH, FW);
    printf("  O: [K=%d, H=%d, W=%d]\n", K, H, W);

    // allocate host memory
    double *h_I = (double *)malloc(input_size);
    double *h_F = (double *)malloc(filter_size);
    double *h_O = (double *)malloc(output_size);

    if (!h_I || !h_F || !h_O)
    {
        fprintf(stderr, "error: Host memory allocation failed\n");
        return 1;
    }

    printf("initializing padded input tensor...\n");
    for (int c = 0; c < C; c++)
    {
        for (int y = 0; y < H + 2 * P; y++)
        {
            for (int x = 0; x < W + 2 * P; x++)
            {
                int idx = c * (H + 2 * P) * (W + 2 * P) + y * (W + 2 * P) + x;

                // Check if we're in the padding region
                if (x < P || x >= W + P || y < P || y >= H + P)
                {
                    h_I[idx] = 0.0; // padding is zero
                }
                else
                {
                    // interior: original coordinates
                    int orig_x = x - P;
                    int orig_y = y - P;
                    h_I[idx] = c * (orig_x + orig_y);
                }
            }
        }
    }

    // initialize filters
    // F[k, c, i, j] = (c + k) * (i + j)
    printf("initializing filters...\n");
    for (int k = 0; k < K; k++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int j = 0; j < FH; j++)
            {
                for (int i = 0; i < FW; i++)
                {
                    int idx = k * C * FH * FW + c * FH * FW + j * FW + i;
                    h_F[idx] = (c + k) * (i + j);
                }
            }
        }
    }

    double *d_I, *d_F, *d_O;
    cudaMalloc(&d_I, input_size);
    cudaMalloc(&d_F, filter_size);
    cudaMalloc(&d_O, output_size);

    // copy data to device (NOT timed)
    printf("copying data to device...\n");
    cudaMemcpy(d_I, h_I, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, h_F, filter_size, cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    dim3 blockDim(16, 16, 1); // 256 threads per block
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x,
                 (H + blockDim.y - 1) / blockDim.y,
                 (K + blockDim.z - 1) / blockDim.z);

    printf("kernel configuration: grid(%d, %d, %d), block(%d, %d, %d)\n",
           gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);

    // warm-up run
    convolution<<<gridDim, blockDim>>>(d_I, d_F, d_O);
    cudaDeviceSynchronize();

    // now forreal this time execution
    printf("running convolution kernel...\n");
    initialize_timer();
    start_timer();

    convolution<<<gridDim, blockDim>>>(d_I, d_F, d_O);
    cudaDeviceSynchronize(); // Synchronization primitive

    stop_timer();
    double execution_time_ms = elapsed_time() * 1000.0;

    // copy result back to host
    cudaMemcpy(h_O, d_O, output_size, cudaMemcpyDeviceToHost);

    // calculate checksum
    double checksum = 0.0;
    for (int i = 0; i < K * H * W; i++)
    {
        checksum += h_O[i];
    }

    printf("%.6f, %.3f [ms]\n", checksum, execution_time_ms);

    // cleanup
    cudaFree(d_I);
    cudaFree(d_F);
    cudaFree(d_O);
    free(h_I);
    free(h_F);
    free(h_O);

    return 0;
}
