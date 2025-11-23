/*
same procedure, but with tiling
some resources i used
https://ajdillhoff.github.io/notes/gpu_pattern_convolution/
https://github.com/debowin/cuda-tiled-2D-convolution 
*/
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include "timer.h"


const int H = 1024;
const int W = 1024;
const int C = 3;
const int FH = 3;
const int FW = 3;
const int K = 64;
const int P = 1;

// these tile dimensions are subjective
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

/*
tiled convolution kernel with shared memory
Each thread block processes a TILE_WIDTH x TILE_HEIGHT tile of the output
for one output channel k
 */
__global__ void tiledConvolution(const double *I, const double *F, double *O)
{
    // Calculate output coordinates
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_WIDTH + tx;
    int y = blockIdx.y * TILE_HEIGHT + ty;
    int k = blockIdx.z;

    // Shared memory for input tile (includes halo for filter)
    // Need extra cells for the filter overlap
    __shared__ double tile[C][TILE_HEIGHT + FH - 1][TILE_WIDTH + FW - 1];

    // Accumulator for this thread's output
    double sum = 0.0;

    // Load input tiles for all channels into shared memory
    for (int c = 0; c < C; c++)
    {
        // Each thread loads one or more elements into shared memory
        // We need to load a (TILE_HEIGHT + FH - 1) x (TILE_WIDTH + FW - 1) region
        
        // Calculate how many elements each thread needs to load
        int halo_height = TILE_HEIGHT + FH - 1;
        int halo_width = TILE_WIDTH + FW - 1;
        int total_elements = halo_height * halo_width;
        int threads_per_block = TILE_HEIGHT * TILE_WIDTH;
        int elements_per_thread = (total_elements + threads_per_block - 1) / threads_per_block;
        
        int thread_id = ty * TILE_WIDTH + tx;
        
        for (int e = 0; e < elements_per_thread; e++)
        {
            int idx = thread_id + e * threads_per_block;
            if (idx < total_elements)
            {
                int tile_y = idx / halo_width;
                int tile_x = idx % halo_width;
                
                // Global coordinates in padded input
                int global_x = blockIdx.x * TILE_WIDTH + tile_x;
                int global_y = blockIdx.y * TILE_HEIGHT + tile_y;
                
                // Load from global memory (padded input)
                if (global_x < W + 2*P && global_y < H + 2*P)
                {
                    int input_idx = c * (H + 2*P) * (W + 2*P) + global_y * (W + 2*P) + global_x;
                    tile[c][tile_y][tile_x] = I[input_idx];
                }
                else
                {
                    tile[c][tile_y][tile_x] = 0.0;
                }
            }
        }
    }
    
    // Synchronize to ensure all data is loaded
    __syncthreads();

    // Compute convolution if within output bounds
    if (x < W && y < H && k < K)
    {
        // Perform convolution using shared memory
        for (int c = 0; c < C; c++)
        {
            for (int j = 0; j < FH; j++)
            {
                for (int i = 0; i < FW; i++)
                {
                    // Shared memory coordinates
                    int tile_y = ty + j;
                    int tile_x = tx + i;
                    
                    // Flipped filter index
                    int fi = FW - 1 - i;
                    int fj = FH - 1 - j;
                    int filter_idx = k * C * FH * FW + c * FH * FW + fj * FW + fi;
                    
                    // Accumulate from shared memory
                    sum += F[filter_idx] * tile[c][tile_y][tile_x];
                }
            }
        }
        
        // Write output
        int output_idx = k * H * W + y * W + x;
        O[output_idx] = sum;
    }
}

int main()
{
    // Calculate memory sizes
    size_t input_size = C * (H + 2*P) * (W + 2*P) * sizeof(double);
    size_t filter_size = K * C * FH * FW * sizeof(double);
    size_t output_size = K * H * W * sizeof(double);

    printf("tiled Convolution :\n");
    printf("  I:  [C=%d, H=%d, W=%d] -> Padded: [%d, %d, %d]\n",
           C, H, W, C, H + 2*P, W + 2*P);
    printf("  F: [K=%d, C=%d, FH=%d, FW=%d]\n", K, C, FH, FW);
    printf("  O: [K=%d, H=%d, W=%d]\n", K, H, W);
    printf("  chosen tile size: %dx%d\n", TILE_WIDTH, TILE_HEIGHT);

    // Allocate host memory
    double *h_I = (double *)malloc(input_size);
    double *h_F = (double *)malloc(filter_size);
    double *h_O = (double *)malloc(output_size);

    if (!h_I || !h_F || !h_O) {
        fprintf(stderr, "Error: Host memory allocation failed\n");
        return 1;
    }

    // Initialize padded input tensor
    printf("Initializing padded input tensor...\n");
    for (int c = 0; c < C; c++) {
        for (int y = 0; y < H + 2*P; y++) {
            for (int x = 0; x < W + 2*P; x++) {
                int idx = c * (H + 2*P) * (W + 2*P) + y * (W + 2*P) + x;
                
                if (x < P || x >= W + P || y < P || y >= H + P) {
                    h_I[idx] = 0.0;  // Padding
                } else {
                    int orig_x = x - P;
                    int orig_y = y - P;
                    h_I[idx] = c * (orig_x + orig_y);
                }
            }
        }
    }

    // Initialize filters
    printf("Initializing filters...\n");
    for (int k = 0; k < K; k++) {
        for (int c = 0; c < C; c++) {
            for (int j = 0; j < FH; j++) {
                for (int i = 0; i < FW; i++) {
                    int idx = k * C * FH * FW + c * FH * FW + j * FW + i;
                    h_F[idx] = (c + k) * (i + j);
                }
            }
        }
    }

    // Allocate device memory
    double *d_I, *d_F, *d_O;
    cudaMalloc(&d_I, input_size);
    cudaMalloc(&d_F, filter_size);
    cudaMalloc(&d_O, output_size);

    // Copy data to device
    printf("Copying data to device...\n");
    cudaMemcpy(d_I, h_I, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, h_F, filter_size, cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    dim3 blockDim(TILE_WIDTH, TILE_HEIGHT, 1);
    dim3 gridDim((W + TILE_WIDTH - 1) / TILE_WIDTH,
                 (H + TILE_HEIGHT - 1) / TILE_HEIGHT,
                 K);

    printf("Kernel configuration: grid(%d, %d, %d), block(%d, %d, %d)\n",
           gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);

    // Calculate shared memory size
    size_t shared_mem_size = C * (TILE_HEIGHT + FH - 1) * (TILE_WIDTH + FW - 1) * sizeof(double);
    printf("Shared memory per block: %.2f KB\n", shared_mem_size / 1024.0);

    // Warm-up run
    tiledConvolution<<<gridDim, blockDim>>>(d_I, d_F, d_O);
    cudaDeviceSynchronize();

    // Timed execution
    printf("Running tiled convolution kernel...\n");
    initialize_timer();
    start_timer();

    tiledConvolution<<<gridDim, blockDim>>>(d_I, d_F, d_O);
    cudaDeviceSynchronize();  // Synchronization primitive

    stop_timer();
    double execution_time_ms = elapsed_time() * 1000.0;

    // Copy result back to host
    cudaMemcpy(h_O, d_O, output_size, cudaMemcpyDeviceToHost);

    // Calculate checksum
    double checksum = 0.0;
    for (int i = 0; i < K * H * W; i++) {
        checksum += h_O[i];
    }

    // Print results in required format
    printf("%.6f, %.3f [ms]\n", checksum, execution_time_ms);

    // Cleanup
    cudaFree(d_I);
    cudaFree(d_F);
    cudaFree(d_O);
    free(h_I);
    free(h_F);
    free(h_O);

    return 0;
}