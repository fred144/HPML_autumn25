/*
used https://arxiv.org/pdf/1410.0759
and refernces therein
uses cuDNN's optimized convolution routines with automatic algorithm selection.
cuDNN will automatically choose the fastest algorithm for the given configuration.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include "timer.h"

// same stuff
const int H = 1024;
const int W = 1024;
const int C = 3;
const int FH = 3;
const int FW = 3;
const int K = 64;
const int P = 1;

// Macro for cuDNN error checking
#define CUDNN_CHECK(call)                                                     \
    {                                                                         \
        cudnnStatus_t err = call;                                             \
        if (err != CUDNN_STATUS_SUCCESS)                                      \
        {                                                                     \
            fprintf(stderr, "cuDNN error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudnnGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

#define CUDA_CHECK(call)                                                     \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    }

int main()
{
    // calculate memory sizes
    // 
    size_t input_size = 1 * C * H * W * sizeof(double); // batch size = 1
    size_t filter_size = K * C * FH * FW * sizeof(double);
    size_t output_size = 1 * K * H * W * sizeof(double);

    printf("cuDNN Convolution:\n");
    printf("  I:  [N=1, C=%d, H=%d, W=%d]\n", C, H, W);
    printf("  F: [K=%d, C=%d, FH=%d, FW=%d]\n", K, C, FH, FW);
    printf("  O: [N=1, K=%d, H=%d, W=%d]\n", K, H, W);
    printf("  padding: %d\n", P);

    // allocate host memory
    double *h_I = (double *)malloc(input_size);
    double *h_F = (double *)malloc(filter_size);
    double *h_O = (double *)malloc(output_size);

    // initialize input tensor (without padding - cuDNN handles padding)
    // I[c, y, x] = c * (x + y) as inm the HW
    printf("initializing input tensor...\n");
    for (int c = 0; c < C; c++)
    {
        for (int y = 0; y < H; y++)
        {
            for (int x = 0; x < W; x++)
            {
                int idx = c * H * W + y * W + x;
                h_I[idx] = c * (x + y);
            }
        }
    }

    // Initialize filters
    // F[k, c, j, i] = (c + k) * (i + j)
    printf("Initializing filters...\n");
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

    // allocate device memory
    double *d_I, *d_F, *d_O;
    CUDA_CHECK(cudaMalloc(&d_I, input_size));
    CUDA_CHECK(cudaMalloc(&d_F, filter_size));
    CUDA_CHECK(cudaMalloc(&d_O, output_size));

    // copy data to device
    printf("Copying data to device...\n");
    CUDA_CHECK(cudaMemcpy(d_I, h_I, input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_F, h_F, filter_size, cudaMemcpyHostToDevice));

    // create cuDNN handle
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    // Create tensor descriptors
    cudnnTensorDescriptor_t input_desc, output_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));

    // set input tensor descriptor: NCHW format, double precision
    // batch size
    // channels
    // height
    // width
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, C, H, W));

    // set output tensor descriptor
    // batch size
    // output channels
    // height (same due to padding)
    // width (same due to padding)
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, K, H, W));

    // create filter descriptor
    // output channels
    // input channels
    // height
    // width
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, FH, FW)); 

    // ccreate convolution descriptor
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    // Set convolution descriptor with padding
    // Note: We use CUDNN_CROSS_CORRELATION mode, but we'll flip the filter manually
    // to achieve true convolution behavior
    // pad height
    // pad width
    // vertical stride
    // horizontal stride
    // dilation height
    // dilation width
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, P, P, 1, 1, 1,  1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE));

    // since the problem asks for convolution (flipped filter), we need to flip the filter
    // create a flipped filter on the host and copy it
    double *h_F_flipped = (double *)malloc(filter_size);
    for (int k = 0; k < K; k++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int j = 0; j < FH; j++)
            {
                for (int i = 0; i < FW; i++)
                {
                    int src_idx = k * C * FH * FW + c * FH * FW + j * FW + i;
                    int dst_idx = k * C * FH * FW + c * FH * FW + (FH - 1 - j) * FW + (FW - 1 - i);
                    h_F_flipped[dst_idx] = h_F[src_idx];
                }
            }
        }
    }

    // copy flipped filter to device
    CUDA_CHECK(cudaMemcpy(d_F, h_F_flipped, filter_size, cudaMemcpyHostToDevice));
    free(h_F_flipped);

    // find the fastest convolution algorithm-- requetion one algorith
    printf("finding fastest convolution algorithm...\n");
    cudnnConvolutionFwdAlgoPerf_t perf;
    int returnedAlgoCount;
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(cudnn, input_desc, filter_desc, conv_desc, output_desc, 1,  &returnedAlgoCount, &perf));

    printf("selected algorithm: %d\n", perf.algo);
    printf("expected time from warmup: %.3f ms\n", perf.time);

    // get workspace size
    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, filter_desc, conv_desc, output_desc, perf.algo, &workspace_size));

    // allocate workspace
    void *d_workspace = nullptr;
    if (workspace_size > 0)
    {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    }

    // scaling factors for convolution (alpha * conv + beta * output)
    double alpha = 1.0;
    double beta = 0.0;

    // warm-up run
    CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha, input_desc, d_I, filter_desc, d_F, conv_desc, perf.algo, d_workspace, workspace_size, &beta, output_desc, d_O));
    CUDA_CHECK(cudaDeviceSynchronize());

    // timed execution
    printf("running cuDNN convolution...\n");
    initialize_timer();
    start_timer();

    CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha, input_desc, d_I, filter_desc, d_F, conv_desc, perf.algo, d_workspace, workspace_size, &beta, output_desc, d_O));
    CUDA_CHECK(cudaDeviceSynchronize());

    stop_timer();
    double execution_time_ms = elapsed_time() * 1000.0;

    // copy result back to host
    CUDA_CHECK(cudaMemcpy(h_O, d_O, output_size, cudaMemcpyDeviceToHost));

    // calculate checksum
    double checksum = 0.0;
    for (int i = 0; i < K * H * W; i++)
    {
        checksum += h_O[i];
    }

    // print results in required format
    printf("FINAL: %.6f, %.3f [ms]\n", checksum, execution_time_ms);

    // cleanup
    if (d_workspace)
        cudaFree(d_workspace);
    cudaFree(d_I);
    cudaFree(d_F);
    cudaFree(d_O);
    free(h_I);
    free(h_F);
    free(h_O);

    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);

    return 0;
}