///
/// matmultKernel01_step1.cu
/// Step 1: Each thread computes 4 values (2x2 block), FOOTPRINT_SIZE=32
/// No other optimizations yet
///

#include "matmultKernel.h"

#define FOOTPRINT_SIZE 32

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    // Step 1: Each thread computes 4 elements (2x2 block)
    float Cvalue00 = 0.0f;  // (thread_row*2, thread_col*2)
    float Cvalue01 = 0.0f;  // (thread_row*2, thread_col*2+1)
    float Cvalue10 = 0.0f;  // (thread_row*2+1, thread_col*2)
    float Cvalue11 = 0.0f;  // (thread_row*2+1, thread_col*2+1)

    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m){
        // Get submatrices (same as original)
        float* Asub = &A.elements[A.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * m];
        float* Bsub = &B.elements[B.stride * BLOCK_SIZE * m + BLOCK_SIZE * block_col];

        __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

        // Step 1: Still loading single elements per thread (non-coalesced)
        shared_A[thread_row][thread_col] = Asub[thread_row * A.stride + thread_col];
        shared_B[thread_row][thread_col] = Bsub[thread_row * B.stride + thread_col];

        __syncthreads();

        // Step 1: Compute 4 values instead of 1
        for(int e=0; e<BLOCK_SIZE; ++e) {
            // Use same shared memory values for all 4 computations
            // (This will be optimized in later steps)
            float a_val = shared_A[thread_row][e];
            float b_val = shared_B[e][thread_col];
            
            Cvalue00 += a_val * b_val;  // All same for now
            Cvalue01 += a_val * b_val;
            Cvalue10 += a_val * b_val;
            Cvalue11 += a_val * b_val;
        }

        __syncthreads();
    }

    // Step 1: Write 4 values instead of 1
    float* Csub = &C.elements[C.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * block_col];
    
    int row0 = thread_row * 2;
    int row1 = thread_row * 2 + 1;
    int col0 = thread_col * 2;
    int col1 = thread_col * 2 + 1;
    
    Csub[row0 * C.stride + col0] = Cvalue00;
    Csub[row0 * C.stride + col1] = Cvalue01;
    Csub[row1 * C.stride + col0] = Cvalue10;
    Csub[row1 * C.stride + col1] = Cvalue11;
}