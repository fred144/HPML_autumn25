///
/// matmultKernel01_step2.cu
/// Step 2: Add coalesced memory reads and writes
///

#include "matmultKernel.h"

#define FOOTPRINT_SIZE 32

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    float Cvalue00 = 0.0f;
    float Cvalue01 = 0.0f;
    float Cvalue10 = 0.0f;
    float Cvalue11 = 0.0f;

    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m){
        float* Asub = &A.elements[A.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * m];
        float* Bsub = &B.elements[B.stride * BLOCK_SIZE * m + BLOCK_SIZE * block_col];

        __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

        // Step 2: COALESCED MEMORY LOADING
        // Each thread loads the specific elements needed for its 2x2 block
        int load_row_A0 = thread_row * 2;
        int load_row_A1 = thread_row * 2 + 1;
        int load_col_A0 = thread_col * 2;
        int load_col_A1 = thread_col * 2 + 1;
        
        // Load from matrix A - coalesced pattern
        if (load_row_A0 < BLOCK_SIZE && load_col_A0 < BLOCK_SIZE)
            shared_A[load_row_A0][load_col_A0] = Asub[load_row_A0 * A.stride + load_col_A0];
        if (load_row_A0 < BLOCK_SIZE && load_col_A1 < BLOCK_SIZE)
            shared_A[load_row_A0][load_col_A1] = Asub[load_row_A0 * A.stride + load_col_A1];
        if (load_row_A1 < BLOCK_SIZE && load_col_A0 < BLOCK_SIZE)
            shared_A[load_row_A1][load_col_A0] = Asub[load_row_A1 * A.stride + load_col_A0];
        if (load_row_A1 < BLOCK_SIZE && load_col_A1 < BLOCK_SIZE)
            shared_A[load_row_A1][load_col_A1] = Asub[load_row_A1 * A.stride + load_col_A1];
            
        // Load from matrix B - coalesced pattern
        if (load_row_A0 < BLOCK_SIZE && load_col_A0 < BLOCK_SIZE)
            shared_B[load_row_A0][load_col_A0] = Bsub[load_row_A0 * B.stride + load_col_A0];
        if (load_row_A0 < BLOCK_SIZE && load_col_A1 < BLOCK_SIZE)
            shared_B[load_row_A0][load_col_A1] = Bsub[load_row_A0 * B.stride + load_col_A1];
        if (load_row_A1 < BLOCK_SIZE && load_col_A0 < BLOCK_SIZE)
            shared_B[load_row_A1][load_col_A0] = Bsub[load_row_A1 * B.stride + load_col_A0];
        if (load_row_A1 < BLOCK_SIZE && load_col_A1 < BLOCK_SIZE)
            shared_B[load_row_A1][load_col_A1] = Bsub[load_row_A1 * B.stride + load_col_A1];

        __syncthreads();

        // Step 2: Proper computation using correct shared memory elements
        for(int e=0; e<BLOCK_SIZE; ++e) {
            float a_val0 = shared_A[thread_row * 2][e];      // Correct row for Cvalue00 & Cvalue01
            float a_val1 = shared_A[thread_row * 2 + 1][e];  // Correct row for Cvalue10 & Cvalue11
            float b_val0 = shared_B[e][thread_col * 2];      // Correct col for Cvalue00 & Cvalue10
            float b_val1 = shared_B[e][thread_col * 2 + 1];  // Correct col for Cvalue01 & Cvalue11
            
            Cvalue00 += a_val0 * b_val0;
            Cvalue01 += a_val0 * b_val1;
            Cvalue10 += a_val1 * b_val0;
            Cvalue11 += a_val1 * b_val1;
        }

        __syncthreads();
    }

    // Step 2: COALESCED MEMORY WRITING
    float* Csub = &C.elements[C.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * block_col];
    
    int row0 = thread_row * 2;
    int row1 = thread_row * 2 + 1;
    int col0 = thread_col * 2;
    int col1 = thread_col * 2 + 1;
    
    // Consecutive threads write consecutive columns - coalesced
    Csub[row0 * C.stride + col0] = Cvalue00;
    Csub[row0 * C.stride + col1] = Cvalue01;
    Csub[row1 * C.stride + col0] = Cvalue10;
    Csub[row1 * C.stride + col1] = Cvalue11;
}