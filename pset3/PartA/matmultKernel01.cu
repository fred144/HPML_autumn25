//
// matmultKernel01.cu
// optimized matrix multiplication - each thread computes 4 elements (2x2 block)
// FOOTPRINT_SIZE = 32, BLOCK_SIZE = 16
// 


#include "matmultKernel.h"

// FOOTPRINT_SIZE is set to 32 via compiler flag in Makefile
// BLOCK_SIZE remains 16 as set in matmultKernel.h

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){

  // matrix blocks
  float *Asub, *Bsub, *Csub;
  
  // thread and block indices
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  /* each THREAD BLOCK computes one FOOTPRINT_SIZE sub-matrix Csub of C
  each thread within the block computes a 2x2 region in our case*/
  Csub = &C.elements[C.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * block_col];

  /* each thread computes 4 elements in a 2x2 pattern */

 
  float Cvalue[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};  // initialize accumulators for the 2x2 block, changed from matmult00

  /*loop over all sub-matrices in block_row of A and block_col of B*/ 
  for (int m = 0; m < (A.width / BLOCK_SIZE); ++m){
    
    // get Asub and Bsub descriptors
    // we need to load 32 rows from A and 32 cols from B
    // but our thread block is only 16x16, so we load in 2 phases
    
    Asub = &A.elements[A.stride * FOOTPRINT_SIZE * block_row + BLOCK_SIZE * m];
    Bsub = &B.elements[B.stride * BLOCK_SIZE * m + FOOTPRINT_SIZE * block_col];

    /* 
    shared memory for 32x16 of A and 16x32 of B
    but we can only use BLOCK_SIZE dimensions efficiently
    so we'll load two BLOCK_SIZE x BLOCK_SIZE tiles
    */
    __shared__ float shared_A[BLOCK_SIZE * 2][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE * 2];
    
    // load from A: 2 values per thread (to cover 32 rows)
    shared_A[thread_row][thread_col] = 
        Asub[thread_row * A.stride + thread_col];
    shared_A[thread_row + BLOCK_SIZE][thread_col] = 
        Asub[(thread_row + BLOCK_SIZE) * A.stride + thread_col];

    // load from B: 2 values per thread (to cover 32 cols)
    shared_B[thread_row][thread_col] = 
        Bsub[thread_row * B.stride + thread_col];
    shared_B[thread_row][thread_col + BLOCK_SIZE] = 
        Bsub[thread_row * B.stride + thread_col + BLOCK_SIZE];

    // synchronize to ensure all elements are loaded
    __syncthreads();

    // compute the 2x2 block of outputs
    // each thread computes positions:
    // (thread_row, thread_col), (thread_row, thread_col+16)
    // (thread_row+16, thread_col), (thread_row+16, thread_col+16)
    
    #pragma unroll
    for(int e = 0; e < BLOCK_SIZE; ++e) {
      // load values from shared memory once per iteration
      float a0 = shared_A[thread_row][e];
      float a1 = shared_A[thread_row + BLOCK_SIZE][e];
      float b0 = shared_B[e][thread_col];
      float b1 = shared_B[e][thread_col + BLOCK_SIZE];
      
      // compute 2x2 block updates (unrolled)
      Cvalue[0][0] += a0 * b0;  // (thread_row, thread_col)
      Cvalue[0][1] += a0 * b1;  // (thread_row, thread_col+16)
      Cvalue[1][0] += a1 * b0;  // (thread_row+16, thread_col)
      Cvalue[1][1] += a1 * b1;  // (thread_row+16, thread_col+16)
    }

    // synchronize to ensure all Cvalues have been incremented
    __syncthreads();
  }

  // write the 2x2 block to global memory (coalesced writes)
  // position (thread_row, thread_col)
  Csub[thread_row * C.stride + thread_col] = Cvalue[0][0];
  
  // position (thread_row, thread_col + BLOCK_SIZE)
  Csub[thread_row * C.stride + thread_col + BLOCK_SIZE] = Cvalue[0][1];
  
  // position (thread_row + BLOCK_SIZE, thread_col)
  Csub[(thread_row + BLOCK_SIZE) * C.stride + thread_col] = Cvalue[1][0];
  
  // position (thread_row + BLOCK_SIZE, thread_col + BLOCK_SIZE)
  Csub[(thread_row + BLOCK_SIZE) * C.stride + thread_col + BLOCK_SIZE] = Cvalue[1][1];
}

