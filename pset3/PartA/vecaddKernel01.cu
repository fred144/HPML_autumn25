///
/// vecAddKernel01.cu
/// optimized coalesced vector addition (Q2)

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    // total num of threads in the grid
    int totalThreads = gridDim.x * blockDim.x;
    //thread unique ID
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    
    // per thread, processes N elements with coalesced memory access consecutive threads access consecutive as opposed to stided pattern in Add Vectors memory locations
    for (int i = 0; i < N; i++) {
        int index = threadId + i * totalThreads;
        C[index] = A[index] + B[index];
    }
}