#define gpuSafeExec(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/*
    Clears shared memory which is not full of previous
    numbers. Shmem is remembers values between consecutive
    kernel calls.
*/
__device__ void flushShmem(float *shmem, int shmemSize){
    for (int i = 0; i < shmemSize; i ++)
        shmem[i] = 0.0f;
    return;
}

__global__ void galaxy_similarity_reduce(const sGalaxy A, const sGalaxy B, int n , float* output, int shmemSize) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    //clear SHMEM
    if (tid == 0)
    {
        flushShmem(sdata, shmemSize);

        for (int i = 0; i < shmemSize; i++)
        {
            if (sdata[i] != 0.0f)
            {
                printf("sdata[%d] = %f", i,sdata[i]);
            }  
        }
    }

    //wait for shem flush
    __syncthreads();
    
    
    for(int j = i+1; j < n; j++){
        float da = sqrt((A.x[i]-A.x[j])*(A.x[i]-A.x[j])
                    + (A.y[i]-A.y[j])*(A.y[i]-A.y[j])
                    + (A.z[i]-A.z[j])*(A.z[i]-A.z[j]));
        float db = sqrt((B.x[i]-B.x[j])*(B.x[i]-B.x[j])
                    + (B.y[i]-B.y[j])*(B.y[i]-B.y[j])
                    + (B.z[i]-B.z[j])*(B.z[i]-B.z[j]));
        sdata[tid] += (da-db) * (da-db);
    }

    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}


float solveGPU(sGalaxy A, sGalaxy B, int n) {
    float *hostOutput; 
    float *deviceOutput; 

    int blockSize;
    int minGridSize;
    int gridSize;

    //use cuda occupancy calculator to determine grid and block sizes
    gpuSafeExec(cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, 
                                       galaxy_similarity_reduce, 0, 0)); 
    int numOutputElements = n / (blockSize / 2);
    if (n % (blockSize / 2)) {
        numOutputElements++;
    }

    hostOutput = (float *)malloc(numOutputElements * sizeof(float));
    // Round up according to array size 
    gridSize = (n + blockSize - 1) / blockSize; 

    gpuSafeExec(cudaMalloc((void **)&deviceOutput, numOutputElements * sizeof(float)));

    galaxy_similarity_reduce <<<gridSize, blockSize, blockSize*sizeof(float) >>>(A, B, n, deviceOutput, blockSize);

    gpuSafeExec(cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost));

    for (int ii = 1; ii < numOutputElements; ii++) {
        hostOutput[0] += hostOutput[ii]; //accumulates the sum in the first element
    }
    
    //use overall square root out of GPU, to avoid race condition
    float retval = sqrt(1/((float)n*((float)n-1)) * hostOutput[0]);

    gpuSafeExec(cudaFree(deviceOutput));
    free(hostOutput);

    return retval;
}