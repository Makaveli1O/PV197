#define gpuSafeExec(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void galaxy_similarity_reduce(const sGalaxy A, const sGalaxy B, int n , float* output) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
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
    int numThreadsPerBlock = 1024;

    float *hostOutput; 
    float *deviceOutput; 

    int numInputElements = n;
    int numOutputElements; // number of elements in the output list, initialised below

    numOutputElements = numInputElements / (numThreadsPerBlock / 2);
    if (numInputElements % (numThreadsPerBlock / 2)) {
        numOutputElements++;
    }

    hostOutput = (float *)malloc(numOutputElements * sizeof(float));


    //const dim3 blockSize(numThreadsPerBlock, 1, 1);
    //const dim3 gridSize(numOutputElements, 1, 1);

    int blockSize;   // The launch configurator returned block size 
    int minGridSize; // The minimum grid size needed to achieve the 
                   // maximum occupancy for a full device launch 
    int gridSize;    // The actual grid size needed, based on input size 

    gpuSafeExec(cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, 
                                       galaxy_similarity_reduce, 0, 0)); 
    // Round up according to array size 
    gridSize = (n + blockSize - 1) / blockSize; 

    gpuSafeExec(cudaMalloc((void **)&deviceOutput, numOutputElements * sizeof(float)));

    galaxy_similarity_reduce <<<gridSize, blockSize, numThreadsPerBlock*sizeof(float) >>>(A, B, n, deviceOutput);

    gpuSafeExec(cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost));

    for (int ii = 1; ii < numOutputElements; ii++) {
        hostOutput[0] += hostOutput[ii]; //accumulates the sum in the first element
    }
    
    //use overall square root out of GPU, to avoid race condition
    float retval = sqrt(1/((float)n*((float)n-1)) * hostOutput[0]);

   

    return retval;
}