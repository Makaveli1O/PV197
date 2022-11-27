/*
    Ensures safe cuda application executions
*/
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

struct sPoint{
    float x;
    float y;
    float z;
};

/*
Solved by 1d array reduction described by NVIDIA docs.
Might be improved with 2d array reduction?
*/
const int blocksize = 4;
__global__ void galaxy_similarity_reduction(const sGalaxy A, const sGalaxy B, int n , float* output) {
    __shared__ float sdata[blocksize];
    __shared__ sPoint As[blocksize];
    __shared__ sPoint Bs[blocksize];
    __shared__ sPoint Asj[blocksize];
    __shared__ sPoint Bsj[blocksize];

    unsigned int tid = threadIdx.x;
    unsigned int tx_g = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tx = threadIdx.x;

    As[tx].x = A.x[tx_g];
    As[tx].y = A.y[tx_g];
    As[tx].z = A.z[tx_g];
    Bs[tx].x = B.x[tx_g];
    Bs[tx].y = B.y[tx_g];
    Bs[tx].z = B.z[tx_g];

    //clear SHMEM
    if (tid == 0)
    {
        flushShmem(sdata, blocksize);
    }

    //wait for shem flush
    __syncthreads();

    for (int tile = blockIdx.x; tile < n / blocksize; tile++)
    {
        for (int j = 0; j < blocksize; j++)
        {
            Asj[tx].x = A.x[j + (blocksize * tile)];
            Asj[tx].y = A.y[j + (blocksize * tile)];
            Asj[tx].z = A.z[j + (blocksize * tile)];
            Bsj[tx].x = B.x[j + (blocksize * tile)];
            Bsj[tx].y = B.y[j + (blocksize * tile)];
            Bsj[tx].z = B.z[j + (blocksize * tile)];
            
        }
        __syncthreads();

            for (int j = 0; j < blocksize; j++){
                int idx = j + (blocksize * tile);
                if (idx > blocksize)
                {
                    printf("Loading required!\n");
                    //Bsj[tx].x = B.x[j + (blocksize * tile)];
                    //Bsj[tx].y = B.y[j + (blocksize * tile)];
                    //Bsj[tx].z = B.z[j + (blocksize * tile)];
                }
                
                if (idx < tx_g || idx == tx_g) continue;
                printf("idx: %d , j:%d \n", idx, j);
                
                float da = sqrt((As[tx].x-A.x[idx])*(As[tx].x-A.x[idx])
                            + (As[tx].y-A.y[idx])*(As[tx].y-A.y[idx])
                            + (As[tx].z-A.z[idx])*(As[tx].z-A.z[idx]));
                float db = sqrt((Bs[tx].x-B.x[idx])*(Bs[tx].x-B.x[idx])
                            + (Bs[tx].y-B.y[idx])*(Bs[tx].y-B.y[idx])
                            + (Bs[tx].z-B.z[idx])*(Bs[tx].z-B.z[idx]));
                /*
                float da = sqrt((As[tx].x-As[j].x)*(As[tx].x-As[j].x)
                            + (As[tx].y-As[j].y)*(As[tx].y-As[j].y)
                            + (As[tx].z-As[j].z)*(As[tx].z-As[j].z));
                float db = sqrt((Bs[tx].x-Bs[j].x)*(Bs[tx].x-Bs[j].x)
                            + (Bs[tx].y-Bs[j].y)*(Bs[tx].y-Bs[j].y)
                            + (Bs[tx].z-Bs[j].z)*(Bs[tx].z-Bs[j].z));
                */
                sdata[tx] += (da-db) * (da-db);
            }
        __syncthreads();
    }
    

    
    
    //do the math

    for(int j = tx_g + 1; j < n; j++){
        //printf("j: %d\n",j);
        //float da = sqrt((As[tx].x-A.x[j])*(As[tx].x-A.x[j])
        //            + (As[tx].y-A.y[j])*(As[tx].y-A.y[j])
        //            + (As[tx].z-A.z[j])*(As[tx].z-A.z[j]));
        //float db = sqrt((Bs[tx].x-B.x[j])*(Bs[tx].x-B.x[j])
        //            + (Bs[tx].y-B.y[j])*(Bs[tx].y-B.y[j])
        //            + (Bs[tx].z-B.z[j])*(Bs[tx].z-B.z[j]));
        //sdata[tid] += (da-db) * (da-db);
    }
    

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

    //determine correct number of output elements after reduction
    int numOutputElements = n / (blocksize / 2);
    if (n % (blocksize / 2)) {
        numOutputElements++;
    }

    hostOutput = (float *)malloc(numOutputElements * sizeof(float));
    // Round up according to array size 
    int gridSize = (n + blocksize - 1) / blocksize; 
    //printf("blocksize : %d gridSize: %d\n", blocksize, gridSize);
    //allocate GPU memory
    gpuSafeExec(cudaMalloc((void **)&deviceOutput, numOutputElements * sizeof(float)));
    //std::cerr << "galaxy_similarity_reduction<<<" << gridSize << "," << blocksize << "," << 0 << ">>>\n";
    galaxy_similarity_reduction<<<gridSize, blocksize>>>(A, B, n, deviceOutput);
    //move GPU results to CPU via PCIe
    gpuSafeExec(cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost));

    //accumulate the sum in the first element
    for (int i = 1; i < numOutputElements; i++) {
        hostOutput[0] += hostOutput[i]; 
    }
    
    //use overall square root out of GPU, to avoid race condition
    float retval = sqrt(1/((float)n*((float)n-1)) * hostOutput[0]);

    //cleanup
    gpuSafeExec(cudaFree(deviceOutput));
    free(hostOutput);

    return retval;
}