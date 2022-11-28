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

// 12 bytes, should avoid bank conflicts*
struct sPoint{
    float x;
    float y;
    float z;
};

__device__ float diff(float Axi, float Axj, float Ayi, float Ayj, float Azi, float Azj
, float Bxi, float Bxj, float Byi, float Byj, float Bzi, float Bzj){
        float da = sqrt((Axi-Axj)*(Axi-Axj)
                    + (Ayi-Ayj)*(Ayi-Ayj)
                    + (Azi-Azj)*(Azi-Azj));
        float db = sqrt((Bxi-Bxj)*(Bxi-Bxj)
                    + (Byi-Byj)*(Byi-Byj)
                    + (Bzi-Bzj)*(Bzi-Bzj));
        return (da-db) * (da-db);
} 

/*
Solved by 1d array reduction described by NVIDIA docs.
Might be improved with 2d array reduction?
*/
const int blocksize = 256;
__global__ void galaxy_similarity_reduction(const sGalaxy A, const sGalaxy B, const int n , float* output) {
    __shared__ float sdata[blocksize];
    __shared__ float3 As[blocksize];
    __shared__ float3 Bs[blocksize];

    unsigned int tx_g = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tx = threadIdx.x;
    unsigned int bx = blockIdx.x;

    //clear SHMEM
    if (tx == 0)
    {
        flushShmem(sdata, blocksize);
    }

    //wait for shem flush
    __syncthreads();

    for (int tile = 0; tile < n / blocksize; tile++)
    {
        As[tx].x = A.x[tile*blocksize+tx];
        As[tx].y = A.y[tile*blocksize+tx];
        As[tx].z = A.z[tile*blocksize+tx];
        Bs[tx].x = B.x[tile*blocksize+tx];
        Bs[tx].y = B.y[tile*blocksize+tx];
        Bs[tx].z = B.z[tile*blocksize+tx];

        float Ax = A.x[tile*blocksize+tx];
        float Ay = A.y[tile*blocksize+tx];
        float Az = A.z[tile*blocksize+tx];
        float Bx = B.x[tile*blocksize+tx];
        float By = B.y[tile*blocksize+tx];
        float Bz = B.z[tile*blocksize+tx];
        float tmp = 0.0f;
        __syncthreads();
        for (int j = 1; j < blocksize; j++){
            int idx = j + (blocksize * tile); //global index   
            if (idx < tx_g || idx == tx_g){continue;}
            
            float da = sqrt((Ax-As[j].x)*(Ax-As[j].x)
                        + (Ay-As[j].y)*(Ay-As[j].y)
                        + (Az-As[j].z)*(Az-As[j].z));
            float db = sqrt((Bx-Bs[j].x)*(Bx-Bs[j].x)
                        + (By-Bs[j].y)*(By-Bs[j].y)
                        + (Bz-Bs[j].z)*(Bz-Bs[j].z));  
            tmp += (da-db) * (da-db);
        }
        sdata[tx] += tmp;
        __syncthreads();
    }
    
    for (unsigned int stride = blockDim.x/2; stride > 0; stride>>=1)
    {
        if (tx < stride)
        {
            sdata[tx] += sdata[tx + stride];
        }
        
        __syncthreads();
    }
    

    if (tx == 0) output[blockIdx.x] = sdata[0];
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