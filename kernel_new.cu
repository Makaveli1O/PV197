#include <iostream>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

#include <cub/cub.cuh>

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

struct sPoint{
    float x;
    float y;
    float z;
};

#include <assert.h>
const int blocksize = 256;
__global__ void galaxy_similarity_reduction(const sGalaxy A, const sGalaxy B, const int n , float* output) {
    __shared__ float sdata[blocksize];
    __shared__ sPoint As[blocksize];
    __shared__ sPoint Bs[blocksize];
    __shared__ sPoint Asj[blocksize+1];
    __shared__ sPoint Bsj[blocksize+1];

    //get unique block index
    const unsigned long long int blockId = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + gridDim.x * gridDim.y * blockIdx.z; //3D
    //get unique thread index
    const unsigned long long int global_tid = blockId * blockDim.x + threadIdx.x; 
    //get thread id within block
    const unsigned long long int block_tid = threadIdx.x;

    //clear SHMEM
    if (global_tid == 0)
        flushShmem(sdata, blocksize);
    __syncthreads();

    //load I into shmem for this block, then iterate j
    As[block_tid].x = A.x[global_tid];
    As[block_tid].y = A.y[global_tid];
    As[block_tid].z = A.z[global_tid];
    Bs[block_tid].x = B.x[global_tid];
    Bs[block_tid].y = B.y[global_tid];
    Bs[block_tid].z = B.z[global_tid];
    int i = global_tid;
    __syncthreads();
    for(int j = i+1; j < n; j++){
        float da = sqrt((A.x[i]-A.x[j])*(A.x[i]-A.x[j])
                    + (A.y[i]-A.y[j])*(A.y[i]-A.y[j])
                    + (A.z[i]-A.z[j])*(A.z[i]-A.z[j]));
        float db = sqrt((B.x[i]-B.x[j])*(B.x[i]-B.x[j])
                    + (B.y[i]-B.y[j])*(B.y[i]-B.y[j])
                    + (B.z[i]-B.z[j])*(B.z[i]-B.z[j]));
        sdata[block_tid] += (da-db) * (da-db);
    }
    __syncthreads();
    //if (block_tid == 0)
    //{
        /*
        float tmp = 0.0f;
        for (int t = blockIdx.x; t < n / blocksize; t++)
        {
            for(int j = block_tid+1; j < blocksize + 1; j++)
            {
                printf("As[%d] = Asj[%ld] \n",block_tid,j + (blockId * blocksize) + (t * blocksize / 2));
                int idx = j + (blockId * blocksize) + (t * blocksize/2);
                tmp += diff(   A.x[block_tid], A.x[idx], A.y[block_tid], A.y[idx], A.z[block_tid], A.z[idx],
                            B.x[block_tid], B.x[idx], B.y[block_tid], B.y[idx], B.z[block_tid], B.z[idx]
                    );
                //printf("j = %d + (%ld * %d) = %ld\n", j, blockId, blocksize,j + (blockId * blocksize));
                
                //printf("As[%ld] = %ld \n",block_tid,idx);
                
                Asj[block_tid].x = A.x[idx];
                Asj[block_tid].y = A.y[idx];
                Asj[block_tid].z = A.z[idx];
                Bsj[block_tid].x = B.x[idx];
                Bsj[block_tid].y = B.y[idx];
                Bsj[block_tid].z = B.z[idx];
                
                

            }
        }
        */
    //}
    //sdata[block_tid] += tmp;
    for (unsigned int stride = blockDim.x/2; stride > 0; stride>>=1)
    {
        if (block_tid < stride)
        {
            sdata[block_tid] += sdata[block_tid + stride];
        }
        
        __syncthreads();
        
    }
    

    //write accumulated result of this block to global memory
    if (block_tid == 0) output[blockId] = sdata[0];
}


float solveGPU(const sGalaxy A, const sGalaxy B, const int n) {
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