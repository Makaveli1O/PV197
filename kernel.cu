//TODO kernel implementation

/*
Performs iteration paraller calculation no GPU
*/

#define NUM_BLOCKS = 10
#define NUM_THREADS = 256

__global__ void increment_gpu(sGalaxy A, sGalaxy B, int n ,float* result){
    //determine unique index within grid, miximizing block usage
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n && j > i && i != j){
        float tmp = 0.0f;
        //printf("a.x[i->%d]: %f  a.x[j->%d]: %f\n", i,A.x[i], j, A.x[j]);
        float da = sqrt((A.x[i]-A.x[j])*(A.x[i]-A.x[j])
                    + (A.y[i]-A.y[j])*(A.y[i]-A.y[j])
                    + (A.z[i]-A.z[j])*(A.z[i]-A.z[j]));
        float db = sqrt((B.x[i]-B.x[j])*(B.x[i]-B.x[j])
                    + (B.y[i]-B.y[j])*(B.y[i]-B.y[j])
                    + (B.z[i]-B.z[j])*(B.z[i]-B.z[j]));
        tmp = (da-db) * (da-db);
        atomicAdd(result, tmp);

        printf("[%d][%d](%f-%f) * (%f-%f) -- tmp = %f -- result %f\n", i,j,da,db,da,db,tmp, *result);
        if(i == n-2 && j == n-1){
            float diff = *result;
            *result = sqrt(1/((float)n*((float)n-1)) * diff);
            printf("RESULT ::  : %f", *result);
        }
        return ;
    }
}


float solveGPU(sGalaxy A, sGalaxy B, int n) {
    float h_result[n];
    float *d_result;

    //allocate space on GPU for d_result variable
    cudaMalloc((void**)&d_result, sizeof(float));

    //define grid and block sizes
    dim3 block_size(16,16);
    dim3 grid_size((n+15)/16,  (n+15)/16);

    //Launch kernel with pass by referennce attribute and N size
    increment_gpu <<<grid_size, block_size>>>(A, B, n,d_result);

    //copy data from GPU memory to CPU memory via PCIe bus
    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    return *h_result;
}