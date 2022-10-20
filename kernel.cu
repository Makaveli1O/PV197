//TODO kernel implementation

/*
Performs iteration paraller calculation no GPU
*/

#define NUM_BLOCKS = 10
#define NUM_THREADS = 256

__global__ void increment_gpu(sGalaxy A, sGalaxy B, int n, float* result){
    //determine unique index within grid, miximizing block usage
    __shared__ float d_result;
    //use odd/even numbers to i/j index?
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = blockIdx.x * blockDim.x + threadIdx.x;


    //int j = i + 1;
    if (i < n - 1){
        d_result = 0.0f;
        // porovnat A.x[i+1] tu a ked sa buildi ci su rovnake!  
        //printf("a.x[i->%d]: %f  a.x[j->%d]: %f\n", i,A.x[i], j, A.x[j]);
        
        float da = sqrt((A.x[i]-A.x[j])*(A.x[i]-A.x[j])
                    + (A.y[i]-A.y[j])*(A.y[i]-A.y[j])
                    + (A.z[i]-A.z[j])*(A.z[i]-A.z[j]));
        float db = sqrt((B.x[i]-B.x[j])*(B.x[i]-B.x[j])
                    + (B.y[i]-B.y[j])*(B.y[i]-B.y[j])
                    + (B.z[i]-B.z[j])*(B.z[i]-B.z[j]));
        float tmp = (da-db) * (da-db);
        printf("[%d][%d](%f-%f) * (%f-%f)\n", i,j,da,db,da,db);

        d_result += tmp;
        *result = sqrt(1/((float)n*((float)n-1)) * d_result);
    }
    return;

}


float solveGPU(sGalaxy A, sGalaxy B, int n) {
    float h_result[n];
    float *d_result;

    //allocate space on GPU for d_result variable
    cudaMalloc((void**)&d_result, sizeof(float));

    //define grid and block sizes
    dim3 block_size(n,n);
    dim3 grid_size(1,1);

    //Launch kernel with pass by referennce attribute and N size
    increment_gpu <<<grid_size, block_size>>>(A, B, n, d_result);

    //copy data from GPU memory to CPU memory via PCIe bus
    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    return *h_result;
}