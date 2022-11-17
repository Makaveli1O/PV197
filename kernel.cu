#include <iostream>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

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

struct MatrixIndex{
    int i = 0;
    int j = 0;
};

std::ostream& operator<<(std::ostream& os, const MatrixIndex& m){
    os << "(" << m.i << "," << m.j << ")";
    return os;
}

struct ConvertLinearIndexToTriangularMatrixIndex{
    int dim;

    __host__ __device__
    ConvertLinearIndexToTriangularMatrixIndex(int dimension) : dim(dimension){}

    __host__ __device__
    MatrixIndex operator()(int linear) const {
        MatrixIndex result;
       //check if those indices work for you

        //compute i and j from linear index https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
        result.i = dim - 2 - floor(sqrt(-8*linear + 4*dim*(dim-1)-7)/2.0 - 0.5);
        result.j = linear + result.i + 1 - dim*(dim-1)/2 + (dim-result.i)*((dim-result.i)-1)/2;

        return result;
    }
};

struct ComputeDelta{
    sGalaxy A;
    sGalaxy B;

    __host__ __device__
    ComputeDelta(sGalaxy _A, sGalaxy _B){
        /* init A and B*/
        A = _A;
        B = _B;
    }

    __host__ __device__
    float operator()(const MatrixIndex& index) const{
        
        float da = 0;
        float db = 0;

        da = sqrt((A.x[index.i]-A.x[index.j])*(A.x[index.i]-A.x[index.j])
                    + (A.y[index.i]-A.y[index.j])*(A.y[index.i]-A.y[index.j])
                    + (A.z[index.i]-A.z[index.j])*(A.z[index.i]-A.z[index.j]));
        db = sqrt((B.x[index.i]-B.x[index.j])*(B.x[index.i]-B.x[index.j])
                    + (B.y[index.i]-B.y[index.j])*(B.y[index.i]-B.y[index.j])
                    + (B.z[index.i]-B.z[index.j])*(B.z[index.i]-B.z[index.j]));

        return (da-db) * (da-db);
    }
};

float solveGPU(sGalaxy A, sGalaxy B, int n) {
    const int dim = n;
    const int elems = round(n*(n-1)/2); //upper triangular(without diagonal) number of elements formula
    auto matrixIndexIterator = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        ConvertLinearIndexToTriangularMatrixIndex{dim}
    );


    //for(int i = 0; i < elems; i++){
    //    std::cout << matrixIndexIterator[i] << " ";
    //}
    
    float result = thrust::transform_reduce(
        matrixIndexIterator, 
        matrixIndexIterator + elems, 
        ComputeDelta{A,B}, 
        float(0), 
        thrust::plus<float>{}
    );
    return sqrt(1/((float)n*((float)n-1)) * result);
}