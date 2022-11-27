#include <stdio.h>
float solveCPU2(sGalaxy A, sGalaxy B, int n, int blocksize){
    float diff = 0.0f;
    float *tmpDiffs = (float*)malloc((n - 1) * sizeof(float));
    if (tmpDiffs == NULL){
        fprintf(stderr, "Malloc has failed!\n");
    }
    for(int i = 0; i < n - 1; i++){
        tmpDiffs[i] = 0.0f;
    }
    
    for (int tj = 0; tj < n / blocksize; tj++)
    {
        //printf("%d\n",tj);
        for (int lj = 0; lj < blocksize; lj++)
        {
            int j = (tj * blocksize) + lj;
            if (j >= n) break;
            
            for (int i = 0; i < j; i++)
            {
                //printf("%d \n",j);
                float da = sqrt((A.x[i]-A.x[j])*(A.x[i]-A.x[j])
                    + (A.y[i]-A.y[j])*(A.y[i]-A.y[j])
                    + (A.z[i]-A.z[j])*(A.z[i]-A.z[j]));
                float db = sqrt((B.x[i]-B.x[j])*(B.x[i]-B.x[j])
                    + (B.y[i]-B.y[j])*(B.y[i]-B.y[j])
                    + (B.z[i]-B.z[j])*(B.z[i]-B.z[j]));
                tmpDiffs[i] += (da-db)*(da-db);
            }
        }
    }
    
    for (int i = 0; i < n-1; i++)
    {
        diff += tmpDiffs[i];
    }
    

    free(tmpDiffs);
    return sqrt(1/((float)n*((float)n-1)) * diff);
}