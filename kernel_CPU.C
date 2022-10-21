float solveCPU(sGalaxy A, sGalaxy B, int n) {
	float diff = 0.0f;
	for (int i = 0; i < n-1; i++) {
		float tmp = 0.0f;
		for (int j = i+1; j < n; j++) {
			float da = sqrt((A.x[i]-A.x[j])*(A.x[i]-A.x[j])
				+ (A.y[i]-A.y[j])*(A.y[i]-A.y[j])
				+ (A.z[i]-A.z[j])*(A.z[i]-A.z[j]));
			float db = sqrt((B.x[i]-B.x[j])*(B.x[i]-B.x[j])
				+ (B.y[i]-B.y[j])*(B.y[i]-B.y[j])
				+ (B.z[i]-B.z[j])*(B.z[i]-B.z[j]));
			//XXX for large galaxies, more precise version of sum should be implemented, not required in this example 
			tmp += (da-db) * (da-db);
			//printf("[%d][%d](%f-%f) * (%f-%f) -- tmp = %f\n", i,j,da,db,da,db,tmp);
		}
		diff += tmp;
	}
	//printf("total diff: %f \n",diff);
	return sqrt(1/((float)n*((float)n-1)) * diff);
}
