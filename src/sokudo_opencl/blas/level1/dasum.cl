R"(
__kernel void run(
	__global double *a,
	const unsigned long int s,
	const unsigned long int n,
	const unsigned long int m,
	const unsigned long int incx
) {
	unsigned long int tid = get_global_id(0);
	unsigned long int index = tid * s * m * incx;
	if (index >= n) {
		return;
	}
	
	unsigned long int target = index;
	unsigned long int j = s;
	double sum = 0;
	while (index < n && j--) {
		sum += a[index];
		index += m * incx;
	}
	
	a[target] = sum;
}

)"