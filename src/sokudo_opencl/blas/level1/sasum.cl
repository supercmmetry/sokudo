R"(
__kernel void run(
	__global float *a,
	const unsigned long int s,
	const unsigned long int n,
	const unsigned long int m
) {
	unsigned long int tid = get_global_id(0);
	unsigned long int index = tid * s * m;
	if (index >= n) {
		return;
	}
	
	unsigned long int target = index;
	unsigned long int j = s;
	unsigned int sum = 0;
	while (index < n && j--) {
		sum += a[index];
		index += m;
	}
	
	a[target] = sum;
}

)"