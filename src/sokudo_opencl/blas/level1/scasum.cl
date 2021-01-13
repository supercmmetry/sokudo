R"(
	
float abs(float x) {
	return x >= 0 ? x : -x;
}

__kernel void run(
	__global float2 *a,
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
	float sum = 0;
	while (index < n && j--) {
		float2 q = a[index];
		sum += abs(q.x) + abs(q.y);
		a[index].x = 0;
		a[index].y = 0;
		index += m * incx;
	}
	
	a[target].x = sum;
}

)"