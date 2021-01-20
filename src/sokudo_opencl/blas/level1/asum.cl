R"(
__kernel void sasum(
	__global float *a,
	const unsigned long int s,
	const unsigned long int n,
	const unsigned long int m,
	const unsigned long int incx
) {
	unsigned long int tid = get_global_id(0);
	unsigned long int index = tid * s * m * incx;
	unsigned long int ulim = n * incx;
	
	if (index >= ulim) {
		return;
	}
	
	unsigned long int di = m * incx;
	unsigned long int target = index;
	unsigned long int j = s;
	float sum = 0;
	while (index < ulim && j--) {
		sum += a[index];
		index += di;
	}
	
	a[target] = sum;
}


__kernel void dasum(
	__global double *a,
	const unsigned long int s,
	const unsigned long int n,
	const unsigned long int m,
	const unsigned long int incx
) {
	unsigned long int tid = get_global_id(0);
	unsigned long int index = tid * s * m * incx;
	unsigned long int ulim = n * incx;
	
	if (index >= ulim) {
		return;
	}
	
	unsigned long int di = m * incx;
	unsigned long int target = index;
	unsigned long int j = s;
	double sum = 0;
	while (index < ulim && j--) {
		sum += a[index];
		index += di;
	}
	
	a[target] = sum;
}


float absf(float x) {
	return x >= 0 ? x : -x;
}

__kernel void scasum(
	__global float2 *a,
	const unsigned long int s,
	const unsigned long int n,
	const unsigned long int m,
	const unsigned long int incx
) {
	unsigned long int tid = get_global_id(0);
	unsigned long int index = tid * s * m * incx;
	unsigned long int ulim = n * incx;
	if (index >= ulim) {
		return;
	}
	
	unsigned long int di = m * incx;
	unsigned long int target = index;
	unsigned long int j = s;
	float sum = 0;
	while (index < ulim && j--) {
		float2 q = a[index];
		sum += absf(q.x) + absf(q.y);
		a[index].x = 0;
		a[index].y = 0;
		index += di;
	}
	
	a[target].x = sum;
}

double absd(double x) {
	return x >= 0 ? x : -x;
}

__kernel void dcasum(
	__global double2 *a,
	const unsigned long int s,
	const unsigned long int n,
	const unsigned long int m,
	const unsigned long int incx
) {
	unsigned long int tid = get_global_id(0);
	unsigned long int index = tid * s * m * incx;
	unsigned long int ulim = n * incx;
	
	if (index >= ulim) {
		return;
	}
	
	unsigned long int di = m * incx;
	unsigned long int target = index;
	unsigned long int j = s;
	double sum = 0;
	while (index < ulim && j--) {
		double2 q = a[index];
		sum += absd(q.x) + absd(q.y);
		a[index].x = 0;
		a[index].y = 0;
		index += di;
	}
	
	a[target].x = sum;
}
)"