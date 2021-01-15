R"(
__kernel void samax(
	__global float *a,
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
	float max = 0;
	while (index < n && j--) {
		float x = a[index];
		max = x > max ? x : max;
		index += m * incx;
	}
	
	a[target] = max;
}


__kernel void damax(
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
	double max = 0;
	while (index < n && j--) {
		double x = a[index];
		max = x > max ? x : max;
		index += m * incx;
	}
	
	a[target] = max;
}


float absf(float x) {
	return x >= 0 ? x : -x;
}

__kernel void scamax(
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
	float max = 0;
	while (index < n && j--) {
		float2 q = a[index];
		float x = absf(q.x) + absf(q.y);
		max = x > max ? x : max;
		a[index].x = 0;
		a[index].y = 0;
		index += m * incx;
	}
	
	a[target].x = max;
}

double absd(double x) {
	return x >= 0 ? x : -x;
}

__kernel void dcamax(
	__global double2 *a,
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
	double max = 0;
	while (index < n && j--) {
		double2 q = a[index];
		double x = = absd(q.x) + absd(q.y);
		max = x > max ? x : max;
		a[index].x = 0;
		a[index].y = 0;
		index += m * incx;
	}
	
	a[target].x = max;
}
)"