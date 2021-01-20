R"(
__kernel void saxpy(
	__global float *alpha,
	__global float *x,
	const unsigned long int incx,
	__global float *y,
	const unsigned long int incy,
	const unsigned long int n,
	const unsigned long int s
) {
	unsigned long int tid = get_global_id(0);
	unsigned long int index_x = tid * incx * s;
	unsigned long int index_y = tid * incy * s;
	unsigned long int ulim_x = n * incx;
	unsigned long int ulim_y = n * incy;
	
	if (index_x >= ulim_x || index_y >= ulim_y) {
		return;
	}
	
	float a = *alpha;
	unsigned long int j = s;
	while (index_x < ulim_x && index_y < ulim_y && j--) {
		y[index_y] = a * x[index_x] + y[index_y];
		index_x += incx;
		index_y += incy;
	}
}


__kernel void daxpy(
	__global double *alpha,
	__global double *x,
	const unsigned long int incx,
	__global double *y,
	const unsigned long int incy,
	const unsigned long int n,
	const unsigned long int s
) {
	unsigned long int tid = get_global_id(0);
	unsigned long int index_x = tid * incx * s;
	unsigned long int index_y = tid * incy * s;
	unsigned long int ulim_x = n * incx;
	unsigned long int ulim_y = n * incy;
	
	if (index_x >= ulim_x || index_y >= ulim_y) {
		return;
	}
	
	double a = *alpha;
	unsigned long int j = s;
	while (index_x < ulim_x && index_y < ulim_y && j--) {
		y[index_y] = a * x[index_x] + y[index_y];
		index_x += incx;
		index_y += incy;
	}
}

float2 scmul(float2 z1, float2 z2) {
	float2 z;
	z.x = z1.x * z2.x - z1.y * z2.y;
	z.y = z1.x * z2.y + z1.y * z2.x;
	return z;
}

float2 scadd(float2 z1, float2 z2) {
	float2 z;
	z.x = z1.x + z2.x;
	z.y = z1.y + z2.y;
	return z;
}

__kernel void scaxpy(
	__global float2 *alpha,
	__global float2 *x,
	const unsigned long int incx,
	__global float2 *y,
	const unsigned long int incy,
	const unsigned long int n,
	const unsigned long int s
) {
	unsigned long int tid = get_global_id(0);
	unsigned long int index_x = tid * incx * s;
	unsigned long int index_y = tid * incy * s;
	unsigned long int ulim_x = n * incx;
	unsigned long int ulim_y = n * incy;
	
	if (index_x >= ulim_x || index_y >= ulim_y) {
		return;
	}
	
	float2 a = *alpha;
	unsigned long int j = s;
	while (index_x < ulim_x && index_y < ulim_y && j--) {
		y[index_y] = scadd(scmul(a, x[index_x]), y[index_y]);
		index_x += incx;
		index_y += incy;
	}
}

double2 dcmul(double2 z1, double2 z2) {
	double2 z;
	z.x = z1.x * z2.x - z1.y * z2.y;
	z.y = z1.x * z2.y + z1.y * z2.x;
	return z;
}

double2 dcadd(double2 z1, double2 z2) {
	double2 z;
	z.x = z1.x + z2.x;
	z.y = z1.y + z2.y;
	return z;
}

__kernel void dcaxpy(
	__global double2 *alpha,
	__global double2 *x,
	const unsigned long int incx,
	__global double2 *y,
	const unsigned long int incy,
	const unsigned long int n,
	const unsigned long int s
) {
	unsigned long int tid = get_global_id(0);
	unsigned long int index_x = tid * incx * s;
	unsigned long int index_y = tid * incy * s;
	unsigned long int ulim_x = n * incx;
	unsigned long int ulim_y = n * incy;
	
	if (index_x >= ulim_x || index_y >= ulim_y) {
		return;
	}
	
	double2 a = *alpha;
	unsigned long int j = s;
	while (index_x < ulim_x && index_y < ulim_y && j--) {
		y[index_y] = dcadd(dcmul(a, x[index_x]), y[index_y]);
		index_x += incx;
		index_y += incy;
	}
}
)"