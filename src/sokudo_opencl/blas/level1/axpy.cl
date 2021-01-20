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
	float a = *alpha;
	unsigned long int j = s;
	while (index_x < n && index_y < n && j--) {
		y[index_y] = a * x[index_x] + y[index_y];
		index_x += incx;
		index_y += incy;
	}
}

)"