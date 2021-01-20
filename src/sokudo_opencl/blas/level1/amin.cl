R"(
__kernel void samin(
	__global float *a,
	const unsigned long int s,
	const unsigned long int n,
	const unsigned long int m,
	const unsigned long int incx,
	__global unsigned long int *b
) {
	unsigned long int tid = get_global_id(0);
	unsigned long int index = tid * s * m * incx;
	unsigned long int ulim = n * incx;
	
	if (index >= ulim) {
		return;
	}
	
	unsigned long int dtx = m / s;
	unsigned long int itx = tid * m;
	unsigned long int di = m * incx;
	
	unsigned long int target = index;
	unsigned long int j = s;
	float min = FLT_MAX;
	unsigned long int i = 0;
	unsigned long int tx = itx;
	
	if (m == 1) {
		while (index < ulim && j--) {
			float x = a[index];
			if (x < min) {
				min = x;
				i = index; 
			}
			index += di;
		}
		b[tid] = i + 1;
	} else {
		i = b[itx];
		while (index < ulim && j--) {
			float x = a[index];
			if (x < min) {
				min = x;
				i = b[tx];
			}
			index += di;
			tx += dtx;
		}
		
		b[itx] = i;
	}
	
	a[target] = min;
}


__kernel void damin(
	__global double *a,
	const unsigned long int s,
	const unsigned long int n,
	const unsigned long int m,
	const unsigned long int incx,
	__global unsigned long int *b
) {
	unsigned long int tid = get_global_id(0);
	unsigned long int index = tid * s * m * incx;
	unsigned long int ulim = n * incx;
	
	if (index >= ulim) {
		return;
	}
	
	unsigned long int dtx = m / s;
	unsigned long int itx = tid * m;
	unsigned long int di = m * incx;
	
	unsigned long int target = index;
	unsigned long int j = s;
	double min = DBL_MAX;
	unsigned long int i = 0;
	unsigned long int tx = itx;
	
	if (m == 1) {
		while (index < ulim && j--) {
			double x = a[index];
			if (x < min) {
				min = x;
				i = index; 
			}
			index += di;
		}
		b[tid] = i + 1;
	} else {
		i = b[itx];
		while (index < ulim && j--) {
			double x = a[index];
			if (x < min) {
				min = x;
				i = b[tx];
			}
			index += di;
			tx += dtx;
		}
		
		b[itx] = i;
	}
	
	a[target] = min;
}


float absf(float x) {
	return x >= 0 ? x : -x;
}

__kernel void scamin(
	__global float2 *a,
	const unsigned long int s,
	const unsigned long int n,
	const unsigned long int m,
	const unsigned long int incx,
	__global unsigned long int *b
) {
	unsigned long int tid = get_global_id(0);
	unsigned long int index = tid * s * m * incx;
	unsigned long int ulim = n * incx;
	
	if (index >= ulim) {
		return;
	}
	
	unsigned long int dtx = m / s;
	unsigned long int itx = tid * m;
	unsigned long int di = m * incx;
	
	unsigned long int target = index;
	unsigned long int j = s;
	float min = FLT_MAX;
	unsigned long int i = 0;
	unsigned long int tx = itx;
	
	if (m == 1) {
		while (index < ulim && j--) {
			float2 q = a[index];
			float x = absf(q.x) + absf(q.y);
			a[index].x = 0;
			a[index].y = 0;
			if (x < min) {
				min = x;
				i = index; 
			}
			index += di;
		}
		b[tid] = i + 1;
	} else {
		i = b[itx];
		while (index < ulim && j--) {
			float2 q = a[index];
			float x = absf(q.x) + absf(q.y);
			a[index].x = 0;
			a[index].y = 0;
			if (x < min) {
				min = x;
				i = b[tx];
			}
			index += di;
			tx += dtx;
		}
		
		b[itx] = i;
	}
	
	a[target].x = min;
}

double absd(double x) {
	return x >= 0 ? x : -x;
}

__kernel void dcamin(
	__global double2 *a,
	const unsigned long int s,
	const unsigned long int n,
	const unsigned long int m,
	const unsigned long int incx,
	__global unsigned long int *b
) {
	unsigned long int tid = get_global_id(0);
	unsigned long int index = tid * s * m * incx;
	unsigned long int ulim = n * incx;
	
	if (index >= ulim) {
		return;
	}
	
	unsigned long int dtx = m / s;
	unsigned long int itx = tid * m;
	unsigned long int di = m * incx;
	
	unsigned long int target = index;
	unsigned long int j = s;
	double min = DBL_MAX;
	unsigned long int i = 0;
	unsigned long int tx = itx;
	
	if (m == 1) {
		while (index < ulim && j--) {
			double2 q = a[index];
			double x = absd(q.x) + absd(q.y);
			a[index].x = 0;
			a[index].y = 0;
			if (x < min) {
				min = x;
				i = index; 
			}
			index += di;
		}
		b[tid] = i + 1;
	} else {
		i = b[itx];
		while (index < ulim && j--) {
			double2 q = a[index];
			double x = absd(q.x) + absd(q.y);
			a[index].x = 0;
			a[index].y = 0;
			if (x < min) {
				min = x;
				i = b[tx];
			}
			index += di;
			tx += dtx;
		}
		
		b[itx] = i;
	}
	
	a[target].x = min;
}
)"