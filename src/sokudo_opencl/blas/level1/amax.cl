R"(
__kernel void samax(
	__global float *a,
	const unsigned long int s,
	const unsigned long int n,
	const unsigned long int m,
	const unsigned long int incx,
	__global unsigned long int *b
) {
	unsigned long int tid = get_global_id(0);
	unsigned long int index = tid * s * m * incx;
	unsigned long int dtx = m / s;
	unsigned long int itx = tid * m;
	unsigned long int di = m * incx;
	
	
	if (index >= n) {
		return;
	}
	
	unsigned long int target = index;
	unsigned long int j = s;
	float max = FLT_MIN;
	unsigned long int i = 0;
	unsigned long int tx = itx;
	
	if (m == 1) {
		while (index < n && j--) {
			float x = a[index];
			if (x > max) {
				max = x;
				i = index; 
			}
			index += di;
		}
		b[tid] = i + 1;
	} else {
		i = b[itx];
		while (index < n && j--) {
			float x = a[index];
			if (x > max) {
				max = x;
				i = b[tx];
			}
			index += di;
			tx += dtx;
		}
		
		b[itx] = i;
	}
	
	a[target] = max;
}


__kernel void damax(
	__global double *a,
	const unsigned long int s,
	const unsigned long int n,
	const unsigned long int m,
	const unsigned long int incx,
	__global unsigned long int *b
) {
	unsigned long int tid = get_global_id(0);
	unsigned long int index = tid * s * m * incx;
	unsigned long int dtx = m / s;
	unsigned long int itx = tid * m;
	unsigned long int di = m * incx;
	
	
	if (index >= n) {
		return;
	}
	
	unsigned long int target = index;
	unsigned long int j = s;
	double max = DBL_MIN;
	unsigned long int i = 0;
	unsigned long int tx = itx;
	
	if (m == 1) {
		while (index < n && j--) {
			double x = a[index];
			if (x > max) {
				max = x;
				i = index; 
			}
			index += di;
		}
		b[tid] = i + 1;
	} else {
		i = b[itx];
		while (index < n && j--) {
			double x = a[index];
			if (x > max) {
				max = x;
				i = b[tx];
			}
			index += di;
			tx += dtx;
		}
		
		b[itx] = i;
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
	const unsigned long int incx,
	__global unsigned long int *b
) {
	unsigned long int tid = get_global_id(0);
	unsigned long int index = tid * s * m * incx;
	unsigned long int dtx = m / s;
	unsigned long int itx = tid * m;
	unsigned long int di = m * incx;
	
	
	if (index >= n) {
		return;
	}
	
	unsigned long int target = index;
	unsigned long int j = s;
	float max = FLT_MIN;
	unsigned long int i = 0;
	unsigned long int tx = itx;
	
	if (m == 1) {
		while (index < n && j--) {
			float2 q = a[index];
			float x = absf(q.x) + absf(q.y);
			a[index].x = 0;
			a[index].y = 0;
			if (x > max) {
				max = x;
				i = index; 
			}
			index += di;
		}
		b[tid] = i + 1;
	} else {
		i = b[itx];
		while (index < n && j--) {
			float2 q = a[index];
			float x = absf(q.x) + absf(q.y);
			a[index].x = 0;
			a[index].y = 0;
			if (x > max) {
				max = x;
				i = b[tx];
			}
			index += di;
			tx += dtx;
		}
		
		b[itx] = i;
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
	const unsigned long int incx,
	__global unsigned long int *b
) {
	unsigned long int tid = get_global_id(0);
	unsigned long int index = tid * s * m * incx;
	unsigned long int dtx = m / s;
	unsigned long int itx = tid * m;
	unsigned long int di = m * incx;
	
	
	if (index >= n) {
		return;
	}
	
	unsigned long int target = index;
	unsigned long int j = s;
	double max = DBL_MIN;
	unsigned long int i = 0;
	unsigned long int tx = itx;
	
	if (m == 1) {
		while (index < n && j--) {
			double2 q = a[index];
			double x = absd(q.x) + absd(q.y);
			a[index].x = 0;
			a[index].y = 0;
			if (x > max) {
				max = x;
				i = index; 
			}
			index += di;
		}
		b[tid] = i + 1;
	} else {
		i = b[itx];
		while (index < n && j--) {
			double2 q = a[index];
			double x = absd(q.x) + absd(q.y);
			a[index].x = 0;
			a[index].y = 0;
			if (x > max) {
				max = x;
				i = b[tx];
			}
			index += di;
			tx += dtx;
		}
		
		b[itx] = i;
	}
	
	a[target].x = max;
}
)"