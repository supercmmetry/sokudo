//R"(
__kernel void run(__global long *a, __global long *b, __global long *c, __global ulong n) {
	ulong index = get_global_id(0);
	if (index < n) {
		c[index] = a[index] + b[index];
	}
}

//)"