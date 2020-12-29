R"(
__kernel void run(__global long int *a, __global long int *b, __global long int *c, const unsigned long int n) {
	unsigned long int index = get_global_id(0);
	if (index < n) {
		c[index] = a[index] + b[index];
	}
}

)"