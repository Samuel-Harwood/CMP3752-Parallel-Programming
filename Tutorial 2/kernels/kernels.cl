

kernel void cumulative_histogram(global const unsigned char* image, global int* cumulative_histogram, const int nr_bins) {
	const int id = get_global_id(0);
	const int local_id = get_local_id(0);
	const int group_id = get_group_id(0);
	const int image_size = get_global_size(0);

	// Local memory for storing the local scan result (YOU WILL HAVE TO CHANGE THIS TO USE 16BIT)	
	local int local_histogram[256]; //for some reason putting nr_bins here gives an error!											//8192 and nr_bins = 65536, local work size 1024

	if (image[id] < nr_bins)
		local_histogram[local_id] = 1;
	else
		local_histogram[local_id] = 0; //initalise with zeros

	barrier(CLK_LOCAL_MEM_FENCE);

	//parallel reduction
	for (int stride = 1; stride < get_local_size(0); stride *= 2) { 
		int index = local_id + stride;
		if (index < nr_bins) {
			local_histogram[index] += local_histogram[index - stride];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}


	// partial to global
	if (id < nr_bins) {
		cumulative_histogram[id] = local_histogram[local_id];
	}
	barrier(CLK_GLOBAL_MEM_FENCE); //kinda neccessary to avoid race conditions

	//Blelloch up sweep
	//Wow what a clever use of this algorithm
	//start at 1, double with each iteration
	for (int stride = 1; stride * 2 <= image_size; stride *= 2) { 
		//index = global id - size of stride
		int index = id - stride;
		if (index >= 0 && index < nr_bins) { 
			//replacing atomic add
			cumulative_histogram[id] += cumulative_histogram[index]; 
		}
		barrier(CLK_GLOBAL_MEM_FENCE); //Neccesary synchronisation
	}
}




//for reference
kernel void scan_bl(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	int t;
	// Up-sweep
	for (int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride * 2)) == 0)
			B[id] += B[id - stride];
		barrier(CLK_GLOBAL_MEM_FENCE); // Sync the step
	}
	// Down-sweep
	if (id == 0) B[N - 1] = 0; // Exclusive scan
	barrier(CLK_GLOBAL_MEM_FENCE); // Sync the step
	for (int stride = N / 2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride * 2)) == 0) {
			t = B[id];
			B[id] += B[id - stride]; // Reduce
			B[id - stride] = t; // Move
		}
		barrier(CLK_GLOBAL_MEM_FENCE); // Sync the step
	}


}
