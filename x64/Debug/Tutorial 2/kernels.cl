kernel void cumulative_histogram(global const unsigned char* image, global int* cumulative_histogram,const int nr_bins) {
	const int id = get_global_id(0);
	const int local_id = get_local_id(0);
	const int group_id = get_group_id(0);
	const int image_size = get_global_size(0);
	const int global_id = get_global_id(0);
	// Local memory for storing the local scan result (YOU WILL HAVE TO CHANGE THIS TO USE 16BIT)	
	local int local_histogram[256]; //for some reason putting nr_bins here gives an error!	//8192 and nr_bins = 65536, local work size 1024
	if (image[id] < nr_bins)
		local_histogram[local_id] = 1;
	else
		local_histogram[local_id] = 0; //initalise with zeros

	barrier(CLK_LOCAL_MEM_FENCE);

	//An attempt to not use atomic_inc. It runs but just gives a slightly lighter image. I thought it was cool idk
	//for (int stride = 1; stride < get_local_size(0); stride *= 2) { 
	//	int index = local_id + stride;
	//	if (index < nr_bins) {
	//		local_histogram[index] += local_histogram[index - stride];
	//	}
	//	barrier(CLK_LOCAL_MEM_FENCE);
	//}
	atomic_inc(&local_histogram[image[global_id]]); //increment count 
	barrier(CLK_LOCAL_MEM_FENCE); //synchronisation

	// partial to global
	if (id < nr_bins) {
		cumulative_histogram[id] = local_histogram[local_id];
	}
	barrier(CLK_GLOBAL_MEM_FENCE); //kinda neccessary to avoid race conditions

	////Blelloch up sweep
	////Wow what a clever use of this algorithm
	////start at 1, double with each iteration
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


//so this kernel works with 8-bit monochrome. Im pretty sure it does the exact same thing as the one above, just slightly slower.
//Would've been nice to see what the final output should look like.
//kernel void cumulative_histogram(global const uchar* image, global int* cumulative_histogram, const int nr_bins) {
//	const int global_id = get_global_id(0);
//	const int local_id = get_local_id(0);
//
//	local uint local_hist[256]; //8bit 
//	local_hist[local_id] = 0;  // Initialize local histogram counts to zero
//    
//	atomic_inc(&local_hist[image[global_id]]); //increment count 
//	barrier(CLK_LOCAL_MEM_FENCE); //synchronisation
//
//
//	uint sum = 0; //Compute the cumulative sum
//	for (int i = 0; i <= local_id; ++i) {
//		sum += local_hist[i];
//	}
//	atomic_add(&cumulative_histogram[local_id], sum); //add the sum to corresponding bin
//	barrier(CLK_GLOBAL_MEM_FENCE);
//}


kernel void colour_cumulative_histogram(global const uchar* image, global int* cumulative_histogram, const int nr_bins) {
	const int id = get_global_id(0);
	const int local_id = get_local_id(0);
	const int group_id = get_group_id(0);
	const int image_size = get_global_size(0);
	const int channels = get_global_size(2); // Number of color channels

	// Local memory for storing the local scan result
	local int local_histogram[256]; // Assuming 8-bit color

	// Initialize local histogram
	for (int i = 0; i < 256; ++i) {
		local_histogram[i] = 0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// Compute histogram for each color channel separately
	for (int c = 0; c < channels; ++c) {
		for (int i = local_id; i < nr_bins; i += get_local_size(0)) {
			local_histogram[i] = 0; // Reset local histogram
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		// Compute local histogram
		for (int i = id; i < image_size; i += get_global_size(0)) {
			uchar pixel = image[i + c * image_size]; // Access pixel intensity for current channel
			if (pixel < nr_bins) {
				atomic_add(&local_histogram[pixel], 1);
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		// Parallel reduction
		for (int stride = 1; stride < get_local_size(0); stride *= 2) {
			int index = local_id + stride;
			if (index < nr_bins) {
				local_histogram[index] += local_histogram[index - stride];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		// Partial to global
		for (int i = local_id; i < nr_bins; i += get_local_size(0)) {
			cumulative_histogram[i + c * nr_bins] = local_histogram[i];
		}

		barrier(CLK_GLOBAL_MEM_FENCE);

		// Blelloch up sweep
		for (int stride = 1; stride * 2 <= image_size; stride *= 2) {
			int index = id - stride;
			if (index >= 0 && index < nr_bins) {
				cumulative_histogram[id + c * nr_bins] += cumulative_histogram[index + c * nr_bins];
			}
			barrier(CLK_GLOBAL_MEM_FENCE);
		}
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
