
kernel void hist_with_print(global const unsigned char* image, global int* histogram, const int nr_bins) {
	int id = get_global_id(0);
	const int image_size = get_global_size(0); // Get total number of pixels in the image
	if (id < nr_bins) {
		histogram[id] = -1;
	}

	// Calculate histogram
	if (id < image_size) {
		int bin_index = image[id]; // Assuming image contains intensity values
		if (bin_index >= nr_bins) {
			bin_index = nr_bins - 1; // Put numbers greater than or equal to nr_bins in the last bin
		}
		atomic_inc(&histogram[bin_index]);
	}
	// Barrier to ensure all threads finish histogram calculation before printing
	barrier(CLK_GLOBAL_MEM_FENCE);
	if (id == 0) {
		for (int i = 0; i < nr_bins; ++i) {
			printf("Bin %d: %d\n", i, histogram[i]);
		}
	}
}


kernel void blelloch_upsweep(global const unsigned char* image, global int* cumulative_histogram, const int nr_bins) {
	const int id = get_global_id(0);
	const int local_id = get_local_id(0);
	const int group_id = get_group_id(0);
	const int image_size = get_global_size(0);
	// Local memory for partial histogram

	local int local_histogram[256];

	// Initialize local histogram to 0
	if (local_id < nr_bins) {
		local_histogram[local_id] = 0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// Calculate partial 
	for (int i = id; i < image_size; i += image_size) {
		int bin_index = image[i];
		if (bin_index >= 0 && bin_index < nr_bins) {
			atomic_inc(&local_histogram[bin_index]);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// partial to global
	for (int i = local_id; i < nr_bins; i += get_local_size(0)) {
		atomic_add(&cumulative_histogram[i], local_histogram[i]);
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	// Blelloch scan up-sweep
	//wow what a clever use of this algorithm
	for (int stride = 1; stride < nr_bins; stride *= 2) {
		for (int index = id; index < nr_bins; index += image_size) {
			if (index >= stride) {
				cumulative_histogram[index] += cumulative_histogram[index - stride];
			}
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

}



kernel void hillis_steele_scan(global const unsigned char* image, global int* cumulative_histogram, const int nr_bins) {
	const int id = get_global_id(0);
	const int image_size = get_global_size(0);

	// Local memory for partial sums
	local int local_sums[256];

	// Each work-item loads its value into local memory
	local_sums[id] = (id > 0) ? cumulative_histogram[id - 1] : 0;

	// Up-sweep phase
	for (int stride = 1; stride < nr_bins; stride *= 2) {
		barrier(CLK_LOCAL_MEM_FENCE);
		if (id >= stride) {
			local_sums[id] += local_sums[id - stride];
		}
	}

	// Write the last element of each block to global memory
	if (id == image_size - 1) {
		cumulative_histogram[id] = local_sums[id];
	}

	// Down-sweep phase
	for (int stride = nr_bins / 2; stride > 0; stride /= 2) {
		barrier(CLK_LOCAL_MEM_FENCE);
		if (id >= stride) {
			int temp = local_sums[id - stride];
			local_sums[id - stride] = local_sums[id];
			local_sums[id] += temp;
		}
	}

	// Write the final results to global memory
	barrier(CLK_LOCAL_MEM_FENCE);
	cumulative_histogram[id] = local_sums[id];
}



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



kernel void filter_r(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0)/3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue

	//this is just a copy operation, modify to filter out the individual colour channels
	B[id] = A[id];
}

//simple ND identity kernel
kernel void identityND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	B[id] = A[id];
}

//2D averaging filter
kernel void avg_filterND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	uint result = 0;

	//simple boundary handling - just copy the original pixel
	if ((x == 0) || (x == width-1) || (y == 0) || (y == height-1)) {
		result = A[id];	
	} else {
		for (int i = (x-1); i <= (x+1); i++)
		for (int j = (y-1); j <= (y+1); j++) 
			result += A[i + j*width + c*image_size];

		result /= 9;
	}

	B[id] = (uchar)result;
}

//2D 3x3 convolution kernel
kernel void convolutionND(global const uchar* A, global uchar* B, constant float* mask) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	float result = 0;

	//simple boundary handling - just copy the original pixel
	if ((x == 0) || (x == width-1) || (y == 0) || (y == height-1)) {
		result = A[id];	
	} else {
		for (int i = (x-1); i <= (x+1); i++)
		for (int j = (y-1); j <= (y+1); j++) 
			result += A[i + j*width + c*image_size]*mask[i-(x-1) + j-(y-1)];
	}

	B[id] = (uchar)result;
}