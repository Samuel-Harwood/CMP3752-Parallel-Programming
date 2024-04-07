

kernel void cumulative_histogram(global const unsigned char* image, global int* cumulative_histogram, const int nr_bins) {
	const int id = get_global_id(0);
	const int local_id = get_local_id(0);
	const int group_id = get_group_id(0);
	const int image_size = get_global_size(0);

	// Local memory for storing the local scan result
	local int local_scan[256];

	// Load input data into local memory and perform reduction
	local_scan[local_id] = (image[id] < nr_bins) ? 1 : 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	// Perform parallel reduction within work group
	for (int stride = 1; stride <= get_local_size(0); stride *= 2) {
		int index = local_id + stride;
		if (index < 256) {
			local_scan[index] += local_scan[index - stride];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Write the local scan result back to global memory
	if (id < nr_bins) {
		cumulative_histogram[id] = local_scan[local_id];
	}
	barrier(CLK_GLOBAL_MEM_FENCE);

	// Perform inclusive scan on the global memory to get the cumulative histogram
	for (int stride = 1; stride * 2 <= get_global_size(0); stride *= 2) {
		int index = id - stride;
		if (index >= 0 && index < nr_bins) {
			cumulative_histogram[id] += cumulative_histogram[index];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}




//kernel void blelloch_upsweep(global const unsigned short* image, global ulong* cumulative_histogram, const int nr_bins) {
//	const int id = get_global_id(0);
//	const int local_id = get_local_id(0);
//	const int group_id = get_group_id(0);
//	const int image_size = get_global_size(0);
//	// Declare private variables for intermediate calculations
//	private int private_histogram[65535]; // Size equal to nr_bins
//
//	// Initialize private histogram to 0
//	if (local_id < nr_bins) {
//		private_histogram[local_id] = 0;
//	}
//
//	barrier(CLK_LOCAL_MEM_FENCE);
//
//	// Calculate partial histogram in private memory
//	for (int i = id; i < image_size; i += image_size) {
//		int bin_index = image[i];
//		if (bin_index >= 0 && bin_index < nr_bins) {
//			private_histogram[bin_index]++;
//		}
//	}
//
//	barrier(CLK_LOCAL_MEM_FENCE);
//
//	// Add partial histogram to cumulative histogram
//	for (int i = local_id; i < nr_bins; i += get_local_size(0)) {
//		cumulative_histogram[i] += private_histogram[i];
//	}
//
//	barrier(CLK_GLOBAL_MEM_FENCE);
//
//	// Blelloch scan up-sweep
//	for (int stride = 1; stride < nr_bins; stride *= 2) {
//		for (int index = id; index < nr_bins; index += image_size) {
//			if (index >= stride) {
//				cumulative_histogram[index] += cumulative_histogram[index - stride];
//			}
//		}
//		barrier(CLK_GLOBAL_MEM_FENCE);
//	}
//}

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
