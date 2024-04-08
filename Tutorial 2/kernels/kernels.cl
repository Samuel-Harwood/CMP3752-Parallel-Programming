kernel void cumulative_histogram(global const unsigned char* image, global int* cumulative_histogram,const int nr_bins) {
	const int id = get_global_id(0);
	const int local_id = get_local_id(0);
	const int group_id = get_group_id(0);
	const int image_size = get_global_size(0);
	const int global_id = get_global_id(0);

	// Local memory for storing the local scan result (YOU WILL HAVE TO CHANGE THIS TO USE 16BIT OR CHANGE BIN VALUES)	
	//8192 is max for 16bit
	local int local_histogram[256]; //for some reason putting nr_bins here gives an error!	

	//An attempt at dynamically allocating local memory
	//This assignment is the embodiment of trying everything and nothing working
	//__local int* local_histogram;
	//if (local_id == 0) {
	//	local_histogram = (__local int*)malloc(sizeof(int) * nr_bins); 
	//}
	
	
	if (image[id] < nr_bins)
		local_histogram[local_id] = 1;
	else
		local_histogram[local_id] = 0; //initalise with zeros

	barrier(CLK_LOCAL_MEM_FENCE);

	//An attempt to replace atomic_inc. It runs but just gives a slightly darker image.
	// I thought it was working correctly lol
	//Your are welcome to check if it runs or not (it should for 8bit)
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


//kernel void cumulative_histogram(global const unsigned char* image, global int* cumulative_histogram, const int nr_bins) {
//	const int id = get_global_id(0);
//
//	// Initialize local histogram for this work-item
//	int local_histogram = 0;
//
//	// Process the pixel at index 'id'
//	if (id < get_global_size(0)) {
//		unsigned char pixel_value = image[id];
//		if (pixel_value < nr_bins) {
//			local_histogram = 1;
//		}
//	}
//
//	// Synchronize all work-items to ensure all pixels are processed
//	barrier(CLK_GLOBAL_MEM_FENCE);
//
//	// Accumulate local histograms into the cumulative histogram
//	if (id < nr_bins) {
//		for (int i = 0; i <= id; ++i) {
//			atomic_add(&cumulative_histogram[id], local_histogram);
//		}
//	}
//}






//so this kernel works with 8-bit monochrome. Im pretty sure it does the exact same thing as the one above, just slightly slower and inefficient synchronisation.
//kernel void cumulative_histogram(global const uchar* image, global int* cumulative_histogram, const int nr_bins) {
//	const int global_id = get_global_id(0);
//	const int local_id = get_local_id(0);
//
//	local uint local_hist[1024]; //8bit 
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









