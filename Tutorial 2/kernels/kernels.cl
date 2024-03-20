
//a simple OpenCL kernel which copies all pixels from A to B
kernel void identity(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	B[id] = A[id];
}

kernel void calculate_histogram(global const uchar* A, global uint* histogram) {
	// Initialize histogram bins to zero
	local uint local_histogram[256];
	for (int i = 0; i < 256; ++i) {
		local_histogram[i] = 0;
	}

	// Calculate histogram in local memory
	int id = get_global_id(0);
	uchar intensity = A[id];
	atomic_inc(&local_histogram[intensity]);

	// Accumulate local histograms into global histogram
	if (get_local_id(0) == 0) {
		for (int i = 0; i < 256; ++i) {
			atomic_add(&histogram[i], local_histogram[i]);
		}
	}
}

kernel void hist_simple(global const int* A, global int* H, private int nr_bins) {
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index
	if (bin_index >= nr_bins) {
		bin_index = nr_bins - 1; // Put numbers greater than or equal to nr_bins in the last bin
	}
	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}

kernel void hist_with_print(global const unsigned char* image, global int* histogram, const int nr_bins) {
	int id = get_global_id(0);
	const int image_size = get_global_size(0); // Get total number of pixels in the image
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