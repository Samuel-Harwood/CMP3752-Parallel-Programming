kernel void cumulative_histogram(global const unsigned char* image, global int* cumulative_histogram,const int nr_bins) {
	const int id = get_global_id(0);
	const int local_id = get_local_id(0);
	const int group_id = get_group_id(0);
	const int image_size = get_global_size(0);
	const int global_id = get_global_id(0);

	// Local memory for storing the local scan result (YOU WILL HAVE TO CHANGE THIS TO USE 16BIT OR CHANGE BIN VALUES)	
	//8192 is max for 16bit
	local int local_histogram[256]; //for some reason putting nr_bins here gives an error!	

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


kernel void sixteen_bit_cumulative_histogram(global const unsigned char* image, global int* cumulative_histogram, const int nr_bins) {
	const int id = get_global_id(0);
	const int image_size = get_global_size(0);

	// Increment the corresponding bin in the cumulative histogram
	atomic_inc(&cumulative_histogram[image[id]]); 
	//Couldn't solve for how to use local memory on such a large image
	// Synchronise
	barrier(CLK_GLOBAL_MEM_FENCE);

	// Compute cumulative
	if (id == 0) {
		for (int i = 1; i < nr_bins; ++i) {
			cumulative_histogram[i] += cumulative_histogram[i - 1];
		}
	}
}


//For this kernel, i would recommend only using 256 bins. Any number of bns will run but give weird results.
kernel void cumulative_histogram_colour(global const unsigned char* image, global int* cumulative_histogram_r, global int* cumulative_histogram_g, global int* cumulative_histogram_b, const int nr_bins) {
    const int id = get_global_id(0);
    const int image_size = get_global_size(0);

    // Initialise local histograms
    int local_histogram_r = 0;
    int local_histogram_g = 0;
    int local_histogram_b = 0;

    if (id < image_size) {
        unsigned char pixel_value_r = image[id * 3];     // Red 
        unsigned char pixel_value_g = image[id * 3 + 1]; // Green 
        unsigned char pixel_value_b = image[id * 3 + 2]; // Blue

        //value of 1 is added to all bins with a value < nr_bins, showing presence of a pixel
        //this is the code which gives weird results if nr_bins is not 256
        if (pixel_value_r < nr_bins) {
            local_histogram_r = 1;
        }
        if (pixel_value_g < nr_bins) {
            local_histogram_g = 1;
        }
        if (pixel_value_b < nr_bins) {
            local_histogram_b = 1;
        }
    }

    // Synchronise
    barrier(CLK_GLOBAL_MEM_FENCE);

    //local to cumulative
    if (id < nr_bins) {
        for (int i = 0; i <= id; ++i) {
            atomic_add(&cumulative_histogram_r[id], local_histogram_r); //add local histograms together 
            atomic_add(&cumulative_histogram_g[id], local_histogram_g);
            atomic_add(&cumulative_histogram_b[id], local_histogram_b);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

    }

}



//so this kernel works with 8-bit monochrome. Im pretty sure it does the exact same thing as the one above, just slightly slower and inefficient synchronisation.
kernel void atomic_cumulative_histogram(global const unsigned char* image, global int* cumulative_histogram, const int nr_bins) {
	const int global_id = get_global_id(0);
	const int local_id = get_local_id(0);

	local uint local_hist[256]; //8bit size
	local_hist[local_id] = 0;  // Initialize local histogram counts to zero

	atomic_inc(&local_hist[image[global_id]]); //increment count 
	barrier(CLK_LOCAL_MEM_FENCE); //synchronisation


	uint sum = 0; //Compute the cumulative sum
	for (int i = 0; i <= local_id; ++i) {
		sum += local_hist[i];
	}
	atomic_add(&cumulative_histogram[local_id], sum); //add the sum to corresponding bin
	barrier(CLK_GLOBAL_MEM_FENCE);
}





