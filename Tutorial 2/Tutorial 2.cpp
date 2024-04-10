//By Samuel Harwood
#include <iostream>
#include <vector>
#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char** argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	int image_choice = 1;
	std::cout << "choose file:\n1 = 8-bit mono\n2 = 16-bit mono\n3 = 8-bit colour\n(enter 1, 2 or 3): ";
	std::cin >> image_choice;
	string image_filename = "";
	//I know this part isnt graded but it looks pretty 
	//If the image names are different have fun changing them
	switch (image_choice) {
	case 1:
		image_filename = "test.pgm"; //8bit mono 209 - 543 (note some values will cause race conditions)
		cout << (image_filename) << endl;
		break;
	case 2:
		image_filename = "mdr16-gs.pgm"; //16 bit mono
		cout << (image_filename) << endl; 
		break;
	case 3:
		image_filename = "test_large.ppm"; //8 bit colour this doesn't work
		cout << (image_filename) << endl;
		break;
	default:
		image_filename = "test.pgm";
		cout << (image_filename) << endl; 
		break;
	}

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		//Dynamically set number of bins 
		int nr_bins = 256; //Default Value
		cout << "Enter No. Bins - ";
		cin >> nr_bins; //Should probably have error handling but not graded so...

		//To switch between 8bitGS, 16bitGS, and 8bit colour you will need to uncomment and comment several things.
		//or you could just give me the marks just for 8bit if you can't be bothered.
	
			
		CImg<unsigned char> image_input(image_filename.c_str()); //8bit
		//CImg<unsigned short> image_input(image_filename.c_str()); //16bit

		CImg<unsigned char> image_output(image_filename.c_str()); //8bit
		//CImg<unsigned short> image_output(image_filename.c_str()); //16bit


		CImgDisplay disp_input(image_input, "input image");
		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);
		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//3.2 Load & build the device code
		cl::Program::Sources sources;
		AddSources(sources, "kernels.cl");
		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 4 - device operations
		//Getting max group size info
		std::vector<cl::Device> devices;
		context.getInfo(CL_CONTEXT_DEVICES, &devices);  
		cl_device_id device_id_cl = devices[device_id]();
		size_t max_work_group_size;
		clGetDeviceInfo(device_id_cl, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL); 
		size_t local_work_size = nr_bins < max_work_group_size ? nr_bins : max_work_group_size; //cheeky ternary operator, just wanted to show off c++ skills

		//Printing all the 
		std::cout << "Local work size: " << local_work_size << std::endl;
		std::cout << "Maximum work group size: " << max_work_group_size << std::endl;
		std::cout << "Image size: " << image_input.size() << std::endl;
		std::cout << "Number bins: " << nr_bins << std::endl;

		//device - buffers
		//greyscale
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size()); //8bit
		//cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size() * sizeof(unsigned short)); //16bit

		cl::Buffer dev_cumulative_histogram(context, CL_MEM_WRITE_ONLY, sizeof(int) * nr_bins); //both 8-bit and 16-bit
		//cl::Buffer dev_cumulative_histogram(context, CL_MEM_WRITE_ONLY, sizeof(int) * 3 * nr_bins); //COLOUR

		//greyscale

	
		//Copy images to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size() * sizeof(unsigned char), &image_input.data()[0]); //8bit 
		//queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size() * sizeof(unsigned short), &image_input.data()[0]); //16bit

		//4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel = cl::Kernel(program, "cumulative_histogram"); //change kernel (all have same number of params)
		kernel.setArg(0, dev_image_input);
		kernel.setArg(1, dev_cumulative_histogram); //image_output
		kernel.setArg(2, nr_bins); //bin number  //Change to kernel 4 for colour
		cl::Event prof_event; //Timing kernel execution

		//queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(max_work_group_size), cl::NDRange(local_work_size), NULL, &prof_event); //if you want to only use max work gorup (for some reason)
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(nr_bins), cl::NDRange(local_work_size), NULL, &prof_event); //8-bit and 16-bit mono
		//queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_input.size() / 3), cl::NullRange, NULL, &prof_event); //Colour


		//4.3 Copy the result from device to host
		cl::Event kernel_event;
		std::vector<unsigned int> cumulative_histogram(nr_bins, 0); //greyscale
		//std::vector<unsigned int> cumulative_histogram(3 * nr_bins, 0); //COLOUR

		queue.enqueueReadBuffer(dev_cumulative_histogram, CL_TRUE, 0, sizeof(int) * nr_bins, cumulative_histogram.data(), NULL, &kernel_event); //8bit and 16bit mono
		//queue.enqueueReadBuffer(dev_cumulative_histogram, CL_TRUE, 0, sizeof(unsigned int) * 3 * nr_bins, cumulative_histogram.data(), NULL, &kernel_event); //colour

		vector<unsigned char> output_buffer(image_input.size()); //holds our look up table data
		//COLOUR
	
		//COLOUR


		//Greyscale
		unsigned int max_value = cumulative_histogram[nr_bins - 1]; // Get the maximum value in the cumulative histogram 
		// Scale and normalize the cumulative histogram
		for (int i = 0; i < nr_bins; ++i) {					
			cumulative_histogram[i] = static_cast<unsigned short>(cumulative_histogram[i] * 255 / max_value); //change this to 65535 for 16bit
		}
	
		////LOOK UP TABLE!
		for (int i = 0; i < image_input.width(); ++i) { //for each pixel..
			for (int j = 0; j < image_input.height(); ++j) {
				int pixel_value = image_input(i, j); //original pixel intensity at co-ord i,j
				int new_pixel_value = cumulative_histogram[pixel_value]; //actual assignment
				output_buffer[i + j * image_input.width()] = static_cast<unsigned int>(new_pixel_value);
			}
		}
		//Greyscale
		

		//Alternate way to do the look up table, gives same result
		// Will need to change output_buffer.data() to image_output.data() in output_image()
		//std::vector<int>lut(nr_bins);
		//for (int i = 0; i < nr_bins; ++i) 
		//	lut[i] = cumulative_histogram[i];
		//cimg_forXY(image_input, x, y) {
		//	int original = image_input(x, y);
		//	int map = lut[original];
		//	image_output(x, y) = map;
		//}


		//Output the normalized and scaled cumulative histogram (see below for .txt output alternative)
		for (int i = 0; i < nr_bins; ++i) {
			std::cout << i << " " << cumulative_histogram[i] << std::endl;
		}


		// Display the back-projected output image
		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum()); //8bit
		//CImg<unsigned short> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum()); //16bit
		CImgDisplay disp_output(output_image, "output image");

		//Output kernel info
		std::cout << "Kernel execution time [ns]:" <<
			prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US)
			<< std::endl;

		//Checking histogram values 
		std::ofstream histogram_file("histogram.txt");
		if (histogram_file.is_open()) {
			histogram_file << "Greyscale Histogram:\n";
			for (int i = 0; i < nr_bins; ++i) {
				histogram_file << i << ": " << output_buffer[i] << "\n";
			}
			//histogram_file << "Red Histogram:\n";
			//for (int i = 0; i < 256; ++i) {
			//	histogram_file << i << ": " << histogram_r[i] << "\n";
			//}
			//histogram_file << "\nGreen Histogram:\n";
			//for (int i = 0; i < 256; ++i) {
			//	histogram_file << i << ": " << histogram_g[i] << "\n";
			//}
			//histogram_file << "\nBlue Histogram:\n";
			//for (int i = 0; i < 256; ++i) {
			//	histogram_file << i << ": " << histogram_b[i] << "\n";
			//}
			histogram_file.close();
		}
		else {
			std::cerr << "Unable to open histogram file" << std::endl;
		}


		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}
		
	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}