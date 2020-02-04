#include <iostream>
#include <vector>

#include "Utils.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

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

		//Part 4 - memory allocation
		//host - input
		//std::vector<int> A = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }; //C++11 allows this type of initialisation
		//std::vector<int> B = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 };
		std::vector<int> A(100);
		std::vector<int> B(100);
		for (int i = 0; i < A.size(); i++)
		{
			A[i] = i;
			B[i] = (i + 1) % 3;
		}
		
		size_t vector_elements = A.size();//number of elements
		size_t vector_size = A.size()*sizeof(int);//size in bytes

		//host - output
		std::vector<int> C(vector_elements);

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, vector_size);

		//Part 5 - device operations

		//5.1 Copy arrays A and B to device memory
		cl::Event copyEventA, copyEventB;
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0], nullptr, &copyEventA);
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vector_size, &B[0], nullptr, &copyEventB);

		//5.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel_mult = cl::Kernel(program, "multadd");
		kernel_mult.setArg(0, buffer_A);
		kernel_mult.setArg(1, buffer_B);
		kernel_mult.setArg(2, buffer_C);

		cl::Kernel kernel_add = cl::Kernel(program, "add");
		kernel_add.setArg(0, buffer_C);
		kernel_add.setArg(1, buffer_B);
		kernel_add.setArg(2, buffer_C);

		cl::Event profEventMult, profEventAdd;

		// the "global" parameter, cl::NDRange(vector_elements), 
		// defines the number of kernel launches (how many time the kernel function will be executed, each time with an incremented global ID in range 0 < id < global)
		queue.enqueueNDRangeKernel(kernel_mult, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, nullptr, &profEventMult);

		//queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, nullptr, &profEventAdd);

		//5.3 Copy the result from device to host
		cl::Event readEventC;
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, vector_size, &C[0], nullptr, &readEventC);

		std::cout << "A = " << A << std::endl;
		std::cout << "B = " << B << std::endl;
		std::cout << "C = " << C << std::endl;

		std::cout << "Copying buffer A to device memory took " << copyEventA.getProfilingInfo<CL_PROFILING_COMMAND_END>() - copyEventA.getProfilingInfo<CL_PROFILING_COMMAND_START>() << "ns to complete." << std::endl;
		std::cout << "Copying buffer B to device memory took " << copyEventB.getProfilingInfo<CL_PROFILING_COMMAND_END>() - copyEventB.getProfilingInfo<CL_PROFILING_COMMAND_START>() << "ns to complete." << std::endl;
		std::cout << "Reading buffer C took " << readEventC.getProfilingInfo<CL_PROFILING_COMMAND_END>() - readEventC.getProfilingInfo<CL_PROFILING_COMMAND_START>() << "ns to complete." << std::endl;
		std::cout << "Kernel took " << /*(profEventAdd.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profEventAdd.getProfilingInfo<CL_PROFILING_COMMAND_START>()) +*/ (profEventMult.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profEventMult.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << "ns to complete." << std::endl;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}