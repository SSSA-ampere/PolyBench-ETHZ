/**
 * jacobi1D.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define POLYBENCH_TIME 1

//select the OpenCL device to use (can be GPU, CPU, or Accelerator such as Intel Xeon Phi)
#define OPENCL_DEVICE_SELECTION CL_DEVICE_TYPE_GPU

#include "jacobi1D.h"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define MAX_SOURCE_SIZE (0x100000)

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

char str_temp[1024];

cl_platform_id platform_id;
cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel1;
cl_kernel clKernel2;
cl_command_queue clCommandQue;
cl_program clProgram;
cl_mem a_mem_obj;
cl_mem b_mem_obj;
FILE *fp;
char *source_str;
size_t source_size;

//#define RUN_ON_CPU


void compareResults(DATA_TYPE POLYBENCH_1D(a1,N,n), DATA_TYPE POLYBENCH_1D(a2,N,n), DATA_TYPE POLYBENCH_1D(b1,N,n), 
	DATA_TYPE POLYBENCH_1D(b2,N,n))
{
	int i, j, fail;
	fail = 0;   

	for (i=0; i<N; i++) 
	{
		if (percentDiff(a1[i], a2[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
		{
			fail++;
		}
	}

	for (i=0; i<N; i++) 
	{
		if (percentDiff(b1[i], b2[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
		{
			fail++;
		}
	}

	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void read_cl_file()
{
	// Load the kernel source code into the array source_str
	fp = fopen("jacobi1D.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}

void init_array(DATA_TYPE POLYBENCH_1D(A,N,n), DATA_TYPE POLYBENCH_1D(B,N,n))
{
	int i, j;

	for (i = 0; i < N; i++)
	{
		A[i] = ((DATA_TYPE) 4 * i + 10) / N;
		B[i] = ((DATA_TYPE) 7 * i + 11) / N;
	}
}


void cl_initialization()
{
	
	// Get platform and device information
	errcode = clGetPlatformIDs(1, &platform_id, &num_platforms);
	if(errcode == CL_SUCCESS) printf("number of platforms is %d\n",num_platforms);

	errcode = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("platform name is %s\n",str_temp);

	errcode = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("platform version is %s\n",str_temp);

	errcode = clGetDeviceIDs( platform_id, OPENCL_DEVICE_SELECTION, 1, &device_id, &num_devices);
	if(errcode == CL_SUCCESS) printf("device id is %d\n",device_id);

	errcode = clGetDeviceInfo(device_id,CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("device name is %s\n",str_temp);
	
	// Create an OpenCL context
	clGPUContext = clCreateContext( NULL, 1, &device_id, NULL, NULL, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating context\n");
 
	//Create a command-queue
	clCommandQue = clCreateCommandQueue(clGPUContext, device_id, 0, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating command queue\n");
}


void cl_mem_init(DATA_TYPE POLYBENCH_1D(A,N,n), DATA_TYPE POLYBENCH_1D(B,N,n))
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, N * sizeof(DATA_TYPE), NULL, &errcode);
	b_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, N * sizeof(DATA_TYPE), NULL, &errcode);
		
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, N * sizeof(DATA_TYPE), A, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0, N * sizeof(DATA_TYPE), B, 0, NULL, NULL);
	if(errcode != CL_SUCCESS)printf("Error in writing buffers\n");
}


void cl_load_prog()
{
	// Create a program from the kernel source
	clProgram = clCreateProgramWithSource(clGPUContext, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode);

	if(errcode != CL_SUCCESS) printf("Error in creating program\n");

	// Build the program
	errcode = clBuildProgram(clProgram, 1, &device_id, NULL, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in building program\n");
		
	// Create the OpenCL kernel
	clKernel1 = clCreateKernel(clProgram, "runJacobi1D_kernel1", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");
	clKernel2 = clCreateKernel(clProgram, "runJacobi1D_kernel2", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");
	clFinish(clCommandQue);
}


void cl_launch_kernel1()
{
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = N;
	globalWorkSize[1] = 1;
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clFinish(clCommandQue);
}


void cl_launch_kernel2()
{
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = N;
	globalWorkSize[1] = 1;
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clFinish(clCommandQue);
}


void cl_clean_up()
{
	// Clean up
	errcode = clFlush(clCommandQue);
	errcode = clFinish(clCommandQue);
	errcode = clReleaseKernel(clKernel1);
	errcode = clReleaseKernel(clKernel2);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(a_mem_obj);
	errcode = clReleaseMemObject(b_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


void jacobi1D(DATA_TYPE POLYBENCH_1D(A,N,n), DATA_TYPE POLYBENCH_1D(B,N,n))
{
	int t, i, j;

	for (t = 0; t < TSTEPS; t++)
	{
		for (i = 2; i < N - 1; i++)
			B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);

		for (j = 2; j < N - 1; j++)
			A[j] = B[j];
	}
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(A,N,n))

{
  int i;

  for (i = 0; i < n; i++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, A[i]);
      if (i % 20 == 0) fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}


int main(void) 
{	
	POLYBENCH_1D_ARRAY_DECL(a,DATA_TYPE,N,n);
	POLYBENCH_1D_ARRAY_DECL(b,DATA_TYPE,N,n);
	POLYBENCH_1D_ARRAY_DECL(a_outputFromGpu,DATA_TYPE,N,n);
	POLYBENCH_1D_ARRAY_DECL(b_outputFromGpu,DATA_TYPE,N,n);

	init_array(POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));
	init_array(POLYBENCH_ARRAY(a_outputFromGpu), POLYBENCH_ARRAY(b_outputFromGpu));

	read_cl_file();
	cl_initialization();
	cl_mem_init(POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));
	cl_load_prog();

	/* Start timer. */
  	polybench_start_instruments;
	
	int t;
	for (t = 0; t < TSTEPS ; t++)
	{
		cl_launch_kernel1();
		cl_launch_kernel2();
	}

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;
	
	errcode = clEnqueueReadBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, N * sizeof(DATA_TYPE), POLYBENCH_ARRAY(a), 0, NULL, NULL);
	errcode = clEnqueueReadBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0, N * sizeof(DATA_TYPE), POLYBENCH_ARRAY(b), 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");
		

	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		jacobi1D(POLYBENCH_ARRAY(a_outputFromGpu), POLYBENCH_ARRAY(b_outputFromGpu));
	
		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;

		compareResults(POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(a_outputFromGpu), POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(b_outputFromGpu));

	#else //print output to stderr so no dead code elimination

		print_array(N, POLYBENCH_ARRAY(a_outputFromGpu));

	#endif //RUN_ON_CPU


	cl_clean_up();

	POLYBENCH_FREE_ARRAY(a);
	POLYBENCH_FREE_ARRAY(b);
	POLYBENCH_FREE_ARRAY(a_outputFromGpu);
	POLYBENCH_FREE_ARRAY(b_outputFromGpu);

	return 0;
}

#include "../../common/polybench.c"
