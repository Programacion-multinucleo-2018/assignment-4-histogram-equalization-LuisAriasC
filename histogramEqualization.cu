/*
  Author: Luis Carlos Arias Camacho
  Student ID: A01364808
  ASSIGNMENT 4
*/

#include <iostream>
#include <cstdio>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "common.h"
#include <cuda_runtime.h>
#include <chrono>

#define img_dest "Images/"
#define default_image "dog.jpeg"
#define C_SIZE 256

using namespace std;

//This function is used to equalize an histogram
  // src_histogram - imput histogram as a one dimentional array of ints
  // eq_histogram - output normalized histogram as a one dimentional array of ints
  // size - size of the histograms
void normalize(int * src_histogram, int * eq_histogram, int size){
    int step = size / C_SIZE;
    int sum = 0;

    for(int i=0; i < C_SIZE; i++){
        sum += src_histogram[i];
        eq_histogram[i] = sum / step;
    }
}



// Histogram equalization on cpu
  // imput - input image
  //output - output image
  //imageName - path to achieve the image
void equalizer_cpu(const cv::Mat &input, cv::Mat &output, string imageName){

  int width = input.cols;
  int height = input.rows;
  int size_ = width * height;

  //Histogram
  int histo[C_SIZE]{};

  //Fill histogram
  for (int i = 0; i < size_; i++)
    histo[input.ptr()[i]]++;

  //Normalize histogram
  int step = size_ / C_SIZE;
  int sum = 0;
  int n_histo[C_SIZE]{};
  for(int i=0; i < C_SIZE; i++){
      sum += histo[i];
      n_histo[i] = sum / step;
  }

  //Write image with normalized histogram on output
  for (int i = 0; i < size_; i++)
    output.ptr()[i] = n_histo[input.ptr()[i]];

  //Save the image
  cv::imwrite("Images/eq_cpu_" + imageName , output);
}



//This function converts a colored imege to a grayscale image
  // input - input image one dimensional array
  // ouput - output image one dimensional array
  // width, height - width and height of the images
  // colorWidthStep - number of color bytes (cols * colors)
  // grayWidthStep - number of gray bytes
__global__ void bgr_to_gray_kernel(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep, int grayWidthStep){
	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	if ((xIndex < width) && (yIndex < height)){
		const int color_tid = yIndex * colorWidthStep + (3 * xIndex);
		const int gray_tid = yIndex * grayWidthStep + xIndex;
		const unsigned char blue = input[color_tid];
		const unsigned char green = input[color_tid + 1];
		const unsigned char red = input[color_tid + 2];
		const float gray = red * 0.3f + green * 0.59f + blue * 0.11f;
		output[gray_tid] = static_cast<unsigned char>(gray);
	}
}



// Get histogram with gpu and atomic operations
  //output - output image int array
  //histo - histogram of the images as an array
  // width, height - width and height of the images
  // grayWidthStep - number of gray bytes
__global__ void get_histogram_kernel(unsigned char* output, int* histo,int width, int height, int grayWidthStep){

	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	if ((xIndex < width) && (yIndex < height)){
    const int tid = yIndex * grayWidthStep + xIndex;
    atomicAdd(&histo[(int)output[tid]], 1);
    __syncthreads();
	}
}


// Histogram equalization on gpu
  // imput - input image
  //output - output image
  //hist - input image histogram
  // width, height - width and height of the images
  // grayWidthStep - number of gray bytes
__global__ void equalizer_kernel(unsigned char* input, unsigned char* output, int * hist, int width, int height, int grayWidthStep){

  //Initialize shared memory for block
  __shared__ int hist_s[256];

    //2D Index of current thread
	unsigned int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

  //Index in shared memory
  unsigned int sxy = threadIdx.y * blockDim.x + threadIdx.x;
  //Thread ID
  const int tid  = yIndex * grayWidthStep + xIndex;

  //Fill in shared memory histogram
  if (sxy < 256){
    hist_s[sxy] = 0;
    hist_s[sxy] = hist[sxy];
  }
  __syncthreads();

  //Generate output image
  if((xIndex < width) && (yIndex < height))
      output[tid] = hist_s[input[tid]];
}


//Call this function to run the image equalization
  // input - input image
  // output - black & white output image
  // eq_output - equalized output image
  // imageName - path to reach the input image
void histogram_equalization(const cv::Mat& input, cv::Mat& output, cv::Mat& eq_output, string imageName){

  //Get size of the image
	size_t colorBytes = input.step * input.rows;
	size_t grayBytes = output.step * output.rows;
  int imSize = input.cols * input.rows;

  //Set device and cpu image arrays and histograms
	unsigned char *d_input, *d_output, *de_output;
  int * d_histogram, * df_histogram;
  int * histogram = (int *)malloc(C_SIZE * sizeof(int));
  int * f_histogram = (int *)malloc(C_SIZE * sizeof(int));
  for (int i = 0; i < C_SIZE; i++)
    f_histogram[i] = histogram[i] = 0;

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, colorBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, grayBytes), "CUDA Malloc Failed");
  SAFE_CALL(cudaMalloc<unsigned char>(&de_output, grayBytes), "CUDA Malloc Failed");
  SAFE_CALL(cudaMalloc<int>(&d_histogram, C_SIZE * sizeof(int)), "CUDA Malloc Failed");
  SAFE_CALL(cudaMalloc<int>(&df_histogram, C_SIZE * sizeof(int)), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
  SAFE_CALL(cudaMemset(d_histogram, 0, C_SIZE * sizeof(int)), "Error setting d_MatC to 0");

  const dim3 block(16, 16);
	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows/ block.y));

	// Launch the color conversion kernel
  printf("Converting image to black & white\n");
	bgr_to_gray_kernel <<<grid, block >>>(d_input, d_output, input.cols, input.rows, static_cast<int>(input.step), static_cast<int>(output.step));
  // Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");
  SAFE_CALL(cudaMemcpy(output.ptr(), d_output, grayBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");
  //Write the black & white image
  cv::imwrite("Images/bw_" + imageName , output);

  // Launch equalization on cpu
  printf("Equalization on cpu.\n");
  float cpuTime = 0.0;
  auto start_cpu =  chrono::high_resolution_clock::now();
  equalizer_cpu(output, eq_output, imageName);
  auto end_cpu =  chrono::high_resolution_clock::now();
  chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
  cpuTime = duration_ms.count();

  // Set the eq_output image to 0 in order to reuse it in gpu
  memset(eq_output.ptr(), 0, colorBytes);

  //Launch histogram calculation on cpu
  printf("Equalization on gpu.\n");
  float gpuTime = 0.0;
  auto start_gpu =  chrono::high_resolution_clock::now();
  get_histogram_kernel<<<grid, block >>>(d_output, d_histogram, input.cols, input.rows, static_cast<int>(output.step));
  SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");
  auto end_gpu =  chrono::high_resolution_clock::now();
  chrono::duration<float, std::milli> gpu_duration_ms = end_gpu - start_gpu;
  gpuTime += gpu_duration_ms.count();
  // SAFE_CALL kernel error
  SAFE_CALL(cudaGetLastError(), "Error with last error");

  // Copy device histogram to host histogram
  SAFE_CALL(cudaMemcpy(histogram, d_histogram, C_SIZE * sizeof(int), cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");
  //Normalize histogram
  normalize(histogram, f_histogram, imSize);

  //Copy normalized histogram to device normalized histogram
  SAFE_CALL(cudaMemcpy(df_histogram, f_histogram, C_SIZE * sizeof(int), cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

  //Set output image with normalized histogram
  start_gpu =  chrono::high_resolution_clock::now();
  equalizer_kernel<<<grid, block >>>(d_output, de_output, df_histogram, input.cols, input.rows, static_cast<int>(output.step));
  SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");
  end_gpu =  chrono::high_resolution_clock::now();
  gpu_duration_ms = end_gpu - start_gpu;
  gpuTime += gpu_duration_ms.count();

  //Write the black & white equalized image
  SAFE_CALL(cudaMemcpy(eq_output.ptr(), de_output, grayBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");
  cv::imwrite("Images/eq_gpu_" + imageName , eq_output);

  printf("Time in CPU: %f\n", cpuTime);
  printf("Time in GPU: %f\n", gpuTime);
  printf("Speedup: %f\n", cpuTime / gpuTime );

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
  SAFE_CALL(cudaFree(de_output), "CUDA Free Failed");
  SAFE_CALL(cudaFree(d_histogram), "CUDA Free Failed");
  SAFE_CALL(cudaFree(df_histogram), "CUDA Free Failed");

  //Free the host memory
  free(histogram);
  free(f_histogram);

  // Reset device
  SAFE_CALL(cudaDeviceReset(), "Error reseting");
}

int main(int argc, char *argv[]){

	string inputImage;

	if(argc < 2)
		inputImage = default_image;
  	else
  		inputImage = argv[1];

	// Read input image from the disk
	cv::Mat input = cv::imread(img_dest + inputImage, CV_LOAD_IMAGE_COLOR);

	if (input.empty()){
		cout << "Image Not Found!" << std::endl;
		cin.get();
		return -1;
	}

	//Create output image
	cv::Mat output(input.rows, input.cols, CV_8UC1);
  	//Create equalized output image
  	cv::Mat eq_output(input.rows, input.cols, CV_8UC1);

	//Convert image to gray and equalize
	histogram_equalization(input, output, eq_output, inputImage);

	//Allow the windows to resize
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Blac&WhiteInput", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("Input", input);
	imshow("Blac&WhiteInput", output);
  	imshow("Output", eq_output);
	//Wait for key press
	cv::waitKey();
	return 0;
}
