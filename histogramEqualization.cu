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
#include <math.h>

#define img_dest "Images/"
#define default_image "dog1.jpeg"
#define C_SIZE 256

using namespace std;



//Create input histogram
    // input - input images
    // histo - histogram
    // size - width * height of the image
void create_input_histogram(const cv::Mat &input, int * histo, int size){
  for (int i = 0; i < size; i++)
    histo[input.ptr()[i]]++;
}



//This function is used to normalize an histogram
  // histo - imput histogram as a one dimentional array of ints
  // n_histo - output normalized histogram as a one dimentional array of ints
  // size - size of the histograms
void normalize_histogram(int * histo, int * n_histo, int size){
  float step = size / (C_SIZE - 1);
  float sum = 0;
  for(int i=0; i < C_SIZE; i++){
      sum += (float)histo[i];
      n_histo[i] = (int)floor(sum / step);
  }
}



//Print the histogram
void print_histogram(int * histo){
  for(int i = 0; i < C_SIZE; i++)
    printf("%d : %d\n", i, histo[i]);
}



// Write image in cpu
void write_image(const cv::Mat &input, cv::Mat &output, int * n_histo, int size){
  for (int i = 0; i < size; i++)
    output.ptr()[i] = n_histo[input.ptr()[i]];
}



// Histogram equalization on cpu
  // imput - input image
  //output - output image
  //imageName - path to achieve the image
void equalizer_cpu(const cv::Mat &input, cv::Mat &output, int * histo, int * n_histo, string imageName){

  int width = input.cols;
  int height = input.rows;
  int size_ = width * height;

  //Create the histogram for the input image
  create_input_histogram(input, histo, size_);
  //Create normalized histogram
  normalize_histogram(histo, n_histo, size_);
  //Write image with normalized histogram on output
  write_image(input, output, n_histo, size_);

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

  // 2D Index of current pixel in the image
  const int tid = yIndex * grayWidthStep + xIndex;
  // Shared histogram
  __shared__ int s_histo[C_SIZE];
  // 2D index of shared memory
  int s_x = threadIdx.x+threadIdx.y*blockDim.x;

  //Initialize shared histogram to 0
  if (s_x < C_SIZE)
    s_histo[s_x] = 0;
  __syncthreads();

  //Fill shared histogram with the image info
	if ((xIndex < width) && (yIndex < height))
    atomicAdd(&s_histo[(int)output[tid]], 1);
  __syncthreads();

  // Copy infro from shared memory to global memory
  if (s_x < C_SIZE)
    atomicAdd(&histo[s_x], s_histo[s_x]);

}

//Calculate normalized histogram in kernel
  // histogram - original histogram of the image
  // n_histogram - normalized histogram
  // size - size of the image in pixels
__global__ void get_normalizedHistogram_kernel(int * histogram, int * n_histogram, int size){

  //Get histogram index from kernel
  int hIndex = threadIdx.y * blockDim.x + threadIdx.x;

  //Share the histogram with shared memory
  __shared__ int aux[C_SIZE];

  // Normalize in GPU
  if (hIndex < 256 && blockIdx.x == 0 && blockIdx.y == 0){
      aux[hIndex] = 0;
      aux[hIndex] = histogram[hIndex];
      __syncthreads();

      float step = size / (C_SIZE -1);
      float sum = 0;
      for(int i = 0; i <= hIndex; i++)
        sum += aux[i];
      n_histogram[hIndex] = (int)floor(sum / step);
  }
}


// Histogram equalization on gpu
  // imput - input image
  //output - output image
  //hist - input image histogram
  // width, height - width and height of the images
  // grayWidthStep - number of gray bytes
__global__ void equalizer_kernel(unsigned char* input, unsigned char* output, int * hist, int width, int height, int grayWidthStep){

    //2D Index of current thread
	unsigned int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
  //Thread ID
  const int tid  = yIndex * grayWidthStep + xIndex;

  //Generate output image
  if((xIndex < width) && (yIndex < height)){
    //int index = ;
    output[tid] = hist[input[tid]];
  }
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

  //Initialize histograms
  int hisBytes = C_SIZE * sizeof(int);
  int * histogram = (int *)malloc(hisBytes);
  int * f_histogram = (int *)malloc(hisBytes);
  memset(histogram, 0, hisBytes);
  memset(f_histogram, 0, hisBytes);

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, colorBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, grayBytes), "CUDA Malloc Failed");
  SAFE_CALL(cudaMalloc<unsigned char>(&de_output, grayBytes), "CUDA Malloc Failed");
  SAFE_CALL(cudaMalloc<int>(&d_histogram, hisBytes), "CUDA Malloc Failed");
  SAFE_CALL(cudaMalloc<int>(&df_histogram, hisBytes), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
  SAFE_CALL(cudaMemset(d_output, 0, grayBytes), "Error setting d_output to 0");
  SAFE_CALL(cudaMemset(de_output, 0, grayBytes), "Error setting de_output to 0");
  SAFE_CALL(cudaMemset(d_histogram, 0, hisBytes), "Error setting d_histogram to 0");
  SAFE_CALL(cudaMemset(df_histogram, 0, hisBytes), "Error setting df_histogram to 0");

  const dim3 block(16, 16);
	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows/ block.y));

	// Launch the color conversion kernel
  printf("Converting image to black & white\n");
	bgr_to_gray_kernel <<<grid, block >>>(d_input, d_output, input.cols, input.rows, static_cast<int>(input.step), static_cast<int>(output.step));
  // Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");
  SAFE_CALL(cudaMemcpy(output.ptr(), d_output, grayBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Device To Host Failed");
  //Write the black & white image
  cv::imwrite("Images/bw_" + imageName , output);

  // Launch equalization on CPU
  printf("Equalization on cpu.\n");
  float cpuTime = 0.0;
  auto start_cpu =  chrono::high_resolution_clock::now();
  equalizer_cpu(output, eq_output, histogram, f_histogram, imageName);
  auto end_cpu =  chrono::high_resolution_clock::now();
  chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
  cpuTime = duration_ms.count();

  // Set the eq_output image to 0 in order to reuse it in gpu
  memset(eq_output.ptr(), 0, grayBytes);
  memset(histogram, 0, hisBytes);
  memset(f_histogram, 0, hisBytes);
  printf("\n\n");

  //Launch equalization on GPU
  printf("Equalization on gpu.\n");
  float gpuTime = 0.0;
  auto start_gpu =  chrono::high_resolution_clock::now();
  get_histogram_kernel<<<grid, block >>>(d_output, d_histogram, input.cols, input.rows, static_cast<int>(output.step));
  get_normalizedHistogram_kernel<<<grid, block>>>(d_histogram, df_histogram, imSize);
  equalizer_kernel<<<grid, block>>>(d_output, de_output, df_histogram, output.cols, output.rows, static_cast<int>(output.step));
  auto end_gpu =  chrono::high_resolution_clock::now();
  chrono::duration<float, std::milli> gpu_duration_ms = end_gpu - start_gpu;
  gpuTime += gpu_duration_ms.count();
  SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");
  // SAFE_CALL kernel error
  SAFE_CALL(cudaGetLastError(), "Error with last error");

  // Copy device histograms to host histograms
  SAFE_CALL(cudaMemcpy(histogram, d_histogram, hisBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Device To Device Failed");
  SAFE_CALL(cudaMemcpy(f_histogram, df_histogram, hisBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Device To Device Failed");
  SAFE_CALL(cudaMemcpy(eq_output.ptr(), de_output, grayBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Device To Device Failed");

  //Save the image
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
