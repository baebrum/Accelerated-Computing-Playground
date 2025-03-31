#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

__constant__ int deviceMask[5][5] = {
    {2, 2, 4, 2, 2},
    {1, 1, 2, 1, 1},
    {0, 0, 0, 0, 0},
    {-1, -1, -2, -1, -1},
    {-2, -2, -4, -2, -2}
};

// CUDA Kernel for 2D convolution processing all 3 channels
__global__ void convolutionKernel(const unsigned char* input, int* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int sum[3] = { 0 };  // One sum for each channel (RGB)

    for (int ky = -2; ky <= 2; ++ky) {
        for (int kx = -2; kx <= 2; ++kx) {
            int imgX = min(max(x + kx, 0), width - 1);
            int imgY = min(max(y + ky, 0), height - 1);
            for (int c = 0; c < 3; ++c) {  // Process each channel
                int pixel = input[(imgY * width + imgX) * channels + c];
                sum[c] += pixel * deviceMask[ky + 2][kx + 2];
            }
        }
    }

    // Store the raw convolution result in the output image
    for (int c = 0; c < 3; ++c) {
        output[(y * width + x) * 3 + c] = sum[c];
    }
}

int main() {
    int width, height, channels;
    unsigned char* h_inputImage = stbi_load("peppers.png", &width, &height, &channels, 3);
    if (!h_inputImage) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    // Write original image data to peppers.dat
    std::ofstream datFile("peppers.dat");
    for (int i = 0; i < width * height * 3; ++i) {
        datFile << static_cast<int>(h_inputImage[i]) << std::endl;
    }
    datFile.close();
    std::cout << "Original image data saved to peppers.dat" << std::endl;

    // Image size checks
    int imageSize = width * height * channels;
    int outputSize = width * height * 3;  // 3 channels for RGB output

    // Allocate memory on the device
    unsigned char* d_inputImage;
    int* d_outputImage;
    cudaMalloc(&d_inputImage, imageSize * sizeof(unsigned char));
    cudaMalloc(&d_outputImage, outputSize * sizeof(int));

    // Copy input image to device
    cudaMemcpy(d_inputImage, h_inputImage, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // CUDA kernel launch setup
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    convolutionKernel << <gridDim, blockDim >> > (d_inputImage, d_outputImage, width, height, channels);
    cudaDeviceSynchronize();

    // Check for errors in the kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    int* h_outputImage = new int[outputSize];
    cudaMemcpy(h_outputImage, d_outputImage, outputSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Write the raw output image to `peppers.out`
    std::ofstream outFile("peppers.out");
    for (int i = 0; i < outputSize; ++i) {
        outFile << h_outputImage[i] << std::endl;
    }
    outFile.close();

    // Cleanup
    stbi_image_free(h_inputImage);
    delete[] h_outputImage;
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    std::cout << "Edge detection complete. Output saved to peppers.out" << std::endl;

    return 0;
}
