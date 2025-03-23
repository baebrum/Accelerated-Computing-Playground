#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>

#define TILE_WIDTH 16

// Usage function to display instructions
void usage(const char* programName) {
    std::cout << "Usage: " << programName << " <m> <p>\n";
    std::cout << "  <m>: Number of rows of matrix A (and matrix C) (max 15)\n";
    std::cout << "  <p>: Number of columns of matrix A and rows of matrix B (max 15)\n";
    std::cout << "If no arguments are provided, defaults to m=15, p=15\n";
    std::cout << "Matrix dimensions must be powers of 2.\n";
}

// Basic Matrix Multiplication Kernel
__global__ void matrixMultiply(float* A, float* B, float* C, int m, int p, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float value = 0.0f;
        for (int k = 0; k < p; ++k) {
            value += A[row * p + k] * B[k * n + col];
        }
        C[row * n + col] = value;
    }
}

// Tiled Matrix Multiplication Kernel using Shared Memory
__global__ void matrixTiledMultiply(float* A, float* B, float* C, int m, int p, int n) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float value = 0.0f;

    for (int t = 0; t < (p + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (row < m && t * TILE_WIDTH + threadIdx.x < p)
            tileA[threadIdx.y][threadIdx.x] = A[row * p + t * TILE_WIDTH + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < n && t * TILE_WIDTH + threadIdx.y < p)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * n + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = value;
    }
}

// Random Matrix Generator
void randomMatrix(float* A, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        A[i] = rand() % 1000 + 1;  // Generate random values between 1 and 1000
    }
}

// Basic Matrix Multiplication Function
void basicMatrixMultiply(float* hA, float* hB, float* hC, int m, int p, int n) {
    float* dA, * dB, * dC;

    cudaMalloc(&dA, sizeof(float) * m * p);
    cudaMalloc(&dB, sizeof(float) * p * n);
    cudaMalloc(&dC, sizeof(float) * m * n);

    cudaMemcpy(dA, hA, sizeof(float) * m * p, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(float) * p * n, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixMultiply << <gridDim, blockDim >> > (dA, dB, dC, m, p, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cout << "Basic matrix multiplication took " << milliseconds << " ms" << std::endl;
    std::cout << milliseconds << ",";

    cudaMemcpy(hC, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

// Tiled Matrix Multiplication Function
void tiledMatrixMultiply(float* hA, float* hB, float* hC, int m, int p, int n) {
    float* dA, * dB, * dC;

    cudaMalloc(&dA, sizeof(float) * m * p);
    cudaMalloc(&dB, sizeof(float) * p * n);
    cudaMalloc(&dC, sizeof(float) * m * n);

    cudaMemcpy(dA, hA, sizeof(float) * m * p, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(float) * p * n, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((n + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixTiledMultiply << <dimGrid, dimBlock >> > (dA, dB, dC, m, p, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cout << "Tiled matrix multiplication took " << milliseconds << " ms" << std::endl;
    std::cout << milliseconds;

    cudaMemcpy(hC, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

// Function to check GPU memory and ensure we don't exceed it
bool checkMemoryAvailability(int m, int p, int n) {
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);

    size_t requiredMemory = sizeof(float) * m * p + sizeof(float) * p * n + sizeof(float) * m * n;
    if (requiredMemory > freeMem) {
        std::cout << "Error: Insufficient memory on the GPU." << std::endl;
        return false;
    }

    return true;
}

int main(int argc, char* argv[]) {

    // If the user passes the -h or --help flag, show usage and exit
    if (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        usage(argv[0]);
        return 0;
    }

    // Default values
    int m = 15;
    int p = 15;
    int n = m;

    if (argc == 3) {
        // Check if arguments are valid integers
        m = atoi(argv[1]);
        p = atoi(argv[2]);

        // Ensure that m and p do not exceed 15
        if (m > 15 || p > 15) {
            std::cerr << "Error: m and p cannot exceed 15." << std::endl;
            usage(argv[0]);
            return -1;
        }

        if (m < 0) {
            m = 0;
        }
        if (p < 0) {
            p = 0;
        }

        m = pow(2, m);
        p = pow(2, p);

        n = m;  // Since m == n for this case
    }
    else {
        // If no arguments are provided, use default values and inform the user
        std::cout << "No arguments provided. Using default dimensions m=15, p=15." << std::endl;
    }

    std::cout << "Matrix dimensions: m = " << m << ", p = " << p << ", n = " << n << std::endl;

    // Check if the GPU has enough memory for the operation
    if (!checkMemoryAvailability(m, p, n)) {
        return -1;  // Exit if memory is insufficient
    }

    float* hA = new float[m * p];
    float* hB = new float[p * n];
    float* hC = new float[m * n];

    // Generate random matrices A and B
    randomMatrix(hA, m, p);
    randomMatrix(hB, p, n);

    // Test basic matrix multiplication
    std::cout << "Basic Matrix Multiplication (ms)," << "Tiled Matrix Multiplication (ms)" << std::endl;
    basicMatrixMultiply(hA, hB, hC, m, p, n);

    // Test tiled matrix multiplication
    // std::cout << "Running Tiled Matrix Multiplication..." << std::endl;
    tiledMatrixMultiply(hA, hB, hC, m, p, n);
    std::cout << std::endl;

    // Clean up
    delete[] hA;
    delete[] hB;
    delete[] hC;

    return 0;
}
