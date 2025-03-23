#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 256

// CUDA kernel for reduction (summation)
__global__ void reduction(long long* input, long long* output, int len) {
    __shared__ long long partialSum[2 * BLOCK_SIZE];
    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;

    if (start + t < len)
        partialSum[t] = input[start + t];
    else
        partialSum[t] = 0;

    if (start + BLOCK_SIZE + t < len)
        partialSum[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
    else
        partialSum[BLOCK_SIZE + t] = 0;

    // Traverse the reduction tree
    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
        __syncthreads();
        if (t < stride)
            partialSum[t] += partialSum[t + stride];
    }

    if (t == 0)
        output[blockIdx.x] = partialSum[0];
}

int main(int argc, char** argv) {
    // Default value for N
    int numInputElements = 1 << 20; // Default: 1 million elements

    // Check if an argument for N is passed in the command line
    if (argc > 1) {
        numInputElements = atoi(argv[1]);
    }

    size_t size = numInputElements * sizeof(long long);

    // Allocate memory for host input array and output array
    long long* hostInput = (long long*)malloc(size);
    long long* hostOutput = (long long*)malloc(sizeof(long long) * ((numInputElements + BLOCK_SIZE - 1) / BLOCK_SIZE));

    // Initialize the input array with values 1, 2, 3, ..., N (instead of just 1.0f)
    for (int i = 0; i < numInputElements; i++) {
        hostInput[i] = (long long)(i + 1);  // Values 1, 2, 3, ..., N
    }

    // Device memory pointers
    long long* deviceInput, * deviceOutput;

    cudaMalloc(&deviceInput, size);
    cudaMalloc(&deviceOutput, sizeof(long long) * ((numInputElements + BLOCK_SIZE - 1) / BLOCK_SIZE));

    // Copy input array to device
    cudaMemcpy(deviceInput, hostInput, size, cudaMemcpyHostToDevice);

    // --- Host-side summation (sequential) ---
    clock_t start = clock();
    long long hostSum = 0;
    for (int i = 1; i <= numInputElements; i++) {
        hostSum += i;
    }
    clock_t end = clock();

    double hostElapsedTime = ((double)(end - start)) / CLOCKS_PER_SEC * 1000; // in milliseconds
    printf("%d, %f, %lld, cpu\n", numInputElements, hostElapsedTime, hostSum);

    // --- Device-side summation (using CUDA kernel) ---
    dim3 dimGrid((numInputElements + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE), 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    cudaEvent_t startEvent, stopEvent;
    float elapsedTime;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);

    reduction << <dimGrid, dimBlock >> > (deviceInput, deviceOutput, numInputElements);

    // Wait for kernel to finish
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

    // Copy back the result from the device to the host
    cudaMemcpy(hostOutput, deviceOutput, sizeof(long long) * dimGrid.x, cudaMemcpyDeviceToHost);

    // Reduce the partial results from the device output
    long long deviceSum = 0;
    for (int i = 0; i < dimGrid.x; i++) {
        deviceSum += hostOutput[i];
    }

    printf("%d, %f, %lld, gpu\n", numInputElements, elapsedTime, deviceSum);

    // Calculate speedup: timeSequential / timeParallel
    double speedup = hostElapsedTime / elapsedTime;
    printf("Speedup: %f\n", speedup);

    // Cleanup
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    free(hostInput);
    free(hostOutput);

    return 0;
}
