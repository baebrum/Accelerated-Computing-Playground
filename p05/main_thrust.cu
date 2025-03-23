#include <iostream>
#include <vector>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <cstdlib>
#include <iomanip>

#define BLOCK_SIZE 256

int main(int argc, char** argv) {
    // Default value for N
    int numInputElements = 1 << 20; // Default: 1 million elements

    // Check if an argument for N is passed in the command line
    if (argc > 1) {
        numInputElements = atoi(argv[1]);
    }

    // Initialize the input array with values 1, 2, 3, ..., N
    std::vector<long long> hostInput(numInputElements);
    for (int i = 0; i < numInputElements; ++i) {
        hostInput[i] = static_cast<long long>(i + 1);  // Values 1, 2, 3, ..., N
    }

    // --- Host-side summation (sequential) ---
    auto start = std::chrono::high_resolution_clock::now();
    long long hostSum = 0;
    for (int i = 0; i < numInputElements; ++i) {
        hostSum += hostInput[i];
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> hostDuration = end - start;
    double hostElapsedTime = hostDuration.count() * 1000; // in milliseconds
    std::cout << numInputElements << ", "
        << hostElapsedTime << ", " << hostSum << ", cpu" << std::endl;

    // --- Device-side summation (using Thrust for reduction) ---
    thrust::device_vector<long long> deviceInput = hostInput;

    start = std::chrono::high_resolution_clock::now();
    long long deviceSum = thrust::reduce(deviceInput.begin(), deviceInput.end(), 0LL, thrust::plus<long long>());
    end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> deviceDuration = end - start;
    double deviceElapsedTime = deviceDuration.count() * 1000; // in milliseconds
    std::cout << numInputElements << ", "
        << deviceElapsedTime << ", " << deviceSum << ", gpu" << std::endl;

    // Calculate speedup: timeSequential / timeParallel
    double speedup = hostElapsedTime / deviceElapsedTime;
    std::cout << "Speedup: " << speedup << std::endl;

    return 0;
}
