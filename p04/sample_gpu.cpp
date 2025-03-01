#include <iostream>
#include <omp.h>

int main() {
    const int size = 100000;
    int a[size], b[size], c[size];

    // Initialize arrays a and b
    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Offload the computation to the GPU
#pragma omp target teams distribute parallel for map(to: a[0:size], b[0:size]) map(from: c[0:size])
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }

    // Verify the result
    for (int i = 0; i < 10; i++) {
        std::cout << "c[" << i << "] = " << c[i] << std::endl;
    }

    return 0;
}
