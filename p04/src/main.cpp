#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <omp.h>
#include <sstream>

// Function prototypes
void initialize_grid(double* T, unsigned N);
void kernel(double* T, unsigned N, int max_iterations, const std::string& executable_name, int num_threads);
void handleCommandLineArguments(int argc, char** argv, unsigned& N, int& max_iterations, int& num_threads);
void printUsage();
void save_grid_to_file(double* T, unsigned N, const std::string& filename);

double MAX_RESIDUAL = 1.e-8;
// Define macros for accessing 2D array elements
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
#define T(i, j) (T[(i)*(N+2) + (j)])
#define T_new(i, j) (T_new[(i)*(N+2) + (j)])

int main(int argc, char** argv) {
    unsigned N = 10;  // Default grid size
    int max_iterations = 1000;  // Default max iterations
    int num_threads = 4;  // Default number of threads

    handleCommandLineArguments(argc, argv, N, max_iterations, num_threads);

    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);

    // Print configuration
    std::cout << "Grid size: " << N << " x " << N << std::endl;
    std::cout << "Max iterations: " << max_iterations << std::endl;
    std::cout << "Number of threads: " << num_threads << std::endl;

    double* T = new double[(N + 2) * (N + 2)];

    initialize_grid(T, N);

    std::string executable_name = argv[0];  // Get the name of the executable

    // Run Jacobi iteration for a maximum number of iterations
    kernel(T, N, max_iterations, executable_name, num_threads);

    delete[] T;
    return 0;
}

// Function to initialize the grid and boundary conditions
void initialize_grid(double* T, unsigned N) {
    for (unsigned i = 0; i <= N + 1; i++) {
        for (unsigned j = 0; j <= N + 1; j++) {
            if (j == 0 || j == N + 1)
                T(i, j) = 1.0; // Boundary condition: walls are 1
            else
                T(i, j) = 0.0; // Interior is 0
        }
    }
}

// Jacobi iteration kernel
void kernel(double* T, unsigned N, int max_iterations, const std::string& executable_name, int num_threads) {
    int iteration = 0;
    double residual = 1.e6;

    // Dynamically allocate T_new on the host
    double* T_new = new double[(N + 2) * (N + 2)];

    // If GPU is enabled, use target offloading with OpenMP
#ifdef BUILD_GPU
// Enter the data for both T and T_new to the GPU
#pragma omp target enter data map(to: T[0:(N+2)*(N+2)]) map(to: T_new[0:(N+2)*(N+2)])
#endif

    // Start timing the kernel
    auto start_time = std::chrono::high_resolution_clock::now();

    while (residual > MAX_RESIDUAL && iteration < max_iterations) {
        // Jacobi iteration: update T_new based on neighbors
#ifdef BUILD_GPU
#ifdef BUILD_GPU_SIMD
        // Offload the computation to the GPU using OpenMP target teams with SIMD
#pragma omp target teams distribute parallel for collapse(2) simd
#else
        // Offload the computation to the GPU using OpenMP target teams without SIMD
#pragma omp target teams distribute parallel for collapse(2)
#endif
#else
        // Use CPU OpenMP parallelization for the Jacobi iteration loop
#pragma omp parallel for collapse(2)
#endif
        for (unsigned i = 1; i <= N; i++) {
            for (unsigned j = 1; j <= N; j++) {
                T_new(i, j) = 0.25 * (T(i + 1, j) + T(i - 1, j) + T(i, j + 1) + T(i, j - 1));
            }
        }

        // Compute residual and update T
        residual = 0.0;
#ifdef BUILD_GPU
#ifdef BUILD_GPU_SIMD
        // Offload the computation to the GPU using OpenMP target teams with SIMD for residual computation
#pragma omp target teams distribute parallel for reduction(max: residual) collapse(2) simd
#else
        // Offload the computation to the GPU using OpenMP target teams without SIMD for residual calculation
#pragma omp target teams distribute parallel for reduction(max: residual) collapse(2)
#endif
#else
        // Use CPU OpenMP parallelization for residual calculation
#pragma omp parallel for reduction(max : residual) collapse(2)
#endif
        for (unsigned i = 1; i <= N; i++) {
            for (unsigned j = 1; j <= N; j++) {
                residual = MAX(fabs(T_new(i, j) - T(i, j)), residual);
                T(i, j) = T_new(i, j);
            }
        }

        iteration++;
    }
#ifdef BUILD_GPU
    // Ensure data is copied back from device to host
#pragma omp target update from(T[0:(N+2)*(N+2)])
#endif

    // Stop timing the kernel
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    // Print output with executable name and iterations
    std::cout << executable_name << " - Residual after " << iteration << " iterations: " << residual << std::endl;
    std::cout << executable_name << " - Time taken for Jacobi iterations: " << duration.count() << " seconds." << std::endl;

    std::stringstream filename;
    filename << executable_name << "_" + std::to_string(N) + "_" + std::to_string(max_iterations) + "_" + std::to_string(num_threads) + "_grid_output.txt";
    save_grid_to_file(T, N, filename.str());

    // Clean up GPU memory and host memory
#ifdef BUILD_GPU
// Exit data on GPU
#pragma omp target exit data map(from: T[0:(N+2)*(N+2)]) map(from: T_new[0:(N+2)*(N+2)])
#endif

    delete[] T_new;
}

// Function to handle command line arguments
void handleCommandLineArguments(int argc, char** argv, unsigned& N, int& max_iterations, int& num_threads) {
    if (argc == 1) {
        std::cout << "Using default N = " << N << ", max_iterations = " << max_iterations << ", and num_threads = " << num_threads << "." << std::endl;
        return;
    }

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) {
            N = std::stoi(argv[++i]);
            if (N <= 0) {
                std::cerr << "Error: Invalid value for N. Defaulting to N = 10." << std::endl;
                N = 10;
            }
        }
        else if (arg == "-i" && i + 1 < argc) {
            max_iterations = std::stoi(argv[++i]);
            if (max_iterations <= 0) {
                std::cerr << "Error: Invalid number of iterations. Defaulting to max_iterations = 1000." << std::endl;
                max_iterations = 1000;
            }
        }
        else if (arg == "-t" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
            if (num_threads <= 0) {
                std::cerr << "Error: Invalid number of threads. Defaulting to num_threads = 1." << std::endl;
                num_threads = 1;
            }
        }
        else if (arg == "-h" || arg == "--help") {
            printUsage();
            exit(0);
        }
        else {
            std::cerr << "Error: Unrecognized argument " << argv[i] << std::endl;
            printUsage();
            exit(1);
        }
    }
}

// Print usage information
void printUsage() {
    std::cout << "Usage: ./jacobi [-n N] [-i ITERATIONS] [-t THREADS] [-h | --help]" << std::endl;
    std::cout << "  -n N          : Set the grid size N (NxN). Default is N = 10." << std::endl;
    std::cout << "  -i ITERATIONS : Set the maximum number of iterations. Default is 1000." << std::endl;
    std::cout << "  -t THREADS    : Set the number of OpenMP threads. Default is 4." << std::endl;
    std::cout << "  -h, --help    : Display this help message." << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  ./jacobi" << std::endl;
    std::cout << "  ./jacobi -n 20" << std::endl;
    std::cout << "  ./jacobi -i 500" << std::endl;
    std::cout << "  ./jacobi -n 20 -i 500" << std::endl;
    std::cout << "  ./jacobi -t 8" << std::endl;
    std::cout << "  ./jacobi -n 20 -i 500 -t 8" << std::endl;
}

// Function to save the grid to a CSV file for Python ingestion
void save_grid_to_file(double* T, unsigned N, const std::string& filename) {
    std::ofstream outfile(filename);

    if (outfile.is_open()) {
        for (unsigned i = 0; i <= N + 1; i++) {
            for (unsigned j = 0; j <= N + 1; j++) {
                outfile << T(i, j);
                if (j < N + 1) outfile << " ";  // Space between columns
            }
            outfile << std::endl;  // Newline after each row
        }
        outfile.close();
    }
    else {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
    }
}
