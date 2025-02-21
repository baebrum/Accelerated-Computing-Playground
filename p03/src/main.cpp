#include <iostream>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <omp.h>
#include <random>
#include <cmath>
#include <chrono>
#include <string>

#define _USE_MATH_DEFINES

double f(double x);
double simpsons(double a, double b, int N, double h);
void handleCommandLineArguments(int argc, char** argv, int& N, int& num_threads, bool& output_csv);
void printUsage();

int main(int argc, char** argv) {
    int N = 100;  // Default N value
    int num_threads = 2;  // Default number of threads
    bool output_csv = false;  // Default to not output CSV format to standard output

    // Handle command-line arguments
    handleCommandLineArguments(argc, argv, N, num_threads, output_csv);

    // Set the number of threads for OpenMP
    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);

    double a = 0;
    double b = M_PI / 2;
    double h = (b - a) / N;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Calculate the integral using Simpson's rule
    double result = simpsons(a, b, N, h);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> wall_clock_time = end_time - start_time;

    if (output_csv) {
        // Output in CSV-friendly format
        std::cout << N << "," << num_threads << "," << wall_clock_time.count() << std::endl;
    }
    else {
        // Exact value of the integral
        double exact_value = 5 * M_PI * M_PI / 24;

        // Compute the numerical error
        double error = std::abs(result - exact_value);

        // Output the result and error
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "Number of intervals: " << N << std::endl;;
        std::cout << "Number of threads: " << num_threads << std::endl;;
        std::cout << "Approximated integral: " << result << std::endl;;
        std::cout << "Exact integral: " << exact_value << std::endl;;
        std::cout << std::scientific << "Numerical error: " << error << std::endl;;

        // Verify if the error is within an acceptable range
        if (error < 1e-14) {
            std::cout << "Error is within acceptable tolerance." << std::endl;;
        }
        else {
            std::cout << "Error exceeds acceptable tolerance." << std::endl;;
        }
    }

    return 0;
}

double f(double x) {
    return acos(cos(x) / (1 + (2 * cos(x))));
}

// compsoite 1/3 Simpson's rule integration
double simpsons(double a, double b, int N, double h) {
    // Precompute the first and last terms
    double f_a = f(a);
    double f_b = f(b);
    double odd_sum = 0.0, even_sum = 0.0;

#pragma omp parallel for reduction(+:odd_sum, even_sum)
    for (int i = 1; i < N; ++i) {
        double x = a + i * h;
        if (i % 2 == 0) {
            even_sum += 2 * f(x);  // Even index
        }
        else {
            odd_sum += 4 * f(x);   // Odd index
        }
    }

    // Return the total calculated value
    return (h / 3) * (f_a + odd_sum + even_sum + f_b);
}

void handleCommandLineArguments(int argc, char** argv, int& N, int& num_threads, bool& output_csv) {
    if (argc == 1) {
        std::cout << "Using default N = " << N << " and num_threads = " << num_threads << "." << std::endl;;
        return;
    }

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) {
            N = std::stoi(argv[++i]);
            if (N <= 0) {
                std::cerr << "Error: Invalid value for N. Defaulting to N = 100." << std::endl;;
                N = 100;
            }
        }
        else if (arg == "-t" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
            if (num_threads <= 0) {
                std::cerr << "Error: Invalid number of threads. Defaulting to num_threads = 2." << std::endl;;
                num_threads = 2;
            }
        }
        else if (arg == "-o" && i + 1 < argc && argv[i + 1] == std::string("csv")) {
            output_csv = true;
            i++; // Skip the next argument (csv)
        }
        else if (arg == "-h" || arg == "--help") {
            printUsage();
            exit(0);
        }
        else {
            std::cerr << "Error: Unrecognized argument " << argv[i] << std::endl;;
            printUsage();
            exit(1);
        }
    }
}

// Print usage information
void printUsage() {
    std::cout << "Usage: ./simpsons [-n N] [-t THREADS] [-o csv] [-h | --help]" << std::endl;;
    std::cout << "  -n N        : Set the number of intervals (N). Default is N = 100." << std::endl;;
    std::cout << "  -t THREADS  : Set the number of threads for OpenMP. Default is THREADS = 2." << std::endl;;
    std::cout << "  -o csv      : Output results in CSV-friendly format (default is error verification)." << std::endl;;
    std::cout << "  -h, --help  : Display this help message." << std::endl;;
    std::cout << std::endl;;
    std::cout << "Examples:" << std::endl;;
    std::cout << "  ./simpsons" << std::endl;;
    std::cout << "  ./simpsons -n 64" << std::endl;;
    std::cout << "  ./simpsons -t 8" << std::endl;;
    std::cout << "  ./simpsons -o csv" << std::endl;;
    std::cout << "  ./simpsons -n 64 -t 8" << std::endl;;
}
