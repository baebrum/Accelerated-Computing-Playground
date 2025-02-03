#include <iostream>
#include <iomanip>
#include <omp.h>
#include <cstdlib>

#define ORDER 1000
#define AVAL 3.0
#define BVAL 5.0

const double PRECISION_THRESHOLD = 1e-16;

int main(int argc, char* argv[])
{
    // Check if number of threads is provided as a command-line argument
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <number_of_threads>" << std::endl;
        return 1;
    }

    // Get the number of threads from the command line argument
    int num_threads = std::atoi(argv[1]);
    if (num_threads <= 0) {
        std::cerr << "Invalid number of threads. Please specify a positive integer." << std::endl;
        return 1;
    }

    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);

    std::cout << "Setting OpenMP to use " << num_threads << " threads." << std::endl;

    int Ndim, Pdim, Mdim; /* A[N][P], B[P][M], C[N][M] */
    int i, j, k;
    double tmp;

    Ndim = ORDER;
    Pdim = ORDER;
    Mdim = ORDER;

    double* A = new double[Ndim * Pdim]; // A[N][P]
    double* B = new double[Pdim * Mdim]; // B[P][M]
    double* C = new double[Ndim * Mdim]; // C[N][M]
    double* C_Repeated = new double[Ndim * Mdim]; // C_Repeated[N][M]

    // Initialize matrices A, B, C, and C_Repeated
    for (int i = 0; i < Ndim; i++)
        for (int j = 0; j < Pdim; j++)
            *(A + (i * Ndim + j)) = AVAL;
    for (int i = 0; i < Pdim; i++)
        for (int j = 0; j < Mdim; j++)
            *(B + (i * Pdim + j)) = BVAL;
    for (int i = 0; i < Ndim; i++) {
        for (int j = 0; j < Mdim; j++) {
            *(C + (i * Ndim + j)) = 0.0;
            *(C_Repeated + (i * Ndim + j)) = 0.0;
        }
    }

    // Calculate expected result for C (sequential multiplication)
    for (int i = 0; i < Ndim; i++) {
        for (int j = 0; j < Mdim; j++) {
            double tmp = 0.0;
            for (int k = 0; k < Pdim; k++) {
                tmp += *(A + (i * Ndim + k)) * *(B + (k * Pdim + j));
            }
            *(C + (i * Ndim + j)) = tmp;
        }
    }

    // Perform matrix multiplication in parallel for C_Repeated
    double start_time = omp_get_wtime();
#pragma omp parallel for private(tmp, i, j, k)
    for (int i = 0; i < Ndim; i++) {
        for (int j = 0; j < Mdim; j++) {
            double tmp = 0.0;
            for (int k = 0; k < Pdim; k++) {
                tmp += *(A + (i * Ndim + k)) * *(B + (k * Pdim + j));
            }
            *(C_Repeated + (i * Ndim + j)) = tmp;
        }
    }

    double run_time = omp_get_wtime() - start_time;
    double error = 0.0;

    // Calculate squared error between C and C_Repeated
    for (int i = 0; i < Ndim; i++) {
        for (int j = 0; j < Mdim; j++) {
            double diff = *(C + (i * Ndim + j)) - *(C_Repeated + (i * Ndim + j));
            error += diff * diff;
        }
    }

    double dN = (double)ORDER;
    double mflops = 2.0 * dN * dN * dN / (1000000.0 * run_time);

    std::cout << std::fixed << std::setprecision(16);  // Set output precision to 16 decimal places
    std::cout << "Order " << ORDER << " multiplication in " << run_time << " seconds" << std::endl;
    std::cout << mflops << " mflops" << std::endl;

    if (error < PRECISION_THRESHOLD) {
        std::cout << "No errors (errsq = " << error << ")." << std::endl;
    }
    else {
        std::cout << "Errors detected! (errsq = " << error << ")" << std::endl;
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_Repeated;
}
