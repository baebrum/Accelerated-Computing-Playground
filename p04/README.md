Explore OpenMP Offloading onto the NVIDIA DGX A100 GPU using 2D Jacobi iteration as an example computational kernel to offload. Implement 2D Jacobi iteration on a N x N grid with N = 1000. Use the initial condition T0 shown with two opposite walls initialized to 1 and the interior grid initialized to 0, as shown in the following code:

```c
// initialize grid and boundary conditions
for (unsigned i = 0; i <= n_cells + 1; i++)
  for (unsigned j = 0; j <= n_cells + 1; j++)
    if((j == 0) || (j == (n_cells + 1)))
      T(i, j) = 1.0;
    else
      T(i, j) = 0.0;
```

First implement and verify a serial solution (no hardware parallelization) of your code until the solution residual reaches a value in the order of 1 x 10-8 or smaller and plot the solution in Matlab (or another visualization package such as Matplotlib if you prefer).

Here is the 2D Jacobi iteration kernel:

```c
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
#define T(i, j) (T[(i)*(n_cells+2) + (j)])
#define T_new(i, j) (T_new[(i)*(n_cells+2) + (j)])

double MAX_RESIDUAL = 1.e-8;

void kernel(double *T, int max_iterations) {

  int iteration = 0;
  double residual = 1.e6;
  double *T_new = (double *)malloc(SIZE * sizeof(double));
  while (residual > MAX_RESIDUAL && iteration < max_iterations) {
    for (unsigned i = 1; i <= N; i++)
      for (unsigned j = 1; j <= N; j++)
        T_new(i, j) =
            0.25 * (T(i + 1, j) + T(i - 1, j) + T(i, j + 1) + T(i, j - 1));
    residual = 0.0;
    for (unsigned int i = 1; i <= N; i++) {
      for (unsigned int j = 1; j <= N; j++) {
        residual = MAX(fabs(T_new(i, j) - T(i, j)), residual);
        T(i, j) = T_new(i, j);
      }
    }
    iteration++;
  }
  printf("residual = %.9e\n", residual);
  free(T_new);
}
```

Second, implement and verify an OpenMP solution using CPU hardware threads. Use the collapse clause. How many hardware threads are needed to obtain a speedup > 1?

Third, implement and verify an OpenMP solution using GPU teams without vectorization (without parallel for simd). How does the parallel GPU teams implementation compare with the parallel CPU hardware thread implementation, in terms of speedup?

Fourth, implement and verify an OpenMP solution using GPU teams with parallel for simd. How does the parallel for simd clause improve speedup on the GPU?

Show your 3D surface plots, residual values, and a brief discussion of what you discovered about OpenMP GPU offloading. Here is an excerpt of code to print the T grid in Matlab format:

```c
  printf("T = [...\n");
  for (unsigned i = 0; i <= (n_cells + 1); i++) {
    for (unsigned j = 0; j <= (n_cells + 1); j++)
      printf("%f ",T(i, j));
    printf(";...\n");
  }
  printf("];\n");
```