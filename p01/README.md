Observe how wall clock runtime changes as hardware thread count increases as a power of 2. Using the DGEMM (Double Precision General Matrix Multiply), invoke the serial (i.e., non-parallel) implementation to obtain a serial value for runtime. Next, modify the serial code to parallelize the computation of product matrix C using the

```c++
#pragma omp parallel for private(tmp, i, j, k)
```
OpenMP directive

Invoke your parallel implementation with a thread count as a power of 2. Output elapsed time data in a format that can be parsed by MATLAB and plot runtime in units of seconds as a function of thread count. Your MATLAB plot should use the following template for axis tick marks and labels.

Continue increasing the hardware thread count as a power of 2 beyond the maximum supported by the microprocessor. At what particular thread count do you find the run time to be minimum or optimal? In a a single page PDF document show:

1. runtime plot

2. code listing

3. program output

4. MATLAB plot script

5. and a brief discussion (a few sentences) of the optimal thread count value you discovered.