Implement a basic dense matrix multiplication CUDA kernel and a tiled dense matrix multiplication CUDA kernel that uses shared memory tiles.

Given "large" dense float matrices **A** of dimension `m × p`, and **B** of dimension `p × n`, compute product matrix **C** of dimension `m × n`.

### The basic strategy is to:

#### 1. Allocate device memory for A, B, and C:

```cpp
cudaMalloc(&dA, sizeof(float) * m * p);
cudaMalloc(&dB, sizeof(float) * p * n);
cudaMalloc(&dC, sizeof(float) * m * n);
```

#### 2. Copy host memory to device

```cpp
cudaMemcpy(dA, hA, sizeof(float) * m * p, cudaMemcpyHostToDevice);
cudaMemcpy(dB, hB, sizeof(float) * p * n, cudaMemcpyHostToDevice);
```

#### 3. Initialize thread block and kernel grid dimensions

```cpp
dim3 gridDim(ceil((float)n / blockDim.x), ceil((float)m / blockDim.y));
```

and

```cpp
dim3 dimGrid((n - 1) / TILE_WIDTH + 1, (m - 1) / TILE_WIDTH + 1, 1);
dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
```

#### 4. Invoke CUDA kernel

```cpp
matrixMultiply<<<dimGrid, dimBlock>>>(dA, dB, dC, m, p, n);
```

and

```cpp
matrixTiledMultiply<<<dimGrid, dimBlock>>>(dA, dB, dC, m, p, n);
```

#### 5. Copy results from device to host

```cpp
cudaDeviceSynchronize();
cudaMemcpy(hC, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
```

#### 6. Deallocate device memory

```cpp
cudaFree(dA);
cudaFree(dB);
cudaFree(dC);
```

### Compute the elapsed time of both kernel invocations separately

You can measure kernel elapsed time by applying CUDA events:

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
...
cudaEventRecord(start);
matrixMultiply<<<dimGrid, dimBlock>>>(dA, dB, dC, m, p, n);
cudaEventRecord(stop);
...
cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
```

### Final Tasks

- Generate random valued matrices with dimensions `m`, `p`, and `n` that are powers of 2, with `m = n`.
- Test your two kernel implementations.
- Plot kernel elapsed time as a function of `m = n`:
  - Basic kernel (blue plot line)
  - Tiled kernel (red plot line)
- Write a few sentences discussing your findings and the performance impact of using shared memory tiles.

**Submit a single PDF** showing:
- Your single plot
- Your discussion
- Your code listing