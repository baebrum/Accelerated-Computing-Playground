Explore **reduction** in Nvidia CUDA. Consider the following CUDA kernel that reduces (by summation) input array `input` of `len` floats, and places a reduced sum in `output[0]`:

```cpp
#define BLOCK_SIZE 512 // You can change this

__global__ void reduction(float *input, float *output, int len) {
    // Load a segment of the input vector into shared memory
    __shared__ float partialSum[2 * BLOCK_SIZE];
    unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
    if (start + t < len)
        partialSum[t] = input[start + t];
    else
        partialSum[t] = 0;

    if (start + BLOCK_SIZE + t < len)
        partialSum[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
    else
        partialSum[BLOCK_SIZE + t] = 0;

    // Traverse the reduction tree
    /*
        strides will assume values:
        512
        256
        128
        64
        32
        16
        8
        4
        2
        1
    */
    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
        __syncthreads();
        if (t < stride)
            partialSum[t] += partialSum[t + stride];
    }

    // Write the computed sum of the block to the output vector at the correct index
    if (t == 0)
        output[blockIdx.x] = partialSum[0];
}
```

---

### Grid and Block Configuration

You can define a grid and block configuration to invoke this reduction kernel as follows:

```cpp
dim3 dimGrid(numOutputElements, 1, 1);
dim3 dimBlock(BLOCK_SIZE, 1, 1);
reduction<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numInputElements);
```

---

### Task

Write a program to add **N sequential floats**:

$$
\sum_{i=1}^{N} i
$$


Run the summation:

- On the **host (CPU)** sequentially
- On the **device (GPU)** using the **reduction kernel**

Measure time for both summations.

---

### Measure CPU Time

```cpp
clock_t t;
t = clock();

// host computation

t = clock() - t;
printf("elapsed time: %f ms\n", ((double)t)/CLOCKS_PER_SEC * 1000);
```

---

### Measure GPU Time

```cpp
cudaEvent_t start, stop;
float elapsedTime;

cudaEventCreate(&start);
cudaEventRecord(start, 0);

// call device kernel

cudaEventCreate(&stop);
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);

cudaEventElapsedTime(&elapsedTime, start, stop);
printf("elapsed time: %f ms\n", elapsedTime);
```

---

### Submission Requirements

Upload a single PDF including:

1. Your complete code
2. Transcript of running your program on `dgx.sdsu.edu` with different values of `N`
3. An answer to the following question:
   - **How long must an array be (as a power of 2) to see a runtime improvement of GPU device-accelerated reduction over CPU sequential summation?**
   - (i.e., for which value of `N` (a power of 2) will elapsed time for host (CPU) summation exceed elapsed time for device (GPU) reduction summation?)

---

### Final Output

Report the **speedup** as the ratio:

Speedup = time_sequential / time_parallel