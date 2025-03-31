# GPU Accelerated 2D Discrete Convolution with CUDA

Explore **GPU-accelerated 2D discrete convolution** using **NVIDIA CUDA** to convolve a PNG image with a **Sobel 5x5 horizontal convolution kernel** for edge detection.

---

## Step 1: Choose an Image

Select a PNG image to work with. For example, the `512 x 384` RGB image `peppers.png` is included in the **MATLAB Example Image Data Set**:

**Image Info:**
- **File:** `peppers.png`
- **Size:** 281 KB
- **Resolution:** 512 x 384

---

## Step 2: Convert Image to ASCII Data with MATLAB

You can use MATLAB to generate an ASCII file (`peppers.dat`) of one-byte unsigned integer pixel values in **row-major format**:

```matlab
I = imread('peppers.png');
[m,n,c] = size(I);
fileID = fopen('peppers.dat','w');
for i = 1:m
    for j = 1:n
        for k = 1:c
            fprintf(fileID,'%d\n', I(i,j,k));
        end
    end
end
fclose(fileID);
```

---

## Step 3: Sobel 5x5 Horizontal Convolution Kernel

```c
int hostMask[5][5] = {
    {  2,  2,  4,  2,  2 },
    {  1,  1,  2,  1,  1 },
    {  0,  0,  0,  0,  0 },
    { -1, -1, -2, -1, -1 },
    { -2, -2, -4, -2, -2 }
};
```

---

## Step 4: Example CUDA Host Driver Code

Here’s a full C/CUDA snippet that:
- Reads the image
- Transfers data to GPU
- Launches a convolution kernel
- Retrieves and saves the output

```c
int main(int argc, char *argv[])
{
  unsigned int *hostInputImage;
  int *hostOutputImage;
  unsigned int inputLength = 589824; // 384 * 512 * 3

  hostInputImage = (unsigned int *)malloc(inputLength * sizeof(unsigned int));
  hostOutputImage = (int *)malloc(inputLength * sizeof(int));

  FILE *f = fopen("peppers.dat", "r");
  unsigned int pixelValue, i = 0;
  while (!feof(f) && i < inputLength) {
    fscanf(f, "%d", &pixelValue);
    hostInputImage[i++] = pixelValue;
  }
  fclose(f);

  int maskRows = 5, maskColumns = 5;
  int imageChannels = 3, imageWidth = 512, imageHeight = 384;

  int hostMask[5][5] = {
    {  2,  2,  4,  2,  2 },
    {  1,  1,  2,  1,  1 },
    {  0,  0,  0,  0,  0 },
    { -1, -1, -2, -1, -1 },
    { -2, -2, -4, -2, -2 }
  };

  unsigned int *deviceInputImage;
  int *deviceOutputImage;
  int *deviceMask;

  cudaMalloc((void **)&deviceInputImage, inputLength * sizeof(int));
  cudaMalloc((void **)&deviceOutputImage, inputLength * sizeof(int));
  cudaMalloc((void **)&deviceMask, maskRows * maskColumns * sizeof(int));

  cudaMemcpy(deviceInputImage, hostInputImage, inputLength * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMask, hostMask, maskRows * maskColumns * sizeof(int), cudaMemcpyHostToDevice);

  dim3 dimGrid(ceil((float)imageWidth / TILE_WIDTH), ceil((float)imageHeight / TILE_WIDTH));
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

  convolution<<<dimGrid, dimBlock>>>(
    deviceInputImage, deviceMask, deviceOutputImage,
    imageChannels, imageWidth, imageHeight
  );

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaMemcpy(hostOutputImage, deviceOutputImage, inputLength * sizeof(int), cudaMemcpyDeviceToHost);

  f = fopen("peppers.out", "w");
  for (int i = 0; i < inputLength; ++i)
    fprintf(f, "%d\n", hostOutputImage[i]);
  fclose(f);

  cudaFree(deviceInputImage);
  cudaFree(deviceOutputImage);
  cudaFree(deviceMask);

  free(hostInputImage);
  free(hostOutputImage);
  return 0;
}
```

---

## Step 5: Implement the CUDA Kernel

Implement the 2D CUDA convolution kernel.

---

## Step 6: Plot the Output in MATLAB

You can use the following MATLAB code to visualize the output file (`peppers.out`) as a **3D surface plot**:

```matlab
clc
close all
m = 384; n = 512; c = 3;
P = zeros(m,n,c);
fid = fopen('peppers.out','r');
i = 1; j = 1; k = 1;
for l = 1:m*n*c
    p = fscanf(fid,'%d', 1);
    P(i,j,k) = p;
    if mod(k,c) == 0
        k = 1; j = j + 1;
        if j == n + 1
            j = 1; i = i + 1;
            if i == m + 1
                break;
            end
        end
    else
        k = k + 1;
    end
end
fclose(fid);

[M, N] = meshgrid(1:n, 1:m);
c = 1;
surf(M, N, P(:,:,c));
xlabel('Image Width')
ylabel('Image Height')
title(['Sobel 5x5 Convolution Output, channel ', num2str(c)])
grid on
grid minor
set(gcf, 'color', 'w')
axis equal
colorbar
shading interp
rotate3d
view(0, 90)
set(gcf, 'position', [10, 10, 1280, 768])
figure(1)
```
