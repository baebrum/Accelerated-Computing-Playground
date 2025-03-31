cmake_minimum_required(VERSION 3.18)
project(MyCudaThrustProject LANGUAGES CUDA CXX)

# Set CUDA and C++ Standards
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# find Stb
find_package(Stb REQUIRED)

# Target NVIDIA A100 GPU
set(CMAKE_CUDA_ARCHITECTURES 80)

# add executable
add_executable(sobel_convolution src/main.cu)

# Include stb_image from vcpkg installation
target_include_directories(sobel_convolution PRIVATE
    ${Stb_INCLUDE_DIR}
)
