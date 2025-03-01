cmake_minimum_required(VERSION 3.12)

project(JacobiIteration)

# Define the build options
option(BUILD_GPU "Build for GPU" OFF)
option(BUILD_GPU_SIMD "Enable SIMD for GPU offloading" OFF)

# Set the compiler to nvc++ (for GPU builds) or another one for CPU
if(BUILD_GPU)

    # GPU build (using nvc++ for NVIDIA GPUs)
    set(CMAKE_CXX_COMPILER nvc++)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mp=gpu -gpu=cc80")

    # Enable SIMD if requested
    if(BUILD_GPU_SIMD)
        add_executable(jacobi_gpu_simd main.cpp)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_BUILD_GPU_SIMD")
        target_compile_definitions(jacobi_gpu_simd PRIVATE BUILD_GPU BUILD_GPU_SIMD)
        install(TARGETS jacobi_gpu_simd DESTINATION ${CMAKE_SOURCE_DIR})
    else()
        add_executable(jacobi_gpu main.cpp)
        target_compile_definitions(jacobi_gpu PRIVATE BUILD_GPU)
        install(TARGETS jacobi_gpu DESTINATION ${CMAKE_SOURCE_DIR})
    endif()

else()
    add_executable(jacobi_cpu main.cpp)

    find_package(OpenMP REQUIRED)
    target_link_libraries(jacobi_cpu PUBLIC OpenMP::OpenMP_CXX)
    install(TARGETS jacobi_cpu DESTINATION ${CMAKE_SOURCE_DIR})
endif()
