#!/bin/bash

# Default values
BUILD_GPU="OFF"
BUILD_GPU_SIMD="OFF"
CLEAN=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --clean)
            CLEAN=true
            ;;
        -DBUILD_GPU=ON)
            BUILD_GPU="ON"
            ;;
        -DBUILD_GPU=OFF)
            BUILD_GPU="OFF"
            ;;
        -DBUILD_GPU_SIMD=ON)
            BUILD_GPU_SIMD="ON"
            ;;
        -DBUILD_GPU_SIMD=OFF)
            BUILD_GPU_SIMD="OFF"
            ;;
        *)
            echo "Usage: $0 [--clean] [-DBUILD_GPU=ON or -DBUILD_GPU=OFF] [-DBUILD_GPU_SIMD=ON or -DBUILD_GPU_SIMD=OFF]"
            exit 1
            ;;
    esac
done

# Clean the build directory if --clean is passed
if [ "$CLEAN" == true ]; then
    echo "Cleaning build directory..."
    rm -rf _build
fi

# Run cmake with the appropriate BUILD_GPU and BUILD_GPU_SIMD options
cmake -S . -B _build/ -DBUILD_GPU=$BUILD_GPU -DBUILD_GPU_SIMD=$BUILD_GPU_SIMD
find -type f -exec touch {} +
cmake --build _build/ --target install
