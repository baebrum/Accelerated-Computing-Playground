#!/bin/bash
set -euo pipefail

BUILD_DIR="_build"
PARALLEL_JOBS=${VCPKG_MAX_CONCURRENCY:-32}

# setup vcpkg
./setup_vcpkg.sh

source ~/.bashrc

if [ -z "${VCPKG_ROOT:-}" ]; then
    echo "Error: VCPKG_ROOT is not set."
    exit 1
fi

if [ "${1:-}" = "clean" ]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"

echo "Updating file timestamps before configuring..."
find . -type f -exec touch {} +

echo "Configuring project..."
cmake -S . -B "$BUILD_DIR" -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"

echo "Updating file timestamps after configuring..."
find . -type f -exec touch {} +

echo "Building project ($PARALLEL_JOBS parallel jobs..."
cmake --build "$BUILD_DIR" --parallel "$PARALLEL_JOBS" -- -Wno-time

echo "Build complete!"
