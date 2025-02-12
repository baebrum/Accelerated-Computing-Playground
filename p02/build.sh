#!/bin/bash

rm -rf _build/
cmake -S . -B _build/
find -type f -exec touch {} +
cmake --build _build/