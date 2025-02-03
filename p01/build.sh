#!/bin/bash

rm -rf _build/
cmake -S src/ -B _build/
find -type f -exec touch {} +
cmake --build _build/