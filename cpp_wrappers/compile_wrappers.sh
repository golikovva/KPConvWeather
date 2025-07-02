#!/bin/bash

# Compile cpp subsampling
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR/cpp_subsampling"
python3 setup.py build_ext --inplace
cd "$DIR"

# Compile cpp neighbors
cd "$DIR/cpp_neighbors"
python3 setup.py build_ext --inplace
cd "$DIR"

