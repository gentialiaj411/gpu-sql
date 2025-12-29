@echo off
mkdir build 2>nul
cd build
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler"
cmake --build .
cd ..

