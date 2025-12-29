# GPU-SQL: GPU-Accelerated SQL Query Engine

A GPU-accelerated SQL filter engine using CUDA.

## Building

```powershell
mkdir build && cd build
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler"
cmake --build .
.\gpu_sql_benchmark.exe
```

## Project Structure

```
src/
├── core/           # Columnar storage (types, column, table)
├── cpu/            # CPU baseline implementation
├── kernels/        # CUDA kernels (filter, utils)
└── main.cpp        # Benchmark runner
```
