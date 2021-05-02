# CULiP - CUDA Library Profiler

CULiP is a library for profiling the execution time of CUDA official library functions

## Supported Libraries and functions

- cuBLAS
  - `cublasSgemm`

## Dependencies
- CUDA >= 10.0
- CMake >= 3.18
- C++ >= 14

## Usage

1. Clone CULiP
```bash
git clone https://github.com/enp1s0/CULiP
cd CULiP
```

2. Build
```bash
mkdir build
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install/CULiP
```

3. Link
```
nvcc main.cu -L/path/to/install/CULiP -lculip_cublas -lcublas -o foo.bar
```

4. Set an environment variable `CULIP_CUBLAS_LIB_PATH` before executing the application
```bash
export CULIP_CUBLAS_LIB_PATH=/nfs/shared/packages/x86_64/cuda/cuda-11.3/lib64/libcublas.so.11
./foo.bar
```

The execution time is printed on stdout.
```
[CULiP Result][cublasSgemm_v2-m1024-n1024-k1024] 155182ns
```

## How it works

<img alt='culip_how_it_works' src='./docs/CULiP.svg'>

## License
MIT
