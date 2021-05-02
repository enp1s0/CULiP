# CULiP - CUDA Library Profiler

CULiP is a library for profiling running time breakdown of CUDA official libraries

## Supported Libraries and functions

- cuBLAS
  - `cublasSgemm`

## Usage
All you need is building this library and link it before linking libcublas and so on.

1. Clone CULiP
```bash
git clone https://github.com/enp1s0/CULiP
cd CULiP
```

2. Build
```bash
mkdir build
cmake .. --DCMAKE_INSTALL_PREFIX=/path/to/install/CULiP
```

3. Link
```
nvcc main.cu -L/path/to/install/CULiP -lculip_cublas -lcublas
```

## License
MIT
