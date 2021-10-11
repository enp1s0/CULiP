<img alt='culip_icon' src='./docs/CULiP-icon.svg' width=100>

# CULiP - CUDA Library Profiler

CULiP is a library for profiling the execution time of CUDA official library functions

## Supported libraries and functions

- cuBLAS
	- GEMM
		- `cublasDgemm`
		- `cublasSgemm`
		- `cublasHgemm`
		- `cublasCgemm`
		- `cublasZgemm`
		- `cublasGemmEx`
		- `cublasDgemmBatched`
		- `cublasSgemmBatched`
		- `cublasHgemmBatched`
		- `cublasCgemmBatched`
		- `cublasZgemmBatched`
		- `cublasGemmExBatched`
	- GEMV
		- `cublasDgemm`
		- `cublasSgemm`
		- `cublasCgemm`
		- `cublasZgemm`
	- GBMV
		- `cublasDgbmm`
		- `cublasSgbmm`
		- `cublasCgbmm`
		- `cublasZgbmm`

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
make
make install
```

3. Link the CULiP libraries **before linking libcublas and so on**
```
nvcc main.cu -L/path/to/install/CULiP/lib -lculip_cublas -lcublas -o foo.bar
```

4. Execute the application
```bash
./foo.bar
```

Then the execution time is printed on stdout.
```
[CULiP Result][cublasSgemm_v2-m1024-n1024-k1024] 155182ns
```

To disable profiling at runtime, define an environment variable `CULIP_PROFILING_CUBLAS_DISABLE`.
```bash
# Disable cuBLAS profiling
export CULIP_PROFILING_CUBLAS_DISABLE=1

# Enable cuBLAS profiling
export CULIP_PROFILING_CUBLAS_DISABLE=0
# or
unset CULIP_PROFILING_CUBLAS_DISABLE
```

### In the case that you can not change set up compiler options
You can make a symbolic link for libculip_cublas.so named libcublas.so so that CULiP library is loaded instead of real cuBLAS library.
Then set an environment variable `CULIP_CUBLAS_LIB_PATH` so that CULiP finds the original cuBLAS library.

```bash
mkdir lib
ln -s /path/to/install/CULiP/lib/libculip_cublas.so lib/libcublas.so
export LD_LIBRARY_PATH="./lib:$LD_LIBRARY_PATH"
export CULIP_CUBLAS_LIB_PATH=$(dirname $(which nvcc))/../lib64/libcublas.so
./a.out
```

## Profiling control API

CULiP provides profiling control API.
By default, all profiling is enabled.

```cpp
// nvcc -I/path/to/install/CULiP/include ...
#include <CULiP/cublas.hpp>

// Disable profiling of all functions
CULiP_profiling_cublas_disable_all();

// Enable profiling of all functions
CULiP_profiling_cublas_enable_all();

// Disable profiling of a function (e.g. `cublasSgemm`)
CULiP_profiling_cublas_disable(CULiP_cublasSgemm);

// Enable profiling of a function (e.g. `cublasSgemm`)
CULiP_profiling_cublas_enable(CULiP_cublasSgemm);
```

## How it works

<img alt='culip_how_it_works' src='./docs/CULiP.svg'>

## Aggregator

CULiP also provides result aggregator `CULiP_aggregator`.

```bash
./foo.bar > result.log
cat result.log | CULiP_aggregator
```

Aggregation result:
```
#####################################
#       CULiP Profiling Result      #
#  https://github.com/enp1s0/CULiP  #
#####################################

- cublasDgemm_v2 : [143904971774 ns; 1.439050e+02 s; 99.40%]
              params    count                   sum          avg          max          min
  m1048576-n128-k128       96  41173.198ms( 28.61%)    428.887ms    433.445ms    427.390ms
  m128-n128-k1048576       96  29626.117ms( 20.59%)    308.605ms    313.811ms    306.485ms
   m524288-n128-k128       96  20592.321ms( 14.31%)    214.503ms    219.402ms    213.780ms
   m128-n128-k524288       96  12510.157ms(  8.69%)    130.314ms    134.215ms    128.985ms
   m262144-n128-k128       96  10294.148ms(  7.15%)    107.231ms    110.439ms    106.706ms
   m128-n128-k262144       96   6687.382ms(  4.65%)     69.660ms     70.357ms     68.977ms
   m128-n128-k131072       96   6627.523ms(  4.61%)     69.037ms     70.009ms     67.999ms
   m131072-n128-k128       96   5152.191ms(  3.58%)     53.669ms     55.125ms     53.376ms
    m128-n128-k65536       96   3028.057ms(  2.10%)     31.542ms     33.302ms     31.127ms
...

- cublasSgemm_v2 : [869182648 ns; 8.691826e-01 s;  0.60%]
              params    count                   sum          avg          max          min
  m128-n128-k1048576       64    180.648ms( 20.78%)      2.823ms      3.347ms      2.503ms
   m128-n128-k524288       64     93.173ms( 10.72%)      1.456ms      1.658ms      1.276ms
  m1048576-n128-k128       32     75.253ms(  8.66%)      2.352ms      2.359ms      2.344ms
   m128-n128-k262144       64     46.822ms(  5.39%)      0.732ms      0.828ms      0.658ms
    m1048576-n96-k32       32     44.452ms(  5.11%)      1.389ms      1.397ms      1.384ms
   m524288-n128-k128       32     37.917ms(  4.36%)      1.185ms      1.191ms      1.179ms
    m32-n96-k1048576       32     36.564ms(  4.21%)      1.143ms      1.190ms      1.093ms
    m1048544-n64-k32       32     31.889ms(  3.67%)      0.997ms      1.003ms      0.992ms
   m128-n128-k131072       64     24.763ms(  2.85%)      0.387ms      0.424ms      0.349ms
    m32-n64-k1048544       32     23.477ms(  2.70%)      0.734ms      0.749ms      0.716ms
...
```

## License
MIT
