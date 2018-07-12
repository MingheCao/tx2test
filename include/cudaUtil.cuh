#ifndef CUDAUTIL_H
#define CUDAUTIL_H

//#define NDEBUG

#include <assert.h>
#include <stdlib.h>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include <cublas_v2.h>

#define CudaCheck(call) { call; cudaError_t __ABC123 = cudaGetLastError(); if(__ABC123 != cudaSuccess) { printf("%s:%d\t%s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(__ABC123)); } assert(__ABC123 == cudaSuccess); }
#define MAX(a, b) (((a) > (b)) ? (a) : (b) )
#define MIN(a, b) (((a) < (b)) ? (a) : (b) )

#ifndef PI
#define PI 3.141592653589793238
#endif

#ifdef __cplusplus
extern "C" {
#endif

float calcElapsedSec(struct timeval* start, struct timeval* stop);

int strToPositiveInteger(const char* str, size_t* value);

#ifdef __cplusplus
}
#endif

void printDevProp();

float* mallocOnGpu(const size_t N);

float* sendToGpu(const size_t N, const float* A);

void backToHost(float* device, float* host, const size_t N);

void print_matrix(const float *A, int rowBegin,int colBegin,
                  int rowEnd,int colEnd,int rows,int cols);

void gpuRowSum(cublasHandle_t &handle,const int rows,const int cols,const float *A,float *y);

size_t largestPowTwoLessThanEq(size_t N);

void calcDim(int N, cudaDeviceProp* devProp, dim3* block, dim3* grid);

void dimToConsole(dim3* block, dim3* grid);

/*
 * Methods to support vector array summation.
 *
 */
__device__ void devVecAdd(size_t pointDim, float* dest, float* src);

__device__ void devVecMinus(const size_t dim, float* z, const float* x, const float* y);

__global__ void kernElementWiseSum(const size_t numPoints, const size_t pointDim, float* dest, float* src);

__global__ void kernBlockWiseSum(const size_t numPoints, const size_t pointDim, float* dest);

__host__ void cudaArraySum(cudaDeviceProp* deviceProp, size_t numPoints, const size_t pointDim, float* device_A, cudaStream_t stream = 0);

/*
 * Methods to support scalar array maximum.
 *
 */
__global__ void kernElementWiseMax(const size_t numPoints, float* dest, float* src);

__global__ void kernBlockWiseMax(const size_t numPoints, float* dest);

__host__ void cudaArrayMax(cudaDeviceProp* deviceProp, size_t numPoints, float* device_A, cudaStream_t stream = 0);

__global__
void kernReduceSum(unsigned int num,const float *d_in, float *d_out);

__host__ __device__
void cudaReduceSum(dim3 grid,dim3 block,
             unsigned int num, const float* dIn,float *dOut);

__global__
void kernReduceMax(unsigned int num,const float *d_in, float *d_out);

__host__ __device__
void cudaReduceMax(dim3 grid,dim3 block,
                   unsigned int num, const float* dIn,float *dOut);
__global__
void kernReduceMin(unsigned int num,const float *d_in, float *d_out);

#endif
