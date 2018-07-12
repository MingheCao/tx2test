#include <errno.h>
#include "cudaUtil.cuh"

extern "C"
float calcElapsedSec(struct timeval* start, struct timeval* stop) {
    assert(start != NULL);
    assert(stop != NULL);

    float sec = stop->tv_sec - start->tv_sec;
    float usec = stop->tv_usec - start->tv_usec;
    if(stop->tv_sec > start->tv_sec) {
        if(start->tv_usec > stop->tv_usec) {
            sec = sec - 1;
            usec = 1e6 - start->tv_usec;
            usec += stop->tv_usec;
        }
    }

    return sec + (usec * 1e-6);
}

extern "C"
int strToPositiveInteger(const char* str, size_t* value) {
    assert(value != NULL);
    *value = 0;

    if(str == NULL || str[0] == '\0' || str[0] == '-') {
        return 0;
    }

    errno = 0;
    *value = strtoul(str, NULL, 10);
    if(errno != 0 || *value == 0) {
        return 0;
    }

    return 1;
}


void printDevProp()
{
	int deviceId;
	cudaDeviceProp deviceProp;

	CudaCheck(cudaGetDevice(&deviceId));
	CudaCheck(cudaGetDeviceProperties(&deviceProp, deviceId));

	printf("name: %s\n", deviceProp.name);
	printf("multiProcessorCount: %d\n", deviceProp.multiProcessorCount);
	printf("concurrentKernels: %d\n", deviceProp.concurrentKernels);

}

float* mallocOnGpu(const size_t N) {
	float* device_A;
	float ABytes = N * sizeof(float);
	CudaCheck(cudaMalloc(&device_A, ABytes));
	return device_A;
}

float* sendToGpu(const size_t N, const float* host) {
	float* device;
	const size_t hostBytes = N * sizeof(float);
	CudaCheck(cudaMalloc(&device, hostBytes));
	CudaCheck(cudaMemcpy(device, host, hostBytes, cudaMemcpyHostToDevice));
	return device;
}

void backToHost(float* device, float* host, const size_t N)
{
	CudaCheck(cudaMemcpy(host, device, N * sizeof(float), cudaMemcpyDeviceToHost));
	cudaFree(device);
}

 //Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const float *A, int rowBegin,int colBegin,
                  int rowEnd,int colEnd,int rows,int cols) {

     assert(rowBegin<=rows);
     assert(rowEnd<=rows);
     assert(colBegin<=cols);
     assert(colEnd<=cols);

     thrust::device_ptr<const float> ptr=thrust::device_pointer_cast(A);

     for(int i = rowBegin-1; i < rowEnd; ++i){
         for(int j = colBegin-1; j < colEnd; ++j){
             std::cout << *(ptr + j * rows + i) << "  ";
         }
         std::cout << std::endl;
     }
         std::cout << std::endl;
 }


size_t largestPowTwoLessThanEq(size_t N) {
    // Assigns the largest value (M = 2^n) < N to N and returns the residual.
    if(N == 0) {
        return 0;
    } // PC: N > 0

    size_t M = 1;
    while(M < N) {
        M *= 2;
    } // PC: M >= N

    if(M == N) {
        return M;
    } // PC: M > N

    return M / 2;
}

void calcDim(int N, cudaDeviceProp* devProp, dim3* block, dim3* grid) {
    assert(devProp != NULL);
    assert(block != NULL);
    assert(grid != NULL);

    // make a 2D grid of 1D blocks
    const int numThreadRows = 1;
    const int numThreadCols = devProp->maxThreadsPerBlock/2;
    block->x = numThreadCols, N;
    block->y = numThreadRows;

    const int numThreadsPerBlock = numThreadRows * numThreadCols;
    const int residualThreads = N % numThreadsPerBlock;
    int numBlocksPerGrid = (N - residualThreads) / numThreadsPerBlock;
    if(residualThreads > 0) {
        ++numBlocksPerGrid;
    }

    const int numBlockCols = min( numBlocksPerGrid, devProp->maxGridSize[0] );
    const int residualBlocks = numBlocksPerGrid % numBlockCols;
    int numBlockRows = (numBlocksPerGrid - residualBlocks) / numBlockCols;
    if(residualBlocks > 0) {
        ++numBlockRows;
    }

    grid->x = numBlockCols;
    grid->y = numBlockRows;

    assert(block->x * block->y * block->z > 0);
    assert(block->x * block->y * block->z <= devProp->maxThreadsPerBlock);

    assert(grid->x <= devProp->maxGridSize[0]);
    assert(grid->y <= devProp->maxGridSize[1]);
    assert(grid->z <= devProp->maxGridSize[2]);
}

void dimToConsole(dim3* block, dim3* grid) {
    assert(block != NULL);
    assert(grid != NULL);

    printf("block: (%d, %d, %d)\n", block->x, block->y, block->z);
    printf("grid: (%d, %d, %d)\n", grid->x, grid->y, grid->z);
}


// ----------------------------------------------------------------------------
// Find sum of a vector array
// ----------------------------------------------------------------------------

__device__ void devVecAdd(size_t pointDim, float* dest, float* src) {
	for(size_t i = 0; i < pointDim; ++i) {
		dest[i] += src[i];
	}
}

/*
 * Computes \sum_{i}^{N} x_i y_i for x, y \in \mathbb{R}^{N}.
 */
__device__ float devVecDot(const size_t N, const float *x, const float *y) {
    assert(N > 0);
    assert(x != NULL);
    assert(y != NULL);
    // x == y allowed

    float sum = 0;
    for (size_t i = 0; i < N; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

/*
 * Computes z_{i} \gets x_{i} - y_{i} for x, y \in \mathbb{R}^N.
 */
__device__ void devVecMinus(const size_t dim, float *z, const float *x, const float *y) {
    assert(dim > 0);
    assert(x != NULL);
    assert(y != NULL);
    // x == y allowed

    for (size_t i = 0; i < dim; ++i) {
        z[i] = x[i] - y[i];
    }
}


__global__ void kernElementWiseSum(const size_t numPoints, const size_t pointDim, float* dest, float* src) {
	// Called to standardize arrays to be a power of two

	// Assumes a 2D grid of 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;

	if(i < numPoints) {
		devVecAdd(pointDim, &dest[i * pointDim], &src[i * pointDim]);
	}
}

__global__ void kernBlockWiseSum(const size_t numPoints, const size_t pointDim, float* dest) {
	// Assumes a 2D grid of 1024x1 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;

	// call repeatedly for each dimension where dest is assumed to begin at dimension d

	__shared__ float blockSum[1024];

	if(threadIdx.x >= numPoints) {
		blockSum[threadIdx.x] = 0;
	} else {
		blockSum[threadIdx.x] = dest[i * pointDim];
	}

	__syncthreads();

	// Do all the calculations in block shared memory instead of global memory.
	for(int s = blockDim.x / 2; threadIdx.x < s; s /= 2) {
		blockSum[threadIdx.x] += blockSum[threadIdx.x + s];
		__syncthreads();
	}

	if(threadIdx.x == 0) {
		// Just do one global write
		dest[i * pointDim] = blockSum[0];
	}	
}

__global__ void kernMoveMem(const size_t numPoints, const size_t pointDim, const size_t s, float* A) {
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;

	// Before
	// [abc......] [def......] [ghi......] [jkl......]

	// shared memory
	// [adgj.....]

	// After
	// [a..d..g..] [j........] [ghi......] [.........]

	__shared__ float mem[1024];
	mem[threadIdx.x] = A[s * i * pointDim];
	__syncthreads();
	A[i * pointDim] = mem[threadIdx.x];
}

__global__ void kernBlockWiseSum2(const size_t numPoints, const size_t pointDim, float* dest) {
    // Assumes a 2D grid of 1024x1 1D blocks
    int b = blockIdx.y * gridDim.x + blockIdx.x;
    int i = b * blockDim.x + threadIdx.x;

    // call repeatedly for each dimension where dest is assumed to begin at dimension d

    __shared__ float blockSum[1024];

    if(threadIdx.x >= numPoints) {
        blockSum[threadIdx.x] = 0;
    } else {
        blockSum[threadIdx.x] = dest[i * pointDim];
    }

    __syncthreads();

    // Do all the calculations in block shared memory instead of global memory.
    for(int s = blockDim.x / 2; threadIdx.x < s; s /= 2) {
        blockSum[threadIdx.x] += blockSum[threadIdx.x + s];
        __syncthreads();
    }

    if(threadIdx.x == 0) {
        // Just do one global write
        dest[i * pointDim] = blockSum[0];
    }
}

__host__
void cudaArraySum(cudaDeviceProp* deviceProp, size_t numPts, const size_t dim, float* d_in, cudaStream_t stream) {
	assert(deviceProp != NULL);
	assert(numPts > 0);
	assert(dim > 0);
	assert(d_in != NULL);

	size_t M = largestPowTwoLessThanEq(numPts);
	if(M != numPts) {
		dim3 block , grid;
		calcDim(numPts - M, deviceProp, &block, &grid);
		kernElementWiseSum<<<grid, block>>>(
                numPts - M, dim, d_in, d_in + M * dim
		);
        numPts = M;
	}

	while(numPts > 1) {
		dim3 block, grid;
		calcDim(numPts, deviceProp, &block, &grid);

		for(size_t d = 0; d < dim; ++d) {
			kernBlockWiseSum<<<grid, block>>>(numPts, dim, d_in + d);
			
			if(numPts > block.x) {
				dim3 block2, grid2;
				calcDim(grid.x, deviceProp, &block2, &grid2);
				kernMoveMem<<<grid2, block2>>>(numPts, dim, block.x, d_in + d);
			}
		}

        numPts /= block.x;
	}
}

// ----------------------------------------------------------------------------
// Find maximum of a scalar array
// ----------------------------------------------------------------------------

__global__ void kernElementWiseMax(const size_t numPoints, float* dest, float* src) {
	// Called to standardize arrays to be a power of two

	// Assumes a 2D grid of 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;

	if(i < numPoints) {
		if(dest[i] < src[i]) {
			dest[i] = src[i];
		}
	}
}

__global__ void kernBlockWiseMax(const size_t numPoints, float* dest) {
	// Assumes a 2D grid of 1024x1 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;

	__shared__ float blockMax[1024];

	if(threadIdx.x >= numPoints) {
		blockMax[threadIdx.x] = -INFINITY;
	} else {
		blockMax[threadIdx.x] = dest[i];
	}

	__syncthreads();

	// Do all the calculations in block shared memory instead of global memory.
	for(int s = blockDim.x / 2; threadIdx.x < s; s /= 2) {
		if(blockMax[threadIdx.x] < blockMax[threadIdx.x + s]) {
			blockMax[threadIdx.x] = blockMax[threadIdx.x + s];
		}
		__syncthreads();
	}

	if(threadIdx.x == 0) {
		// Just do one global write
		dest[i] = blockMax[0];
	}
}

__host__ void cudaArrayMax(cudaDeviceProp* deviceProp, size_t numPoints, float* device_A, cudaStream_t stream) {
	assert(deviceProp != NULL);
	assert(numPoints > 0);
	assert(device_A != NULL);

	size_t M = largestPowTwoLessThanEq(numPoints);
	if(M != numPoints) {
		dim3 block , grid;
		calcDim(M, deviceProp, &block, &grid);
		kernElementWiseMax<<<grid, block>>>(
			numPoints - M, device_A, device_A + M
		);
		numPoints = M;
	}

	while(numPoints > 1) {
		dim3 block, grid;
		calcDim(numPoints, deviceProp, &block, &grid);

		kernBlockWiseMax<<<grid, block>>>(numPoints, device_A);
		
		if(numPoints > block.x) {
			dim3 block2, grid2;
			calcDim(grid.x, deviceProp, &block2, &grid2);
			kernMoveMem<<<grid2, block2>>>(numPoints, 1, block.x, device_A);
		}

		numPoints /= block.x;
	}
}

__global__
void kernReduceSum(unsigned int num,const float *d_in, float *d_out)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    volatile __shared__ float sdata[1024];
    sdata[tid] = 0;

    if (idx + blockDim.x< num) {
        sdata[tid] = d_in[idx] + d_in[idx+blockDim.x];
    }
    else if(idx<num){
        sdata[tid] = d_in[idx];
    }
    __syncthreads();

    if (blockDim.x >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockDim.x >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) {
        if (blockDim.x >= 64) sdata[tid] += sdata[tid + 32];
        if (blockDim.x >= 32) sdata[tid] += sdata[tid + 16];
        if (blockDim.x >= 16) sdata[tid] += sdata[tid + 8];
        if (blockDim.x >= 8) sdata[tid] += sdata[tid + 4];
        if (blockDim.x >= 4) sdata[tid] += sdata[tid + 2];
        if (blockDim.x >= 2) sdata[tid] += sdata[tid + 1];
    }
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

__host__ __device__
void cudaReduceSum(dim3 grid,dim3 block,
                  unsigned int num, const float* dIn,float *dOut)
{
    kernReduceSum<<<(grid.x+1)/2,block>>>(num,dIn,dOut);
    if((grid.x+1)/2!=1) {
        const int n = (grid.x / 2) + (grid.x % 2);
        kernReduceSum << < 1, block >> > (n, dOut, dOut);
    }
}

__global__
void kernReduceMax(unsigned int num,const float *d_in, float *d_out)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    volatile __shared__ float sdata[1024];
    sdata[tid] = -INFINITY;

    if (idx + blockDim.x< num) {
        sdata[tid] = max(d_in[idx],d_in[idx+blockDim.x]);
    }
    else if(idx < num){
        sdata[tid] = d_in[idx];
    }
    __syncthreads();

    if (blockDim.x >= 1024) { if (tid < 512) { sdata[tid] = max(sdata[tid], sdata[tid + 512]); } __syncthreads(); }
    if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] = max(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
    if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] = max(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
    if (blockDim.x >= 128) { if (tid < 64) { sdata[tid] = max(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
    if (tid < 32) {
        if (blockDim.x >= 64) sdata[tid] = max(sdata[tid], sdata[tid + 32]);
        if (blockDim.x >= 32) sdata[tid] = max(sdata[tid], sdata[tid + 16]);
        if (blockDim.x >= 16) sdata[tid] = max(sdata[tid], sdata[tid + 8]);
        if (blockDim.x >= 8) sdata[tid] = max(sdata[tid], sdata[tid + 4]);
        if (blockDim.x >= 4) sdata[tid] = max(sdata[tid], sdata[tid + 2]);
        if (blockDim.x >= 2) sdata[tid] = max(sdata[tid], sdata[tid + 1]);
    }
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

__host__ __device__
void cudaReduceMax(dim3 grid,dim3 block,
                   unsigned int num, const float* dIn,float *dOut)
{
    kernReduceMax<<<(grid.x+1)/2,block>>>(num,dIn,dOut);
    if((grid.x+1)/2!=1) {
        const int n = (grid.x / 2) + (grid.x % 2);
        kernReduceMax << < 1, block >> > (n, dOut, dOut);
    }
}

__global__
void kernReduceMin(unsigned int num,const float *d_in, float *d_out)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    volatile __shared__ float sdata[1024];
    sdata[tid] = INFINITY;

    if (idx + blockDim.x< num) {
        sdata[tid] = min(d_in[idx],d_in[idx+blockDim.x]);
    }
    else if(idx < num){
        sdata[tid] = d_in[idx];
    }
    __syncthreads();

    if (blockDim.x >= 1024) { if (tid < 512) { sdata[tid] = min(sdata[tid], sdata[tid + 512]); } __syncthreads(); }
    if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] = min(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
    if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] = min(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
    if (blockDim.x >= 128) { if (tid < 64) { sdata[tid] = min(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
    if (tid < 32) {
        if (blockDim.x >= 64) sdata[tid] = min(sdata[tid], sdata[tid + 32]);
        if (blockDim.x >= 32) sdata[tid] = min(sdata[tid], sdata[tid + 16]);
        if (blockDim.x >= 16) sdata[tid] = min(sdata[tid], sdata[tid + 8]);
        if (blockDim.x >= 8) sdata[tid] = min(sdata[tid], sdata[tid + 4]);
        if (blockDim.x >= 4) sdata[tid] = min(sdata[tid], sdata[tid + 2]);
        if (blockDim.x >= 2) sdata[tid] = min(sdata[tid], sdata[tid + 1]);
    }
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}


// ----------------------------------------------------------------------------
// Archived Code
// ----------------------------------------------------------------------------

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
//void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
//    int lda=m,ldb=k,ldc=m;
//    const float alf = 1;
//    const float bet = 0;
//    const float *alpha = &alf;
//    const float *beta = &bet;
//
//    // Create a handle for CUBLAS
//    cublasHandle_t handle;
//    cublasCreate(&handle);
//
//    // Do the actual multiplication
//    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
//
//    // Destroy the handle
//    cublasDestroy(handle);
//}

//// Multiply the arrays A and B on GPU and save the result in C
//// C(m,n) = A(m,k) * B(k,n)
//void gpu_blas_mmul(cublasHandle_t &handle,const float *A, const float *B, float *C, const int m, const int k, const int n) {
//    int lda=m,ldb=k,ldc=m;
//    const float alf = 1;
//    const float bet = 0;
//    const float *alpha = &alf;
//    const float *beta = &bet;
//
//    // Do the actual multiplication
//    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
//
//}
//
//void gpuRowSum(cublasHandle_t &handle,const int rows,const int cols,const float *A,float *y){
//    const int lda=rows;
//    const float alf=1.0;
//    const float bet=0;
//    const float *alpha=&alf;
//    const float *beta=&bet;
//
//    thrust::device_vector<float> d_tx(cols,1.0);
//    float *d_x = thrust::raw_pointer_cast(d_tx.data());
//
//    CudaCheck(cublasSgemv(handle,CUBLAS_OP_N,rows,cols,alpha,A,lda,d_x,1,beta,y,1));
//
//}
//
//
//int cuSolverChol(cusolverDnHandle_t &handle,float *Sigma,const size_t dim){
//
//    cusolverStatus_t cusolverStatus;
//
//    float *Work;
//    int *d_info,Lwork,h_info;
//
//    CudaCheck(cusolverDnCreate(&handle));
//    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
//
//    CudaCheck(cudaMalloc((void**)&d_info,sizeof(int)));
//
//    cusolverStatus = cusolverDnSpotrf_bufferSize(
//            handle,uplo,dim, Sigma, dim,&Lwork);
//
//    CudaCheck(cudaMalloc((void**)&Work,Lwork*sizeof(float)));
//
//    cusolverStatus=cusolverDnSpotrf(handle,uplo,dim,
//                                    Sigma,dim,Work,Lwork,d_info);
//
//    CudaCheck(cudaMemcpy(&h_info,
//                         d_info,
//                         sizeof(int),
//                         cudaMemcpyDeviceToHost));
//    return h_info;
//}
//
//// kmeans CUDA
//// In the assignment step, each point (thread) computes its distance to each
//// cluster centroid and adds its x and y values to the sum of its closest
//// centroid, as well as incrementing that centroid's count of assigned points.
//__global__ void kernKmeansClusterAoS(const size_t dim,
//                                     const size_t pts,
//                                     const float *data_,
//                                     const size_t k,
//                                     const float *means_,
//                                     float *new_sums_,
//                                     int *counts) {
//    // copy means_ into shared memory.
//    extern __shared__ float shared_means[];
//
//    const int i = blockIdx.x * blockDim.x + threadIdx.x;
//    if (i >= pts) return;
//
//    // Let the first k threads copy over the cluster means.
////    if (threadIdx.x < k*dim) {
////            shared_means[threadIdx.x] = means_[threadIdx.x];
////    }
//
//    // Wait for those k threads.
//    __syncthreads();
//
//    float best_distance = FLT_MAX;
//    float distance = 0;
//    int best_cluster = 0;
//    for (size_t cluster = 0; cluster < k; ++cluster) {
//        for (size_t d = 0; d < dim; ++d) {
//            distance += (data_[i * dim + d] - means_[cluster * dim + d]) *
//                        (data_[i * dim + d] - means_[cluster * dim + d]);
//        }
//        if (distance < best_distance) {
//            best_distance = distance;
//            best_cluster = cluster;
//        }
//        distance = 0;
//    }
//
//    for (size_t d = 0; d < dim; ++d) {
//        atomicAdd(&new_sums_[best_cluster * dim + d], data_[i * dim + d]);
//    }
//    atomicAdd(&counts[best_cluster], 1);
//}
//
//__global__ void kernKmeansNewMuAoS(size_t dim,
//                                   float *means_,
//                                   const float *new_sum_,
//                                   const int *counts) {
//    const int cluster = threadIdx.x;
//    const int count = max(1, counts[cluster]);
//    for (size_t d = 0; d < dim; ++d) {
//        means_[cluster * dim + d] = new_sum_[cluster * dim + d] / count;
//    }
//}
//
//void gpuKeans(
//        const size_t pts,const size_t dim,const size_t k,
//        const thrust::host_vector<float> &h_X,
//        thrust::host_vector<float> &h_means,
//        const size_t maxItr) {
//
//    assert(dim > 0 && dim <= 1024);
//    assert(k > 0 && k <= 1024);
//    assert(k > dim);
//    assert(maxItr >= 1);
//
//    thrust::device_vector<float> d_X=h_X;
//    thrust::device_vector<float> d_means=h_means;
//    thrust::device_vector<float> d_sums(k * dim);
//    thrust::device_vector<int> d_counts(k, 0);
//
//    const int threads = 1024;
//    const int blocks = (pts + threads - 1) / threads;
//    const int sharedMem = k * dim;
//
//    for (size_t i = 0; i < maxItr; ++i) {
//        thrust::fill(d_sums.begin(), d_sums.end(), 0);
//        thrust::fill(d_counts.begin(), d_counts.end(), 0);
//
//        kmeansAssignClusters << < blocks, threads, sharedMem >> > (
//                dim, pts,
//                        thrust::raw_pointer_cast(d_X.data()),
//                        k,
//                        thrust::raw_pointer_cast(d_means.data()),
//                        thrust::raw_pointer_cast(d_sums.data()),
//                        thrust::raw_pointer_cast(d_counts.data()));
//
//        cudaDeviceSynchronize();
//
//        kernKmeansClusterAoS << < 1, k >> > (dim,
//                thrust::raw_pointer_cast(d_means.data()),
//                thrust::raw_pointer_cast(d_sums.data()),
//                thrust::raw_pointer_cast(d_counts.data()));
//
//        cudaDeviceSynchronize();
//    }
//    h_means=d_means;
//}

