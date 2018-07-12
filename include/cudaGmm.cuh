#ifndef CUDAGMM_HU
#define CUDAGMM_HU

#include <stdlib.h>

#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define mat3d(r, c, d) ((r)*(c)*(d))
#define mat3dIdx(i, j, d, r, c) ((r)*(c)*(d))+(((j)*(r))+(i))

#define maxSharedMem  32 *1024;

__global__ void kmeansUpdateMean(size_t dim, float* means_, const float* new_sum_, const int* counts);

__global__ void kmeansAssignClusters(const size_t dim,
									 const size_t pts,
									 const float* data_,
									 const size_t k,
									 const float*  means_,
									 float*  new_sums_,
									 int*  counts);

void gpuKeans(
        const size_t pts,const size_t dim,const size_t k,
        const thrust::host_vector<float> &h_X,
        thrust::host_vector<float> &h_means,
        const size_t maxItr);

__device__ void devVecMinus(const size_t N, float* z, float* x, const float* y);

__device__ float devVecDot(const size_t N, const float* x, const float* y);

__device__ void devSolveLowerTri(const size_t N, const float* L, float* x, const float* b);

__device__ void devSolveLowerTriT(const size_t N, const float* L, float* x, const float* b);

__global__ void kernLogMVNormDist(
		const size_t pts, const size_t dim,
		const float* X, const float* mu,
		const float* sigmaL, float* logProb
);

__global__ void kernCalcLogLikelihoodAndGammaNK(
	const size_t numPoints, const size_t numComponents,
	const float* logpi, float* logPx, float* loggamma
);


__global__ void kernCalcMu(
	const size_t numPoints, const size_t pointDim,
	const float* X, const float* loggamma, const float* logGammaK,
	float* dest
);

__global__ void kernCalcSigma(
	const size_t numPoints, const size_t pointDim,
	const float* X, const float* mu, const float* loggamma, const float* logGammaK,
	float* dest
);

__global__ void kernPrepareCovariances(
	const size_t numComponents, const size_t pointDim,
	float* Sigma, float* SigmaL
);

struct eliUnspt
{
	const float a;

	eliUnspt(float _a) : a(_a) {}

	__host__ __device__
	float operator()(const float& x) const
	{
		return x<=a;
	}
};

void cudaEM(
		const unsigned int pts,const unsigned int dim,const unsigned int comps,
		const thrust::host_vector<float> &h_X,
		thrust::host_vector<float> &h_Pi,
		thrust::host_vector<float> &h_Mu,
		thrust::host_vector<float> &h_Sigma,
		const float tol,
		const unsigned int maxItr);

__host__
void cudaHierarchicalEM(
        const unsigned int totalPts,
        const unsigned int DIM,
        const thrust::host_vector<float> &h_X,
        thrust::host_vector<float> &h_Pi,
        thrust::host_vector<float> &h_Mu,
        thrust::host_vector<float> &h_Sigma,
        const unsigned int J,
        const unsigned int maxLevel,
        const float tol_prt,
        const float tol_spt,
        const float tol_cov,
        const unsigned int maxItr);

#endif
