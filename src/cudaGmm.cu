#include <cmath>
#include <cstdlib>
#include <random>
#include <iostream>
#include <cfloat>

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>

#include <thrust/replace.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#include "cudaUtil.cuh"
#include "cudaGmm.cuh"


__global__ void kernKmeansAssign(
        unsigned int pts,unsigned int dim, unsigned int k,
        const float* __restrict__ X,
        const float* __restrict__ Mu,
        float* __restrict__ working) {
    assert(dim <= 3);

    extern __shared__ float sdata[];

    const unsigned int tid = threadIdx.x;
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pts) return;

    // Load the mean values into shared memory.
    if (tid < k) {
        for (int d = 0; d < dim; d++) {
            sdata[d * k + tid] = Mu[d * k + tid];
        }
    }

    __syncthreads();

    float x[6];
    for(int d=0;d<dim;d++) {
        x[d] = X[pts * d + idx];
    }

    float best_distance = FLT_MAX;
    int best_cluster = -1;
    float distance = 0;

    for (int cluster = 0; cluster < k; cluster++) {
        for(int d=0;d<dim;d++){
            float* Mud=&sdata[d*k];
            distance += (x[d] - Mud[cluster]) * (x[d] - Mud[cluster]);
        }

        if (distance < best_distance) {
            best_distance = distance;
            best_cluster = cluster;
        }
        distance=0;
    }

    __syncthreads();

    // reduction

    for (int cluster = 0; cluster < k; cluster++) {
        for(int d=0;d<dim;d++) {
            float *pd=&sdata[blockDim.x * d];
            pd[tid] =
                    (best_cluster == cluster) ? X[pts*d+idx] : 0;
        }
        float *cnt=&sdata[blockDim.x * dim];
        cnt[tid] = (best_cluster == cluster) ? 1 : 0;

        __syncthreads();


        // Reduction for this cluster.
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                if(idx+stride<pts){
                    for(int d=0;d<dim;d++) {
                        float *pd=&sdata[blockDim.x * d];
                        pd[tid] += pd[tid + stride];
                    }
                    float *cnt=&sdata[blockDim.x * dim];
                    cnt[tid] += cnt[tid + stride];
                }
            }
            __syncthreads();
        }

        // Now sdata[0] holds the sum for x.

        if (tid == 0) {
            const int clusterIdx = blockIdx.x * k + cluster;

            for(int d=0;d<dim;d++) {
                float* newSumd=&working[gridDim.x *k *d];
                newSumd[clusterIdx]=sdata[blockDim.x * d];
            }
            float *counts=&working[gridDim.x *k *dim];
            counts[clusterIdx] = sdata[blockDim.x * dim];
        }
        __syncthreads();
    }
}

__global__ void kernKmeansMu(unsigned int num,
                             unsigned int dim,
                             unsigned int k,
                             float* __restrict__ Mu,
                             float* __restrict__ working) {

    const int tid = threadIdx.x;
    if (tid >= k) return;

    extern __shared__ float sdata[];

    for(int d=0;d<=dim;d++){
        sdata[k*d +tid] = 0;
    }
    __syncthreads();

    // Load into shared memory for more efficient reduction.

    for(int d=0;d<=dim;d++) {

        float *wd = &working[num * d];
        float *sdatad = &sdata[k * d];

        unsigned int idx = tid;

        while (idx < num) {
            sdatad[tid] += wd[idx];
            idx += k;
        }
        __syncthreads();
    }

    float *cnt=&sdata[k*dim];

    for(int d=0;d<=dim;d++) {
        float* Mud=&Mu[k*d];
        float* sdatad=&sdata[k*d];

        Mud[tid]=sdatad[tid]/cnt[tid];
    }
}

void deviceKeans(
        dim3 grid,dim3 block,
        unsigned int pts,
        unsigned int dim,
        unsigned int k,
        const float *X,
        float *Mu,
        const unsigned int maxItr) {

    float *Working;

    const unsigned int size=grid.x*k*(dim+1)*sizeof(float);
    cudaMalloc((void**)&Working, size);

    unsigned int sharedMem = block.x * (dim+1)*sizeof(float);

    for (int i = 0; i < maxItr; ++i) {
//        cudaMemset(Working, 0, size);

        sharedMem = block.x * (dim+1)*sizeof(float);
        kernKmeansAssign<<<grid,block, sharedMem>>>(
                pts,dim,k,
                        X,Mu,Working);

//        cudaDeviceSynchronize();

        sharedMem=k*(dim+1)*sizeof(float);
        kernKmeansMu<<<1,k,sharedMem>>>(k*grid.x,dim,k,Mu,Working);

//        cudaDeviceSynchronize();
    }
    cudaFree(Working);
}

/*
 * Solves the lower triangular system L^T x = b for x, b \in \mathbb{R}^{N},
 * L \in \mathbb{R}^{N \times N} and L_{i, j} = 0 for j > i.
 */
__device__ void devSolveLowerTri(const unsigned int dim, const float *L, float *x, const float *b) {
    assert(dim > 0);
    assert(L != NULL);
    assert(x != NULL);
    assert(b != NULL);
    // x == b allowed

    for (int d = 0; d < dim; ++d) {
        float sum = 0.0;
        if (d > 0) {
            for (int i = 0; i <= d - 1; ++i) {
                sum += L[d * dim + i] * x[i];
            }
        }

        x[d] = (b[d] - sum) / L[d * dim + d];
    }
}

/*
 * Solves the upper triangular system L^T x = b for x, b \in \mathbb{R}^{N},
 * L \in \mathbb{R}^{N \times N} and L_{i, j} = 0 for j > i.
 */
__device__ void devSolveLowerTriT(const unsigned int dim, const float *L, float *x, const float *b) {
    assert(dim > 0);
    assert(L != NULL);
    assert(x != NULL);
    assert(b != NULL);
    // x == b allowed

    // treat L as an upper triangular matrix U
    for (int d = 0; d < dim; d++) {
        int ip = dim - 1 - d;
        float sum = 0;
        for (int j = ip + 1; j < dim; ++j) {
            sum += L[j * dim + ip] * x[j];
        }

        x[ip] = (b[ip] - sum) / L[ip * dim + ip];
    }
}

__host__
void Eig(cusolverDnHandle_t cusolverH,
         const unsigned int dim,
         float *Sigma,float *working){

    cusolverStatus_t cusolver_status;

    assert(dim<31);

    float *d_W=&working[0];
    int *d_Info = (int *)&working[dim];
    float *d_work = &working[32];
    int  h_lwork = 0;

    cusolver_status = cusolverDnSsyevd_bufferSize(
            cusolverH,
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            dim,
            Sigma,
            dim,
            d_W,
            &h_lwork);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cusolver_status = cusolverDnSsyevd(
            cusolverH,
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            dim,
            Sigma,
            dim,
            d_W,
            d_work,
            h_lwork,
            d_Info);
    cudaDeviceSynchronize();
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    int info_gpu = 0;
    CudaCheck(cudaMemcpy(&info_gpu, d_Info, sizeof(int), cudaMemcpyDeviceToHost));
    assert(info_gpu==0);
}

__host__
void eigBatch(cusolverDnHandle_t cusolverH,
              const unsigned int dim,
              const unsigned int comps,
              float *Sigma,float *working){

    cusolverStatus_t cusolver_status;
    syevjInfo_t syevj_params = NULL;

    float *d_W=&working[0];
    int *d_info=(int *)&working[dim*comps];
    float *d_work=&working[dim*comps+comps];

    int lwork=0;

    cusolver_status = cusolverDnCreateSyevjInfo(&syevj_params);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cusolver_status = cusolverDnXsyevjSetTolerance(
            syevj_params,
            1.e-7);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cusolver_status = cusolverDnXsyevjSetMaxSweeps(
            syevj_params,
            15);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; /* compute eigenvectors */
    const cublasFillMode_t  uplo = CUBLAS_FILL_MODE_LOWER;

    cusolver_status = cusolverDnSsyevjBatched_bufferSize(
            cusolverH,
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            dim,
            Sigma,
            dim,
            d_W,
            &lwork,
            syevj_params,
            comps);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cusolver_status = cusolverDnSsyevjBatched(
            cusolverH,
            jobz,
            uplo,
            dim,
            Sigma,
            dim,
            d_W,
            d_work,
            lwork,
            d_info,
            syevj_params,
            comps
    );
    CudaCheck(cudaDeviceSynchronize());
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    int h_info[comps];

    CudaCheck(cudaMemcpy(
            h_info, d_info,
            comps*sizeof(int),
            cudaMemcpyDeviceToHost));

    for(int k=0;k<comps;k++){
        assert(h_info[k]==0);
    }

}

__global__ void Regularize(const unsigned int pts, const unsigned int dim,
                           const unsigned int comps, float *Sigma,float *working){
    const unsigned int tid=threadIdx.x;
    const unsigned int b=blockIdx.x;

    if(b>=comps) return;
    if(tid>=dim) return;

    extern __shared__ float sWork[];

    float *SigmaK=&Sigma[b*dim*dim];
    float *eig=&sWork[0];
    float *sig=&sWork[dim];

    eig[tid] = working[b*dim+tid];

    for(int d=0;d<dim;d++) {
        sig[tid + d*dim] = SigmaK[tid + d*dim];
    }

    float minEig=eig[0];
    float maxEig=eig[dim-1];

    if( minEig<0 || (maxEig/minEig) >1e5 || maxEig<1e-10) {
        const float minEigval = max(maxEig / 1e5, 1e-10);

        eig[tid] = max(eig[tid], minEigval);
    }

    for(int d=0;d<dim;d++){
        sig[tid + d*dim]=eig[d]*sig[tid + d*dim];
    }

    float *QDK=&working[b*dim*dim];
    float *Q=&working[comps*dim*dim+b*dim*dim];

    for(int d=0;d<dim;d++){
        QDK[tid + d*dim]=sig[tid + d*dim];
        Q[tid + d*dim]=SigmaK[tid + d*dim];
    }

}

__device__ void devChol(const unsigned int dim,
                        float *Sigma, float *SigmaL){

    // L is the resulting lower diagonal portion of A = LL^T
    float *A = &Sigma[0];
    float *L = &SigmaL[0];

    for (int d = 0; d < dim * dim; ++d) {
        L[d] = 0;
    }


    for (int k = 0; k < dim; ++k) {
        float sum = 0;
        for (int s = 0; s < k; ++s) {
            const float l = L[k * dim + s];
            const float ll = l * l;
            sum += ll;
        }

        assert(sum >= 0);

        sum = A[k * dim + k] - sum;
        if (sum <= DBL_EPSILON) {
            printf("Sigma is not positive definite. (sum = %E)\n", sum);
            assert(sum > 0);
        }

        L[k * dim + k] = sqrt(sum);
        for (int d = k + 1; d < dim; ++d) {
            float subsum = 0;
            for (int s = 0; s < k; ++s)
                subsum += L[d * dim + s] * L[k * dim + s];

            L[d * dim + k] = (A[d * dim + k] - subsum) / L[k * dim + k];
        }
    }
}
/*
 * Computes log( p(x | mu, Sigma ) ) for multivariate normal distribution with
 * parameters mu (mean), and Sigma (covariance).
 */

__global__ void kernLogMVNormDist(
        const unsigned int pts, const unsigned int dim,
        const unsigned int comps, const float *X,
        const float *pi,const float *mu,
        const float *sigmaL, float *logProb
) {
    // Assumes a 2D grid of 1024x1 1D blocks
    int b = blockIdx.y * gridDim.x + blockIdx.x;
    int i = b * blockDim.x + threadIdx.x;

    extern __shared__ float work[];

    float *Pi=&work[0];
    float *Mu=&work[1];
    float *SigL=&work[dim+1];

    if(threadIdx.x == 0){
        *Pi=*pi;
    }

    if(threadIdx.x < dim){
        Mu[threadIdx.x] = mu[threadIdx.x*comps];
    }

    if(threadIdx.x < dim*dim){
        SigL[threadIdx.x] = sigmaL[threadIdx.x];
    }
    __syncthreads();

    if (i >= pts) return;

    float u[16];
    float v[16];

    for(int d=0;d<dim;d++){
        v[d]=X[i+pts*d]-Mu[d];
    }

    devSolveLowerTri(dim, SigL, u, v);
    devSolveLowerTriT(dim, SigL, u, u);

    float det = 1.0;
    for (int d = 0; d < dim; ++d) {
        det *= SigL[d * dim + d];
    }
    det *= det;

    logProb[i] = log(*Pi) -0.5 * log(2.0 * PI) * dim - 0.5 * log(det)
                 - 0.5 * devVecDot(dim, u, v);
}

__global__
void kernEStep(
        const dim3 grid, const dim3 block,
        const unsigned int comps,
        const unsigned int pts, const unsigned int dim,
        const float *X, const float *Pi, const float *Mu,
        float *Sigma, float *SigmaL, float *logProb){

    const unsigned int tid = threadIdx.x;

    extern __shared__ float work[];

    float *sig=&work[tid*dim*dim];
    float *sigL=&work[comps*dim*dim+tid*dim*dim];

    for(int i=0;i<dim*dim;i++){
        sig[i]=Sigma[tid*dim*dim+i];
    }

    devChol(dim,sig,sigL);
    cudaDeviceSynchronize();

    for(int i=0;i<dim*dim;i++){
        SigmaL[tid*dim*dim+i] = sigL[i];
    }

    __syncthreads();

    const int shareMemSize=(1+dim+dim*dim)*sizeof(float);
    kernLogMVNormDist<<<grid, block,shareMemSize>>>(
            pts, dim,comps,
                    X,
                    &Pi[tid],
                    &Mu[tid],
                    &SigmaL[tid*dim*dim],
                    &logProb[tid * pts]);
}

__global__ void kernLlh(
        const unsigned int pts, const unsigned int comps,
        float *Pi, float *logPx, float *loggamma,const float *ptWeight
) {
    // loggamma[k * pts + i] =
    // On Entry: log p(x_i | mu_k, Sigma_k)
    // On exit: [log pi_k] + [log p(x_i | mu_k, sigma_k)] - [log p(x_i)]

    // Assumes a 2D grid of 1024x1 1D blocks
    int b = blockIdx.y * gridDim.x + blockIdx.x;
    int i = b * blockDim.x + threadIdx.x;
    if (i >= pts) return;

    float maxArg = -INFINITY;
    for (int k = 0; k < comps; ++k) {
        const float logProbK = loggamma[k * pts + i];
        if (logProbK > maxArg) {
            maxArg = logProbK;
        }
//        if(pts==237){
//            printf("probk[%d][%d]=%f \n",i,k,logProbK);
//        }
    }

    float sum = 0.0;
    for (int k = 0; k < comps; ++k) {
        const float logProbK = loggamma[k * pts + i];
        sum += exp(logProbK - maxArg);
    }

    assert(sum >= 0);
    const float logpx = maxArg + log(sum);

    const float w=ptWeight[i];

    for (int k = 0; k < comps; ++k) {
        loggamma[k * pts + i] += -logpx +log(w);
    }

    logPx[i] = logpx;

}

__global__ void kernExpBias(const unsigned int pts,const float *src,float *dest, const float maxArg) {
    int b = blockIdx.y * gridDim.x + blockIdx.x;
    int i = b * blockDim.x + threadIdx.x;
    if (i >= pts) return;

    const float arg = exp(src[i] - maxArg);
    dest[i]=arg;
}

__global__ void kernT1(
        const unsigned int pts, const unsigned int dim,
        const float *X, const float *gamma,const float T0, float *dest
) {
    // Assumes a 2D grid of 1024x1 1D blocks
    int b = blockIdx.y * gridDim.x + blockIdx.x;
    int i = b * blockDim.x + threadIdx.x;
    if (i >= pts) return;

    const float a = exp(gamma[i]) / T0;

    for (int d = 0; d < dim; d++) {
        const float arg=a * X[i + d * pts];
        dest[i + d * pts] = arg;
    }
}

__global__ void kernT2(
        const unsigned int comps,
        const unsigned int pts, const unsigned int dim,
        const float *X, const float *mu,const float *gamma,const float T0,
        float *dest
) {
    int b = blockIdx.y * gridDim.x + blockIdx.x;
    int i = b * blockDim.x + threadIdx.x;
    if (i >= pts) return;

    volatile __shared__ float Mu[16];

    if(threadIdx.x<dim){
        Mu[threadIdx.x]=mu[threadIdx.x*comps];
    }
    __syncthreads();

    const float a = exp(gamma[i]) / T0;
    const float *x = &X[i];
    float *y = &dest[i];

    float u[16];
    for (int d = 0; d < dim; d++) {
        u[d] = x[d*pts] - Mu[d];
    }

    for (int d = 0; d < dim; d++) {
        float *yp = &y[d * dim * pts];
        for (int j = 0; j < dim; ++j) {
            yp[j*pts] = a * u[d] * u[j];
        }
    }
}

__global__ void kernMStep(dim3 grid,dim3 block,
                          const unsigned int comps,
                          const unsigned int pts,
                          const unsigned int dim,
                          const float *X, float* loggamma,
                          float *working, float *Pi,float *Mu,
                          float* Sigma, const float weight){

    const unsigned int tid = threadIdx.x;

    float *loggammaK=&loggamma[tid*pts];
    float *workingK=&working[tid*pts*dim*dim];

    cudaReduceMax(grid,block,pts,loggammaK,workingK);
    cudaDeviceSynchronize();

    const float maxArg=workingK[0];

    kernExpBias << < grid, block>> >
                           (pts, loggammaK, workingK, maxArg);

    cudaReduceSum(grid,block,pts,workingK,workingK);
    cudaDeviceSynchronize();

    const float t0=exp(maxArg + log(*workingK));

    kernT1<<<grid,block>>>(pts,dim,X,loggammaK,t0,workingK);
    cudaDeviceSynchronize();

    Pi[tid]=t0 / weight;

    for(int d=0;d<dim;d++){
        cudaReduceSum(grid,block,pts,&workingK[d*pts],workingK);
        cudaDeviceSynchronize();
        Mu[tid+comps*d]=*workingK;
    }

    kernT2 << < grid, block>> > (
            comps,pts, dim, X, &Mu[tid],
                    loggammaK, t0, workingK);

    for(int i=0;i<dim * dim;i++){
        cudaReduceSum(grid,block,pts,&workingK[i*pts],workingK);
        cudaDeviceSynchronize();
        Sigma[tid*dim*dim+i]=*workingK;
    }
}

void deviceEM(
        const dim3 grid,
        const dim3 block,
        const unsigned int dim,
        const unsigned int pts,
        const unsigned int comps,
        const float *d_X,
        float *d_Pi,
        float *d_Mu,
        float *d_Sigma,
        float *d_gamma,
        const float *d_ptWeight,//---------all ptWeight =1 for usual use, 0<w<=1 for soft partition.
        const unsigned int maxItr,
        const float tol) {

    float *d_SigmaL = mallocOnGpu(dim * dim * comps);
    float *d_loggamma = mallocOnGpu(pts * comps);
    float *d_working = mallocOnGpu(comps * pts * dim * dim);

    unsigned int itr = 0;
    float prellh = -INFINITY,curllh = -INFINITY;

    cusolverStatus_t cusolver_status;
    cusolverDnHandle_t cusolverH;
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cublasHandle_t cublasH;
    cublasCreate(&cublasH);

    cudaReduceSum(grid,block,pts,d_ptWeight,d_working);

    float w;
    CudaCheck(cudaMemcpy(
            &w, d_working,
            sizeof(float),
            cudaMemcpyDeviceToHost));

    do {
        // --------------------------------------------------------------------------
        // Regularization
        // --------------------------------------------------------------------------

        eigBatch(cusolverH, dim,comps,d_Sigma, d_working);

        Regularize<<<comps,dim,dim*(dim+1)>>>(pts,dim,comps,d_Sigma,d_working);

        const float alf = 1;
        const float bet = 0;

        for (int k = 0; k < comps; ++k) {
            cublasSgemm(cublasH,
                        CUBLAS_OP_N,
                        CUBLAS_OP_T,
                        dim, dim, dim,
                        &alf, &d_working[k * dim * dim],
                        dim, &d_working[comps * dim * dim+k * dim * dim],
                        dim, &bet,
                        &d_Sigma[k * dim * dim], dim);
        }
        cudaDeviceSynchronize();

        // --------------------------------------------------------------------------
        // E-Step
        // --------------------------------------------------------------------------

        kernEStep<<<1,comps,2*dim*dim*comps*sizeof(float)>>>(grid,block,
                comps,pts,dim, d_X,d_Pi,d_Mu,d_Sigma,d_SigmaL,d_loggamma);

        kernLlh << < grid, block>> > (
                pts, comps,
                        d_Pi, d_working, d_loggamma,d_ptWeight);

        cudaReduceSum(grid,block,pts,d_working,d_working);

        prellh = curllh;
        CudaCheck(cudaMemcpy(
                &curllh, d_working,
                sizeof(float),
                cudaMemcpyDeviceToHost));

        cudaDeviceSynchronize();

        if (fabs(curllh - prellh) / pts < tol) {
            CudaCheck(cudaMemcpy(d_gamma,
                                 d_loggamma,
                                 pts * comps * sizeof(float),
                                 cudaMemcpyDeviceToDevice));

            break;
        }

        // --------------------------------------------------------------------------
        // M-Step
        // --------------------------------------------------------------------------

        kernMStep<<<1,comps>>>(grid,block, comps,pts,dim,
                d_X,d_loggamma,d_working,d_Pi,d_Mu,d_Sigma,w);

    } while (++itr < maxItr);
    std::cout << "Iter = " << itr << "\t";
    std::cout << "currllh = " << curllh / pts << "\t" << " prellh = " << prellh / pts << "\t" << " diff = "
              << fabs(curllh - prellh) / pts << std::endl;

    cublasDestroy(cublasH);
    cusolverDnDestroy(cusolverH);

    cudaFree(d_SigmaL);
    cudaFree(d_working);
    cudaFree(d_loggamma);
}

void cudaEM(
        const unsigned int pts,const unsigned int dim,const unsigned int comps,
        const thrust::host_vector<float> &h_X,
        thrust::host_vector<float> &h_Pi,
        thrust::host_vector<float> &h_Mu,
        thrust::host_vector<float> &h_Sigma,
        const float tol,
        const unsigned int maxItr) {

    int deviceId;
    cudaDeviceProp deviceProp;
    CudaCheck(cudaGetDevice(&deviceId));
    CudaCheck(cudaGetDeviceProperties(&deviceProp, deviceId));

    dim3 grid, block;
    calcDim(pts, &deviceProp, &block, &grid);

    const thrust::device_vector<float> d_X(h_X.begin(), h_X.end());
    thrust::device_vector<float> d_Pi(comps, 0);
    thrust::device_vector<float> d_Mu(comps*dim, 0);
    thrust::device_vector<float> d_Sigma(comps*dim*dim, 0);
    thrust::device_vector<float> d_logGamma(pts*comps, 0);
    thrust::device_vector<float> d_ptWeight(pts, 1);

    thrust::fill(d_Pi.begin(),d_Pi.end(),1.0f / (float(comps)));

    for (int j = 0; j < comps; ++j) {
        unsigned int u = pts / comps * j;
        for (int d = 0; d < dim; d++) {
            d_Mu[mat3dIdx(j, d, 0, comps, dim)] = d_X[
                    mat3dIdx(u, d, 0, pts, dim)];
        }
    }

    for (int j = 0; j < comps; j++) {
        for (int i = 0; i < dim; i++) {
            d_Sigma[dim * dim * j + (i * dim + i)] = 2;
        }
    }

    deviceEM(grid,block, dim, pts, comps,
             thrust::raw_pointer_cast(&d_X[0]),
             thrust::raw_pointer_cast(&d_Pi[0]),
             thrust::raw_pointer_cast(&d_Mu[0]),
             thrust::raw_pointer_cast(&d_Sigma[0]),
             thrust::raw_pointer_cast(&d_logGamma[0]),
             thrust::raw_pointer_cast(&d_ptWeight[0]),
             100, tol);
}

__global__
void kernPartition(
        const unsigned int rows,
        const unsigned int cols,
        const float tol,
        const float *loggamma, float *prt) {
    int b = blockIdx.y * gridDim.x + blockIdx.x;
    int i = b * blockDim.x + threadIdx.x;
    if (i >= rows) {
        return;
    }

    float cnt = 0;
    float sum=0;

    for (int k = 0; k < cols; k++) {
        sum += exp(loggamma[i + k * rows]);
    }

    for (int k = 0; k < cols; k++) {
        prt[i + k * rows] = 0;
        const float a = exp(loggamma[i + k * rows])/sum;
        if (a >= tol) {
            prt[i + k * rows] = 1.0;
            cnt = cnt + 1;
        }
    }

    for (int k = 0; k < cols; k++) {
        prt[i + k * rows] /= cnt;
    }
}

__global__
void kernFindIdx(
        const unsigned int totalPts,
        const unsigned int rows,
        const unsigned int cols,
        const float *Prt,
        unsigned int *lIdx, float *weight,
        unsigned int *numPts) {

    const unsigned int b = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int comp = b * blockDim.x + threadIdx.x;
    if (comp >= cols) {
        return;
    }

    unsigned int *lIdxK = &lIdx[totalPts * comp];
    float *wK = &weight[totalPts*comp];

    unsigned int num = 0;

    for (int i = 0; i < rows; i++) {
        float arg = Prt[i + comp * rows];
        if (arg != 0) {
            lIdxK[num]=i;
            wK[num]=arg;
            num++;
        }
    }
    numPts[comp] = num;
}

__global__ void kernGetPts(
        const unsigned int totalPts,
        const unsigned int pts,
        const unsigned int dim,
        const float *X,
        const unsigned int *gIdx,
        unsigned int *lIdx,
        float *points)
{
    int b = blockIdx.y * gridDim.x + blockIdx.x;
    int i = b * blockDim.x + threadIdx.x;
    if (i >= pts) {
        return;
    }

    const unsigned int arg=lIdx[i];
    lIdx[i]=gIdx[arg];

    for(int d=0;d<dim;d++){
        points[i+d*pts]=X[lIdx[i]+d*totalPts];
    }
}

__global__ void kernfindCornerPts(dim3 grid,dim3 block,
                                  unsigned int pts,unsigned int dim,unsigned int comp,
                                  const float *X,float *Mu,float *working){

    unsigned int tid =  threadIdx.x;
    if(tid >= 2 *dim) return;

    __shared__ unsigned int minMaxIdx[24];

    unsigned int *maxIdx=&minMaxIdx[0];
    unsigned int *minIdx=&minMaxIdx[12];

    if(tid==0){
        maxIdx[0]=1;
        maxIdx[1]=4;
        maxIdx[2]=5;
        maxIdx[3]=8;
        maxIdx[4]=1;
        maxIdx[5]=2;
        maxIdx[6]=5;
        maxIdx[7]=6;
        maxIdx[8]=1;
        maxIdx[9]=2;
        maxIdx[10]=3;
        maxIdx[11]=4;

        minIdx[0]=2;
        minIdx[1]=3;
        minIdx[2]=6;
        minIdx[3]=7;
        minIdx[4]=3;
        minIdx[5]=4;
        minIdx[6]=7;
        minIdx[7]=8;
        minIdx[8]=5;
        minIdx[9]=6;
        minIdx[10]=7;
        minIdx[11]=8;

    }
    __syncthreads();

    unsigned int d=tid/2;

    const float *x=&X[d*pts];
    float *wK=&working[tid * grid.x/2];

    if(tid %2 ==0){
        kernReduceMax<<<grid.x/2,block>>>(pts,x,wK);
        kernReduceMax<<<1,block>>>(grid.x/2,wK,wK);
        cudaDeviceSynchronize();

        for(int j=0;j<4;j++){
            Mu[d*comp+maxIdx[d*4+j]-1]=*wK;
        }
    }
    else{
        kernReduceMin<<<grid.x/2,block>>>(pts,x,wK);
        kernReduceMin<<<1,block>>>(grid.x/2,wK,wK);
        cudaDeviceSynchronize();

        for(int j=0;j<4;j++){
            Mu[d*comp+minIdx[d*4+j]-1]=*wK;
        }
    }

}

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
        const unsigned int maxItr) {

    int deviceId;
    cudaDeviceProp deviceProp;
    CudaCheck(cudaGetDevice(&deviceId));
    CudaCheck(cudaGetDeviceProperties(&deviceProp, deviceId));

    dim3 grid, block;

    //-----------------Init Para ------------------------

    h_Pi[0] = 1; // nodeIdx=0 at level 0, assume pi=1;

    const thrust::host_vector<float> h_pi(J, 1.0f / (float(J)));
    thrust::host_vector<float> h_sigma(DIM * DIM * J, 0);
    thrust::host_vector<float> h_mu(DIM * J);

    for (int j = 0; j < J; j++) {
        for (int i = 0; i < DIM; i++) {
            h_sigma[DIM * DIM * j + (i * DIM + i)] = 2;
        }
    }

    const unsigned int maxLevMix = pow(J, maxLevel);
    const unsigned int pMaxLevMix = pow(J, maxLevel - 1);

    thrust::host_vector<int> h_numPts(pMaxLevMix);
    h_numPts[0] = totalPts;

    const thrust::device_vector<float> d_X(h_X.begin(), h_X.end());
    thrust::device_vector<float> d_Pi(mat3d(1, J, pMaxLevMix), 0);
    thrust::device_vector<float> d_Mu(mat3d(DIM, J, pMaxLevMix), 0);
    thrust::device_vector<float> d_Sigma(mat3d(DIM * DIM, J, pMaxLevMix), 0);
    thrust::device_vector<float> d_logGamma(mat3d(totalPts, J, pMaxLevMix), 0);
    thrust::device_vector<float> d_Prt(mat3d(totalPts, J, pMaxLevMix), 0);
    thrust::device_vector<unsigned int> d_numPts(mat3d(1, 1, pMaxLevMix), 0);
    thrust::device_vector<unsigned int> d_gIdx(mat3d(totalPts, J, pMaxLevMix), 0);
    thrust::device_vector<unsigned int> d_lIdx(mat3d(totalPts, J, pMaxLevMix), 0);
    thrust::device_vector<float> d_ptWeight(mat3d(totalPts, J, pMaxLevMix), 1);

    thrust::device_vector<float> d_working(mat3d(DIM, totalPts, pMaxLevMix), 0);

    thrust::sequence(d_gIdx.begin(), d_gIdx.begin() + totalPts, 0, 1);

    //-------------for debug -------------------
    float *raw_Pi = thrust::raw_pointer_cast(&d_Pi[0]);
    float *raw_Mu = thrust::raw_pointer_cast(&d_Mu[0]);
    float *raw_Sigma = thrust::raw_pointer_cast(&d_Sigma[0]);
    float *raw_logGamma = thrust::raw_pointer_cast(&d_logGamma[0]);
    float *raw_Prt = thrust::raw_pointer_cast(&d_Prt[0]);
    float *raw_working = thrust::raw_pointer_cast(&d_working[0]);
    unsigned int *raw_numPts = thrust::raw_pointer_cast(&d_numPts[0]);
    unsigned int *raw_gIdx = thrust::raw_pointer_cast(&d_gIdx[0]);
    unsigned int *raw_lIdx = thrust::raw_pointer_cast(&d_lIdx[0]);
    float *raw_ptWeight = thrust::raw_pointer_cast(&d_ptWeight[0]);

    //---------------------------------------------

    for (int l = 0; l < maxLevel; l++) {
        const unsigned int currLevMix = pow(J, l);
        const unsigned int currLevHeadIdx = (pow(J, l) - 1) / (J - 1);
        const unsigned int childLevHeadIdx = (pow(J, l + 1) - 1) / (J - 1);

        thrust::fill(d_Pi.begin(), d_Pi.end(), 0);
        thrust::fill(d_Mu.begin(), d_Mu.end(), 0);
        thrust::fill(d_Sigma.begin(), d_Sigma.end(), 0);

        std::cout<<std::endl<<"Training Level "<<l+1<<" : "<<std::endl<<std::endl;

        if (l == 0) {
            //-------------First Train -----------------
            const unsigned int pts = h_numPts[0];

            calcDim(pts, &deviceProp, &block, &grid);
            dimToConsole(&block, &grid);

            thrust::copy(h_pi.begin(), h_pi.end(), d_Pi.begin());
            thrust::copy(h_sigma.begin(), h_sigma.end(), d_Sigma.begin());

            for (int j = 0; j < J; ++j) {
                unsigned int u = pts / J * j;
                for (int d = 0; d < DIM; d++) {
                    d_Mu[mat3dIdx(j, d, 0, J, DIM)] = d_X[
                            mat3dIdx(u, d, 0, pts, DIM)];
                }
            }

//            kernfindCornerPts << < 1, 2 * DIM >> > (grid, block,
//                    totalPts, DIM, J,
//                    thrust::raw_pointer_cast(&d_X[0]),
//                    thrust::raw_pointer_cast(&d_Mu[0]),
//                    thrust::raw_pointer_cast(&d_working[0]));

//            deviceKeans(grid, block, pts, DIM, J,
//                        thrust::raw_pointer_cast(&d_X[0]),
//                        thrust::raw_pointer_cast(&d_Mu[0]), 10);

            deviceEM(grid,block, DIM, pts, J,
                     thrust::raw_pointer_cast(&d_X[0]),
                     thrust::raw_pointer_cast(&d_Pi[0]),
                     thrust::raw_pointer_cast(&d_Mu[0]),
                     thrust::raw_pointer_cast(&d_Sigma[0]),
                     thrust::raw_pointer_cast(&d_logGamma[0]),
                     thrust::raw_pointer_cast(&d_ptWeight[0]),
                     100, tol_cov);

            thrust::copy(d_Pi.begin(), d_Pi.begin() + mat3d(1, J, 1), h_Pi.begin()+1);
//            thrust::copy(d_Mu.begin(), d_Mu.begin() + mat3d(J, DIM, 1), h_Mu.begin()+1);
//            thrust::copy(d_Sigma.begin(), d_Sigma.begin() + mat3d(DIM * DIM, J, 1), h_Sigma.begin()+DIM*DIM);

            kernPartition << < grid, block >> > (pts, J, tol_prt,
                            thrust::raw_pointer_cast(&d_logGamma[0]),
                            thrust::raw_pointer_cast(&d_Prt[0]));

            kernFindIdx << < 1, J >> > (totalPts, pts, J,
                    thrust::raw_pointer_cast(&d_Prt[0]),
                    thrust::raw_pointer_cast(&d_lIdx[0]),
                    thrust::raw_pointer_cast(&d_ptWeight[0]),
                    thrust::raw_pointer_cast(&d_numPts[0]));

            h_numPts = d_numPts;
            continue;
        }

        for (int m = 0; m < currLevMix; m++) {
            const unsigned int currMix = currLevHeadIdx + m;
            const unsigned int pts = h_numPts[m];

            if (h_Pi[currMix] == 0 || pts< J*J ) {
                continue;
            }

            calcDim(pts, &deviceProp, &block, &grid);

            std::cout<<"Curr Node:"<<currMix<<std::endl;
            dimToConsole(&block, &grid);

            thrust::copy(h_pi.begin(), h_pi.end(),
                         d_Pi.begin() + mat3d(1, J, m));
            thrust::copy(h_sigma.begin(), h_sigma.end(),
                         d_Sigma.begin() + mat3d(DIM * DIM, J, m));

            thrust::device_vector<float> d_pts(pts * DIM, 0);

            kernGetPts << < block, grid >> > (totalPts,pts, DIM,
                    thrust::raw_pointer_cast(&d_X[0]),
                    thrust::raw_pointer_cast(&d_gIdx[mat3d(totalPts, 1, m/J)]),
                    thrust::raw_pointer_cast(&d_lIdx[mat3d(totalPts, 1, m)]),
                    thrust::raw_pointer_cast(&d_pts[0]));

            for (int j = 0; j < J; ++j) {
                unsigned int u = pts / J * j;
                for (int d = 0; d < DIM; d++) {
                    d_Mu[mat3dIdx(j, d, m, J, DIM)] = d_pts[
                            mat3dIdx(u, d, 0, pts, DIM)];
                }
            }

//            deviceKeans(grid, block, pts, DIM, J,
//                        thrust::raw_pointer_cast(&d_pts[0]),
//                        thrust::raw_pointer_cast(&d_Mu[mat3d(J, DIM, m)]), 10);

            deviceEM(grid,block, DIM, pts, J,
                     thrust::raw_pointer_cast(&d_pts[0]),
                     thrust::raw_pointer_cast(&d_Pi[mat3d(J, 1, m)]),
                     thrust::raw_pointer_cast(&d_Mu[mat3d(J, DIM, m)]),
                     thrust::raw_pointer_cast(&d_Sigma[mat3d(DIM * DIM, J, m)]),
                     thrust::raw_pointer_cast(&d_logGamma[mat3d(totalPts, J, m)]),
                     thrust::raw_pointer_cast(&d_ptWeight[mat3d(totalPts, 1, m)]),
                     100, tol_cov);
        }
        CudaCheck(cudaDeviceSynchronize());

        d_gIdx=d_lIdx;

        thrust::replace_if(d_Pi.begin(),
                           d_Pi.begin() + mat3d(1, J, currLevMix),
                           eliUnspt(tol_spt), 0);
        thrust::copy(d_Pi.begin(),
                     d_Pi.begin() + mat3d(1, J, currLevMix),
                     h_Pi.begin() + mat3d(1, J, currLevHeadIdx));

        if (l == maxLevel - 1) {
            thrust::copy(d_Mu.begin(),
                         d_Mu.begin() + mat3d(DIM, J, currLevMix),
                         h_Mu.begin());
            thrust::copy(d_Sigma.begin(),
                         d_Sigma.begin() + mat3d(DIM * DIM, J, currLevMix),
                         h_Sigma.begin());
            std::cout << "Training Hierarchical Model Done." << std::endl;
            return;
        }

        for (int m = 0; m < currLevMix; m++) {
            const unsigned int currMix = currLevHeadIdx + m;
            const unsigned int pts = h_numPts[m];

            if (h_Pi[currMix] == 0 || pts< J*J ) {
                continue;
            }

            kernPartition << < grid, block >> > (
                    pts, J, tol_prt,
                            thrust::raw_pointer_cast(&d_logGamma[mat3d(totalPts, J, m)]),
                            thrust::raw_pointer_cast(&d_Prt[mat3d(totalPts, J, m)]));

            kernFindIdx << < 1, J >> > (totalPts, pts, J,
                    thrust::raw_pointer_cast(&d_Prt[mat3d(totalPts, J, m)]),
                    thrust::raw_pointer_cast(&d_lIdx[mat3d(totalPts, J, m)]),
                    thrust::raw_pointer_cast(&d_ptWeight[mat3d(totalPts, J, m)]),
                    thrust::raw_pointer_cast(&d_numPts[mat3d(1, J, m)]));
        }
        CudaCheck(cudaDeviceSynchronize());
        h_numPts = d_numPts;
    }
}

//    for(int i = 0; i < d_Num.size(); i++)
//    {
//        std::cout << "d_Num[" << i << "] = " << d_Num[i] << std::endl;
//    }



