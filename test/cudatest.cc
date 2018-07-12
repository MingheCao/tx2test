#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <chrono>

#include "cudaGmm.cuh"
#include "cudaUtil.cuh"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>

#include <thrust/gather.h>

int main(int argc, char **argv) {

    const unsigned int maxLevel=2;
    const unsigned int J=8;
    const float tol_spt = 0.01;   //check unsupported
    const float tol_prt = 0.3;   //check unsupported

    const unsigned int DIM = 3;
    const unsigned int maxIter = 100;
    const float tol_cov = 0.001;  //lambda_s, converge threthold

    thrust::host_vector<float> Data;
    float b;

    if (!freopen("../res/bunny32768.dat", "r", stdin))
    { std::cerr << "error opening file!" << std::endl; }
    while (std::cin >> b) {
        Data.push_back(b);
    }

    fclose(stdin);
    assert(Data.size() % DIM== 0);
//    const unsigned int numPts=(Data.size() / DIM)/512 *512;

    const unsigned int numPts = largestPowTwoLessThanEq(Data.size() / DIM);
    std::cout << "Input Data Size = " << DIM << " x " << numPts << std::endl;

    Eigen::MatrixXf DataMat(DIM,numPts);

    DataMat = Eigen::Map<Eigen::MatrixXf>(&Data[0], DIM, numPts);
    DataMat=100*DataMat;
    DataMat.transposeInPlace();

    thrust::host_vector<float> useData(numPts*DIM);

    Eigen::Map<Eigen::MatrixXf>(&useData[0], DataMat.rows(), DataMat.cols()) = DataMat;

    //---------------------------------Test Cuda---------------------------------------

    unsigned int totalMix = (pow(J, maxLevel + 1) - 1) / (J - 1);
    thrust::host_vector<float> Pi(totalMix);
    thrust::host_vector<float> Mu(DIM*totalMix);
    thrust::host_vector<float> Sigma(DIM*DIM*totalMix);

    std::cout << std::endl << "Calling CUDA training:" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    cudaHierarchicalEM(
            numPts,
            DIM,
            useData,
            Pi,
            Mu,
            Sigma,
            J,
            maxLevel,
            tol_prt,
            tol_spt,
            tol_cov,
            maxIter);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration =
            std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
    std::cout << "CUDA Time elapsed: " << duration.count() << "s" << std::endl;

    start = std::chrono::high_resolution_clock::now();

//    cudaEM(
//            numPts,DIM,160,
//            useData,
//            Pi,
//            Mu,
//            Sigma,
//            tol_cov,
//            100);
    end = std::chrono::high_resolution_clock::now();

    duration =
            std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
    std::cout << "CUDA Time elapsed: " << duration.count() << "s" << std::endl;

    const int maxLevMix=pow(J,maxLevel);
    const int maxLevHeadIdx= (pow(J, maxLevel) - 1) / (J - 1);

    Eigen::MatrixXf matPi(maxLevMix,1);
    Eigen::MatrixXf matMu(maxLevMix,DIM);
    Eigen::MatrixXf matSigma(DIM*DIM,maxLevMix);

    matPi = Eigen::Map<Eigen::MatrixXf>(&Pi[maxLevHeadIdx], maxLevMix, 1);
//    matMu = Eigen::Map<Eigen::MatrixXf>(&Mu[0], maxLevMix, DIM);
    matSigma = Eigen::Map<Eigen::MatrixXf>(&Sigma[0], DIM*DIM, maxLevMix);

    for(auto i=0;i<maxLevMix/J;++i){
        for(auto j=0;j<J;++j){
            for(auto d=0;d<DIM;++d){
                matMu(i*J+j,d)=Mu[J*DIM*i+j+d*J];
            }
        }
    }

    std::ofstream fWeight("../result/Weight.dat");
    if (fWeight.is_open()) {
        fWeight << matPi;
    }
    fWeight.close();
    std::ofstream fMu("../result/Mu.dat");
    if (fMu.is_open()) {
        fMu << matMu;
    }
    fMu.close();

    std::ofstream fSigma("../result/Sigma.dat");
    if (fSigma.is_open()) {
        fSigma << matSigma;
    }
    fSigma.close();

    return EXIT_SUCCESS;
}
