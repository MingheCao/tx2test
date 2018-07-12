cmake_minimum_required(VERSION 3.5)

project(cudagmm)
#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
#find_package(ArrayFire REQUIRED)
#find_package(catkin REQUIRED)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS "-gencode" "arch=compute_62,code=compute_62") #TX2
list(APPEND CUDA_NVCC_FLAGS "-gencode" "arch=compute_50,code=compute_50")
list(APPEND CUDA_NVCC_FLAGS "-rdc=true" "-std=c++11")
list(APPEND CUDA_NVCC_FLAGS "-g" "-G")

include_directories(include
                    ${CUDA_INCLUDE_DIRS}
                    ${EIGEN3_INCLUDE_DIRS}
                    "/usr/include/eigen3"
					"/usr/include"
                    ${Boost_INCLUDE_DIRS}
                    )

link_directories(
		/usr/local/cuda/lib64
       		 src
       		 lib
)

file(GLOB CU_Files "src/*.cu")

cuda_add_library(Cuda_Lib_
            ${CU_Files})

#file(GLOB C_Files "src/*.cc")
#add_library(C_Lib_ ${C_Files})

cuda_add_executable(${PROJECT_NAME}HierEM
		test/cudatest.cc
		)
target_link_libraries(${PROJECT_NAME}HierEM
#		C_Lib_
		Cuda_Lib_
		${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
		${CUDA_cusolver_LIBRARY}
		${CUDA_cudadevrt_LIBRARY}
		stdc++
#		afcuda
		)
