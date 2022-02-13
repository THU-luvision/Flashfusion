cmake_minimum_required(VERSION 3.2)
project(flashfusion)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
message( "cmake path: " ${CMAKE_MODULE_PATH})

set(OpenCV_DIR "./third_party_library/opencv-3.3.0/build")
set(Pangolin_DIR "./third_party_library/Pangolin-master/build")
set(Sophus_DIR "./third_party_library/Sophus/build")
set(SOPHUS_INCLUDE_DIR "./third_party_library/Sophus")


set(GCFUSION_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Shaders" CACHE PATH "Where the shaders live")


find_package(OpenCV REQUIRED)
find_package(OpenGL)
find_package(GLUT)
find_package(Chisel REQUIRED)
find_package(OpenNI2 REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin 0.1 REQUIRED)
find_package(PkgConfig)
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
find_package(Boost COMPONENTS system thread  filesystem REQUIRED)

set(realsense2_DIR "./third_party_library/librealsense-master/build")
set(REALSENSE2_INCLUDE_DIRS "./third_party_library/librealsense-master/include")
find_library(REALSENSE2_FOUND realsense2 HINTS ${realsense2_DIR} REQUIRED)

if(NOT REALSENSE2_FOUND)
    message(WARN "Failed to find_library(realsense2)")
else()
    message( STATUS "Realsense lib found: " ${REALSENSE2_FOUND})
    message( STATUS "Realsense incl dir: " ${REALSENSE2_INCLUDE_DIRS})
endif()



include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS}  )
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories( ${SOPHUS_INCLUDE_DIR} )
include_directories( ${Boost_INCLUDE_DIR} )
include_directories(${CHISEL_INCLUDE_DIR})
include_directories(${OPENNI2_INCLUDE_DIR})
include_directories(${REALSENSE2_INCLUDE_DIRS})

add_definitions(-DWITH_REALSENSE)


set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -w")
set(CMAKE_CXX_FLAGS ${ADDITIONAL_CMAKE_CXX_FLAGS} "-O3 -msse2 -msse3  -msse4.1 -Wall -std=c++14 -DSHADER_DIR=${GCFUSION_SHADER_DIR}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -ftree-vectorize -lboost_system -lpthread -lboost_thread -lboost_filesystem -mavx2")

file(GLOB SOURCE_FILES
"*.cpp"
"*.h"
"Tools/*.h"
"Tools/*.cpp"
"GCSLAM/*.cpp"
"GCSLAM/*.h"
"GCSLAM/MILD/*.cpp"
"GCSLAM/MILD/*.h"
"GCSLAM/ORBSLAM/*.cpp"
"GCSLAM/ORBSLAM/*.h"
"GCFusion/*.cpp"
"GCFusion/*.h"
"Shaders/*.cpp"
"Shaders/*.h"
)

add_executable(FlashFusion ${SOURCE_FILES})
target_link_libraries(FlashFusion ${OpenCV_LIBS} 
                      ${Eigen_LIBRARIES}
                      ${Pangolin_LIBRARIES}
                      ${CHISEL_LIBRARY}
                      ${OPENNI2_LIBRARY}
		      ${OPENGL_LIBRARIES} 
		      ${GLUT_LIBRARY}
                      ${REALSENSE2_FOUND}
		      ${pthread} 
		      ${Boost_LIBRARIES})
