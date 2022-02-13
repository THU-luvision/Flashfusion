cmake_minimum_required(VERSION 2.8.3)

project(CHISEL)




set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

set(Pangolin_DIR "../../third_party_library/Pangolin-master/build")

find_package(Pangolin 0.1 REQUIRED)
find_package(SuiteSparse REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(Boost COMPONENTS system thread REQUIRED )


include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${SUITESPARSE_INCLUDE_DIRS})
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS}  )
include_directories( ${Boost_INCLUDE_DIR} )
include_directories( ./ )

file(GLOB_RECURSE CHISEL_srcs *.cpp )

if(WIN32)
  file(GLOB_RECURSE CHISEL_hdrs *.h)
endif()


SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})


if(WIN32)
  set(ADDITIONAL_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif()
execute_process(COMMAND cat /proc/cpuinfo OUTPUT_VARIABLE CPU_INFO)
string(REGEX MATCHALL "avx2" AVX_STRING ${CPU_INFO})
list(LENGTH AVX_STRING AVX_STRING_LENGTH)
message(STATUS "AVX_STRING: " ${AVX_STRING})

set(CMAKE_CXX_FLAGS ${ADDITIONAL_CMAKE_CXX_FLAGS} "-O3 -msse2 -msse3  -msse4.1 -Wall -std=c++14")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -ftree-vectorize -lboost_system -lpthread -lboost_thread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
  
if(WIN32)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  add_definitions(-DNOMINMAX)
  set (EXTRA_WINDOWS_LIBS ${EXTRA_WINDOWS_LIBS} ws2_32)
endif()

add_library(chisel SHARED 
	    ${CHISEL_srcs}
            ${containers}
	    ${CHISEL_hdrs}
)

target_link_libraries(chisel
                      ${Eigen_LIBRARIES}
                      ${Pangolin_LIBRARIES}
                      ${SUITESPARSE_LIBRARIES}
		      ${EXTRA_WINDOWS_LIBS}
		      ${OPENGL_LIBRARIES} 
		      ${GLUT_LIBRARY}
		      ${Boost_LIBRARIES}
)


INSTALL(TARGETS chisel
            RUNTIME DESTINATION bin
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
)
