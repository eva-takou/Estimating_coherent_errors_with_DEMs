cmake_minimum_required(VERSION 3.12)
project(sample_outcomes)


set(CMAKE_CXX_STANDARD 14)        # Require at least C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
set(CUSTOM_OPT_FLAGS
    "-std=c++14 -O3 -march=native -ffast-math  -flto -funroll-loops -fstrict-aliasing -DNDEBUG"
)




# Append flags to release build type
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CUSTOM_OPT_FLAGS}")



if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()


#--- external dependencies ----
add_subdirectory(external/pybind11)

# header-only libraries

# Maybe switch to target_include_directories

#include_directories(
#    ${CMAKE_CURRENT_SOURCE_DIR}/include
#    ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen
#    ${CMAKE_CURRENT_SOURCE_DIR}/external/spectra/include
#    ${CMAKE_CURRENT_SOURCE_DIR}/external/pcg/include
#)

target_include_directories(sample_outcomes_rep_code
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen
        ${CMAKE_CURRENT_SOURCE_DIR}/external/spectra/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/pcg/include
)

#${CMAKE_CURRENT_SOURCE_DIR}/external/xsimd/include

#---- Python module -----

pybind11_add_module(sample_outcomes_rep_code 
                    src/bindings_rep_code.cpp 
                    src/Monte_Carlo_Sim_rep_code.cpp 
                    src/Unitary_Ops.cpp 
                    src/Kets.cpp 
                    src/call_to_pymatching.cpp 
                    src/estimation_functions_rep_code.cpp 
                    src/Measurements.cpp )





#To compile use this:
#build % cmake -DPython_EXECUTABLE=$(which python) ..
#make
