# minimum cmake version
cmake_minimum_required(VERSION 2.8)

# use at least C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# generate position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# define pso project
project(pso)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

# use OpenMP if available
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -Wall")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Wall")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message("No OpenMP detected, fstlib builds without OpenMP but needs it for optimal performance!")
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(pso  
    pso-hpc.c)

target_link_libraries(pso dl m)
