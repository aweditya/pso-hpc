# minimum cmake version
cmake_minimum_required(VERSION 3.12)

# define pso-mpi project
project(pso-mpi)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

# OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -Wall")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message("Install OpenMP and rebuild!")
endif()

# MPICH
find_package(MPI REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(pso-mpi
    pso-mpi.c)

target_link_libraries(pso-mpi dl MPI::MPI_C)
