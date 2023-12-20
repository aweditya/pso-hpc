# minimum cmake version
cmake_minimum_required(VERSION 2.8)

# define pso-hpc project
project(pso-hpc)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

# use OpenMP if available
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -Wall")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message("No OpenMP detected, pso-hpc builds without OpenMP but needs it for optimal performance!")
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(pso-hpc
    pso-hpc.c)

target_link_libraries(pso-hpc dl)
