cmake_minimum_required(VERSION 3.18)
project(nanotrace-cuda LANGUAGES C CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture to Blackwell (Compute Capability 10.0)
set(CMAKE_CUDA_ARCHITECTURES 100)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Options
option(NANOTRACE_WITH_MINIZ "Enable compression support with miniz" ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Library sources
set(NANOTRACE_SOURCES
    src/nanotrace_host.cpp
)

# Add miniz if enabled
if(NANOTRACE_WITH_MINIZ)
    list(APPEND NANOTRACE_SOURCES
        ext/miniz.c
        ext/miniz_tdef.c
        ext/miniz_tinfl.c
    )
endif()

# Library
add_library(nanotrace_host STATIC
    ${NANOTRACE_SOURCES}
)

target_include_directories(nanotrace_host PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Add ext directory for miniz
if(NANOTRACE_WITH_MINIZ)
    target_include_directories(nanotrace_host PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ext
    )
    target_compile_definitions(nanotrace_host PUBLIC NANOTRACE_WITH_MINIZ)
endif()

target_link_libraries(nanotrace_host PUBLIC
    CUDA::cudart
)

# Examples (optional)
option(BUILD_EXAMPLES "Build example programs" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Install
install(DIRECTORY include/nanotrace DESTINATION include)
install(TARGETS nanotrace_host DESTINATION lib)
