cmake_minimum_required(VERSION 3.18)
project(nanotrace-cuda LANGUAGES C CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target architecture-specific Blackwell GeForce and workstation features.
set(CMAKE_CUDA_ARCHITECTURES "120a")

# Find CUDA
find_package(CUDAToolkit REQUIRED)
find_path(LIBIBERTY_INCLUDE_DIR libiberty/demangle.h REQUIRED)
find_library(LIBIBERTY_LIBRARY iberty REQUIRED)

# Options
option(NANOTRACE_WITH_MINIZ "Enable compression support with miniz" ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Library sources
set(NANOTRACE_SOURCES
    src/cpu_thread_trace.cpp
    src/gpu_process_trace.cu
    src/hes.cpp
    src/trace_writer.cpp
    src/session.cpp
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
add_library(nanotrace STATIC
    ${NANOTRACE_SOURCES}
)

target_include_directories(nanotrace PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_include_directories(nanotrace PRIVATE
    ${LIBIBERTY_INCLUDE_DIR}
)

# Add ext directory for miniz
if(NANOTRACE_WITH_MINIZ)
    target_include_directories(nanotrace PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ext
    )
    target_compile_definitions(nanotrace PRIVATE NANOTRACE_WITH_MINIZ)
endif()

target_link_libraries(nanotrace PUBLIC
    CUDA::cuda_driver
    CUDA::cudart
)

target_link_libraries(nanotrace PRIVATE
    ${LIBIBERTY_LIBRARY}
)

target_compile_definitions(nanotrace PRIVATE HAVE_DECL_BASENAME=1)

if(NANOTRACE_CUPTI_LIBRARY)
    add_library(nanotrace_cupti SHARED IMPORTED)
    set_target_properties(nanotrace_cupti PROPERTIES
        IMPORTED_LOCATION "${NANOTRACE_CUPTI_LIBRARY}"
    )
    target_link_libraries(nanotrace PUBLIC nanotrace_cupti)
else()
    target_link_libraries(nanotrace PUBLIC CUDA::cupti)
endif()

# Examples (optional)
option(BUILD_EXAMPLES "Build example programs" OFF)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Install
install(DIRECTORY include/nanotrace DESTINATION include)
install(TARGETS nanotrace DESTINATION lib)
