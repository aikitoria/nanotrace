# Examples for nanotrace-cuda

add_executable(simple_trace simple_trace.cu)
target_link_libraries(simple_trace PRIVATE nanotrace_host)

add_executable(mixed_trace mixed_trace.cu)
target_link_libraries(mixed_trace PRIVATE nanotrace_host)

add_executable(grayscale_trace grayscale_trace.cu)
target_link_libraries(grayscale_trace PRIVATE nanotrace_host)
