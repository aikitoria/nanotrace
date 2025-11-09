#include <nanotrace/nanotrace.cuh>
#include <nanotrace/nanotrace_host.h>
#include <cuda_runtime.h>
#include <cstdio>

// Define trace type for kernel execution events (0 parameters)
NANOTRACE_DEFINE_TRACE_TYPE(TraceKernelExec, "Kernel Exec", "Kernel execution", 0, nanotrace::lane_type::STATIC);

// Define block type for labeling blocks
NANOTRACE_DEFINE_BLOCK_TYPE(SimpleBlock, "Block {blockLinear}", "Block {blockLinear} on SM");

// Define track type for labeling tracks (warps)
NANOTRACE_DEFINE_TRACK_TYPE(SimpleTrack, "Warp {lane}", "Warp {lane}", 0);

// Define our static trace config type
// 8 lanes, all using TraceKernelExec
using TraceConfig = nanotrace::static_trace_builder<8,
    TraceKernelExec, TraceKernelExec, TraceKernelExec, TraceKernelExec,
    TraceKernelExec, TraceKernelExec, TraceKernelExec, TraceKernelExec
>;

__global__ void simple_kernel(
    nanotrace::static_tensor_handle<8, 2> trace_handle,
    dim3 grid_dims,
    int iterations,
    volatile int* scratch)
{
    // Compute linear block ID from grid
    uint32_t block_id = blockIdx.x;

    // Get warp ID
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_in_warp = threadIdx.x % 32;

    // Only lane 0 of each warp traces
    bool should_trace = (lane_in_warp == 0 && warp_id < 8);

    // Begin lane for this warp
    auto lane = nanotrace::begin_lane(trace_handle, block_id, warp_id, should_trace);

    // Perform some work and record multiple events
    for (int i = 0; i < iterations; ++i) {
        auto s = nanotrace::start();

        // Do substantial work to get realistic trace durations
        volatile int result = 0;
        for (int j = 0; j < 100; ++j) {
            result = result + j * i;
            result = (result * 7 + 13) % 1000000;
        }
        // Write to global memory to prevent full optimization
        if (scratch) {
            scratch[blockIdx.x * 256 + threadIdx.x] = result;
        }

        nanotrace::end(s, trace_handle, lane);
    }

    // Finalize lane (write header with SM ID and event count)
    nanotrace::finish_lane(trace_handle, lane);
}

int main() {
    printf("nanotrace-cuda simple example\n");

    // Grid configuration
    dim3 grid(16, 1, 1);    // 16 blocks
    dim3 block(256, 1, 1);   // 256 threads per block (8 warps)

    // Allocate scratch buffer to prevent compiler optimizations
    int* d_scratch;
    cudaMalloc(&d_scratch, grid.x * block.x * sizeof(int));

    // Create trace tensor
    // Each block has 8 lanes (one per warp), each lane can hold 1024 events
    TraceConfig trace_tensor(1024, grid);

    printf("Launching kernel with grid (%u, %u, %u) and block (%u, %u, %u)\n",
           grid.x, grid.y, grid.z, block.x, block.y, block.z);

    // Warmup: run 10 iterations to warm up GPU
    printf("Warming up GPU (10 iterations)...\n");
    for (int i = 0; i < 10; i++) {
        simple_kernel<<<grid, block>>>(trace_tensor.get_handle(), grid, 100, d_scratch);
    }
    cudaDeviceSynchronize();

    // Reset trace tensor before traced run
    trace_tensor.reset();

    // Traced run: single iteration with fresh trace data
    printf("Running traced iteration...\n");
    simple_kernel<<<grid, block>>>(trace_tensor.get_handle(), grid, 100, d_scratch);

    // Wait for completion
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        cudaFree(d_scratch);
        return 1;
    }

    cudaFree(d_scratch);

    printf("Kernel completed successfully\n");

    // Write trace file
    printf("Writing trace file...\n");
    nanotrace::trace_writer writer("simple_kernel");
    writer.set_block_type<SimpleBlock>();
    writer.set_track_type<SimpleTrack>();
    writer.register_trace_type<TraceKernelExec>();
    writer.add_tensor(trace_tensor);
    writer.write("simple_trace.nanotrace", false);  // Uncompressed for now

    printf("Trace written to simple_trace.nanotrace\n");

    return 0;
}
