#include <nanotrace/nanotrace.cuh>
#include <nanotrace/nanotrace_host.h>
#include <cuda_runtime.h>
#include <cstdio>

// Define trace types
NANOTRACE_DEFINE_TRACE_TYPE(TraceKernel, "Kernel", "Kernel execution", 0, nanotrace::lane_type::STATIC);
NANOTRACE_DEFINE_TRACE_TYPE(TraceLoad, "Load", "Load from src={0} to dst={1}", 2, nanotrace::lane_type::STATIC);
NANOTRACE_DEFINE_TRACE_TYPE(TraceCompute, "Compute", "Compute iteration {0}", 1, nanotrace::lane_type::DYNAMIC);
NANOTRACE_DEFINE_TRACE_TYPE(TraceStore, "Store", "Store to address {0}", 1, nanotrace::lane_type::DYNAMIC);

// Define block type
NANOTRACE_DEFINE_BLOCK_TYPE(BlockTrace, "Block {blockX},{blockY}", "Block ({blockX}, {blockY}) on SM");

// Define track type
NANOTRACE_DEFINE_TRACK_TYPE(WarpTrack, "Warp {lane}", "Warp {lane}", 0);

// Static trace config: 8 lanes with mixed trace types
using TraceConfig1 = nanotrace::static_trace_builder<8,
    TraceKernel,   // Warp 0: 0 params → width 2
    TraceKernel,   // Warp 1: 0 params → width 2
    TraceLoad,     // Warp 2: 2 params → width 4
    TraceLoad,     // Warp 3: 2 params → width 4
    TraceLoad,     // Warp 4: 2 params → width 4
    TraceLoad,     // Warp 5: 2 params → width 4
    TraceKernel,   // Warp 6: 0 params → width 2
    TraceKernel    // Warp 7: 0 params → width 2
>;
// max_event_width = 4

// Dynamic trace config: 4 lanes for mixed event types
using TraceConfig2 = nanotrace::dynamic_trace_builder<4>;

__global__ void mixed_kernel(
    nanotrace::static_tensor_handle<8, 4> static_handle,
    nanotrace::dynamic_tensor_handle<4> dynamic_handle,
    dim3 grid_dims,
    int* data)
{
    // Compute linear block ID from 2D grid
    uint32_t block_id = blockIdx.y * grid_dims.x + blockIdx.x;

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_in_warp = threadIdx.x % 32;
    uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Only lane 0 of each warp traces
    bool should_trace = (lane_in_warp == 0);

    // Warps 0-1: Simple kernel events (static, 0 params)
    if (warp_id < 2) {
        auto lane = nanotrace::begin_lane(static_handle, block_id, warp_id, should_trace);

        for (int i = 0; i < 10; ++i) {
            auto s = nanotrace::start();

            // Do substantial work
            volatile int sum = 0;
            for (int j = 0; j < 500; ++j) {
                sum = sum + j * i;
                sum = (sum * 3 + 7) % 500000;
            }
            if (data) data[global_tid % 1024] = sum;

            nanotrace::end(s, static_handle, lane, TraceKernel{});
        }

        nanotrace::finish_lane(static_handle, lane);
    }
    // Warps 2-5: Load events (static, 2 params)
    else if (warp_id < 6) {
        auto lane = nanotrace::begin_lane(static_handle, block_id, warp_id, should_trace);

        for (int i = 0; i < 5; ++i) {
            auto s = nanotrace::start();

            uint32_t src_addr = global_tid * 0x100 + i;
            uint32_t dst_addr = src_addr + 0x1000;

            // Do memory work
            volatile int val = 0;
            for (int k = 0; k < 300; ++k) {
                if (data) {
                    val = data[(global_tid + k) % 1024];
                    data[(global_tid + k + 1) % 1024] = val + i;
                }
            }

            nanotrace::end(s, static_handle, lane, TraceLoad{}, src_addr, dst_addr);
        }

        nanotrace::finish_lane(static_handle, lane);
    }
    // Warps 6-7: Kernel events (static, 0 params)
    else if (warp_id < 8) {
        auto lane = nanotrace::begin_lane(static_handle, block_id, warp_id, should_trace);

        for (int i = 0; i < 8; ++i) {
            auto s = nanotrace::start();

            // Do substantial work
            volatile int dummy = 0;
            for (int j = 0; j < 400; ++j) {
                dummy = dummy + j * i;
                dummy = (dummy << 1) ^ (dummy >> 3);
            }
            if (data) data[global_tid % 1024] = dummy;

            nanotrace::end(s, static_handle, lane, TraceKernel{});
        }

        nanotrace::finish_lane(static_handle, lane);
    }
    // Warps 8-11: Dynamic events (mixed types)
    else if (warp_id < 12) {
        uint32_t dynamic_lane_id = warp_id - 8;
        auto lane = nanotrace::begin_lane_dynamic(dynamic_handle, block_id, dynamic_lane_id, should_trace);

        for (int i = 0; i < 6; ++i) {
            // Alternate between compute and store events
            if (i % 2 == 0) {
                auto s = nanotrace::start();

                // Do substantial compute work
                volatile int result = 0;
                for (int j = 0; j < 600; ++j) {
                    result = result + j * i;
                    result = (result * 11 + 19) % 1000000;
                }
                if (data) data[global_tid % 1024] = result;

                nanotrace::end(s, dynamic_handle, lane, TraceCompute{}, (uint32_t)i);

            } else {
                auto s = nanotrace::start();

                uint32_t store_addr = global_tid * 0x200 + i;

                // Do memory work
                volatile int val = i;
                for (int k = 0; k < 200; ++k) {
                    if (data) {
                        data[(global_tid + k) % 1024] = val;
                        val = data[(global_tid + k + 1) % 1024] + 1;
                    }
                }

                nanotrace::end(s, dynamic_handle, lane, TraceStore{}, store_addr);
            }
        }

        nanotrace::finish_lane(dynamic_handle, lane);
    }
}

int main() {
    printf("nanotrace-cuda mixed example\n");

    // Grid configuration (2D grid)
    dim3 grid(8, 4, 1);     // 32 blocks (8x4)
    dim3 block(384, 1, 1);  // 384 threads per block (12 warps)

    // Allocate dummy data
    int* d_data;
    cudaMalloc(&d_data, 1024 * sizeof(int));
    cudaMemset(d_data, 0, 1024 * sizeof(int));

    // Create trace tensors
    // Static lanes: max 10 events (warps 0-1)
    TraceConfig1 static_tensor(10, grid);
    static_tensor.set_track_type<WarpTrack>();
    // Dynamic lanes: 6 events (warps 8-11)
    TraceConfig2 dynamic_tensor(6, grid);
    dynamic_tensor.set_track_type<WarpTrack>();

    printf("Launching kernel with grid (%u, %u, %u) and block (%u, %u, %u)\n",
           grid.x, grid.y, grid.z, block.x, block.y, block.z);

    // Warmup: run 10 iterations to warm up GPU
    printf("Warming up GPU (10 iterations)...\n");
    for (int i = 0; i < 10; i++) {
        mixed_kernel<<<grid, block>>>(
            static_tensor.get_handle(),
            dynamic_tensor.get_handle(),
            grid,
            d_data);
    }
    cudaDeviceSynchronize();

    // Reset trace tensors before traced run
    static_tensor.reset();
    dynamic_tensor.reset();

    // Traced run: single iteration with fresh trace data
    printf("Running traced iteration...\n");
    mixed_kernel<<<grid, block>>>(
        static_tensor.get_handle(),
        dynamic_tensor.get_handle(),
        grid,
        d_data);

    // Wait for completion
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    printf("Kernel completed successfully\n");

    // Write trace file
    printf("Writing trace file...\n");
    nanotrace::trace_writer writer("mixed_kernel");
    writer.set_block_type<BlockTrace>();
    writer.register_trace_type<TraceKernel>();
    writer.register_trace_type<TraceLoad>();
    writer.register_trace_type<TraceCompute>();
    writer.register_trace_type<TraceStore>();

    writer.add_tensor(static_tensor);
    writer.add_tensor(dynamic_tensor);

    writer.write("mixed_trace.nanotrace", false);  // Uncompressed for now

    printf("Trace written to mixed_trace.nanotrace\n");

    cudaFree(d_data);

    return 0;
}
