#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <stdio.h>
#include <assert.h>
#include <nanotrace/nanotrace.cuh>
#include <nanotrace/nanotrace_gpu.h>
#include <nanotrace/nanotrace_host.h>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
namespace cg = cooperative_groups;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

constexpr int TILE_HEIGHT = 64;
constexpr int TILE_WIDTH = 128;
constexpr int TILE_SIZE_BYTES = TILE_HEIGHT * TILE_WIDTH * sizeof(float);

constexpr int TENSOR_WIDTH = 131072;
constexpr int TENSOR_HEIGHT = 32768;
constexpr size_t TENSOR_NUM_ELEMENTS = static_cast<size_t>(TENSOR_WIDTH) * TENSOR_HEIGHT;
constexpr size_t TENSOR_SIZE_BYTES = TENSOR_NUM_ELEMENTS * sizeof(float);
constexpr double PEAK_BANDWIDTH_GB_S = 1792.0;

// Define trace types
NANOTRACE_DEFINE_TRACE_TYPE_WITH_PARAMETERS(TileTransfer, "Tile {0},{1}", "Transfer tile ({0},{1})",
                                            nanotrace::lane_type::STATIC, "tile_x", "tile_y");
NANOTRACE_DEFINE_BLOCK_TYPE(TMABlock, "Block {blockLinear}", "Block {blockLinear} on SM");
NANOTRACE_DEFINE_TRACK_TYPE(BufferTrack, "Buffer {lane}", "Buffer {lane}", 0);

// L2 cache flush kernel
__global__ void flush_l2_kernel(uint8_t* buffer, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    uint8_t sum = 0;
    for (size_t i = idx; i < size; i += stride) {
        sum += buffer[i];
    }

    // Write back to prevent optimization
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        buffer[0] = sum;
    }
}

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
    cudaDriverEntryPointQueryResult driver_status;
    void* cuTensorMapEncodeTiled_ptr = nullptr;
    CUDA_CHECK(cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled",
        &cuTensorMapEncodeTiled_ptr,
        12000,
        cudaEnableDefault,
        &driver_status
    ));
    assert(driver_status == cudaDriverEntryPointSuccess);
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}

constexpr int NUM_BUFFERS = 3;

__global__ void tma_bandwidth_kernel(
    const __grid_constant__ CUtensorMap tensor_map,
    int total_tiles,
    nanotrace::static_tensor_handle<3, 4> trace_handle
) {
    extern __shared__ __align__(128) float smem_buffer_raw[];
    float (*smem_buffer)[TILE_HEIGHT][TILE_WIDTH] =
        reinterpret_cast<float (*)[TILE_HEIGHT][TILE_WIDTH]>(smem_buffer_raw);

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar[NUM_BUFFERS];

    const int tid = threadIdx.x;
    const int block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

    const int tiles_per_block = (total_tiles + gridDim.x - 1) / gridDim.x;
    const int tile_start = block_id * tiles_per_block;
    const int tile_end = min(tile_start + tiles_per_block, total_tiles);

    // Initialize 3 lane contexts (one per buffer) - only thread 0 traces
    bool should_trace = (tid == 0);
    nanotrace::lane_context_static<4> lanes[3] = {
        nanotrace::begin_lane(trace_handle, block_id, 0, should_trace),
        nanotrace::begin_lane(trace_handle, block_id, 1, should_trace),
        nanotrace::begin_lane(trace_handle, block_id, 2, should_trace)
    };

    nanotrace::start_token trace_start[3] = {
        nanotrace::start_zero(),
        nanotrace::start_zero(),
        nanotrace::start_zero()
    };
    int tile_coords_x[NUM_BUFFERS];
    int tile_coords_y[NUM_BUFFERS];

    if (tid == 0) {
        for (int i = 0; i < NUM_BUFFERS; i++) {
            init(&bar[i], 1);
        }
    }
    __syncthreads();

    constexpr int tiles_per_row = TENSOR_WIDTH / TILE_WIDTH;

    int stage = 0;
    int tile_buffer[NUM_BUFFERS];
    for (int i = 0; i < NUM_BUFFERS; i++) {
        tile_buffer[i] = -1;
    }

    int current_tile_iter = tile_start;
    for (int i = 0; i < (NUM_BUFFERS - 1); i++) {
        int tile_idx = (current_tile_iter < tile_end) ? current_tile_iter : -1;
        if (tile_idx >= 0) current_tile_iter++;

        tile_buffer[stage] = tile_idx;

        if (tile_idx >= 0 && tid == 0) {
            int tile_y = tile_idx / tiles_per_row;
            int tile_x = tile_idx % tiles_per_row;
            int coord_x = tile_x * TILE_WIDTH;
            int coord_y = tile_y * TILE_HEIGHT;

            tile_coords_x[stage] = tile_x;
            tile_coords_y[stage] = tile_y;

            trace_start[stage] = nanotrace::start();

            cg::invoke_one(cg::coalesced_threads(), [&]() {
                const cuda::std::int32_t coordinates[]{ coord_x, coord_y };
                cuda::ptx::cp_async_bulk_tensor(
                    cuda::ptx::space_cluster,
                    cuda::ptx::space_global,
                    &smem_buffer[stage],
                    &tensor_map,
                    coordinates,
                    cuda::device::barrier_native_handle(bar[stage]));
                (void)cuda::device::barrier_arrive_tx(bar[stage], 1, TILE_SIZE_BYTES);
            });
        }
        stage = (stage + 1) % NUM_BUFFERS;
    }
    __syncthreads();

    int parity = 0;

    while (true) {
        int future_tile = (current_tile_iter < tile_end) ? current_tile_iter : -1;
        if (future_tile >= 0) current_tile_iter++;

        if (future_tile >= 0 && tid == 0) {
            int tile_y = future_tile / tiles_per_row;
            int tile_x = future_tile % tiles_per_row;
            int coord_x = tile_x * TILE_WIDTH;
            int coord_y = tile_y * TILE_HEIGHT;

            tile_coords_x[stage] = tile_x;
            tile_coords_y[stage] = tile_y;

            trace_start[stage] = nanotrace::start();

            cg::invoke_one(cg::coalesced_threads(), [&]() {
                const cuda::std::int32_t coordinates[]{ coord_x, coord_y };
                cuda::ptx::cp_async_bulk_tensor(
                    cuda::ptx::space_cluster,
                    cuda::ptx::space_global,
                    &smem_buffer[stage],
                    &tensor_map,
                    coordinates,
                    cuda::device::barrier_native_handle(bar[stage]));
                (void)cuda::device::barrier_arrive_tx(bar[stage], 1, TILE_SIZE_BYTES);
            });
        }

        tile_buffer[stage] = future_tile;
        stage = (stage + 1) % NUM_BUFFERS;

        int current_tile = tile_buffer[stage];
        if (current_tile == -1) break;

        while (!cuda::ptx::mbarrier_try_wait_parity(
            cuda::ptx::sem_acquire,
            cuda::ptx::scope_cta,
            cuda::device::barrier_native_handle(bar[stage]),
            parity)) { }

        if (tid == 0) {
            nanotrace::end(trace_start[stage], trace_handle, lanes[stage], TileTransfer{},
                          tile_coords_x[stage], tile_coords_y[stage]);
        }

        parity ^= (stage == (NUM_BUFFERS - 1));

        if (tid == 0) {
            float dummy = smem_buffer[stage][0][0];
            if (dummy > 1e30f) {
                smem_buffer[stage][0][1] = dummy;
            }
        }

        __syncthreads();
    }

    if (tid == 0) {
        nanotrace::finish_lane(trace_handle, lanes[0]);
        nanotrace::finish_lane(trace_handle, lanes[1]);
        nanotrace::finish_lane(trace_handle, lanes[2]);
    }
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <num_blocks> [output.nanotrace]\n", argv[0]);
        fprintf(stderr, "  num_blocks: Number of thread blocks (SMs to use)\n");
        return 1;
    }

    int num_blocks = atoi(argv[1]);
    const char* output_path = argc == 3 ? argv[2] : "tma_bandwidth_static.nanotrace";
    if (num_blocks <= 0) {
        fprintf(stderr, "Error: num_blocks must be positive\n");
        return 1;
    }

    nanotrace::GpuTrace gpu_trace("TMA bandwidth static schedule");
    if (!gpu_trace) {
        fprintf(stderr, "GPU tracing initialization failed: %s\n", gpu_trace.LastError().c_str());
        return 1;
    }

    printf("=== TMA Bandwidth Benchmark (Static Schedule) ===\n");
    printf("Tensor size: %.2f GiB (%zu elements)\n",
           TENSOR_SIZE_BYTES / (1024.0 * 1024 * 1024),
           TENSOR_NUM_ELEMENTS);
    printf("Tensor dimensions: %d rows x %d cols\n", TENSOR_HEIGHT, TENSOR_WIDTH);
    printf("Tile dimensions: %d rows x %d cols (%d bytes)\n",
           TILE_HEIGHT, TILE_WIDTH, TILE_SIZE_BYTES);

    int tiles_per_row = TENSOR_WIDTH / TILE_WIDTH;
    int tiles_per_col = TENSOR_HEIGHT / TILE_HEIGHT;
    int total_tiles = tiles_per_row * tiles_per_col;

    printf("Total tiles: %d (grid: %d x %d)\n", total_tiles, tiles_per_col, tiles_per_row);
    printf("Blocks: %d (static work distribution)\n", num_blocks);

    float* d_tensor;
    CUDA_CHECK(cudaMalloc(&d_tensor, TENSOR_SIZE_BYTES));
    CUDA_CHECK(cudaMemset(d_tensor, 0x42, TENSOR_SIZE_BYTES));

    CUtensorMap tensor_map{};
    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {TENSOR_WIDTH, TENSOR_HEIGHT};
    uint64_t stride[rank - 1] = {TENSOR_WIDTH * sizeof(float)};
    uint32_t box_size[rank] = {TILE_WIDTH, TILE_HEIGHT};
    uint32_t elem_stride[rank] = {1, 1};

    auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
    CUresult res = cuTensorMapEncodeTiled(
        &tensor_map,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        rank,
        d_tensor,
        size,
        stride,
        box_size,
        elem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    if (res != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(res, &errStr);
        fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n", errStr);
        return 1;
    }

    size_t shared_mem_size = NUM_BUFFERS * TILE_SIZE_BYTES;
    printf("Dynamic shared memory per block: %zu KB (allows %d blocks per SM)\n",
           shared_mem_size / 1024, 228 / (int)(shared_mem_size / 1024));

    if (shared_mem_size > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            tma_bandwidth_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem_size
        ));
        printf("Set max dynamic shared memory to %zu KB\n", shared_mem_size / 1024);
    }

    // Setup nanotrace: 3 lanes per block (one per buffer)
    printf("\nSetting up nanotrace...\n");
    dim3 grid(num_blocks, 1, 1);
    uint32_t tiles_per_block = (total_tiles + num_blocks - 1) / num_blocks;
    uint32_t max_events_per_lane = tiles_per_block;

    using TraceConfig = nanotrace::static_trace_builder<3, TileTransfer, TileTransfer, TileTransfer>;
    TraceConfig trace_tensor(max_events_per_lane, grid);
    trace_tensor.set_track_type<BufferTrack>();

    printf("Trace setup: %d blocks × 3 lanes/block, %u max events/lane\n",
           num_blocks, max_events_per_lane);

    // Allocate L2 flush buffer (128 MB should be enough to flush L2)
    uint8_t* d_flush_buffer;
    size_t flush_size = 128 * 1024 * 1024;  // 128 MB in bytes
    CUDA_CHECK(cudaMalloc(&d_flush_buffer, flush_size));
    CUDA_CHECK(cudaMemset(d_flush_buffer, 0, flush_size));
    CUDA_CHECK(cudaGetLastError());

    printf("\nWarming up...\n");
    int threads_per_block = 32;

    printf("Launching kernel with: blocks=%d, threads=%d, smem=%zu (static load balancing)\n",
           num_blocks, threads_per_block, shared_mem_size);

    // Warmup runs
    for (int i = 0; i < 5; i++) {
        flush_l2_kernel<<<256, 256>>>(d_flush_buffer, flush_size);
        tma_bandwidth_kernel<<<grid, threads_per_block, shared_mem_size>>>(
            tensor_map, total_tiles, trace_tensor.get_handle());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(launch_err));
        return 1;
    }

    printf("Warmup complete\n");

    printf("Running benchmark...\n");

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int num_iters = 10;
    float total_ms = 0.0f;
    for (int i = 0; i < num_iters; i++) {
        // Flush L2 before each timed iteration (no sync - let it overlap)
        flush_l2_kernel<<<256, 256>>>(d_flush_buffer, flush_size);

        CUDA_CHECK(cudaEventRecord(start));
        tma_bandwidth_kernel<<<grid, threads_per_block, shared_mem_size>>>(
            tensor_map, total_tiles, trace_tensor.get_handle());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float iter_ms;
        CUDA_CHECK(cudaEventElapsedTime(&iter_ms, start, stop));
        total_ms += iter_ms;
    }
    float elapsed_ms = total_ms / num_iters;

    double elapsed_s = elapsed_ms / 1000.0;
    double bandwidth_gb_s = TENSOR_SIZE_BYTES / (1000.0 * 1000 * 1000) / elapsed_s;

    printf("\n=== Results ===\n");
    printf("Elapsed time: %.3f ms\n", elapsed_ms);
    printf("Bandwidth: %.2f GB/s (%.1f%% of peak %.0f GB/s)\n",
           bandwidth_gb_s,
           (bandwidth_gb_s / PEAK_BANDWIDTH_GB_S) * 100,
           PEAK_BANDWIDTH_GB_S);
    printf("Tiles processed: %d\n", total_tiles);
    printf("Tiles/ms: %.0f\n", total_tiles / elapsed_ms);

    if (!gpu_trace.Begin()) {
        fprintf(stderr, "GPU trace capture failed: %s\n", gpu_trace.LastError().c_str());
        return 1;
    }

    // Reset and run traced iteration
    printf("\nRunning traced iteration...\n");
    trace_tensor.reset();

    // Flush L2 before traced run
    flush_l2_kernel<<<256, 256>>>(d_flush_buffer, flush_size);

    tma_bandwidth_kernel<<<grid, threads_per_block, shared_mem_size>>>(
        tensor_map, total_tiles, trace_tensor.get_handle());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Writing trace file...\n");
    nanotrace::trace_writer writer("tma_bandwidth_kernel");
    writer.set_block_type<TMABlock>();
    writer.add_tensor(trace_tensor);
    if (!gpu_trace.Write(output_path, writer)) {
        fprintf(stderr, "Trace write failed: %s\n", gpu_trace.LastError().c_str());
        return 1;
    }
    printf("Trace written to %s\n", output_path);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_tensor));
    CUDA_CHECK(cudaFree(d_flush_buffer));

    return 0;
}
