#include <cstdio>

#include <cuda_runtime.h>

#include <nanotrace/nanotrace.cuh>
#include <nanotrace/nanotrace_gpu.h>
#include <nanotrace/nanotrace_host.h>
#include <nanotrace/nanotrace_session.h>

NANOTRACE_DEFINE_TRACE_TYPE(UnifiedWork, "Work", "Kernel work", 0,
    nanotrace::lane_type::STATIC);
NANOTRACE_DEFINE_BLOCK_TYPE(UnifiedBlock, "Block {blockLinear}",
    "Block {blockLinear}");
NANOTRACE_DEFINE_TRACK_TYPE(UnifiedLane, "Warp {lane}", "Warp {lane}", 0);

using UnifiedTrace = nanotrace::static_trace_builder<1, UnifiedWork>;

__global__ void StageKernel(uint32_t* output, uint32_t seed)
{
    uint32_t value = blockIdx.x + seed;

    for (uint32_t i = 0; i < 2048; ++i)
    {
        value = value * 1664525U + 1013904223U;
    }

    output[blockIdx.x] = value;
}

__global__ void UnifiedKernel(
    nanotrace::static_tensor_handle<1, 2> trace, uint32_t* output,
    bool trace_enabled)
{
    bool should_trace = trace_enabled && threadIdx.x == 0;
    nanotrace::lane_context_static<2> lane = nanotrace::begin_lane(
        trace, blockIdx.x, 0, should_trace);

    if (should_trace)
    {
        nanotrace::start_token token = nanotrace::start();
        uint32_t value = blockIdx.x;

        for (uint32_t i = 0; i < 4096; ++i)
        {
            value = value * 1664525U + 1013904223U;
        }

        output[blockIdx.x] = value;
        nanotrace::end(token, trace, lane, UnifiedWork{});
        nanotrace::finish_lane(trace, lane);
    }
}

static bool CheckCuda(cudaError_t result, const char* operation)
{
    if (result == cudaSuccess)
    {
        return true;
    }

    std::fprintf(stderr, "%s failed: %s\n",
        operation, cudaGetErrorString(result));
    return false;
}

int main(int argc, char** argv)
{
    const char* output_path = argc > 1
        ? argv[1] : "unified_trace.nanotrace";
    nanotrace::GpuTrace gpu_trace{ "Unified CPU, HES, and kernel trace" };
    if (!gpu_trace)
    {
        std::fprintf(stderr, "GPU tracing initialization failed: %s\n",
            gpu_trace.LastError().c_str());
        return 1;
    }

    constexpr uint32_t BLOCK_COUNT = 32;
    int device_count = 0;
    uint32_t* device_output = nullptr;

    if (!CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")
        || device_count == 0
        || !CheckCuda(cudaSetDevice(0), "cudaSetDevice"))
    {
        return 1;
    }

    if (!CheckCuda(cudaMalloc(&device_output,
        BLOCK_COUNT * sizeof(uint32_t)), "cudaMalloc"))
    {
        return 1;
    }

    UnifiedTrace device_trace{ 1, dim3{ BLOCK_COUNT, 1, 1 } };
    device_trace.set_track_type<UnifiedLane>();
    device_trace.reset();

    cudaFuncAttributes kernel_attributes{};
    if (!CheckCuda(cudaFuncGetAttributes(
        &kernel_attributes, UnifiedKernel), "cudaFuncGetAttributes")
        || !CheckCuda(cudaDeviceSynchronize(), "pre-launch synchronize"))
    {
        cudaFree(device_output);
        return 1;
    }

    UnifiedKernel<<<BLOCK_COUNT, 32>>>(
        device_trace.get_handle(), device_output, false);
    if (!CheckCuda(cudaDeviceSynchronize(), "warm-up synchronize")
        || !gpu_trace.Begin())
    {
        std::fprintf(stderr, "GPU capture begin failed: %s\n",
            gpu_trace.LastError().c_str());
        cudaFree(device_output);
        return 1;
    }

    device_trace.reset();
    nanotrace::CpuThreadContext cpu{
        gpu_trace.Session(), "Main thread", 16 };
    {
        nanotrace::CpuScope launch_scope{ cpu, "Launch preprocessing" };
        StageKernel<<<BLOCK_COUNT, 32>>>(device_output, 1);
    }

    {
        nanotrace::CpuScope launch_scope{ cpu, "Launch transformation" };
        StageKernel<<<BLOCK_COUNT, 32>>>(device_output, 2);
    }

    {
        nanotrace::CpuScope launch_scope{ cpu, "Launch instrumented kernel" };
        UnifiedKernel<<<BLOCK_COUNT, 32>>>(
            device_trace.get_handle(), device_output, true);
    }

    {
        nanotrace::CpuScope wait_scope{ cpu, "Wait for GPU" };
        if (!CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"))
        {
            cudaFree(device_output);
            return 1;
        }
    }
    cpu.Bookmark("GPU complete");

    nanotrace::trace_writer kernel_trace{ "UnifiedKernel" };
    kernel_trace.set_block_type<UnifiedBlock>();
    kernel_trace.add_tensor(device_trace);

    cpu.Flush();
    if (!gpu_trace.Write(output_path, kernel_trace))
    {
        std::fprintf(stderr, "Trace write failed: %s\n",
            gpu_trace.LastError().c_str());
        cudaFree(device_output);
        return 1;
    }

    if (!CheckCuda(cudaFree(device_output), "cudaFree"))
    {
        return 1;
    }

    std::printf("Wrote %s\n", output_path);
    return 0;
}
