#include <array>
#include <cstdio>

#include <cuda_runtime.h>

#include <nanotrace/nanotrace.cuh>
#include <nanotrace/nanotrace_gpu.h>
#include <nanotrace/nanotrace_host.h>
#include <nanotrace/nanotrace_session.h>

NANOTRACE_DEFINE_TRACE_TYPE(GraphWork, "Graph work", "Graph kernel work", 0,
    nanotrace::lane_type::STATIC);
NANOTRACE_DEFINE_BLOCK_TYPE(GraphBlock, "Block {blockLinear}",
    "Block {blockLinear}");
NANOTRACE_DEFINE_TRACK_TYPE(GraphLane, "Warp {lane}", "Warp {lane}", 0);

using GraphTrace = nanotrace::static_trace_builder<1, GraphWork>;

static constexpr uint32_t BLOCK_COUNT = 32;
static constexpr uint32_t CYCLE_COUNT = 8;

__global__ void GraphBranchKernel(
    uint32_t* output, uint32_t seed, uint32_t iteration_count)
{
    uint32_t value = blockIdx.x + seed;

    for (uint32_t i = 0; i < iteration_count; ++i)
    {
        value = value * 1664525U + 1013904223U;
    }

    output[blockIdx.x] = value;
}

__global__ void TracedGraphKernel(
    nanotrace::static_tensor_handle<1, 2> trace, uint32_t* output)
{
    bool should_trace = threadIdx.x == 0;
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
        nanotrace::end(token, trace, lane, GraphWork{});
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
        ? argv[1] : "multistream_graph.nanotrace";
    nanotrace::GpuTrace gpu_trace{ "Multistream CUDA graph trace" };
    if (!gpu_trace)
    {
        std::fprintf(stderr, "GPU tracing initialization failed: %s\n",
            gpu_trace.LastError().c_str());
        return 1;
    }

    int device_count = 0;
    uint32_t* primary_output = nullptr;
    uint32_t* secondary_output = nullptr;
    cudaStream_t primary_stream = nullptr;
    cudaStream_t secondary_stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    std::array<cudaEvent_t, CYCLE_COUNT> fork_events{};
    std::array<cudaEvent_t, CYCLE_COUNT> join_events{};

    if (!CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")
        || device_count == 0
        || !CheckCuda(cudaSetDevice(0), "cudaSetDevice")
        || !CheckCuda(cudaMalloc(&primary_output,
            BLOCK_COUNT * sizeof(uint32_t)), "cudaMalloc primary output")
        || !CheckCuda(cudaMalloc(&secondary_output,
            BLOCK_COUNT * sizeof(uint32_t)), "cudaMalloc secondary output")
        || !CheckCuda(cudaStreamCreateWithFlags(
            &primary_stream, cudaStreamNonBlocking), "create primary stream")
        || !CheckCuda(cudaStreamCreateWithFlags(
            &secondary_stream, cudaStreamNonBlocking),
            "create secondary stream"))
    {
        return 1;
    }

    for (uint32_t i = 0; i < CYCLE_COUNT; ++i)
    {
        if (!CheckCuda(cudaEventCreateWithFlags(
                &fork_events[i], cudaEventDisableTiming), "create fork event")
            || !CheckCuda(cudaEventCreateWithFlags(
                &join_events[i], cudaEventDisableTiming), "create join event"))
        {
            return 1;
        }
    }

    GraphTrace device_trace{ 1, dim3{ BLOCK_COUNT, 1, 1 } };
    device_trace.set_track_type<GraphLane>();
    device_trace.reset();

    if (!CheckCuda(cudaStreamBeginCapture(primary_stream,
        cudaStreamCaptureModeThreadLocal), "begin graph capture"))
    {
        return 1;
    }

    for (uint32_t cycle = 0; cycle < CYCLE_COUNT; ++cycle)
    {
        GraphBranchKernel<<<BLOCK_COUNT, 32, 0, primary_stream>>>(
            primary_output, cycle * 4 + 1, 2048);
        CheckCuda(cudaEventRecord(fork_events[cycle], primary_stream),
            "record fork event");
        CheckCuda(cudaStreamWaitEvent(
            secondary_stream, fork_events[cycle]), "wait for fork event");
        GraphBranchKernel<<<BLOCK_COUNT, 32, 0, primary_stream>>>(
            primary_output, cycle * 4 + 2, 4096);
        GraphBranchKernel<<<BLOCK_COUNT, 32, 0, secondary_stream>>>(
            secondary_output, cycle * 4 + 3, 6144);
        CheckCuda(cudaEventRecord(join_events[cycle], secondary_stream),
            "record join event");
        CheckCuda(cudaStreamWaitEvent(
            primary_stream, join_events[cycle]), "wait for join event");
        GraphBranchKernel<<<BLOCK_COUNT, 32, 0, primary_stream>>>(
            primary_output, cycle * 4 + 4, 2048);
    }

    TracedGraphKernel<<<BLOCK_COUNT, 32, 0, primary_stream>>>(
        device_trace.get_handle(), primary_output);

    if (!CheckCuda(cudaStreamEndCapture(primary_stream, &graph),
            "end graph capture")
        || !CheckCuda(cudaGraphInstantiate(&graph_exec, graph, 0),
            "instantiate graph")
        || !CheckCuda(cudaGraphUpload(graph_exec, primary_stream),
            "upload graph")
        || !CheckCuda(cudaStreamSynchronize(primary_stream),
            "synchronize graph upload")
        || !gpu_trace.Begin())
    {
        std::fprintf(stderr, "Graph setup failed: %s\n",
            gpu_trace.LastError().c_str());
        return 1;
    }

    device_trace.reset();
    nanotrace::CpuThreadContext cpu{
        gpu_trace.Session(), "Main thread", 8 };
    {
        nanotrace::CpuScope launch_scope{ cpu, "Launch CUDA graph" };
        if (!CheckCuda(cudaGraphLaunch(graph_exec, primary_stream),
            "launch graph"))
        {
            return 1;
        }
    }

    {
        nanotrace::CpuScope wait_scope{ cpu, "Wait for CUDA graph" };
        if (!CheckCuda(cudaStreamSynchronize(primary_stream),
            "synchronize graph"))
        {
            return 1;
        }
    }

    nanotrace::trace_writer kernel_trace{ "TracedGraphKernel" };
    kernel_trace.set_block_type<GraphBlock>();
    kernel_trace.add_tensor(device_trace);

    cpu.Flush();
    if (!gpu_trace.Write(output_path, kernel_trace))
    {
        std::fprintf(stderr, "Trace write failed: %s\n",
            gpu_trace.LastError().c_str());
        return 1;
    }

    for (uint32_t i = 0; i < CYCLE_COUNT; ++i)
    {
        cudaEventDestroy(fork_events[i]);
        cudaEventDestroy(join_events[i]);
    }
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(secondary_stream);
    cudaStreamDestroy(primary_stream);
    cudaFree(secondary_output);
    cudaFree(primary_output);

    std::printf("Wrote %s\n", output_path);
    return 0;
}
