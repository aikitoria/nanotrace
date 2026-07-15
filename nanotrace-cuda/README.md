# nanotrace-cuda

Nanotrace combines three event sources in one timeline:

- low-overhead CPU thread scopes using `CLOCK_MONOTONIC_RAW`;
- hardware kernel timestamps from CUPTI HES on Blackwell;
- explicit intra-kernel ranges recorded with `%globaltimer`.

The host writer emits the compact chunked nanotrace v4 format. Clock snapshots correlate
GPU `%globaltimer` through the CUPTI clock into the CPU reference clock.

## Requirements

- CUDA 13.3 or newer;
- a Blackwell GPU with HES support;
- CMake 3.18 or newer;
- a C++20 compiler.

The standalone build targets architecture-specific `sm_120a`.

## Build

~~~bash
cmake -S . -B build -GNinja -DBUILD_EXAMPLES=ON
cmake --build build
~~~

## Unified tracing

Construct `GpuTrace` before any CUDA context exists. It initializes HES
immediately and hides CUPTI setup, priming, clock snapshots, kernel matching,
and attachment:

~~~cpp
nanotrace::GpuTrace gpu_trace{ "Inference step" };
if (!gpu_trace)
{
    // Handle gpu_trace.LastError().
}

// CUDA contexts may now be created normally.
cudaSetDevice(0);

gpu_trace.Begin();
InstrumentedKernel<<<grid, block>>>(device_trace.get_handle());

nanotrace::trace_writer kernel_trace{ "InstrumentedKernel" };
kernel_trace.add_tensor(device_trace);
gpu_trace.Write("trace.nanotrace", kernel_trace);
~~~

CPU events use a fixed-capacity per-thread buffer:

~~~cpp
nanotrace::TraceSession& session = gpu_trace.Session();
nanotrace::CpuThreadContext cpu{
    session, "Rank 0", 1024 };
{
    nanotrace::CpuScope scope{ cpu, "Execute step" };
    // Launch work.
}
cpu.Bookmark("Step complete");
cpu.Flush();
~~~

Bookmarks appear as labeled vertical rules across the viewer rather than as
zones on the source thread.

Parent CPU tracks explicitly when the application has a hierarchy:

~~~cpp
nanotrace::CpuThreadContext rank{ session, "Rank 0", 1024 };
nanotrace::CpuThreadContext worker{
    session, "Worker 0", 1024, rank.Track(), 0 };

// A parent can also be assigned after both tracks exist.
session.SetTrackParent(worker.Track(), rank.Track());
~~~

Parent IDs are serialized in the trace. The viewer never infers application
relationships from track names.

`GpuTrace::Write()` places explicit intra-kernel events beneath the matching
hardware kernel event. See `examples/unified_trace.cu` for the full flow.

## Device instrumentation

Define event, block, and lane formats:

~~~cpp
NANOTRACE_DEFINE_TRACE_TYPE(Work, "Work", "Kernel work", 0,
    nanotrace::lane_type::STATIC);
NANOTRACE_DEFINE_BLOCK_TYPE(Block, "Block {blockLinear}",
    "Block {blockLinear}");
NANOTRACE_DEFINE_TRACK_TYPE(Warp, "Warp {lane}", "Warp {lane}", 0);

using Trace = nanotrace::static_trace_builder<1, Work>;
~~~

For parameterized events, provide names for the viewer tooltip:

~~~cpp
NANOTRACE_DEFINE_TRACE_TYPE_WITH_PARAMETERS(
    TileTransfer, "Tile {0},{1}", "Transfer tile ({0},{1})",
    nanotrace::lane_type::STATIC, "tile_x", "tile_y");
~~~

Instrument one controlling thread per logical lane:

~~~cpp
nanotrace::lane_context_static<2> lane = nanotrace::begin_lane(
    trace, blockIdx.x, 0, threadIdx.x == 0);
nanotrace::start_token token = nanotrace::start();

// Work being measured.

nanotrace::end(token, trace, lane, Work{});
nanotrace::finish_lane(trace, lane);
~~~

Each enabled lane captures only low 32-bit timestamps while work is in flight.
`finish_lane()` reads one full 64-bit `%globaltimer` anchor and commits it with
the lane header, so no 64-bit anchor stays live in the lane context. Lane rows
are padded to 16-byte alignment for the vectorized header commit.

## Compile-time controls

- `NANOTRACE_DISABLED` removes device instrumentation.
- `NANOTRACE_NO_LOG` suppresses host writer diagnostics.
- `NANOTRACE_WITH_MINIZ` enables the default deflate-compressed v4 body in
  addition to compact varint event and argument records.

## Examples

- `unified_trace`: launches three kernels and correlates CPU scopes, HES kernel
  timestamps, and one explicit intra-kernel trace;
- `multistream_graph_trace`: captures a fork/join CUDA graph that HES reports
  on two driver-selected execution streams;
- `cpu_hierarchy_trace`: writes deterministic parent and worker CPU tracks;
- `simple_trace`, `mixed_trace`, and `grayscale_trace`: device-lane API
  examples;
- `tma_bandwidth_bench_static` and `tma_bandwidth_bench_atomic`: TMA tracing
  benchmarks using the non-deprecated `cuda::ptx` API. They accept the block
  count and an optional output path, for example
  `tma_bandwidth_bench_static 170 output.nanotrace`.

Every CUDA target is compiled for `sm_120a`. The host library is compatible
with warning-as-error and `-fno-exceptions` builds.
