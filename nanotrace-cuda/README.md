# nanotrace-cuda

High-performance CUDA tracing library for GPU kernel execution profiling. Generates `.nanotrace` files compatible with the nanotrace visualizer.

## Features

- **Minimal overhead**: 2 uint32 lane context, vectorized writes with write-through caching
- **Type-safe**: Compile-time trace type definitions with static assertions
- **Flexible**: Support for static lanes (fixed trace type) and dynamic lanes (mixed trace types)
- **Heterogeneous tensors**: Multiple trace tensors with different event widths
- **Clean API**: Simple `start()` → `end()` → `finish_lane()` workflow

## Performance Optimizations

- Write-through caching (`.wt` specifier) prevents L1 cache pollution
- Vectorized stores: v2/v4/v8 based on parameter count
- SM ID written once per lane (not per event)
- Event width optimized per tensor (2/4/8 uint32s)
- No runtime branching for event type lookup
- Compile-time event width validation

## Quick Start

### 1. Define Trace Types

```cpp
#include <nanotrace/nanotrace.cuh>
#include <nanotrace/nanotrace_host.h>

// Static trace types (fixed format per lane)
NANOTRACE_DEFINE_TRACE_TYPE(TraceKernel, "Kernel", "Kernel execution", 0, nanotrace::lane_type::STATIC);
NANOTRACE_DEFINE_TRACE_TYPE(TraceLoad, "Load", "Load from {0} to {1}", 2, nanotrace::lane_type::STATIC);

// Dynamic trace types (format ID written per event)
NANOTRACE_DEFINE_TRACE_TYPE(TraceCompute, "Compute", "Compute iteration {0}", 1, nanotrace::lane_type::DYNAMIC);

// Block type (for block labels)
NANOTRACE_DEFINE_BLOCK_TYPE(MyBlock, "Block {blockX}", "Block {blockX} on SM");

// Track type (for lane/track labels)
NANOTRACE_DEFINE_TRACK_TYPE(MyTrack, "Warp {lane}", "Warp {lane}", 0);
```

### 2. Create Trace Tensor

```cpp
// Static tensor: 8 lanes, all using TraceKernel (0 params → width 2)
using TraceTensor = nanotrace::static_trace_builder<8,
    TraceKernel, TraceKernel, TraceKernel, TraceKernel,
    TraceKernel, TraceKernel, TraceKernel, TraceKernel
>;

dim3 grid(16, 1, 1);  // Match kernel grid
TraceTensor trace_tensor(100, grid);  // 100 events per lane
```

### 3. Instrument Kernel

```cpp
__global__ void my_kernel(
    nanotrace::static_tensor_handle<8, 2> trace_handle,
    dim3 grid_dims)
{
    uint32_t block_id = blockIdx.x;
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_in_warp = threadIdx.x % 32;

    if (lane_in_warp == 0 && warp_id < 8) {
        auto lane = nanotrace::begin_lane(trace_handle, block_id, warp_id);

        for (int i = 0; i < 100; ++i) {
            auto s = nanotrace::start();

            // ... your work here ...

            nanotrace::end(s, trace_handle, lane, TraceKernel{});
        }

        nanotrace::finish_lane(trace_handle, lane);
    }
}
```

### 4. Write Trace File

```cpp
my_kernel<<<grid, block>>>(trace_tensor.get_handle(), grid);
cudaDeviceSynchronize();

nanotrace::trace_writer writer("my_kernel");
writer.set_block_type<MyBlock>();
writer.set_track_type<MyTrack>();
writer.register_trace_type<TraceKernel>();
writer.add_tensor(trace_tensor);
writer.write("output.nanotrace");
```

## API Reference

### Device-Side API

#### Capture Start Time

```cpp
nanotrace::start_token nanotrace::start()
```

Returns an opaque start token with current `%%globaltimer` value.

#### Begin Lane

```cpp
template<uint32_t NumLanes, uint32_t MaxEventWidth>
auto nanotrace::begin_lane(
    static_tensor_handle<NumLanes, MaxEventWidth> handle,
    uint32_t block_id,
    uint32_t lane_index)
```

Initializes lane context for static tensor. Returns `lane_context_static<MaxEventWidth>`.

```cpp
template<uint32_t NumLanes>
auto nanotrace::begin_lane_dynamic(
    dynamic_tensor_handle<NumLanes> handle,
    uint32_t block_id,
    uint32_t lane_index)
```

Initializes lane context for dynamic tensor. Returns `lane_context_dynamic`.

#### End Event (Static Lanes)

```cpp
// 0 parameters (v2 store, 8 bytes)
template<typename TraceType>
void nanotrace::end(start_token, handle, lane, TraceType{})

// 1 parameter (v4 store, 16 bytes)
template<typename TraceType>
void nanotrace::end(start_token, handle, lane, TraceType{}, uint32_t p0)

// 2 parameters (v4 store, 16 bytes)
template<typename TraceType>
void nanotrace::end(start_token, handle, lane, TraceType{}, uint32_t p0, uint32_t p1)

// 3-6 parameters (v8 store, 32 bytes)
template<typename TraceType>
void nanotrace::end(start_token, handle, lane, TraceType{}, uint32_t p0, ..., uint32_t p5)
```

**Note**: The `TraceType` parameter enables compile-time validation that the correct number of parameters are passed.

#### End Event (Dynamic Lanes)

```cpp
// 0 parameters (v4 store, 16 bytes, includes format ID)
template<typename TraceType>
void nanotrace::end(start_token, handle, lane, TraceType{})

// 1 parameter (v4 store, 16 bytes, includes format ID)
template<typename TraceType>
void nanotrace::end(start_token, handle, lane, TraceType{}, uint32_t p0)

// 2-5 parameters (v8 store, 32 bytes, includes format ID)
template<typename TraceType>
void nanotrace::end(start_token, handle, lane, TraceType{}, uint32_t p0, ..., uint32_t p4)
```

#### Finish Lane

```cpp
void nanotrace::finish_lane(handle, lane)
```

Writes lane header with SM ID and event count.

### Host-Side API

#### Static Trace Builder

```cpp
template<uint32_t NumLanes, typename... TraceTypes>
class static_trace_builder {
public:
    static_trace_builder(uint32_t max_events_per_lane, dim3 grid_dims, dim3 cluster_dims = dim3(0, 0, 0));
    auto get_handle() const;
    // ... getters ...
};
```

#### Dynamic Trace Builder

```cpp
template<uint32_t NumLanes>
class dynamic_trace_builder {
public:
    dynamic_trace_builder(uint32_t max_events_per_lane, dim3 grid_dims, dim3 cluster_dims = dim3(0, 0, 0));
    auto get_handle() const;
    // ... getters ...
};
```

#### Trace Writer

```cpp
class trace_writer {
public:
    trace_writer(const char* kernel_name);

    template<typename BlockType>
    void set_block_type();

    template<typename TrackType>
    void set_track_type();

    template<typename TraceType>
    void register_trace_type();

    template<typename Builder>
    void add_tensor(const Builder& builder);

    void write(const char* filename, bool compress = true);
};
```

## Memory Layout

### Static Tensor (event_width = 2, 0 params)

```
Lane: [Header: 2 uint32] [Event 0: 2 uint32] [Event 1: 2 uint32] ...
       [sm_id, write_offset_bytes] [start, end] [start, end]
```

### Static Tensor (event_width = 4, 1-2 params)

```
Lane: [Header: 4 uint32] [Event 0: 4 uint32] [Event 1: 4 uint32] ...
       [sm_id, write_offset_bytes, 0, 0] [start, end, p0, p1] [start, end, p0, p1]
```

### Dynamic Tensor (event_width = 8)

```
Lane: [Header: 8 uint32] [Event 0: 8 uint32] [Event 1: 8 uint32] ...
       [sm_id, write_offset_bytes, ...] [start, end, fmt_id, p0-p4] [start, end, fmt_id, p0-p4]
```

**Note**: The header stores `write_offset_bytes` (the final write position in bytes) instead of event count. The host post-processor computes the event count from this value, which enables overflow detection when `write_offset_bytes` exceeds the allocated lane capacity.

## Building

```bash
mkdir build && cd build
cmake ..
make
```

### Build Options

- `BUILD_EXAMPLES` (default: ON) - Build example programs
- `NANOTRACE_WITH_MINIZ` (default: ON) - Enable compression support using miniz

To build without examples:

```bash
cmake -DBUILD_EXAMPLES=OFF ..
make
```

To build without compression:

```bash
cmake -DNANOTRACE_WITH_MINIZ=OFF ..
make
```

### Requirements

- CMake 3.18+
- CUDA Toolkit 11.0+ (with C++20 support)
- C++20 compiler

## Examples

- `simple_trace.cu`: Basic usage with static tensor
- `mixed_trace.cu`: Static + dynamic tensors with 2D grid

Run examples:

```bash
./simple_trace
./mixed_trace
```

## File Format

Generates `.nanotrace` binary files compatible with the nanotrace visualizer. Format includes:

- File header (magic, version, compression flag) - always uncompressed
- Payload (optionally compressed with deflate):
  - Kernel name
  - Format descriptors (trace type definitions)
  - Block descriptors (SM assignments)
  - Event tracks (lanes with timing data)

When compression is enabled (default), the payload is compressed using miniz (deflate algorithm), significantly reducing file size.

## License

See project root for license information.
