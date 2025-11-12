# Nanotrace

WebGPU-based trace visualizer for GPU kernel execution. Displays execution traces across SM lanes with hierarchical organization (blocks, tracks, zones).

## Project Structure

```
/home/aiki/projects/nanotrace/
├── nanotrace-cuda/                  # CUDA tracing library
│   ├── include/nanotrace/
│   │   ├── nanotrace.cuh            # Device-side API (header-only)
│   │   └── nanotrace_host.h         # Host-side API (builders, writer)
│   ├── src/
│   │   └── nanotrace_host.cpp       # Host-side implementation
│   ├── examples/
│   │   ├── simple_trace.cu                    # Basic example
│   │   ├── mixed_trace.cu                     # Advanced example (mixed static/dynamic)
│   │   ├── grayscale_trace.cu                 # Large-scale example (419K blocks)
│   │   ├── tma_bandwidth_bench_static.cu      # TMA bandwidth (static scheduling)
│   │   └── tma_bandwidth_bench_atomic.cu      # TMA bandwidth (atomic scheduling)
│   ├── CMakeLists.txt
│   └── README.md
├── visualizer/
│   ├── src/
│   │   ├── renderers/
│   │   │   ├── gpu-renderer.ts      # WebGPU shaders and pipelines
│   │   │   ├── label-renderer.ts    # Canvas 2D text rendering
│   │   │   └── timeline-renderer.ts # Timeline with adaptive ticks
│   │   ├── utils/
│   │   │   ├── camera.ts            # Camera with zoom/pan
│   │   │   ├── constants.ts         # All application constants (145+)
│   │   │   ├── file-loader.ts       # Binary parser and hierarchy builder
│   │   │   └── types.ts             # TypeScript interfaces
│   │   ├── interaction-manager.ts   # Mouse input and hit detection
│   │   ├── visualizer.ts            # Main coordinator
│   │   └── main.ts
│   ├── scripts/
│   │   ├── generate.ts              # Sample trace generator
│   │   └── validate.ts
│   └── index.html
├── docs/nanotrace.md                # Binary format spec
└── CLAUDE.md
```

## File Format

Binary `.nanotrace` format (see `docs/nanotrace.md`):
- Magic: "nanotrace\0" + version + compression flag
- Kernel name + grid dimensions (x, y, z) + cluster dimensions (x, y, z)
- Format descriptors (dual strings: label + tooltip, with placeholders)
- Block descriptors (blockId, clusterId, smId, formatDescId - no parameters)
- Event tracks (timing data in nanoseconds)
- Little-endian, deflate compression optional

**Format Descriptor Structure**: Each format has two strings:
- **Label string**: Short format for canvas labels (e.g., "Load {0}")
- **Tooltip string**: Full format for hover tooltips (e.g., "Load from address {0}")

**Block Descriptor Structure** (12 bytes):
- blockId (uint32), clusterId (uint32), smId (uint16), formatDescId (uint16)
- Block strings support special placeholders: `{blockX}`, `{blockY}`, `{blockZ}`, `{blockLinear}`, `{clusterX}`, `{clusterY}`, `{clusterZ}`, `{clusterLinear}`

**Track Structure** (per track in event tracks section):
- Block descriptor ID (uint32), format descriptor ID (uint16), lane ID (uint32), format parameters (uint32[]), event count (uint32)
- Track strings support special placeholder: `{lane}` (automatically substituted from lane ID field)

## CUDA Tracing Library (nanotrace-cuda)

High-performance CUDA library for generating `.nanotrace` files from GPU kernels.

### Key Features
- **Minimal overhead**: 2 uint32 lane context, vectorized writes with `.cs` (cache streaming) specifier
- **Type-safe**: Compile-time trace type definitions with static assertions
- **Flexible**: Static lanes (fixed format) and dynamic lanes (mixed formats)
- **Heterogeneous tensors**: Multiple tensors with different event widths (2/4/8 uint32s)
- **Clean API**: `start()` → `end()` → `finish_lane()` workflow
- **Compression**: Optional deflate compression via miniz (enabled by default)

### Device-Side API (nanotrace.cuh)
Header-only implementation with zero overhead:
- `NANOTRACE_DEFINE_TRACE_TYPE(name, label_str, tooltip_str, param_count, lane_usage)` - Compile-time trace type registration
- `NANOTRACE_DEFINE_BLOCK_TYPE(name, label_str, tooltip_str)` - Compile-time block type registration (0 params)
- `NANOTRACE_DEFINE_TRACK_TYPE(name, label_str, tooltip_str, param_count)` - Compile-time track type registration (supports {lane} placeholder)
- `start()` - Capture timestamp using `%%globaltimer_lo` (32-bit lower portion, 32ns resolution)
- `start_zero()` - Returns a zero-initialized `start_token` (useful for array initialization)
- `begin_lane(handle, block_id, lane_index, enabled=true)` - Initialize lane context (2 uint32s + 1 bool), returns `lane_context_static<MaxEventWidth>`
- `end(start, handle, lane, TraceType{}, [p0-p6])` - Record event (7 overloads for 0-6 params), requires TraceType for compile-time validation, no-op if `!lane.enabled()`
- `finish_lane(handle, lane)` - Write header with SM ID (`%%smid`) and write_offset_bytes (for overflow detection), no-op if `!lane.enabled()`

**Conditional Tracing**: Pass `enabled` boolean to `begin_lane()` to control tracing per-thread/warp:
```cuda
bool should_trace = (threadIdx.x == 0);  // Only thread 0 traces
auto lane = nanotrace::begin_lane(handle, blockIdx.x, 0, should_trace);
auto s = nanotrace::start();
// ... work ...
nanotrace::end(s, handle, lane, TraceType{});  // No-op if !should_trace
nanotrace::finish_lane(handle, lane);  // No-op if !should_trace
```
All operations are forceinlined and use predicated execution for zero overhead when disabled.

**Type Safety**: `lane_context_static<MaxEventWidth>` is typed to match the handle's width, preventing mismatched usage at compile time.

**Important**: Types can be defined in any order. The `trace_writer` automatically maps `__COUNTER__` IDs to sequential file indices, so block/track/trace types don't need to be defined in a specific order.

**Disabling Tracing**:
- `NANOTRACE_DISABLED` - Compile-time disable for device-side tracing (removes all instrumentation overhead)
- `NANOTRACE_NO_LOG` - Disable host-side logging output (statistics during trace write)
- **CRITICAL**: When adding new types, templates, or functions to `nanotrace.cuh`, ALWAYS update the `#ifdef NANOTRACE_DISABLED` section at the top of the file with corresponding empty stubs to maintain API compatibility

### Host-Side API (nanotrace_host.h/cpp)
- `static_trace_builder<NumLanes, TraceTypes...>(max_events, grid_dims, cluster_dims=0)` - Static tensor with per-lane trace types
  - Constructor: `(uint32_t max_events_per_lane, dim3 grid_dims, dim3 cluster_dims = dim3(0,0,0))`
  - Computes `max_event_width` at compile time (2/4/8 based on param counts)
  - Allocates GPU buffer matching grid dimensions
  - Stores grid and cluster dimensions from construction
- `dynamic_trace_builder<NumLanes>(max_events, grid_dims, cluster_dims=0)` - Dynamic tensor (always event_width=8)
  - Constructor: `(uint32_t max_events_per_lane, dim3 grid_dims, dim3 cluster_dims = dim3(0,0,0))`
- `static_trace_builder` / `dynamic_trace_builder` - Configure track types per tensor
  - `set_track_type<TrackType>()` - Set default track type for all lanes in this tensor
  - `set_track_type<TrackType>(uint32_t lane)` - Override track type for specific lane in this tensor
- `trace_writer` - Generates `.nanotrace` file from tensors
  - `set_block_type<BlockType>()` - Set default block format descriptor (call before add_tensor)
  - `register_trace_type<T>()` - Register trace type (throws on duplicate IDs)
  - `add_tensor(builder)` - Add tensor to trace file
  - Copies device buffers to host
  - **Multiple tensors stack lanes within blocks**: All tensors must have same grid dimensions
  - Tensor 0: lanes [0, L₀), Tensor 1: lanes [L₀, L₀+L₁), etc. within each block
  - Writes kernel name, grid/cluster dimensions (6 uint32s)
  - Writes format descriptors (label + tooltip strings), block descriptors, event tracks
  - Automatically maps `__COUNTER__` IDs to file indices for all format types
  - Clamps event durations to minimum 32ns (global timer resolution)
  - Optional deflate compression (enabled by default)
  - **Logs statistics to stdout**: Per-tensor lane counts, max events/lane, total duration, compression ratio
- `builder.reset()` - Reset trace tensor to zeros (cudaMemset)
  - **Best practice**: Warmup GPU (10 iterations) → reset() → traced run
  - Avoids cold-start timing artifacts in traces

### Performance Optimizations
- **Cache streaming**: `.cs` specifier uses evict-first policy to minimize cache pollution for write-once data
- **Vectorized stores**: v2/v4/v8 based on parameter count (64/128/256 bits)
- **Optimized header writes**: v2 (64-bit) header stores instead of v8 (256-bit) padding
- **SM ID per lane**: Written once in header, not per event (saves 4 bytes × millions)
- **Optimized event width**: Static lanes use minimum width (2/4/8 uint32s)
- **No runtime branching**: Event type deduced at compile time from parameters
- **Compile-time validation**: Static assertions for event width ≤ max_event_width
- **Target architecture**: Compiled for sm_100 (Blackwell/CC 10.0) with optimal cache policies

### Memory Layout

**Important**: All event slots have the same width (`max_event_width`) for a given tensor, even if individual events don't fill the entire slot. The `lane.advance()` call always advances by `MaxEventWidth` to ensure proper spacing.

```
Static lane (max_event_width=2, 0 params):
  [Header: sm_id, write_offset_bytes] [Event: start, end] [Event: start, end] ...

Static lane (max_event_width=4, 1-2 params):
  [Header: sm_id, write_offset_bytes, ?, ?] [Event: start, end, p0, p1] ...

Dynamic lane (max_event_width=8, 0-5 params):
  [Header: sm_id, write_offset_bytes, ?, ?, ?, ?, ?, ?] [Event: start, end, format_id, p0-p4] ...
```

- Header (first event slot): Occupies full `max_event_width` but only first 2 uint32s are written (sm_id, write_offset_bytes)
- **write_offset_bytes**: Final write position in bytes, used by host to compute event count and detect overflow
- Each event slot: Occupies full `max_event_width` regardless of actual event size
- Unused portions of slots (marked with `?`) contain uninitialized data

### Overflow Detection

**Constructor checks**: Both `static_trace_builder` and `dynamic_trace_builder` validate that the tensor configuration won't overflow uint32 byte offsets:
```cpp
// Throws std::runtime_error if total_blocks × num_lanes × row_stride_bytes > UINT32_MAX
TraceConfig trace(max_events, grid_dims);
```

**Post-processor checks**: The host-side `parse_all_events()` detects lane overflow:
```cpp
// Throws std::runtime_error if write_offset_bytes > allocated capacity
// Reports: block ID, lane ID, SM ID, allocated vs attempted event counts
```

### Example Usage
```cuda
// Define trace, block, and track types
NANOTRACE_DEFINE_TRACE_TYPE(TraceKernel, "Kernel", "Kernel execution", 0, lane_type::STATIC);
NANOTRACE_DEFINE_BLOCK_TYPE(MyBlock, "Block {blockX}", "Block {blockX} on SM");
NANOTRACE_DEFINE_TRACK_TYPE(MyTrack, "Warp {lane}", "Warp {lane}", 0);

// Create tensor (max_events first, then grid, then optional cluster)
using Tensor = nanotrace::static_trace_builder<8, TraceKernel, ...>;
Tensor trace(100, dim3(16,1,1));  // 100 events/lane, 16 blocks

// Kernel
__global__ void kernel(nanotrace::static_tensor_handle<8,2> handle, dim3 grid) {
    auto lane = nanotrace::begin_lane(handle, blockIdx.x, warp_id);
    auto s = nanotrace::start();
    // ... work (should take >32ns for meaningful duration) ...
    nanotrace::end(s, handle, lane, TraceKernel{});
    nanotrace::finish_lane(handle, lane);
}

// Configure track types on the tensor
trace.set_track_type<MyTrack>();  // Default for all lanes in this tensor
// Optional: Override track type for specific lanes
// trace.set_track_type<SpecialTrack>(7);  // Lane 7 uses SpecialTrack

// Write trace (set_block_type before add_tensor)
nanotrace::trace_writer writer("kernel");
writer.set_block_type<MyBlock>();
writer.register_trace_type<TraceKernel>();
writer.add_tensor(trace);
writer.write("out.nanotrace");  // Compressed by default, logs stats to stdout
```

### Build Requirements
- CMake 3.18+
- CUDA Toolkit 13.0+ with C++20 support
- C++20 compiler (uses fold expressions, std::index_sequence)
- Target architecture: sm_100 (Blackwell/CC 10.0) - set in CMakeLists.txt

## Visualizer Architecture

### Data Hierarchy
SM Lane → Block Lanes → Blocks → Sublanes (tracks) → Zones (events)

### Module Organization

**`utils/types.ts`**: TypeScript interfaces only

**`utils/constants.ts`**: Single source of truth for all constants (145+)
- UI dimensions, layout (world space), navigation limits
- Colors, fonts, timings, GPU buffer sizes
- File format values, conversion factors

**`utils/camera.ts`**: Viewport transformations
- Independent X/Y zoom
- `splitDouble()`: f64 → f32 high/low pairs for GPU precision

**`utils/file-loader.ts`**: Binary parser
- `parseTraceFile()`: Binary → structured data (includes kernel name, grid/cluster dims)
- `buildHierarchy()`: Flat data → nested visualization hierarchy
- Block lane assignment (non-overlapping grouping)

**`renderers/gpu-renderer.ts`**: WebGPU rendering
- 6 WGSL shaders (zones, blocks, borders, lanes, background)
- Template literal injection for shader constants
- `createGPUBuffers()`: Storage buffers with double-precision encoding
- `createPipelines()`: Grouped render passes

**`renderers/label-renderer.ts`**: Canvas 2D text overlay
- Hierarchical culling (viewport + size thresholds)

**`renderers/timeline-renderer.ts`**: Timeline UI
- 4-level tick hierarchy, power-of-10 intervals
- Adaptive units (s/ms/μs/ns)

**`interaction-manager.ts`**: Mouse interaction
- `findZoneAtPosition()`: Hierarchical binary search O(log n)
- Hover/selection state management

**`visualizer.ts`**: Main coordinator
- WebGPU initialization → file loading → render loop
- Delegates to specialized renderers
- Bottom-right stats widget: kernel name, duration, grid/cluster dims, SMs, blocks, zones, zoom, FPS

### High-Precision Rendering

**Problem**: f32 precision (~7 digits) insufficient at extreme zoom (nanoseconds in millisecond coordinates)

**Solution**: "Double-single" emulated f64
1. CPU: `splitDouble()` splits f64 → (f32 high, f32 low)
2. GPU: `ds_add()`, `ds_sub()` for high-precision coordinate math
3. Storage: Zones (12 floats), Blocks (8 floats) with X as high/low pairs
4. Result: Stable rendering at all zoom levels (1ns precision)

## Development

```bash
npm install
npm run dev          # Development server
npm run build        # Production build
npm run preview      # Preview build
```

Build: TypeScript → Vite bundling → minification (Terser/CSS/HTML) → WebP conversion

## Sample Generators

```bash
npm run generate:minimal  # 1 block, 2 events, grid (1,1,1)
npm run generate:small    # ~50K events, 16 SMs, grid (1043,1,1)
npm run generate:large    # ~9M events, 148 SMs, grid (99071,1,1)
```

Samples use seeded random (seed=42), sequential block placement, weighted event distribution.
Grid dimensions match total block count. Cluster dimensions are (0,0,0) for all samples.

**Format**: Generators output new binary format with:
- Dual strings (label + tooltip) for all format descriptors
- Block descriptors with blockId, clusterId, smId, formatDescId (12 bytes each)
- Special block placeholders like `{blockLinear}` automatically expanded by visualizer

**Validation**: Use `npm run validate <file.nanotrace>` to parse and validate binary format, showing:
- Format descriptors (label + tooltip strings)
- Block descriptors (blockId, clusterId, smId, formatDescId)
- Track and event samples with timing information

## B200 Sample Traces

Real traces from nanotrace-cuda examples running on NVIDIA B200 (Blackwell, CC 10.0).
Generated with the latest conditional tracing API improvements (clean `begin_lane(..., enabled)` pattern).

**Files** (in `visualizer/public/samples/`, committed to repo):
- `simple_trace_b200.nanotrace` - 16 blocks, 128 tracks, ~12.8K events
- `mixed_trace_b200.nanotrace` - 32 blocks, 384 tracks, ~2.5K events (mixed static/dynamic tensors)
- `grayscale_trace_b200.nanotrace` - 419K blocks, 419K events (1 per block), all 148 SMs
- `tma_bandwidth_static_148.nanotrace` - 148 blocks, 444 tracks (3 buffers/block), 6801 GB/s (85.0%)
- `tma_bandwidth_static_296.nanotrace` - 296 blocks, 888 tracks (3 buffers/block), 6969 GB/s (87.1%)
- `tma_bandwidth_atomic_148.nanotrace` - 148 blocks, 444 tracks (3 buffers/block), 7096 GB/s (88.7%)
- `tma_bandwidth_atomic_296.nanotrace` - 296 blocks, 888 tracks (3 buffers/block), 7427 GB/s (92.8%)

**Access**: Available via "Load Sample File" menu in visualizer

## Controls

- Right-click drag: Pan
- Scroll: X-axis zoom
- Shift+Scroll: Y-axis zoom
- Ctrl+Scroll: Uniform zoom
- Left-click drag: Time range selection
- Double-click: Snap to zone/block boundaries
- R: Reset view
- Hover: Tooltip with timing details

## Performance

- Load: ~100-500ms (small), 2-4s (10M events)
- Render: 60 FPS
- Hit detection: <1ms (hierarchical binary search)
- Memory: ~300-350MB for 10M zones
