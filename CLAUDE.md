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
│   │   ├── simple_trace.cu          # Basic example
│   │   └── mixed_trace.cu           # Advanced example
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
- Format descriptors (string templates with placeholders)
- Block descriptors (SM assignment + params)
- Event tracks (timing data in nanoseconds)
- Little-endian, deflate compression optional

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
- `NANOTRACE_DEFINE_TRACE_TYPE(name, format_str, param_count, lane_usage)` - Compile-time trace type registration
- `start()` - Capture timestamp using `%%globaltimer_lo` (32-bit lower portion)
- `begin_lane(handle, block_id, lane_index)` - Initialize lane context (2 uint32s)
- `end(start, handle, lane, [p0-p6])` - Record event (7 overloads for 0-6 params)
- `finish_lane(handle, lane)` - Write header with SM ID (`%%smid`) and event count

### Host-Side API (nanotrace_host.h/cpp)
- `static_trace_builder<NumLanes, TraceTypes...>` - Static tensor with per-lane trace types
  - Computes `max_event_width` at compile time (2/4/8 based on param counts)
  - Allocates GPU buffer matching grid dimensions
  - Stores grid dimensions from tensor construction
- `dynamic_trace_builder<NumLanes>` - Dynamic tensor (always event_width=8)
- `trace_writer` - Generates `.nanotrace` file from tensors
  - Copies device buffers to host
  - Writes kernel name, grid/cluster dimensions (6 uint32s)
  - Writes format descriptors, block descriptors, event tracks
  - Optional deflate compression (enabled by default)

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

**Important**: All event slots have the same width (`max_event_width`) for a given tensor, even if individual events don't fill the entire slot. The `lane.advance<MaxW>()` call ensures proper spacing.

```
Static lane (max_event_width=2, 0 params):
  [Header: sm_id, event_count] [Event: start, end] [Event: start, end] ...

Static lane (max_event_width=4, 1-2 params):
  [Header: sm_id, event_count, ?, ?] [Event: start, end, p0, p1] ...

Dynamic lane (max_event_width=8, 0-5 params):
  [Header: sm_id, event_count, ?, ?, ?, ?, ?, ?] [Event: start, end, format_id, p0-p4] ...
```

- Header (first event slot): Occupies full `max_event_width` but only first 2 uint32s are written (sm_id, event_count)
- Each event slot: Occupies full `max_event_width` regardless of actual event size
- Unused portions of slots (marked with `?`) contain uninitialized data

### Example Usage
```cuda
// Define trace types
NANOTRACE_DEFINE_TRACE_TYPE(TraceKernel, "kernel", 0, lane_type::STATIC);

// Create tensor
using Tensor = nanotrace::static_trace_builder<8, TraceKernel, ...>;
Tensor trace(dim3(16,1,1), 1024);  // 16 blocks, 1024 events/lane

// Kernel
__global__ void kernel(nanotrace::static_tensor_handle<8,2> handle, dim3 grid) {
    auto lane = nanotrace::begin_lane(handle, blockIdx.x, warp_id);
    auto s = nanotrace::start();
    // ... work ...
    nanotrace::end(s, handle, lane);
    nanotrace::finish_lane(handle, lane);
}

// Write trace
nanotrace::trace_writer writer("kernel");
writer.register_trace_type<TraceKernel>();
writer.add_tensor(trace);
writer.write("out.nanotrace");
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
npm run generate:large    # ~9M events, 144 SMs, grid (99071,1,1)
```

Samples use seeded random (seed=42), sequential block placement, weighted event distribution.
Grid dimensions match total block count. Cluster dimensions are (0,0,0) for all samples.

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
