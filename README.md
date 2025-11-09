# nanotrace

Low-overhead trace recorder and visualizer for inspecting pipelined CUDA kernels.

## Overview

Nanotrace provides nanosecond-precision tracing for GPU kernel execution with a WebGPU-based visualizer. The system handles traces with millions of events while maintaining interactive performance through GPU-accelerated rendering.

**Live demo**: [aikitoria.github.io/nanotrace](https://aikitoria.github.io/nanotrace)

## Features

**Visualizer:**
- WebGPU instanced rendering with adaptive detail levels
- Four-level hierarchy: SM lanes → block lanes → blocks → event tracks
- Pan, zoom (X/Y/uniform), and time range selection
- Hover tooltips and labels with hierarchical context
- Drag-and-drop file loading with sample traces included
- Handles traces with 10M+ events at 60 FPS

**File Format:**
- Compact binary format with optional deflate compression
- Format descriptors for reusable string templates
- Nanosecond-precision timing data
- See `docs/nanotrace.md` for specification

## Quick Start

### Using the Visualizer

Visit [aikitoria.github.io/nanotrace](https://aikitoria.github.io/nanotrace) or build locally:

```bash
cd visualizer
npm install
npm run dev
```

Load a trace file or try the included samples:
- **Minimal**: 1 block, 2 events (133 bytes)
- **Small Random**: 16 SMs, ~48K events (347 KB)
- **Large Random**: 144 SMs, ~10M events (82 MB)

### Navigation

- **Pan**: Right-click + drag
- **Zoom X-axis**: Scroll (0.001x to 1,000,000x)
- **Zoom Y-axis**: Shift + scroll (0.01x to 2.0x)
- **Zoom uniform**: Ctrl + scroll (0.01x to 2.0x)
- **Select time range**: Left-click + drag
- **Snap selection**: Double-click on zone or block
- **Reset view**: Press R

## File Format

Binary format with little-endian encoding:

```
Header:
  - Magic: "nanotrace\0" (10 bytes)
  - Version: uint8 (currently 1)
  - Compression: uint8 (0=none, 1=deflate)
  - Kernel name: string (uint16 length + UTF-8 bytes)
  - Counts: format descriptors, blocks, tracks, total events

Format Descriptors:
  - Template string with {0}, {1}, ... placeholders
  - Parameter count: uint8

Block Descriptors:
  - SM ID: uint16
  - Format descriptor ID: uint16
  - Parameters: uint32[] (count from format descriptor)

Event Tracks:
  - Block ID: uint32
  - Format descriptor ID: uint16
  - Parameters: uint32[]
  - Event count: uint32
  - Events: [time: uint32, duration: uint32, format_desc_id: uint16, params: uint32[]]
```

Full specification: `docs/nanotrace.md`

## Generating Test Traces

TypeScript generator in `visualizer/scripts/generate.ts`:

```bash
cd visualizer

# Minimal trace (1 block, 2 events)
npm run generate:minimal

# Small random trace (~50K events, 16 SMs)
npm run generate:small

# Large random trace (~9M events, 144 SMs)
npm run generate:large

# Generate all samples
npm run generate:all

# Validate a trace file
npm run validate <file.nanotrace>
```

Samples are generated automatically during `npm run build`.

## Building

Production build:

```bash
cd visualizer
npm run build
```

Output in `visualizer/dist/`. The build process:
- Type-checks TypeScript
- Bundles and minifies JS/CSS/HTML
- Converts images to WebP
- Optimizes assets for deployment

Preview the build:

```bash
npm run preview
```

## Project Structure

```
nanotrace/
├── visualizer/              # WebGPU trace visualizer
│   ├── src/
│   │   ├── renderers/       # GPU, label, timeline renderers
│   │   ├── utils/           # Camera, file loader, types
│   │   ├── styles/          # CSS with design system variables
│   │   ├── interaction-manager.ts
│   │   ├── visualizer.ts
│   │   └── main.ts
│   ├── scripts/             # TypeScript trace generator
│   ├── public/              # Sample traces
│   └── dist/                # Build output
├── docs/
│   └── nanotrace.md         # Binary format specification
└── old-warptrace-cuda/      # Previous CUDA implementation
```

## Implementation Notes

The visualizer uses a modular architecture:

- `gpu-renderer.ts`: WebGPU pipelines with WGSL shaders for instanced rendering
- `label-renderer.ts`: Canvas 2D text overlay with hierarchical culling
- `timeline-renderer.ts`: Adaptive timeline with power-of-10 tick intervals
- `interaction-manager.ts`: Hierarchical binary search for O(log n) hit detection
- `camera.ts`: Zoom/pan with screen-to-world coordinate transforms
- `file-loader.ts`: Binary parser and hierarchy builder

All rendering happens on the GPU via storage buffers. Labels are culled based on hierarchical max-width tracking to maintain 60 FPS with dense traces.

## License

MIT License - see LICENSE file for details.
