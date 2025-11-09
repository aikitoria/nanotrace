# Nanotrace

WebGPU-based trace visualizer for GPU kernel execution. Displays execution traces across SM lanes with hierarchical organization (blocks, tracks, zones).

## Project Structure

```
/home/aiki/projects/nanotrace/
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
- Format descriptors (string templates with placeholders)
- Block descriptors (SM assignment + params)
- Event tracks (timing data in nanoseconds)
- Little-endian, deflate compression optional

## Architecture

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
- `parseTraceFile()`: Binary → structured data
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
npm run generate:minimal  # 1 block, 2 events
npm run generate:small    # ~50K events, 16 SMs
npm run generate:large    # ~9M events, 144 SMs
```

Samples use seeded random (seed=42), sequential block placement, weighted event distribution.

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
