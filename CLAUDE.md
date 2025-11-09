# Nanotrace

## Project Overview

Nanotrace is a tracing library for GPU kernel execution with a WebGPU-based visualizer. The visualizer displays execution traces across SM (Streaming Multiprocessor) lanes with hierarchical organization showing blocks and event tracks. Loads binary `.nanotrace` files containing format descriptors, blocks, and event tracks.

## Project Structure

```
/home/aiki/projects/nanotrace/
├── visualizer/                  # Web-based trace visualizer
│   ├── src/
│   │   ├── renderers/
│   │   │   ├── gpu-renderer.ts      # WebGPU shaders, pipelines, and buffers
│   │   │   ├── label-renderer.ts    # Canvas 2D text rendering (blocks, zones)
│   │   │   └── timeline-renderer.ts # Timeline UI with ticks and labels
│   │   ├── utils/
│   │   │   ├── camera.ts            # Camera class with zoom/pan logic
│   │   │   ├── file-loader.ts       # Binary trace parser and hierarchy builder
│   │   │   ├── types.ts             # TypeScript interfaces and constants
│   │   │   └── vite-env.d.ts        # Vite type declarations
│   │   ├── styles/
│   │   │   └── style.css            # All CSS styles (with variables & utilities)
│   │   ├── assets/
│   │   │   └── avatar.png           # UI asset (converted to WebP on build)
│   │   ├── interaction-manager.ts   # Mouse interaction, hit detection, selection
│   │   ├── visualizer.ts            # Main ZoneVisualizer orchestrator
│   │   └── main.ts                  # Entry point
│   ├── scripts/
│   │   ├── generate.ts              # TypeScript generator for all sample traces
│   │   └── validate.ts              # Validate .nanotrace format
│   ├── public/                  # Static assets (generated .nanotrace files)
│   ├── dist/                    # Vite build output
│   ├── node_modules/
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── index.html               # Vite entry point with HTML structure
│   └── README.md
├── docs/
│   └── nanotrace.md             # Binary format specification
├── old-warptrace-cuda/          # Previous CUDA implementation
├── .github/
│   └── workflows/
│       └── deploy.yml           # GitHub Pages deployment
├── .gitignore
├── LICENSE
├── README.md
└── CLAUDE.md                    # This file
```

## File Format

The binary `.nanotrace` format is documented in `docs/nanotrace.md`. Key features:
- Magic: "nanotrace\0" + 2-byte version number (currently version 1)
- Little-endian binary format
- Format descriptors for reusable string templates (e.g., "Block {0}", "Track {0}", "Event {1}")
- Block descriptors with SM assignments and format parameters
- Event tracks with format descriptors, parameters, and time/duration data
- 20-color palette mapped by format descriptor ID

## Visualizer Implementation

### Core Features

#### File Loading
- Startup modal with file selector for `.nanotrace` files
- Drag & drop support for loading traces
- Sample menu with 3 available samples:
  - Minimal: Minimal trace (1 block, 2 events)
  - Small Random: Small random trace (~48K events, 16 SMs)
  - Large Random: Large random trace (~10M events, 144 SMs)
- Binary parser for .nanotrace format
- Format descriptor parsing with placeholder replacement
- Hierarchy builder that converts trace data to visualization structure
- Auto-zoom on load showing entire kernel with padding
- Error handling with fallback to file selector
- Reload support with proper cleanup

#### Trace Data Structure
- 4-level hierarchy: SM Lane → Block Lanes → Blocks → Sublanes (tracks) → Zones (events)
- Dynamic SM lane count based on trace data
- Block lane assignment groups non-overlapping blocks into lanes
- Format-based naming for blocks, tracks, and zones
- Format-based color assignment (20-color palette)
- Block alignment with padding for labels

#### Core Rendering
- WebGPU instanced rendering using storage buffers
- Adaptive outlines (1px width using `fwidth()`, disabled below 0.5x zoom)
- Block borders with hover highlighting
- Zone rendering with format descriptor-based colors
- Selection highlighting with brightness boost

#### Navigation & Controls
- Pan: Right-click drag
- Selection: Left-click drag for time range
- X-axis zoom: Mouse scroll (0.001x to 1,000,000x)
- Y-axis zoom: Shift + scroll (0.01x to 2.0x)
- Uniform zoom: Ctrl + scroll (0.01x to 2.0x)
- Horizontal pan: Scroll wheel left/right
- Zoom-to-cursor: All zoom modes center on cursor
- Reset view: Press 'R' to return to auto-fit view

#### Selection System
- Drag selection for time ranges
- Real-time highlighting (zones fully in selection brightened by 55%)
- Selection label shows Start/End/Length in nanoseconds
- Double-click snap to zone or block boundaries
- Selection clamped to [0, TIME_RANGE] bounds
- Persistent until cleared

#### Timeline System
- Hierarchical tick marks (4 levels)
- Adaptive time units (s, ms, μs, ns)
- Whole number formatting with comma separators
- Power-of-10 tick intervals
- Independent tick/label spacing (120px ticks, 180px labels)
- Minimum precision of 1ns
- Fixed 30px bar at top with blur effect

#### Labels & Tooltips
- Block labels showing formatted name and duration when zoomed
- Zone labels showing formatted name and duration when zoomed
- SM labels (50px width) on left edge, follow lanes
- Hover tooltip with event name, hierarchy, and timing
- Performance culling with hierarchical max-width tracking

#### Visual Design
- Dark theme with CSS variables for all colors and effects
- Utility classes: `.glass`, `.glass-dark`, `.mono`, `.overlay-full`, `.btn-primary`, `.btn-secondary`
- 20-color palette mapped by format descriptor ID
- Gray block borders with hover highlight
- Selection highlight: brightness boost
- Frosted glass effects on UI elements
- Stats display: kernel name, duration, SM/block/zone counts, zoom levels, FPS, and version/attribution links

### Technical Architecture

#### Module Structure

**`src/utils/types.ts`** (interfaces and constants)
- TypeScript interfaces (FormatDescriptor, Zone, Block, Lane, FindZoneResult, etc.)
- Layout constants (SUBLANE_HEIGHT, LANE_PADDING, etc.)
- No external dependencies

**`src/utils/camera.ts`** (camera system)
- Camera class for zoom, pan, coordinate transformations
- Base zoom with X/Y multiplier system
- Screen-to-world coordinate conversion
- Auto-zoom calculation

**`src/utils/file-loader.ts`** (trace parsing)
- `parseTraceFile()`: Binary parser for .nanotrace format
- `buildHierarchy()`: Converts parsed trace to visualization hierarchy
- `formatString()`: Placeholder replacement for format descriptors
- Block lane assignment algorithm
- 20-color palette mapping

**`src/renderers/gpu-renderer.ts`** (WebGPU rendering)
- 6 WGSL shaders (zones, lanes, blocks, borders, background, selection)
- `createGPUBuffers()`: GPU storage buffer creation with memory tracking
- `createPipelines()`: Creates render pipelines grouped by pass type
  - Returns structured `GPUResources` with `passes` (RenderPass objects) and `buffers`
  - Each RenderPass contains pipeline + bindGroup pair
  - Helper functions: `createUniformBuffer()`, `createSimplePipeline()`, `createBackgroundPipeline()`

**`src/renderers/label-renderer.ts`** (canvas text rendering)
- LabelRenderer class: 2D canvas text rendering for blocks and zones
- `renderBlockLabels()`: Renders block names and durations
- `renderZoneLabels()`: Renders zone names and durations with culling
- Uses hierarchical max-width tracking for performance
- Requires camera, formatDescriptors, and formatString callback

**`src/renderers/timeline-renderer.ts`** (timeline UI)
- TimelineRenderer class: Timeline bar with ticks and labels
- `updateTimeline()`: Creates hierarchical tick marks (4 levels)
- `calculateTimelineInterval()`: Adaptive spacing based on zoom
- `formatTimeLabel()`: Converts times to appropriate units (s, ms, μs, ns)
- Power-of-10 intervals with whole number formatting

**`src/interaction-manager.ts`** (user interaction)
- InteractionManager class: Mouse interaction and hit detection
- `findZoneAtPosition()`: Hierarchical binary search for zone under cursor
- `updateHover()`: Manages tooltip display with zone details
- `updateSelection()`: Updates selection UI (region, lines, label)
- Tracks hover state (hoveredZoneId, hoveredBlockId)
- Manages selection state (isSelecting, hasSelection, start/end positions)

**`src/visualizer.ts`** (main orchestrator)
- ZoneVisualizer class: main application coordinator
- Two-phase initialization (WebGPU → file selection → visualization)
- Event handling (mouse, keyboard, resize)
- Delegates to specialized renderers (LabelRenderer, TimelineRenderer, InteractionManager)
- SM lane label positioning and updates
- WebGPU render loop with uniform buffer updates
- File loading and sample management

**`src/main.ts`** (entry point)
- Initializes ZoneVisualizer
- Sets avatar image source (WebP converted at build time)
- Error handling

**`src/styles/style.css`** (styles with optimization)
- CSS variables (`:root`) for all colors, effects, shadows, fonts
- Utility classes for common patterns (glass effects, overlays, buttons)
- Semantic color naming for maintainability
- Minified on build for production

#### Code Organization

The codebase follows a modular architecture with clear separation of concerns:

**Rendering Pipeline:**
- `renderers/gpu-renderer.ts` → WebGPU setup and shader compilation
- `renderers/label-renderer.ts` → 2D canvas text overlay
- `renderers/timeline-renderer.ts` → Timeline UI component
- `visualizer.ts` → Coordinates all renderers, manages render loop

**Data Flow:**
- `utils/file-loader.ts` → Parses binary → Creates hierarchy → Passes to visualizer
- `visualizer.ts` → Distributes data to specialized renderers
- Each renderer maintains its own state and updates independently

**Interaction Handling:**
- `interaction-manager.ts` → Encapsulates all mouse interaction logic
- Uses hierarchical binary search for O(log n) hit detection
- Manages hover and selection state separately from rendering
- Provides API: `findZoneAtPosition()`, `updateHover()`, `updateSelection()`

#### Key Constants (in `src/utils/types.ts`)
```typescript
export const SUBLANE_HEIGHT = 0.01;        // Fixed height per sublane
export const LANE_PADDING = 0.015;         // Between lanes
export const SUBLANE_PADDING = 0.002;      // Between sublanes
export const LANE_EDGE_PADDING = 0.003;    // Top/bottom of lane
export const BLOCK_LANE_PADDING = 0.01;    // Between block lanes
export const BLOCK_PADDING = 0.00005;      // Between blocks
export const BLOCK_EDGE_PADDING = 0.008;   // Padding above blocks for labels
export const ZONE_GAP = 0.00001;           // Between zones
export const BASE_TIME_RANGE = 1.0;        // Base 1ms range
```

### Development Workflow

The visualizer uses TypeScript with Vite for building:

```bash
# Install dependencies
npm install

# Development server with hot reload
npm run dev

# Production build
npm run build

# Preview production build
npm run preview
```

Build process:
1. TypeScript type checking with `tsc --noEmit`
2. Vite bundles TypeScript, CSS, and assets into `dist/`
3. Minification enabled for JavaScript (Terser), CSS, and HTML
4. Images converted to WebP via vite-imagetools
5. Output includes separate JS, CSS, and optimized asset files

Build configuration (`vite.config.ts`):
- Base path: `/nanotrace/` for GitHub Pages
- Image imports require type declarations in `src/vite-env.d.ts`

Edit files in `visualizer/`:
- TypeScript files in `src/` subdirectories for code changes
- `src/styles/style.css` for styling (use CSS variables where possible)
- `index.html` (root) for Vite entry point and app HTML structure

Run `npm run build` after changes, open `dist/index.html` in Chrome.

### Deployment

GitHub Actions workflow deploys to GitHub Pages at `aikitoria.github.io/nanotrace`:
- Triggers on push to main
- Builds visualizer with npm
- Deploys to GitHub Pages

Configuration in `.github/workflows/deploy.yml`.

## Sample Trace Generator

TypeScript generator in `visualizer/scripts/generate.ts` produces `.nanotrace` files:

**Minimal** (`npm run generate:minimal`)
- 1 block on SM 0
- 1 track with 2 events (0ns and 1000ns, each 1000ns duration)
- 3 format descriptors (Block, Track, Event)
- Uncompressed (133 bytes)

**Small Random** (`npm run generate:small`)
- 16 SMs with 1-3 block lanes each
- 20-50 blocks per block lane
- 3-6 tracks per block, 8-15 events per track
- ~50K events total
- 7 format descriptors, deflate compressed (~400KB)

**Large Random** (`npm run generate:large`)
- 144 SMs with 1-4 block lanes each
- 100-500 blocks per block lane
- 4-8 tracks per block, 10-20 events per track
- ~9M events total
- 7 format descriptors, deflate compressed (~70MB)

All samples use:
- Sequential block placement (realistic GPU behavior)
- Weighted event distribution: 40% Load, 30% Store, 20% Compute, 10% Tile ops
- Seeded random generation (reproducible)
- Output to `visualizer/public/` directory

Samples are generated automatically during `npm run build`.

## Usage

1. **Development**: Run `npm run dev` in `visualizer/` directory
2. **Generate samples**: Run `npm run generate:all` (or generate:minimal/small/large individually)
3. **Validate trace**: Run `npm run validate <file.nanotrace>` to verify format correctness
4. **Load trace**: Click "Select .nanotrace" or drag file onto window
5. **Load sample**: Click "Load Sample File" → Choose "Minimal" (1 block, 2 events), "Small Random" (~48K events), or "Large Random" (~10M events)
6. **Navigate**:
   - Right-click + drag to pan
   - Scroll to zoom timeline
   - Shift + Scroll to zoom Y-axis
   - Ctrl + Scroll to zoom uniformly
   - Press R to reset view
6. **Select**: Left-click + drag for time range, double-click to snap to zone/block
7. **Inspect**: Hover for details, zoom in for labels

## Not Yet Implemented

- Multi-range selection
- Filtering by criteria (SM, block, event type)
- Thread state visualization
- Dependency tracking between zones
- View export or saving
- Performance analysis features
- Zone aggregation or summarization
- Detailed statistics panel for selections

## Performance Characteristics

- Initial load: ~100-500ms for small traces, 2-4s for ~10M events
- Render: ~16ms @ 60 FPS
- Hover latency: <1ms (hierarchical binary search)
- Memory: Scales with trace size (~300-350MB for ~10M zones)
