/**
 * WebGPU rendering infrastructure for high-performance trace visualization.
 *
 * Architecture:
 * - Storage buffers: Zone, block, block lane, and lane data (uploaded once)
 * - Uniform buffers: View-projection matrix, hover/selection state (updated per frame)
 * - Instanced rendering: One draw call per element type (zones, blocks, lanes)
 * - WGSL shaders: 6 render passes (background, lanes, block lanes, blocks, borders, zones)
 *
 * Rendering order (painter's algorithm):
 * 1. Background (solid color)
 * 2. Lanes (SM backgrounds)
 * 3. Block lanes (lighter backgrounds)
 * 4. Block backgrounds
 * 5. Block borders (1px adaptive outlines with hover)
 * 6. Zones (colored rectangles with selection highlighting)
 *
 * Performance: Handles 10M+ zones at 60 FPS via GPU instancing.
 */

import { ZonesSoA, BlocksSoA, BlockLanesSoA, LanesSoA } from '../utils/types.js';
import { NS_TO_MS } from '../utils/soa-helpers.js';
import { Camera } from '../utils/camera.js';
import {
    SUBLANE_HEIGHT,
    ZONE_FILL_BRIGHTNESS,
    ZONE_COLOR_SATURATION,
    ZONE_PASTEL_MIX,
    ZONE_HOVER_COLOR_R,
    ZONE_HOVER_COLOR_G,
    ZONE_HOVER_COLOR_B,
    ZONE_HOVER_BRIGHTNESS,
    SELECTION_BRIGHTNESS_BOOST,
    ZONE_OUTLINE_BRIGHTNESS,
    ZONE_HOVER_OUTLINE_COLOR_R,
    ZONE_HOVER_OUTLINE_COLOR_G,
    ZONE_HOVER_OUTLINE_COLOR_B,
    ZONE_HOVER_OUTLINE_BRIGHTNESS,
    OUTLINE_DISABLE_ZOOM_THRESHOLD,
    OUTLINE_THICKNESS_MULTIPLIER,
    BLOCK_BORDER_COLOR_R,
    BLOCK_BORDER_COLOR_G,
    BLOCK_BORDER_COLOR_B,
    BLOCK_BORDER_HOVER_COLOR_R,
    BLOCK_BORDER_HOVER_COLOR_G,
    BLOCK_BORDER_HOVER_COLOR_B,
    BLOCK_BORDER_HOVER_BRIGHTNESS,
    BLOCK_BORDER_SELECTION_BOOST,
    BLOCK_BORDER_OPACITY,
    LANE_BG_COLOR_R,
    LANE_BG_COLOR_G,
    LANE_BG_COLOR_B,
    BLOCK_LANE_BG_COLOR_R,
    BLOCK_LANE_BG_COLOR_G,
    BLOCK_LANE_BG_COLOR_B,
    BLOCK_BG_COLOR_R,
    BLOCK_BG_COLOR_G,
    BLOCK_BG_COLOR_B,
    CANVAS_BG_COLOR_R,
    CANVAS_BG_COLOR_G,
    CANVAS_BG_COLOR_B,
    MIN_GPU_BUFFER_SIZE,
    UNIFORM_BUFFER_SIZE,
    BACKGROUND_UNIFORM_BUFFER_SIZE,
    ZONE_BUFFER_FLOATS,
    BLOCK_BUFFER_FLOATS,
    LANE_BUFFER_FLOATS,
    BLOCK_LANE_BUFFER_FLOATS
} from '../utils/constants.js';

/** Single render pass with pipeline and bind group. */
export interface RenderPass {
    pipeline: GPURenderPipeline;
    bindGroup: GPUBindGroup;
}

/** All GPU resources (buffers, pipelines, bind groups). */
export interface GPUResources {
    uniformBuffer: GPUBuffer;                 // Per-frame uniform data (view matrix, hover state)
    backgroundUniformBuffer: GPUBuffer;       // Background-specific uniforms
    buffers: {
        position: GPUBuffer;                  // Zone positions/colors (storage buffer)
        lane: GPUBuffer;                      // Lane geometry (storage buffer)
        blockLane: GPUBuffer;                 // Block lane geometry (storage buffer)
        block: GPUBuffer;                     // Block geometry (storage buffer)
        rowLayout: GPUBuffer;                 // Dynamic row Y offsets and visibility
        zoneVisibility: GPUBuffer;            // Dynamic per-zone visibility
    };
    passes: {
        zone: RenderPass;                     // Zone rendering (colored rectangles)
        lane: RenderPass;                     // Lane backgrounds
        blockLane: RenderPass;                // Block lane backgrounds
        blockBg: RenderPass;                  // Block backgrounds
        block: RenderPass;                    // Block borders (adaptive 1px)
        background: RenderPass;               // Full-screen background
    };
    gpuMemoryUsage: number;                   // Total GPU memory allocated (bytes)
}

/**
 * Shared WGSL code: Full uniform structure with double-precision camera support.
 * Used by shaders that need high-precision transformations (zones, blocks).
 */
const WGSL_FULL_UNIFORMS = `
struct Uniforms {
    viewProj: mat4x4<f32>,
    hoveredId: i32,
    zoomX: f32,
    zoomY: f32,
    selectionStart: f32,
    selectionEnd: f32,
    hasSelection: i32,
    hoveredBlockId: i32,
    camera_x_high: f32,
    camera_x_low: f32,
    camera_y: f32,
    scale_x: f32,
    scale_y: f32,
    viewport_width: f32,
    blinkedId: i32,
    blinkIntensity: f32,
}
`;

/**
 * Shared WGSL code: Emulated double-precision arithmetic functions.
 * Implements "double-single" representation for high-precision calculations.
 */
const WGSL_DOUBLE_PRECISION_FUNCTIONS = `
// Emulated double-precision subtraction using double-single representation
// Computes (ah + al) - (bh + bl) with improved precision
fn ds_sub(ah: f32, al: f32, bh: f32, bl: f32) -> f32 {
    let sh = ah - bh;          // High-order difference
    let th = ah - sh;          // Recover precision loss
    let tl = th - bh + al - bl; // Low-order correction
    return sh + tl;            // Reconstruct with better precision
}

// Emulated double-precision addition using double-single representation
// Computes (ah + al) + (bh + bl) with improved precision
fn ds_add(ah: f32, al: f32, bh: f32, bl: f32) -> f32 {
    let sh = ah + bh;          // High-order sum
    let th = sh - ah;          // Recover precision loss
    let tl = bh - th + al + bl; // Low-order correction
    return sh + tl;            // Reconstruct with better precision
}
`;

const WGSL_ROW_LAYOUT = `
struct RowLayout {
    yOffset: f32,
    visible: f32,
    selected: f32,
    padding1: f32,
}

@group(0) @binding(2) var<storage, read> rowLayouts: array<RowLayout>;
`;

/**
 * WGSL shader for zone rendering with hover and selection highlighting.
 * Uses instanced rendering with per-zone data from storage buffer.
 * Adaptive 1px outlines using fwidth() (disabled below 0.5x Y-zoom).
 */
const ZONE_SHADER = `
${WGSL_FULL_UNIFORMS}
${WGSL_ROW_LAYOUT}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> zones: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> zoneVisibility: array<u32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) zoneCoord: vec2<f32>,
    @location(1) @interpolate(flat) isHovered: f32,
    @location(2) @interpolate(flat) color: vec3<f32>,
    @location(3) @interpolate(flat) zoneSize: vec2<f32>,
    @location(4) @interpolate(flat) isSelected: f32,
    @location(5) @interpolate(flat) blinkIntensity: f32,
}

${WGSL_DOUBLE_PRECISION_FUNCTIONS}

@vertex
fn vertexMain(
    @builtin(vertex_index) vertexIndex: u32,
    @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
    // Read zone data from 3 vec4s (12 floats, aligned)
    let zone0 = zones[instanceIndex * 3u];      // [x_high, x_low, y, width]
    let zone1 = zones[instanceIndex * 3u + 1u]; // [height, r, g, b]
    let zone2 = zones[instanceIndex * 3u + 2u]; // [id, row, pad, pad]

    let x_high = zone0.x;
    let x_low = zone0.y;
    let y = zone0.z;
    let width = zone0.w;
    let height = zone1.x;
    let color = zone1.yzw;
    let id = i32(zone2.x);
    let rowIndex = u32(zone2.y);
    let rowLayout = rowLayouts[rowIndex];

    let vertices = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0)
    );

    let vertex = vertices[vertexIndex];

    // Calculate the center with double-single precision, then enforce a
    // one-pixel minimum raster width without changing the zone's true bounds.
    let viewCenterX = ds_add(
        x_high, x_low, uniforms.camera_x_high, uniforms.camera_x_low);
    let ndcCenterX = viewCenterX * uniforms.scale_x;
    let actualNdcWidth = abs(width * uniforms.scale_x);
    let minimumNdcWidth = 2.0 / uniforms.viewport_width;
    let ndcHalfWidth = max(actualNdcWidth, minimumNdcWidth) * 0.5;

    // Y doesn't need double precision (small range)
    let worldCornerY = y + rowLayout.yOffset
        + vertex.y * height * 0.5;
    let viewY = worldCornerY + uniforms.camera_y;

    // Apply scale to get NDC.
    let ndcPos = vec2<f32>(
        ndcCenterX + vertex.x * ndcHalfWidth,
        viewY * uniforms.scale_y);

    var output: VertexOutput;
    let visible = rowLayout.visible >= 0.5
        && zoneVisibility[instanceIndex] != 0u;
    output.position = select(
        vec4<f32>(2.0, 2.0, 0.0, 1.0),
        vec4<f32>(ndcPos, 0.0, 1.0), visible);
    output.zoneCoord = vertex;
    output.isHovered = select(0.0, 1.0, id == uniforms.hoveredId);
    output.color = color;
    output.zoneSize = vec2<f32>(width, height);
    output.blinkIntensity = select(
        0.0, uniforms.blinkIntensity, id == uniforms.blinkedId);

    // Selection calculation using high-precision position
    let zoneCenterX = ds_add(x_high, x_low, 0.0, 0.0); // Reconstruct absolute position
    let zoneStart = zoneCenterX - width * 0.5;
    let zoneEnd = zoneCenterX + width * 0.5;
    let isFullyInside = uniforms.hasSelection != 0 &&
                       rowLayout.selected >= 0.5 &&
                       zoneStart >= uniforms.selectionStart &&
                       zoneEnd <= uniforms.selectionEnd;
    output.isSelected = select(0.0, 1.0, isFullyInside);

    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    let luminance = dot(input.color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let mutedColor = mix(vec3<f32>(luminance), input.color, ${ZONE_COLOR_SATURATION});
    let pastelColor = mix(mutedColor, vec3<f32>(1.0), ${ZONE_PASTEL_MIX});
    let baseFillColor = pastelColor * ${ZONE_FILL_BRIGHTNESS};
    let hoverFillColor = vec3<f32>(${ZONE_HOVER_COLOR_R}, ${ZONE_HOVER_COLOR_G}, ${ZONE_HOVER_COLOR_B}) * ${ZONE_HOVER_BRIGHTNESS};

    // If both hovered and selected, use hover color (don't stack brightness)
    var fillColor: vec3<f32>;
    if (input.isHovered > ${OUTLINE_DISABLE_ZOOM_THRESHOLD}) {
        fillColor = hoverFillColor;
    } else {
        fillColor = mix(baseFillColor, baseFillColor * ${SELECTION_BRIGHTNESS_BOOST}, input.isSelected);
    }
    fillColor = mix(
        fillColor, vec3<f32>(1.0, 0.78, 0.22), input.blinkIntensity);

    if (uniforms.zoomY < ${OUTLINE_DISABLE_ZOOM_THRESHOLD}) {
        return vec4<f32>(fillColor, 1.0);
    }

    let coord = input.zoneCoord;

    let pixelSizeX = fwidth(coord.x);
    let pixelSizeY = fwidth(coord.y);

    let edgeThicknessX = pixelSizeX * ${OUTLINE_THICKNESS_MULTIPLIER};
    let edgeThicknessY = pixelSizeY * ${OUTLINE_THICKNESS_MULTIPLIER};

    let distFromEdgeX = 1.0 - abs(coord.x);
    let distFromEdgeY = 1.0 - abs(coord.y);

    let isEdge = distFromEdgeX < edgeThicknessX || distFromEdgeY < edgeThicknessY;

    let baseOutlineColor = mutedColor * ${ZONE_OUTLINE_BRIGHTNESS};
    let hoverOutlineColor = vec3<f32>(${ZONE_HOVER_OUTLINE_COLOR_R}, ${ZONE_HOVER_OUTLINE_COLOR_G}, ${ZONE_HOVER_OUTLINE_COLOR_B}) * ${ZONE_HOVER_OUTLINE_BRIGHTNESS};

    // Same logic for outlines - don't stack hover and selection
    var outlineColor: vec3<f32>;
    if (input.isHovered > ${OUTLINE_DISABLE_ZOOM_THRESHOLD}) {
        outlineColor = hoverOutlineColor;
    } else {
        outlineColor = mix(baseOutlineColor, baseOutlineColor * ${SELECTION_BRIGHTNESS_BOOST}, input.isSelected);
    }
    outlineColor = mix(
        outlineColor, vec3<f32>(1.0, 0.94, 0.68), input.blinkIntensity);

    let color = select(fillColor, outlineColor, isEdge);

    return vec4<f32>(color, 1.0);
}
`;

const LANE_SHADER = `
${WGSL_FULL_UNIFORMS}
${WGSL_ROW_LAYOUT}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
struct LaneData {
    geometry: vec4<f32>,
    horizontal: vec4<f32>,
}

@group(0) @binding(1) var<storage, read> lanes: array<LaneData>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

${WGSL_DOUBLE_PRECISION_FUNCTIONS}

@vertex
fn vertexMain(
    @builtin(vertex_index) vertexIndex: u32,
    @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
    let lane = lanes[instanceIndex];
    let rowLayout = rowLayouts[instanceIndex];

    let laneY = lane.geometry.x;
    let laneHeight = lane.geometry.y;
    let mainLane = lane.geometry.z < 0.5;
    let start_high = lane.horizontal.x;
    let start_low = lane.horizontal.y;
    let end_high = lane.horizontal.z;
    let end_low = lane.horizontal.w;

    let vertices = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0)
    );

    let vertex = vertices[vertexIndex];

    let worldX_high = select(start_high, end_high, vertex.x > 0.5);
    let worldX_low = select(start_low, end_low, vertex.x > 0.5);

    // Apply camera transformation using double-precision
    let viewX = ds_add(worldX_high, worldX_low, uniforms.camera_x_high, uniforms.camera_x_low);

    // Y doesn't need double precision
    let worldY = laneY + rowLayout.yOffset + vertex.y * laneHeight;
    let viewY = worldY + uniforms.camera_y;

    // Apply scale to get NDC
    // Background rectangles can become millions of pixels wide at high time
    // zoom. Clip their horizontal vertices before rasterization so their
    // horizontal edges remain stable.
    let timeNdcX = clamp(viewX * uniforms.scale_x, -1.0, 1.0);
    let screenNdcX = select(-1.0, 1.0, vertex.x > 0.5);
    let ndcPos = vec2<f32>(
        select(timeNdcX, screenNdcX, mainLane),
        viewY * uniforms.scale_y);

    var output: VertexOutput;
    output.position = select(
        vec4<f32>(2.0, 2.0, 0.0, 1.0),
        vec4<f32>(ndcPos, 0.0, 1.0), rowLayout.visible >= 0.5);
    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(
        ${LANE_BG_COLOR_R}, ${LANE_BG_COLOR_G}, ${LANE_BG_COLOR_B}, 1.0);
}
`;

const BLOCK_LANE_SHADER = `
${WGSL_FULL_UNIFORMS}
${WGSL_ROW_LAYOUT}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
struct BlockLaneData {
    geometry: vec4<f32>,
    horizontal: vec4<f32>,
}

@group(0) @binding(1) var<storage, read> blockLanes: array<BlockLaneData>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

${WGSL_DOUBLE_PRECISION_FUNCTIONS}

@vertex
fn vertexMain(
    @builtin(vertex_index) vertexIndex: u32,
    @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
    let blockLane = blockLanes[instanceIndex];

    let y = blockLane.geometry.x;
    let height = blockLane.geometry.y;
    let rowIndex = u32(blockLane.geometry.z);
    let rowLayout = rowLayouts[rowIndex];
    let mainLane = blockLane.geometry.w < 0.5;
    let start_high = blockLane.horizontal.x;
    let start_low = blockLane.horizontal.y;
    let end_high = blockLane.horizontal.z;
    let end_low = blockLane.horizontal.w;

    let vertices = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0)
    );

    let vertex = vertices[vertexIndex];

    let worldX_high = select(start_high, end_high, vertex.x > 0.5);
    let worldX_low = select(start_low, end_low, vertex.x > 0.5);

    // Apply camera transformation using double-precision
    let viewX = ds_add(worldX_high, worldX_low, uniforms.camera_x_high, uniforms.camera_x_low);

    // Y doesn't need double precision
    let worldY = y + rowLayout.yOffset + vertex.y * height;
    let viewY = worldY + uniforms.camera_y;

    // Apply scale to get NDC
    // Keep off-screen horizontal coordinates within the rasterizer viewport.
    let timeNdcX = clamp(viewX * uniforms.scale_x, -1.0, 1.0);
    let screenNdcX = select(-1.0, 1.0, vertex.x > 0.5);
    let ndcPos = vec2<f32>(
        select(timeNdcX, screenNdcX, mainLane),
        viewY * uniforms.scale_y);

    var output: VertexOutput;
    if (rowLayout.visible < 0.5) {
        output.position = vec4<f32>(2.0, 2.0, 0.0, 1.0);
        return output;
    }
    output.position = vec4<f32>(ndcPos, 0.0, 1.0);
    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(${BLOCK_LANE_BG_COLOR_R}, ${BLOCK_LANE_BG_COLOR_G}, ${BLOCK_LANE_BG_COLOR_B}, 1.0);
}
`;

const BLOCK_BG_SHADER = `
${WGSL_FULL_UNIFORMS}
${WGSL_ROW_LAYOUT}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> blocks: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> zoneVisibility: array<u32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

${WGSL_DOUBLE_PRECISION_FUNCTIONS}

@vertex
fn vertexMain(
    @builtin(vertex_index) vertexIndex: u32,
    @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
    // Read block data from 2 vec4s (8 floats, aligned)
    let block0 = blocks[instanceIndex * 2u];      // [startX_high, startX_low, y, endX_high]
    let block1 = blocks[instanceIndex * 2u + 1u]; // [endX_low, height, row, firstZone]

    let startX_high = block0.x;
    let startX_low = block0.y;
    let y = block0.z;
    let endX_high = block0.w;
    let endX_low = block1.x;
    let height = block1.y;
    let rowLayout = rowLayouts[u32(block1.z)];
    let blockVisible = zoneVisibility[u32(block1.w)] != 0u;

    let vertices = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0)
    );

    let vertex = vertices[vertexIndex];

    // Block starts at startX and extends to endX
    // Left edge (vertex.x=0): X = startX
    // Right edge (vertex.x=1): X = endX
    // Use select() to pick between the two double-single values without arithmetic
    let worldCornerX_high = select(startX_high, endX_high, vertex.x > 0.5);
    let worldCornerX_low = select(startX_low, endX_low, vertex.x > 0.5);

    // Apply camera transformation using double-precision
    let viewX = ds_add(worldCornerX_high, worldCornerX_low, uniforms.camera_x_high, uniforms.camera_x_low);

    // Y doesn't need double precision
    let worldCornerY = y + rowLayout.yOffset + vertex.y * height;
    let viewY = worldCornerY + uniforms.camera_y;

    // Apply scale to get NDC
    // Keep off-screen horizontal coordinates within the rasterizer viewport.
    let ndcPos = vec2<f32>(
        clamp(viewX * uniforms.scale_x, -1.0, 1.0),
        viewY * uniforms.scale_y);

    var output: VertexOutput;
    if (rowLayout.visible < 0.5) {
        output.position = vec4<f32>(2.0, 2.0, 0.0, 1.0);
        return output;
    }
    output.position = select(
        vec4<f32>(2.0, 2.0, 0.0, 1.0),
        vec4<f32>(ndcPos, 0.0, 1.0), blockVisible);
    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(${BLOCK_BG_COLOR_R}, ${BLOCK_BG_COLOR_G}, ${BLOCK_BG_COLOR_B}, 1.0);
}
`;

const BLOCK_BORDER_SHADER = `
${WGSL_FULL_UNIFORMS}
${WGSL_ROW_LAYOUT}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> blocks: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> zoneVisibility: array<u32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) ndcPos: vec2<f32>,
    @location(1) @interpolate(flat) blockStart: vec2<f32>,
    @location(2) @interpolate(flat) blockEnd: vec2<f32>,
    @location(3) @interpolate(flat) isHovered: f32,
    @location(4) @interpolate(flat) isSelected: f32,
}

${WGSL_DOUBLE_PRECISION_FUNCTIONS}

@vertex
fn vertexMain(
    @builtin(vertex_index) vertexIndex: u32,
    @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
    // Read block data from 2 vec4s (8 floats, aligned)
    let block0 = blocks[instanceIndex * 2u];      // [startX_high, startX_low, y, endX_high]
    let block1 = blocks[instanceIndex * 2u + 1u]; // [endX_low, height, row, firstZone]

    let startX_high = block0.x;
    let startX_low = block0.y;
    let y = block0.z;
    let endX_high = block0.w;
    let endX_low = block1.x;
    let height = block1.y;
    let rowLayout = rowLayouts[u32(block1.z)];
    let blockVisible = zoneVisibility[u32(block1.w)] != 0u;

    let vertices = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0)
    );

    let vertex = vertices[vertexIndex];

    // Y doesn't need double precision
    let centerY = y + rowLayout.yOffset + height * 0.5;
    let worldCornerY = centerY + vertex.y * height * 0.5;
    let viewY = worldCornerY + uniforms.camera_y;

    // Selection calculation using high-precision
    // blockStart = startX, blockEnd = endX (use ds_add to collapse to f32 for comparison)
    let blockStart = ds_add(startX_high, startX_low, 0.0, 0.0);
    let blockEnd = ds_add(endX_high, endX_low, 0.0, 0.0);
    let isFullyInside = uniforms.hasSelection != 0 &&
                       rowLayout.selected >= 0.5 &&
                       blockStart >= uniforms.selectionStart &&
                       blockEnd <= uniforms.selectionEnd;

    // Compute the original block bounds in NDC for edge detection. Clamp the
    // rasterized horizontal vertices to the viewport exactly as the block
    // background does: a long block can otherwise produce a quad millions of
    // pixels wide, making fixed-function clipping and f32 derivatives unstable.
    let blockStartView = ds_add(startX_high, startX_low, uniforms.camera_x_high, uniforms.camera_x_low);
    let blockEndView = ds_add(endX_high, endX_low, uniforms.camera_x_high, uniforms.camera_x_low);
    let blockStartNdc = vec2<f32>(
        blockStartView * uniforms.scale_x,
        (y + rowLayout.yOffset + uniforms.camera_y) * uniforms.scale_y);
    let blockEndNdc = vec2<f32>(
        blockEndView * uniforms.scale_x,
        (y + rowLayout.yOffset + height + uniforms.camera_y) * uniforms.scale_y);
    let clippedStartX = clamp(blockStartNdc.x, -1.0, 1.0);
    let clippedEndX = clamp(blockEndNdc.x, -1.0, 1.0);
    let ndcPos = vec2<f32>(
        select(clippedStartX, clippedEndX, vertex.x > 0.0),
        viewY * uniforms.scale_y);

    var output: VertexOutput;
    output.position = select(
        vec4<f32>(2.0, 2.0, 0.0, 1.0),
        vec4<f32>(ndcPos, 0.0, 1.0),
        rowLayout.visible >= 0.5 && blockVisible);
    output.ndcPos = ndcPos;
    output.blockStart = blockStartNdc;
    output.blockEnd = blockEndNdc;
    output.isHovered = select(0.0, 1.0, i32(instanceIndex) == uniforms.hoveredBlockId);
    output.isSelected = select(0.0, 1.0, isFullyInside);
    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    if (uniforms.zoomY < ${OUTLINE_DISABLE_ZOOM_THRESHOLD}) {
        discard;
    }

    // Derivatives stay pixel-sized because the interpolant uses the same
    // clipped NDC geometry as the rasterizer.
    let pixelSizeX = fwidth(input.ndcPos.x);
    let pixelSizeY = fwidth(input.ndcPos.y);

    // Retain the unclipped bounds so off-screen vertical edges remain hidden.
    let distFromLeftEdge = input.ndcPos.x - input.blockStart.x;
    let distFromRightEdge = input.blockEnd.x - input.ndcPos.x;
    let distFromTopEdge = input.blockEnd.y - input.ndcPos.y;
    let distFromBottomEdge = input.ndcPos.y - input.blockStart.y;

    // Edge threshold in NDC (one pixel worth, same as zones)
    let edgeThicknessX = pixelSizeX * ${OUTLINE_THICKNESS_MULTIPLIER};
    let edgeThicknessY = pixelSizeY * ${OUTLINE_THICKNESS_MULTIPLIER};

    // Check if we're on an edge
    let isEdge = distFromLeftEdge < edgeThicknessX ||
                 distFromRightEdge < edgeThicknessX ||
                 distFromTopEdge < edgeThicknessY ||
                 distFromBottomEdge < edgeThicknessY;

    if (!isEdge) {
        discard;
    }

    let baseColor = vec3<f32>(${BLOCK_BORDER_COLOR_R}, ${BLOCK_BORDER_COLOR_G}, ${BLOCK_BORDER_COLOR_B});
    let hoverColor = vec3<f32>(${BLOCK_BORDER_HOVER_COLOR_R}, ${BLOCK_BORDER_HOVER_COLOR_G}, ${BLOCK_BORDER_HOVER_COLOR_B}) * ${BLOCK_BORDER_HOVER_BRIGHTNESS};
    var color = mix(baseColor, hoverColor, input.isHovered);

    // Brighten if within selection (match zone brightness boost)
    color = color + vec3<f32>(${BLOCK_BORDER_SELECTION_BOOST}, ${BLOCK_BORDER_SELECTION_BOOST}, ${BLOCK_BORDER_SELECTION_BOOST}) * input.isSelected;

    return vec4<f32>(color, ${BLOCK_BORDER_OPACITY});
}
`;

const BACKGROUND_SHADER = `
struct Uniforms {
    camera_x_high: f32,
    camera_x_low: f32,
    camera_y: f32,
    scale_x: f32,
    scale_y: f32,
    timeRange_high: f32,
    timeRange_low: f32,
    worldHeight: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

${WGSL_DOUBLE_PRECISION_FUNCTIONS}

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    let vertices = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0)
    );

    let vertex = vertices[vertexIndex];

    // Background spans from X=0 to timeRange
    // Left edge (vertex.x=0): X = 0.0
    // Right edge (vertex.x=1): X = timeRange
    let worldX_high = select(0.0, uniforms.timeRange_high, vertex.x > 0.5);
    let worldX_low = select(0.0, uniforms.timeRange_low, vertex.x > 0.5);

    // Apply camera transformation using double-precision
    let viewX = ds_add(worldX_high, worldX_low, uniforms.camera_x_high, uniforms.camera_x_low);

    // Y doesn't need double precision
    let worldY = vertex.y * uniforms.worldHeight;
    let viewY = worldY + uniforms.camera_y;

    // Apply scale to get NDC
    let ndcPos = vec2<f32>(viewX * uniforms.scale_x, viewY * uniforms.scale_y);

    var output: VertexOutput;
    output.position = vec4<f32>(ndcPos, 0.0, 1.0);
    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(${CANVAS_BG_COLOR_R}, ${CANVAS_BG_COLOR_G}, ${CANVAS_BG_COLOR_B}, 1.0);
}
`;

/**
 * Creates a uniform buffer for per-frame shader data.
 * Uniform buffers are small (typically <256 bytes) and updated frequently.
 * COPY_DST allows writeBuffer() to update data each frame.
 */
function createUniformBuffer(device: GPUDevice, size: number): GPUBuffer {
    return device.createBuffer({
        size,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
}

/**
 * Creates standard bind group layout for instanced rendering.
 *
 * Layout:
 * - Binding 0: Uniform buffer (view-projection matrix, hover state)
 * - Binding 1: Storage buffer (instance data: zones, blocks, lanes)
 *
 * fragmentAccess controls whether uniforms are visible in fragment shader
 * (needed for hover/selection highlighting, not needed for simple geometry).
 */
function createStorageBindGroupLayout(device: GPUDevice, fragmentAccess: boolean = false): GPUBindGroupLayout {
    return device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: fragmentAccess
                    ? GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT
                    : GPUShaderStage.VERTEX,
                buffer: { type: 'uniform' }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.VERTEX,
                buffer: { type: 'read-only-storage' }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.VERTEX,
                buffer: { type: 'read-only-storage' }
            },
            {
                binding: 3,
                visibility: GPUShaderStage.VERTEX,
                buffer: { type: 'read-only-storage' }
            }
        ]
    });
}

/**
 * Creates a render pipeline for instanced geometry.
 *
 * Parameters:
 * - shaderCode: WGSL shader source (must have vertexMain/fragmentMain entry points)
 * - uniformBuffer: Per-frame data (view-projection, hover state)
 * - storageBuffer: Instance data (zones, blocks, lanes)
 * - hasBlending: Enable alpha blending for transparency (block borders, zones)
 * - fragmentAccess: Expose uniforms to fragment shader (for hover/selection)
 *
 * Returns pipeline and bind group ready for use in render pass.
 */
function createSimplePipeline(
    device: GPUDevice,
    format: GPUTextureFormat,
    shaderCode: string,
    uniformBuffer: GPUBuffer,
    storageBuffer: GPUBuffer,
    rowLayoutBuffer: GPUBuffer,
    zoneVisibilityBuffer: GPUBuffer,
    hasBlending: boolean = false,
    fragmentAccess: boolean = false
): RenderPass {
    const shaderModule = device.createShaderModule({ code: shaderCode });
    const bindGroupLayout = createStorageBindGroupLayout(device, fragmentAccess);

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: storageBuffer } },
            { binding: 2, resource: { buffer: rowLayoutBuffer } },
            { binding: 3, resource: { buffer: zoneVisibilityBuffer } }
        ]
    });

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });

    const blendState = hasBlending ? {
        blend: {
            color: {
                srcFactor: 'src-alpha' as GPUBlendFactor,
                dstFactor: 'one-minus-src-alpha' as GPUBlendFactor,
            },
            alpha: {
                srcFactor: 'one' as GPUBlendFactor,
                dstFactor: 'one-minus-src-alpha' as GPUBlendFactor,
            }
        }
    } : {};

    const pipeline = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
            module: shaderModule,
            entryPoint: 'vertexMain',
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fragmentMain',
            targets: [{ format, ...blendState }]
        },
        primitive: {
            topology: 'triangle-list',
        },
    });

    return { pipeline, bindGroup };
}

/**
 * Creates specialized pipeline for full-screen background rectangle.
 *
 * Unlike other pipelines, this only needs uniforms (no storage buffer).
 * Background dimensions (timeRange × worldHeight) are passed via uniforms
 * and updated when trace data changes.
 */
function createBackgroundPipeline(
    device: GPUDevice,
    format: GPUTextureFormat,
    backgroundUniformBuffer: GPUBuffer
): RenderPass {
    const shaderModule = device.createShaderModule({ code: BACKGROUND_SHADER });
    const bindGroupLayout = device.createBindGroupLayout({
        entries: [{
            binding: 0,
            visibility: GPUShaderStage.VERTEX,
            buffer: { type: 'uniform' }
        }]
    });

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{ binding: 0, resource: { buffer: backgroundUniformBuffer } }]
    });

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });

    const pipeline = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
            module: shaderModule,
            entryPoint: 'vertexMain',
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fragmentMain',
            targets: [{ format }]
        },
        primitive: {
            topology: 'triangle-list',
        },
    });

    return { pipeline, bindGroup };
}

/**
 * Uploads trace data to GPU storage buffers for instanced rendering (SoA version).
 *
 * Creates 4 storage buffers:
 * - positionBuffer: Zone data (12 floats per zone, aligned to 3 vec4s):
 *   vec4 #0: [x_high, x_low, y, width]
 *   vec4 #1: [height, r, g, b]
 *   vec4 #2: [id, pad, pad, pad]
 *   Zone X uses double-single precision to avoid Float32 precision loss at extreme zoom
 * - laneBuffer: Lane geometry (8 floats per lane, 2 vec4s):
 *   [y, height, padding, padding, start_high, start_low, end_high, end_low]
 *   Lane width uses double-single precision for high-precision rendering at extreme zoom
 * - blockLaneBuffer: Block lane geometry (8 floats per block lane, 2 vec4s):
 *   [y, height, padding, padding, start_high, start_low, end_high, end_low]
 *   Block lane width uses double-single precision for high-precision rendering at extreme zoom
 * - blockBuffer: Block geometry (8 floats per block, aligned to 2 vec4s):
 *   vec4 #0: [startX_high, startX_low, y, endX_high]
 *   vec4 #1: [endX_low, height, pad, pad]
 *   Block startX and endX use double-single precision to avoid Float32 precision loss at extreme zoom
 *
 * Converts nanoseconds (SoA storage) to milliseconds (GPU rendering).
 * Uses mappedAtCreation for efficient one-time upload (no COPY_DST needed).
 * Buffers are read-only after creation, enabling optimal GPU caching.
 * Returns total GPU memory usage for stats display.
 */
export function createGPUBuffers(
    device: GPUDevice,
    zones: ZonesSoA,
    blocks: BlocksSoA,
    blockLanes: BlockLanesSoA,
    lanes: LanesSoA,
    rowBaseYs: Float32Array,
    rowTimeBounded: Float32Array,
    rowLayout: Float32Array,
    zoneVisibility: Uint32Array
): { positionBuffer: GPUBuffer; laneBuffer: GPUBuffer; blockLaneBuffer: GPUBuffer; blockBuffer: GPUBuffer; rowLayoutBuffer: GPUBuffer; zoneVisibilityBuffer: GPUBuffer; gpuMemoryUsage: number } {
    const positions = new Float32Array(zones.count * ZONE_BUFFER_FLOATS);
    for (let i = 0; i < zones.count; i++) {
        // Compute zone center X in nanoseconds, then convert to milliseconds
        const centerXNs = (zones.startsX[i] + zones.endsX[i]) / 2;
        const centerXMs = centerXNs * NS_TO_MS;
        const [x_high, x_low] = Camera.splitDouble(centerXMs);

        // Compute zone width in nanoseconds, then convert to milliseconds
        const widthNs = zones.endsX[i] - zones.startsX[i];
        const widthMs = widthNs * NS_TO_MS;

        // Unpack colors from Uint8Array (0-255 range) to float (0-1 range)
        const r = zones.colors[i * 3 + 0] / 255;
        const g = zones.colors[i * 3 + 1] / 255;
        const b = zones.colors[i * 3 + 2] / 255;

        // vec4 #0: [x_high, x_low, y, width]
        positions[i * ZONE_BUFFER_FLOATS + 0] = x_high;
        positions[i * ZONE_BUFFER_FLOATS + 1] = x_low;
        positions[i * ZONE_BUFFER_FLOATS + 2] = zones.ys[i];
        positions[i * ZONE_BUFFER_FLOATS + 3] = widthMs;
        // vec4 #1: [height, r, g, b]
        positions[i * ZONE_BUFFER_FLOATS + 4] = SUBLANE_HEIGHT;  // Constant height
        positions[i * ZONE_BUFFER_FLOATS + 5] = r;
        positions[i * ZONE_BUFFER_FLOATS + 6] = g;
        positions[i * ZONE_BUFFER_FLOATS + 7] = b;
        // vec4 #2: [id, pad, pad, pad] - padding already zero-initialized
        positions[i * ZONE_BUFFER_FLOATS + 8] = i;  // Array index is the ID
        positions[i * ZONE_BUFFER_FLOATS + 9] = zones.smIndices[i];
    }

    const positionBuffer = device.createBuffer({
        size: positions.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Float32Array(positionBuffer.getMappedRange()).set(positions);
    positionBuffer.unmap();

    const laneData = new Float32Array(lanes.count * LANE_BUFFER_FLOATS);
    for (let i = 0; i < lanes.count; i++) {
        const startMs = lanes.startsX[i] * NS_TO_MS;
        const endMs = lanes.widths[i] * NS_TO_MS;
        const [startHigh, startLow] = Camera.splitDouble(startMs);
        const [endHigh, endLow] = Camera.splitDouble(endMs);
        laneData[i * LANE_BUFFER_FLOATS + 0] = rowBaseYs[i];
        laneData[i * LANE_BUFFER_FLOATS + 1] = lanes.heights[i];
        laneData[i * LANE_BUFFER_FLOATS + 2] = rowTimeBounded[i];
        laneData[i * LANE_BUFFER_FLOATS + 4] = startHigh;
        laneData[i * LANE_BUFFER_FLOATS + 5] = startLow;
        laneData[i * LANE_BUFFER_FLOATS + 6] = endHigh;
        laneData[i * LANE_BUFFER_FLOATS + 7] = endLow;
    }

    const laneBuffer = device.createBuffer({
        size: laneData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Float32Array(laneBuffer.getMappedRange()).set(laneData);
    laneBuffer.unmap();

    const blockLaneData = new Float32Array(blockLanes.count * BLOCK_LANE_BUFFER_FLOATS);
    for (let i = 0; i < blockLanes.count; i++) {
        const startMs = blockLanes.startsX[i] * NS_TO_MS;
        const endMs = blockLanes.widths[i] * NS_TO_MS;
        const [startHigh, startLow] = Camera.splitDouble(startMs);
        const [endHigh, endLow] = Camera.splitDouble(endMs);
        blockLaneData[i * BLOCK_LANE_BUFFER_FLOATS + 0] = blockLanes.ys[i];
        blockLaneData[i * BLOCK_LANE_BUFFER_FLOATS + 1] = blockLanes.heights[i];
        blockLaneData[i * BLOCK_LANE_BUFFER_FLOATS + 2] =
            blockLanes.smIndices[i];
        blockLaneData[i * BLOCK_LANE_BUFFER_FLOATS + 3] =
            rowTimeBounded[blockLanes.smIndices[i]];
        blockLaneData[i * BLOCK_LANE_BUFFER_FLOATS + 4] = startHigh;
        blockLaneData[i * BLOCK_LANE_BUFFER_FLOATS + 5] = startLow;
        blockLaneData[i * BLOCK_LANE_BUFFER_FLOATS + 6] = endHigh;
        blockLaneData[i * BLOCK_LANE_BUFFER_FLOATS + 7] = endLow;
    }

    const blockLaneBuffer = device.createBuffer({
        size: Math.max(MIN_GPU_BUFFER_SIZE, blockLaneData.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Float32Array(blockLaneBuffer.getMappedRange()).set(blockLaneData);
    blockLaneBuffer.unmap();

    const blockData = new Float32Array(blocks.count * BLOCK_BUFFER_FLOATS);
    for (let i = 0; i < blocks.count; i++) {
        // Convert block startX and endX from nanoseconds to milliseconds
        const startXMs = blocks.startsX[i] * NS_TO_MS;
        const endXMs = blocks.endsX[i] * NS_TO_MS;
        const [startX_high, startX_low] = Camera.splitDouble(startXMs);
        const [endX_high, endX_low] = Camera.splitDouble(endXMs);

        // vec4 #0: [startX_high, startX_low, y, endX_high]
        blockData[i * BLOCK_BUFFER_FLOATS + 0] = startX_high;
        blockData[i * BLOCK_BUFFER_FLOATS + 1] = startX_low;
        blockData[i * BLOCK_BUFFER_FLOATS + 2] = blocks.ys[i];
        blockData[i * BLOCK_BUFFER_FLOATS + 3] = endX_high;
        // vec4 #1: [endX_low, height, pad, pad] - padding already zero-initialized
        blockData[i * BLOCK_BUFFER_FLOATS + 4] = endX_low;
        blockData[i * BLOCK_BUFFER_FLOATS + 5] = blocks.heights[i];
        blockData[i * BLOCK_BUFFER_FLOATS + 6] = blocks.smIndices[i];
        blockData[i * BLOCK_BUFFER_FLOATS + 7] =
            blocks.zonesStartIndices[i];
    }

    const blockBuffer = device.createBuffer({
        size: Math.max(MIN_GPU_BUFFER_SIZE, blockData.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Float32Array(blockBuffer.getMappedRange()).set(blockData);
    blockBuffer.unmap();

    const rowLayoutBuffer = device.createBuffer({
        size: Math.max(MIN_GPU_BUFFER_SIZE, rowLayout.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Float32Array(rowLayoutBuffer.getMappedRange()).set(rowLayout);
    rowLayoutBuffer.unmap();

    const zoneVisibilityBuffer = device.createBuffer({
        size: Math.max(MIN_GPU_BUFFER_SIZE, zoneVisibility.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Uint32Array(zoneVisibilityBuffer.getMappedRange()).set(
        zoneVisibility);
    zoneVisibilityBuffer.unmap();

    const gpuMemoryUsage = positions.byteLength + laneData.byteLength +
        Math.max(MIN_GPU_BUFFER_SIZE, blockLaneData.byteLength) + Math.max(MIN_GPU_BUFFER_SIZE, blockData.byteLength) +
        Math.max(MIN_GPU_BUFFER_SIZE, rowLayout.byteLength)
        + Math.max(MIN_GPU_BUFFER_SIZE, zoneVisibility.byteLength)
        + (64 + 32) + (64 + 16);

    return { positionBuffer, laneBuffer, blockLaneBuffer, blockBuffer,
        rowLayoutBuffer, zoneVisibilityBuffer, gpuMemoryUsage };
}

/**
 * Creates all render pipelines and uniform buffers.
 *
 * Returns GPUResources with 6 render passes:
 * 1. background - Full-screen background rectangle
 * 2. lane - SM lane backgrounds
 * 3. blockLane - Block lane backgrounds (lighter than lanes)
 * 4. blockBg - Block backgrounds (darker than block lanes)
 * 5. block - Block borders with hover highlighting
 * 6. zone - Zone rectangles with selection highlighting
 *
 * Each pass contains a pipeline and bind group ready for immediate use.
 * Uniform buffers (96 bytes main, 80 bytes background) are updated per frame.
 */
export function createPipelines(
    device: GPUDevice,
    format: GPUTextureFormat,
    positionBuffer: GPUBuffer,
    laneBuffer: GPUBuffer,
    blockLaneBuffer: GPUBuffer,
    blockBuffer: GPUBuffer,
    rowLayoutBuffer: GPUBuffer,
    zoneVisibilityBuffer: GPUBuffer,
    cachedPasses?: GPUResources['passes']
): GPUResources {
    const uniformBuffer = createUniformBuffer(device, UNIFORM_BUFFER_SIZE);
    const backgroundUniformBuffer = createUniformBuffer(device, BACKGROUND_UNIFORM_BUFFER_SIZE);

    const RebindStoragePass = (
        pipeline: GPURenderPipeline,
        storageBuffer: GPUBuffer
    ): RenderPass => {
        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: storageBuffer } },
                { binding: 2, resource: { buffer: rowLayoutBuffer } },
                { binding: 3, resource: { buffer: zoneVisibilityBuffer } }
            ]
        });
        return { pipeline, bindGroup };
    };

    const RebindBackgroundPass = (
        pipeline: GPURenderPipeline
    ): RenderPass => {
        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [{
                binding: 0,
                resource: { buffer: backgroundUniformBuffer }
            }]
        });
        return { pipeline, bindGroup };
    };

    const passes = cachedPasses ? {
        zone: RebindStoragePass(cachedPasses.zone.pipeline, positionBuffer),
        lane: RebindStoragePass(cachedPasses.lane.pipeline, laneBuffer),
        blockLane: RebindStoragePass(
            cachedPasses.blockLane.pipeline, blockLaneBuffer),
        blockBg: RebindStoragePass(
            cachedPasses.blockBg.pipeline, blockBuffer),
        block: RebindStoragePass(cachedPasses.block.pipeline, blockBuffer),
        background: RebindBackgroundPass(cachedPasses.background.pipeline)
    } : {
        zone: createSimplePipeline(device, format, ZONE_SHADER,
            uniformBuffer, positionBuffer, rowLayoutBuffer,
            zoneVisibilityBuffer, true, true),
        lane: createSimplePipeline(device, format, LANE_SHADER,
            uniformBuffer, laneBuffer, rowLayoutBuffer,
            zoneVisibilityBuffer),
        blockLane: createSimplePipeline(device, format, BLOCK_LANE_SHADER,
            uniformBuffer, blockLaneBuffer, rowLayoutBuffer,
            zoneVisibilityBuffer),
        blockBg: createSimplePipeline(device, format, BLOCK_BG_SHADER,
            uniformBuffer, blockBuffer, rowLayoutBuffer,
            zoneVisibilityBuffer),
        block: createSimplePipeline(device, format, BLOCK_BORDER_SHADER,
            uniformBuffer, blockBuffer, rowLayoutBuffer,
            zoneVisibilityBuffer, true, true),
        background: createBackgroundPipeline(
            device, format, backgroundUniformBuffer)
    };

    return {
        uniformBuffer,
        backgroundUniformBuffer,
        buffers: {
            position: positionBuffer,
            lane: laneBuffer,
            blockLane: blockLaneBuffer,
            block: blockBuffer,
            rowLayout: rowLayoutBuffer,
            zoneVisibility: zoneVisibilityBuffer
        },
        passes,
        gpuMemoryUsage: 0
    };
}
