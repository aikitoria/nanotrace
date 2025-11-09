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

import { Zone, Block, BlockLane, Lane } from '../utils/types.js';
import { Camera } from '../utils/camera.js';

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

/**
 * WGSL shader for zone rendering with hover and selection highlighting.
 * Uses instanced rendering with per-zone data from storage buffer.
 * Adaptive 1px outlines using fwidth() (disabled below 0.5x Y-zoom).
 */
const ZONE_SHADER = `
${WGSL_FULL_UNIFORMS}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> zones: array<vec4<f32>>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) zoneCoord: vec2<f32>,
    @location(1) @interpolate(flat) isHovered: f32,
    @location(2) @interpolate(flat) color: vec3<f32>,
    @location(3) @interpolate(flat) zoneSize: vec2<f32>,
    @location(4) @interpolate(flat) isSelected: f32,
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
    let zone2 = zones[instanceIndex * 3u + 2u]; // [id, pad, pad, pad]

    let x_high = zone0.x;
    let x_low = zone0.y;
    let y = zone0.z;
    let width = zone0.w;
    let height = zone1.x;
    let color = zone1.yzw;
    let id = i32(zone2.x);

    let vertices = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0)
    );

    let vertex = vertices[vertexIndex];

    // Step 1: Calculate world corner position from center in double-precision
    // Add small vertex offset to low component (vertex offset is typically < 0.001)
    let cornerOffsetX = vertex.x * width * 0.5;
    let worldCornerX_high = x_high;
    let worldCornerX_low = x_low + cornerOffsetX;

    // Step 2: Apply camera transformation using double-precision
    // This is where precision loss would occur in regular f32
    let viewX = ds_add(worldCornerX_high, worldCornerX_low, uniforms.camera_x_high, uniforms.camera_x_low);

    // Y doesn't need double precision (small range)
    let worldCornerY = y + vertex.y * height * 0.5;
    let viewY = worldCornerY + uniforms.camera_y;

    // Step 3: Apply scale to get NDC
    let ndcPos = vec2<f32>(viewX * uniforms.scale_x, viewY * uniforms.scale_y);

    var output: VertexOutput;
    output.position = vec4<f32>(ndcPos, 0.0, 1.0);
    output.zoneCoord = vertex;
    output.isHovered = select(0.0, 1.0, id == uniforms.hoveredId);
    output.color = color;
    output.zoneSize = vec2<f32>(width, height);

    // Selection calculation using high-precision position
    let zoneCenterX = ds_add(x_high, x_low, 0.0, 0.0); // Reconstruct absolute position
    let zoneStart = zoneCenterX - width * 0.5;
    let zoneEnd = zoneCenterX + width * 0.5;
    let isFullyInside = uniforms.hasSelection != 0 &&
                       zoneStart >= uniforms.selectionStart &&
                       zoneEnd <= uniforms.selectionEnd;
    output.isSelected = select(0.0, 1.0, isFullyInside);

    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    let baseFillColor = input.color * 0.55;
    let hoverFillColor = vec3<f32>(0.22, 0.74, 0.97) * 0.95;

    // If both hovered and selected, use hover color (don't stack brightness)
    var fillColor: vec3<f32>;
    if (input.isHovered > 0.5) {
        fillColor = hoverFillColor;
    } else {
        fillColor = mix(baseFillColor, baseFillColor * 1.55, input.isSelected);
    }

    if (uniforms.zoomY < 0.5) {
        return vec4<f32>(fillColor, 1.0);
    }

    let coord = input.zoneCoord;

    let pixelSizeX = fwidth(coord.x);
    let pixelSizeY = fwidth(coord.y);

    let edgeThicknessX = pixelSizeX * 1.0;
    let edgeThicknessY = pixelSizeY * 1.0;

    let distFromEdgeX = 1.0 - abs(coord.x);
    let distFromEdgeY = 1.0 - abs(coord.y);

    let isEdge = distFromEdgeX < edgeThicknessX || distFromEdgeY < edgeThicknessY;

    let baseOutlineColor = input.color * 0.98;
    let hoverOutlineColor = vec3<f32>(0.38, 0.82, 1.0) * 1.15;

    // Same logic for outlines - don't stack hover and selection
    var outlineColor: vec3<f32>;
    if (input.isHovered > 0.5) {
        outlineColor = hoverOutlineColor;
    } else {
        outlineColor = mix(baseOutlineColor, baseOutlineColor * 1.55, input.isSelected);
    }

    let color = select(fillColor, outlineColor, isEdge);

    return vec4<f32>(color, 1.0);
}
`;

const LANE_SHADER = `
struct Uniforms {
    viewProj: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> lanes: array<vec4<f32>>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vertexMain(
    @builtin(vertex_index) vertexIndex: u32,
    @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
    let lane = lanes[instanceIndex];
    let laneY = lane.x;
    let laneHeight = lane.y;
    let laneWidth = lane.z;

    let vertices = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0)
    );

    let vertex = vertices[vertexIndex];
    let worldPos = vec2<f32>(vertex.x * laneWidth, laneY + vertex.y * laneHeight);

    var output: VertexOutput;
    output.position = uniforms.viewProj * vec4<f32>(worldPos, 0.0, 1.0);
    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.071, 0.071, 0.078, 1.0);
}
`;

const BLOCK_LANE_SHADER = `
struct Uniforms {
    viewProj: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> blockLanes: array<vec4<f32>>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vertexMain(
    @builtin(vertex_index) vertexIndex: u32,
    @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
    let blockLane = blockLanes[instanceIndex];
    let y = blockLane.x;
    let height = blockLane.y;
    let width = blockLane.z;

    let vertices = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0)
    );

    let vertex = vertices[vertexIndex];
    let worldPos = vec2<f32>(vertex.x * width, y + vertex.y * height);

    var output: VertexOutput;
    output.position = uniforms.viewProj * vec4<f32>(worldPos, 0.0, 1.0);
    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.086, 0.086, 0.094, 1.0);
}
`;

const BLOCK_BG_SHADER = `
${WGSL_FULL_UNIFORMS}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> blocks: array<vec4<f32>>;

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
    let block0 = blocks[instanceIndex * 2u];      // [startX_high, startX_low, y, width]
    let block1 = blocks[instanceIndex * 2u + 1u]; // [height, pad, pad, pad]

    let startX_high = block0.x;
    let startX_low = block0.y;
    let y = block0.z;
    let width = block0.w;
    let height = block1.x;

    let vertices = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0)
    );

    let vertex = vertices[vertexIndex];

    // Calculate corner position with double-precision for X
    let cornerOffsetX = vertex.x * width;
    let worldCornerX_high = startX_high;
    let worldCornerX_low = startX_low + cornerOffsetX;

    // Apply camera transformation using double-precision
    let viewX = ds_add(worldCornerX_high, worldCornerX_low, uniforms.camera_x_high, uniforms.camera_x_low);

    // Y doesn't need double precision
    let worldCornerY = y + vertex.y * height;
    let viewY = worldCornerY + uniforms.camera_y;

    // Apply scale to get NDC
    let ndcPos = vec2<f32>(viewX * uniforms.scale_x, viewY * uniforms.scale_y);

    var output: VertexOutput;
    output.position = vec4<f32>(ndcPos, 0.0, 1.0);
    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.055, 0.055, 0.063, 1.0);
}
`;

const BLOCK_BORDER_SHADER = `
${WGSL_FULL_UNIFORMS}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> blocks: array<vec4<f32>>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) blockCoord: vec2<f32>,
    @location(1) @interpolate(flat) isHovered: f32,
    @location(2) @interpolate(flat) isSelected: f32,
}

${WGSL_DOUBLE_PRECISION_FUNCTIONS}

@vertex
fn vertexMain(
    @builtin(vertex_index) vertexIndex: u32,
    @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
    // Read block data from 2 vec4s (8 floats, aligned)
    let block0 = blocks[instanceIndex * 2u];      // [startX_high, startX_low, y, width]
    let block1 = blocks[instanceIndex * 2u + 1u]; // [height, pad, pad, pad]

    let startX_high = block0.x;
    let startX_low = block0.y;
    let y = block0.z;
    let width = block0.w;
    let height = block1.x;

    let vertices = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0)
    );

    let vertex = vertices[vertexIndex];

    // Calculate center position with double-precision for X
    let centerOffsetX = width * 0.5;
    let centerX_high = startX_high;
    let centerX_low = startX_low + centerOffsetX;

    // Calculate corner offset from center
    let cornerOffsetX = vertex.x * width * 0.5;
    let worldCornerX_high = centerX_high;
    let worldCornerX_low = centerX_low + cornerOffsetX;

    // Apply camera transformation using double-precision
    let viewX = ds_add(worldCornerX_high, worldCornerX_low, uniforms.camera_x_high, uniforms.camera_x_low);

    // Y doesn't need double precision
    let centerY = y + height * 0.5;
    let worldCornerY = centerY + vertex.y * height * 0.5;
    let viewY = worldCornerY + uniforms.camera_y;

    // Apply scale to get NDC
    let ndcPos = vec2<f32>(viewX * uniforms.scale_x, viewY * uniforms.scale_y);

    // Selection calculation using high-precision
    let blockStart = ds_add(startX_high, startX_low, 0.0, 0.0);
    let blockEnd = blockStart + width;
    let isFullyInside = uniforms.hasSelection != 0 &&
                       blockStart >= uniforms.selectionStart &&
                       blockEnd <= uniforms.selectionEnd;

    var output: VertexOutput;
    output.position = vec4<f32>(ndcPos, 0.0, 1.0);
    output.blockCoord = vertex;
    output.isHovered = select(0.0, 1.0, i32(instanceIndex) == uniforms.hoveredBlockId);
    output.isSelected = select(0.0, 1.0, isFullyInside);
    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    if (uniforms.zoomY < 0.5) {
        discard;
    }

    let coord = input.blockCoord;
    let pixelSizeX = fwidth(coord.x);
    let pixelSizeY = fwidth(coord.y);
    let edgeThicknessX = pixelSizeX * 1.0;
    let edgeThicknessY = pixelSizeY * 1.0;
    let distFromEdgeX = 1.0 - abs(coord.x);
    let distFromEdgeY = 1.0 - abs(coord.y);
    let isEdge = distFromEdgeX < edgeThicknessX || distFromEdgeY < edgeThicknessY;

    if (!isEdge) {
        discard;
    }

    let baseColor = vec3<f32>(0.45, 0.50, 0.62);
    let hoverColor = vec3<f32>(0.22, 0.74, 0.97) * 1.3;
    var color = mix(baseColor, hoverColor, input.isHovered);

    // Brighten if within selection (match zone brightness boost)
    color = color + vec3<f32>(0.55, 0.55, 0.55) * input.isSelected;

    return vec4<f32>(color, 0.7);
}
`;

const BACKGROUND_SHADER = `
struct Uniforms {
    viewProj: mat4x4<f32>,
    timeRange: f32,
    worldHeight: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

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
    let worldPos = vec2<f32>(vertex.x * uniforms.timeRange, vertex.y * uniforms.worldHeight);

    var output: VertexOutput;
    output.position = uniforms.viewProj * vec4<f32>(worldPos, 0.0, 1.0);
    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.039, 0.039, 0.047, 1.0);
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
    hasBlending: boolean = false,
    fragmentAccess: boolean = false
): RenderPass {
    const shaderModule = device.createShaderModule({ code: shaderCode });
    const bindGroupLayout = createStorageBindGroupLayout(device, fragmentAccess);

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: storageBuffer } }
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
 * Uploads trace data to GPU storage buffers for instanced rendering.
 *
 * Creates 4 storage buffers:
 * - positionBuffer: Zone data (12 floats per zone, aligned to 3 vec4s):
 *   vec4 #0: [x_high, x_low, y, width]
 *   vec4 #1: [height, r, g, b]
 *   vec4 #2: [id, pad, pad, pad]
 *   Zone X uses double-single precision to avoid Float32 precision loss at extreme zoom
 * - laneBuffer: Lane geometry (4 floats per lane: y, height, width, padding)
 * - blockLaneBuffer: Block lane geometry (4 floats per block lane)
 * - blockBuffer: Block geometry (8 floats per block, aligned to 2 vec4s):
 *   vec4 #0: [startX_high, startX_low, y, width]
 *   vec4 #1: [height, pad, pad, pad]
 *   Block X uses double-single precision to avoid Float32 precision loss at extreme zoom
 *
 * Uses mappedAtCreation for efficient one-time upload (no COPY_DST needed).
 * Buffers are read-only after creation, enabling optimal GPU caching.
 * Returns total GPU memory usage for stats display.
 */
export function createGPUBuffers(
    device: GPUDevice,
    zones: Zone[],
    blocks: Block[],
    blockLanes: BlockLane[],
    lanes: Lane[]
): { positionBuffer: GPUBuffer; laneBuffer: GPUBuffer; blockLaneBuffer: GPUBuffer; blockBuffer: GPUBuffer; gpuMemoryUsage: number } {
    const positions = new Float32Array(zones.length * 12);
    for (let i = 0; i < zones.length; i++) {
        const zone = zones[i];
        const [x_high, x_low] = Camera.splitDouble(zone.x);
        // vec4 #0: [x_high, x_low, y, width]
        positions[i * 12 + 0] = x_high;
        positions[i * 12 + 1] = x_low;
        positions[i * 12 + 2] = zone.y;
        positions[i * 12 + 3] = zone.width;
        // vec4 #1: [height, r, g, b]
        positions[i * 12 + 4] = zone.height;
        positions[i * 12 + 5] = zone.r;
        positions[i * 12 + 6] = zone.g;
        positions[i * 12 + 7] = zone.b;
        // vec4 #2: [id, pad, pad, pad]
        positions[i * 12 + 8] = zone.id;
        positions[i * 12 + 9] = 0;
        positions[i * 12 + 10] = 0;
        positions[i * 12 + 11] = 0;
    }

    const positionBuffer = device.createBuffer({
        size: positions.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Float32Array(positionBuffer.getMappedRange()).set(positions);
    positionBuffer.unmap();

    const laneData = new Float32Array(lanes.length * 4);
    for (let i = 0; i < lanes.length; i++) {
        laneData[i * 4 + 0] = lanes[i].y;
        laneData[i * 4 + 1] = lanes[i].height;
        laneData[i * 4 + 2] = lanes[i].width;
        laneData[i * 4 + 3] = 0;
    }

    const laneBuffer = device.createBuffer({
        size: laneData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Float32Array(laneBuffer.getMappedRange()).set(laneData);
    laneBuffer.unmap();

    const blockLaneData = new Float32Array(blockLanes.length * 4);
    for (let i = 0; i < blockLanes.length; i++) {
        blockLaneData[i * 4 + 0] = blockLanes[i].y;
        blockLaneData[i * 4 + 1] = blockLanes[i].height;
        blockLaneData[i * 4 + 2] = blockLanes[i].width;
        blockLaneData[i * 4 + 3] = 0;
    }

    const blockLaneBuffer = device.createBuffer({
        size: Math.max(16, blockLaneData.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Float32Array(blockLaneBuffer.getMappedRange()).set(blockLaneData);
    blockLaneBuffer.unmap();

    const blockData = new Float32Array(blocks.length * 8);
    for (let i = 0; i < blocks.length; i++) {
        const [startX_high, startX_low] = Camera.splitDouble(blocks[i].startX);
        // vec4 #0: [startX_high, startX_low, y, width]
        blockData[i * 8 + 0] = startX_high;
        blockData[i * 8 + 1] = startX_low;
        blockData[i * 8 + 2] = blocks[i].y;
        blockData[i * 8 + 3] = blocks[i].width;
        // vec4 #1: [height, pad, pad, pad]
        blockData[i * 8 + 4] = blocks[i].height;
        blockData[i * 8 + 5] = 0;
        blockData[i * 8 + 6] = 0;
        blockData[i * 8 + 7] = 0;
    }

    const blockBuffer = device.createBuffer({
        size: Math.max(16, blockData.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Float32Array(blockBuffer.getMappedRange()).set(blockData);
    blockBuffer.unmap();

    const gpuMemoryUsage = positions.byteLength + laneData.byteLength +
        Math.max(16, blockLaneData.byteLength) + Math.max(16, blockData.byteLength) +
        (64 + 32) + (64 + 16);

    return { positionBuffer, laneBuffer, blockLaneBuffer, blockBuffer, gpuMemoryUsage };
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
    blockBuffer: GPUBuffer
): GPUResources {
    const uniformBuffer = createUniformBuffer(device, 112);  // 64 (mat4x4) + 32 (7 ints/floats) + 20 (5 floats for double-single camera + scales)
    const backgroundUniformBuffer = createUniformBuffer(device, 64 + 16);

    return {
        uniformBuffer,
        backgroundUniformBuffer,
        buffers: {
            position: positionBuffer,
            lane: laneBuffer,
            blockLane: blockLaneBuffer,
            block: blockBuffer
        },
        passes: {
            zone: createSimplePipeline(device, format, ZONE_SHADER, uniformBuffer, positionBuffer, true, true),
            lane: createSimplePipeline(device, format, LANE_SHADER, uniformBuffer, laneBuffer),
            blockLane: createSimplePipeline(device, format, BLOCK_LANE_SHADER, uniformBuffer, blockLaneBuffer),
            blockBg: createSimplePipeline(device, format, BLOCK_BG_SHADER, uniformBuffer, blockBuffer),
            block: createSimplePipeline(device, format, BLOCK_BORDER_SHADER, uniformBuffer, blockBuffer, true, true),
            background: createBackgroundPipeline(device, format, backgroundUniformBuffer)
        },
        gpuMemoryUsage: 0
    };
}
