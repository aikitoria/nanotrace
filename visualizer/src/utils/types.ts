/**
 * Type definitions for trace data structures and visualization hierarchy (SoA version).
 *
 * Data flow:
 * 1. Binary .nanotrace file is parsed directly into SoA structures (TracksSoA, ZonesSoA, BlocksSoA)
 * 2. buildHierarchy() builds hierarchical acceleration structures (BlockLanesSoA, LanesSoA)
 * 3. HierarchyData is uploaded to GPU for rendering
 *
 * Memory layout:
 * - All data stored in TypedArrays (Structure of Arrays)
 * - Times stored as Uint32Array in nanoseconds (not float milliseconds)
 * - Colors packed as Uint8Array (3 bytes per zone)
 * - Variable-length params stored in pooled arrays with offset/count
 * - Zero intermediate JavaScript objects during parsing
 *
 * Coordinate system:
 * - X axis: Time in nanoseconds (SoA storage), converted to milliseconds for rendering
 * - Y axis: World space coordinates, origin at bottom, stacks upward
 * - All spatial constants are in world space units
 */

/**
 * Format descriptor with dual string templates for labels and tooltips.
 * Used for blocks, warps/tracks, and events to reduce memory usage.
 * Example label: "Warp {0}" with params [5] → "Warp 5"
 * Example tooltip: "Warp {0} on SM {1}" with params [5, 3] → "Warp 5 on SM 3"
 */
export interface FormatDescriptor {
    labelString: string;             // Short template for labels (visible on canvas)
    tooltipString: string;           // Full template for tooltips (on hover)
    placeholderCount: number;        // Number of parameters required
}

/** World-space coordinate (used for screen-to-world transformations). */
export interface WorldPosition {
    x: number;                       // Time in milliseconds
    y: number;                       // Vertical position in world space
}

/** Result from hierarchical hit detection (findZoneAtPosition). SoA version returns indices. */
export interface FindZoneResult {
    zoneIdx: number;                 // Zone index in ZonesSoA (-1 if not found)
    blockIdx: number;                // Block index in BlocksSoA (-1 if not found)
}

/** Timeline tick intervals at different hierarchy levels. */
export interface TimelineIntervals {
    major: number;                   // Major tick interval (longest)
    minor: number;                   // Minor tick interval (major / 5)
    tiny: number;                    // Tiny tick interval (major / 10)
    label: number;                   // Label interval (>= major, power of 10)
}

// ============================================================================
// Structure of Arrays (SoA) Data Structures
// ============================================================================

/**
 * TracksSoA - Structure of Arrays for warp/track information.
 * Stores track metadata once per sublane (instead of duplicating per zone).
 * Tracks correspond to Event Tracks in the binary file.
 */
export class TracksSoA {
    count: number = 0;

    // Format descriptor (for track/warp labels)
    formatDescIds!: Uint16Array;        // Format descriptor index
    sublaneIndices!: Uint8Array;        // Lane ID (same as sublane index, for {lane} placeholder)
    paramsOffsets!: Uint32Array;        // Index into paramsPool
    paramsCounts!: Uint8Array;          // Params count per track

    // Hierarchy (which block this track belongs to)
    blockIndices!: Uint32Array;         // Block index into BlocksSoA

    // Shared params pool (for all tracks)
    paramsPool!: Uint32Array;           // Concatenated params for all tracks
}

/**
 * ZonesSoA - Structure of Arrays for zone/event data.
 * Zones are the smallest renderable units representing timed execution regions.
 * Memory layout uses TypedArrays for 5-6x memory reduction vs objects.
 */
export class ZonesSoA {
    count: number = 0;

    // Spatial bounds (integer nanoseconds for precision)
    startsX!: Uint32Array;              // Start time in nanoseconds
    endsX!: Uint32Array;                // End time in nanoseconds
    ys!: Float32Array;                  // Y position in world space
    // No ID array - index is ID
    // No centers cache - compute (startsX[i] + endsX[i]) / 2 when needed
    // Height is constant SUBLANE_HEIGHT

    // Visual (packed RGB bytes)
    colors!: Uint8Array;                // [r0,g0,b0, r1,g1,b1, ...] packed bytes (0-255)

    // Format descriptors (for zone labels)
    formatDescIds!: Uint16Array;        // Format descriptor index
    paramsOffsets!: Uint32Array;        // Index into paramsPool
    paramsCounts!: Uint8Array;          // Params count per zone

    // Track reference (instead of duplicating warp data)
    trackIndices!: Uint32Array;         // Index into TracksSoA

    // Hierarchy (compact types)
    smIndices!: Uint8Array;             // SM index (0-255)
    blockIndices!: Uint32Array;         // Block index into BlocksSoA
    sublaneIndices!: Uint8Array;        // Sublane within block (0-255)

    // Shared params pool (for zone params only)
    paramsPool!: Uint32Array;           // Concatenated params for all zones
}

/**
 * BlocksSoA - Structure of Arrays for block data.
 * Blocks stay in original grid ID order from file.
 * Block lanes use indirection arrays to reference blocks.
 */
export class BlocksSoA {
    count: number = 0;

    // NOTE: Blocks array stays in original grid ID order from file!
    // Block lanes use indirection to reference blocks.

    // Spatial (integer nanoseconds)
    startsX!: Uint32Array;              // Start time in nanoseconds
    endsX!: Uint32Array;                // End time in nanoseconds
    ys!: Float32Array;                  // Bottom-left Y in world space
    heights!: Float32Array;             // Total height including sublanes

    // Sublane structure
    sublanesCounts!: Uint8Array;        // Number of sublanes per block
    sublanesMaxWidths!: Uint32Array;    // Max zone width in ns (for culling)

    // Format descriptors
    formatDescIds!: Uint16Array;        // Format descriptor index

    // Grid and cluster IDs (from binary file)
    gridIds!: Uint32Array;              // Original spatial grid ID (blocks sorted by this)
    clusterIds!: Uint32Array;           // Original cluster ID

    // Hierarchy
    smIndices!: Uint8Array;             // SM index
    blockLaneIndices!: Uint16Array;     // Assigned during hierarchy build

    // Zone ranges (for efficient sublane access)
    zonesStartIndices!: Uint32Array;    // First zone index for this block
    zonesEndIndices!: Uint32Array;      // Last zone index + 1 for this block

    // Track ranges (for efficient track lookup)
    tracksStartIndices!: Uint32Array;   // First track index for this block
    tracksEndIndices!: Uint32Array;     // Last track index + 1 for this block
}

/**
 * BlockLanesSoA - Structure of Arrays for block lane data.
 * Block lanes group non-overlapping blocks horizontally within an SM.
 * Uses indirection arrays since blocks cannot be reordered.
 */
export class BlockLanesSoA {
    count: number = 0;

    // Spatial
    ys!: Float32Array;                  // Bottom edge Y
    heights!: Float32Array;             // Maximum block height in this lane
    widths!: Uint32Array;               // Rightmost block end time (ns)

    // Culling optimization
    maxBlockWidths!: Uint32Array;       // Widest block in ns
    maxZoneWidths!: Uint32Array;        // Widest zone in ns

    // Hierarchy
    smIndices!: Uint8Array;             // SM index

    // Block indirection (blocks cannot be reordered, use indirection!)
    // Flat array of block indices, partitioned by block lane
    blockIndices!: Uint32Array;         // Indices into BlocksSoA
    blockIndicesOffsets!: Uint32Array;  // Start offset for each block lane
    blockIndicesCounts!: Uint16Array;   // Number of blocks in each lane

    // Example:
    // Block lane 0 has blocks [5, 12, 23] (3 blocks)
    // Block lane 1 has blocks [2, 7] (2 blocks)
    //
    // blockIndices:        [5, 12, 23, 2, 7, ...]
    // blockIndicesOffsets: [0, 3, 5, ...]
    // blockIndicesCounts:  [3, 2, ...]
}

/**
 * LanesSoA - Structure of Arrays for SM lane data.
 * Top-level container representing streaming multiprocessors.
 */
export class LanesSoA {
    count: number = 0;

    // Identity (SM hardware index)
    smIndices!: Uint8Array;             // SM ID (matches hardware)

    // Spatial
    ys!: Float32Array;                  // Bottom edge Y
    heights!: Float32Array;             // Total height
    widths!: Uint32Array;               // Rightmost time (ns)

    // BlockLane ranges
    blockLanesStartIndices!: Uint32Array;  // First block lane index
    blockLanesEndIndices!: Uint32Array;    // Last block lane index + 1
}

/**
 * SMAccelerator - Acceleration structure for SM lookups.
 * Maps SM hardware index to lane index in LanesSoA.
 */
export class SMAccelerator {
    private smToLaneIndex: Map<number, number>;

    constructor(lanes: LanesSoA) {
        this.smToLaneIndex = new Map();
        for (let i = 0; i < lanes.count; i++) {
            this.smToLaneIndex.set(lanes.smIndices[i], i);
        }
    }

    getLaneIndex(smId: number): number | undefined {
        return this.smToLaneIndex.get(smId);
    }
}

/**
 * HierarchyData - Complete visualization hierarchy using SoA structures.
 * Replaces the old object-based hierarchy for massive memory reduction.
 */
export interface HierarchyData {
    tracks: TracksSoA;
    zones: ZonesSoA;
    blocks: BlocksSoA;
    blockLanes: BlockLanesSoA;
    lanes: LanesSoA;
    smAccelerator: SMAccelerator;

    worldHeight: number;                // Total height in world space
    totalDurationNs: number;            // Total duration in nanoseconds
    formatDescriptors: FormatDescriptor[];
    kernelName: string;
    gridDims: [number, number, number];
    clusterDims: [number, number, number];
}

