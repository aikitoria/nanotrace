/**
 * Type definitions for trace data structures and visualization hierarchy.
 *
 * Data flow:
 * 1. Binary .nanotrace file is parsed into FormatDescriptor, BlockDescriptor, WarpTrack
 * 2. buildHierarchy() transforms parsed data into visualization hierarchy:
 *    Lane (SM) → BlockLane → Block → Sublane (array of Zone[])
 * 3. Hierarchy is uploaded to GPU for rendering
 *
 * Coordinate system:
 * - X axis: Time in milliseconds (world space), origin at 0
 * - Y axis: World space coordinates, origin at bottom, stacks upward
 * - All spatial constants are in world space units
 */

/**
 * Format descriptor with placeholder string template.
 * Used for blocks, warps/tracks, and events to reduce memory usage.
 * Example: "Warp {0}" with params [5] → "Warp 5"
 */
export interface FormatDescriptor {
    formatString: string;           // Template with {0}, {1}, etc. placeholders
    placeholderCount: number;        // Number of parameters required
}

/**
 * Block descriptor from trace file (before hierarchy building).
 * Represents a GPU thread block scheduled on a specific SM.
 */
export interface BlockDescriptor {
    smId: number;                    // Streaming multiprocessor ID
    formatDescId: number;            // Index into FormatDescriptor array
    params: number[];                // Values to substitute into format string
}

/**
 * Single event (zone) from trace file (before hierarchy building).
 * Represents a timed execution region within a warp/track.
 */
export interface EventData {
    timeOffset: number;              // Start time in nanoseconds
    duration: number;                // Duration in nanoseconds
    formatDescId: number;            // Index into FormatDescriptor array
    params: number[];                // Values to substitute into format string
}

/**
 * Warp track from trace file (before hierarchy building).
 * Represents a sequence of events executed by a single warp within a block.
 */
export interface WarpTrack {
    blockDescId: number;             // Index into BlockDescriptor array
    formatDescId: number;            // Format descriptor for track/warp name
    params: number[];                // Parameters for track/warp name
    events: EventData[];             // Timed events within this track
}

/**
 * Zone (event) in visualization hierarchy.
 * Smallest renderable unit representing a single timed execution region.
 * Uploaded to GPU storage buffer for instanced rendering.
 */
export interface Zone {
    id: number;                      // Global unique ID (for hit detection)
    startX: number;                  // Start time in milliseconds (world space)
    endX: number;                    // End time in milliseconds (world space)
    width: number;                   // Duration in milliseconds (cached)
    height: number;                  // Fixed height in world space
    r: number;                       // Red component [0-1] from color palette
    g: number;                       // Green component [0-1] from color palette
    b: number;                       // Blue component [0-1] from color palette
    formatDescId: number;            // Format descriptor for zone name
    params: number[];                // Parameters for zone name
    warpFormatDescId: number;        // Format descriptor for parent warp name
    warpParams: number[];            // Parameters for parent warp name
    laneIdx: number;                 // Parent SM lane index
    blockLaneIdx: number;            // Parent block lane index
    blockIdx: number;                // Parent block index
    sublaneIdx: number;              // Sublane index within block
    x: number;                       // Center X (world space, for GPU)
    y: number;                       // Center Y (world space, for GPU)
}

/**
 * Block in visualization hierarchy.
 * Represents a GPU thread block with multiple warp tracks (sublanes).
 * Groups non-overlapping zones into horizontal sublanes.
 */
export interface Block {
    id: number;                      // Global unique ID
    startX: number;                  // Start time of earliest zone
    endX: number;                    // End time of latest zone
    width: number;                   // Total duration (cached)
    height: number;                  // Total height including all sublanes and padding
    sublanes: Zone[][];              // Array of sublanes, each sublane is array of zones
    numSublanes: number;             // Number of sublanes (cached)
    maxZoneWidth: number;            // Widest zone (for label culling)
    formatDescId: number;            // Format descriptor for block name
    params: number[];                // Parameters for block name
    laneIdx: number;                 // Parent SM lane index
    blockLaneIdx: number;            // Parent block lane index
    x: number;                       // Center X (world space, for GPU)
    y: number;                       // Bottom-left Y (world space, for GPU)
}

/**
 * Block lane in visualization hierarchy.
 * Groups non-overlapping blocks horizontally within a single SM lane.
 * Allows multiple blocks to render in parallel when they don't overlap in time.
 */
export interface BlockLane {
    id: number;                      // Global unique ID
    blocks: Block[];                 // Non-overlapping blocks sorted by startX
    height: number;                  // Maximum block height in this lane
    width: number;                   // Rightmost block end time
    maxBlockWidth: number;           // Widest block (for label culling)
    maxZoneWidth: number;            // Widest zone across all blocks (for label culling)
    laneIdx: number;                 // Parent SM lane index
    y: number;                       // Bottom edge Y coordinate (world space)
}

/**
 * Lane (SM lane) in visualization hierarchy.
 * Top-level container representing a single streaming multiprocessor.
 * Contains one or more block lanes for parallel block rendering.
 */
export interface Lane {
    index: number;                   // SM ID (matches hardware SM index)
    blockLanes: BlockLane[];         // Block lanes within this SM
    height: number;                  // Total height including all block lanes and padding
    width: number;                   // Rightmost block lane end time
    y: number;                       // Bottom edge Y coordinate (world space)
}

/** World-space coordinate (used for screen-to-world transformations). */
export interface WorldPosition {
    x: number;                       // Time in milliseconds
    y: number;                       // Vertical position in world space
}

/** Result from hierarchical hit detection (findZoneAtPosition). */
export interface FindZoneResult {
    zone: Zone | null;               // Deepest match (null if not hovering zone)
    block: Block | null;             // Block containing zone (null if not hovering block)
    blockIndex: number;              // Global block index (-1 if not hovering block)
}

/** Timeline tick intervals at different hierarchy levels. */
export interface TimelineIntervals {
    major: number;                   // Major tick interval (longest)
    minor: number;                   // Minor tick interval (major / 5)
    tiny: number;                    // Tiny tick interval (major / 10)
    label: number;                   // Label interval (>= major, power of 10)
}

