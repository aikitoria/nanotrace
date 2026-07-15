/**
 * Centralized constants for the Nanotrace visualizer.
 *
 * This file contains all magic values extracted from the codebase, organized into
 * logical sections with JSDoc comments for maintainability and clarity.
 */

// =============================================================================
// UI DIMENSIONS & SPACING
// =============================================================================

/** Width of top-level track labels in pixels (left edge of viewport). */
export const TRACK_LABEL_WIDTH = 120;

/** Height of timeline bar in pixels (fixed at top of viewport). */
export const TIMELINE_HEIGHT = 30;

/**
 * Padding in pixels for initial auto-zoom to fit entire trace.
 * Used to add horizontal margin when calculating initial X-axis zoom.
 */
export const INITIAL_ZOOM_PADDING = 100;

/**
 * Horizontal offset in pixels for tooltip from cursor position.
 * Tooltip appears this many pixels to the right of cursor.
 */
export const TOOLTIP_OFFSET_X = 10;

/**
 * Vertical offset in pixels for tooltip from cursor position.
 * Tooltip appears this many pixels below cursor.
 */
export const TOOLTIP_OFFSET_Y = 10;

/**
 * Horizontal offset in pixels for cursor timestamp label from cursor line.
 * Timestamp appears this many pixels to the right of cursor line.
 */
export const CURSOR_TIMESTAMP_OFFSET = 4;

/**
 * Horizontal offset in pixels for selection label from selection start.
 * Selection label appears this many pixels to the right of selection start line.
 */
export const SELECTION_LABEL_OFFSET = 4;

/**
 * Threshold in pixels for displaying selection UI.
 * If selection width is less than this, only a single line is shown.
 */
export const MIN_SELECTION_WIDTH = 1;

/**
 * Threshold in pixels for committed selection.
 * Drag distance must exceed this for selection to persist after mouseup.
 */
export const MIN_SELECTION_DISTANCE = 3;

/**
 * Viewport culling margin in pixels.
 * Elements within this margin of viewport edges are still rendered.
 */
export const VIEWPORT_MARGIN = 20;

/**
 * Minimum width in pixels for rendering block labels.
 * Blocks narrower than this will not have their labels rendered.
 */
export const MIN_BLOCK_LABEL_WIDTH = 25;

/**
 * Minimum height in pixels for block padding area.
 * Labels are only rendered if padding height exceeds this threshold.
 */
export const MIN_BLOCK_LABEL_PADDING_HEIGHT = 12;

/**
 * Minimum width in pixels for rendering zone labels.
 * Zones narrower than this will not have their labels rendered.
 */
export const MIN_ZONE_LABEL_WIDTH = 25;

/**
 * Minimum height in pixels for rendering zone labels.
 * Zones shorter than this will not have their labels rendered.
 */
export const MIN_ZONE_LABEL_HEIGHT = 15;

/**
 * Horizontal padding in pixels for block labels from left edge.
 * Labels are indented this many pixels from the block's left edge.
 */
export const BLOCK_LABEL_PADDING_X = 3;

/**
 * Vertical padding in pixels for block labels from top edge.
 * Labels are offset this many pixels from the block's top edge.
 */
export const BLOCK_LABEL_PADDING_Y = 2;

/**
 * Horizontal padding in pixels for zone labels from left edge.
 * Labels are indented this many pixels from the zone's left edge.
 */
export const ZONE_LABEL_PADDING_X = 3;

/**
 * Vertical padding in pixels for zone labels from top edge.
 * Labels are offset this many pixels from the zone's top edge.
 */
export const ZONE_LABEL_PADDING_Y = 8;

/**
 * Maximum width reduction in pixels for label text clipping.
 * Labels are clipped to (availableWidth - this value) to prevent overflow.
 */
export const LABEL_CLIP_MARGIN = 6;

// =============================================================================
// LAYOUT CONSTANTS (WORLD SPACE)
// =============================================================================

/**
 * Layout constants for hierarchical trace visualization (all in world space units).
 *
 * These values control the visual spacing and sizing of the hierarchy:
 * SM Lane → Block Lanes → Blocks → Sublanes (tracks) → Zones (events)
 *
 * Tuned for readability and label visibility at typical zoom levels.
 */

/** Fixed height of each sublane (zone track) within a block. */
export const SUBLANE_HEIGHT = 0.014;

/** Vertical spacing between SM lanes. */
export const LANE_PADDING = 0.0015;

/** Extra vertical spacing between GPU and CPU track groups. */
export const TRACK_GROUP_PADDING = 0.02;

/** Sublanes stack directly without vertical spacing. */
export const SUBLANE_PADDING = 0;

/** Padding at top and bottom edges of each SM lane. */
export const LANE_EDGE_PADDING = 0.0015;

/** Vertical spacing between block lanes within an SM lane. */
export const BLOCK_LANE_PADDING = 0.0015;

/** Horizontal gap between adjacent blocks (currently unused in layout). */
export const BLOCK_PADDING = 0.00005;

/** Vertical padding above blocks (reserved for block labels). */
export const BLOCK_EDGE_PADDING = 0;

/** Horizontal gap between adjacent zones (currently unused in layout). */
export const ZONE_GAP = 0.00001;

/** Default time range in milliseconds (1ms) when no trace is loaded. */
export const BASE_TIME_RANGE = 1.0;

/** Label color for all text labels (timeline, SM, blocks, zones). */
export const LABEL_COLOR = '#dedede';

// =============================================================================
// INTERACTION & NAVIGATION
// =============================================================================

/**
 * Zoom factor applied per 100 pixels of normalized wheel movement.
 */
export const ZOOM_FACTOR = 1.1;

/**
 * Minimum X-axis zoom level (timeline zoom).
 * Prevents zooming out beyond this threshold (0.001x).
 */
export const MIN_ZOOM_X = 0.001;

/**
 * Maximum X-axis zoom level (timeline zoom).
 * Prevents zooming in beyond this threshold (20,000x).
 */
export const MAX_ZOOM_X = 20000;

/**
 * Minimum Y-axis zoom level (vertical zoom).
 * Prevents zooming out beyond this threshold (0.01x).
 */
export const MIN_ZOOM_Y = 0.01;

/**
 * Maximum Y-axis zoom level (vertical zoom).
 * Prevents zooming in beyond this threshold (8.0x).
 */
export const MAX_ZOOM_Y = 8.0;

/**
 * Pan speed factor for horizontal scrolling via scroll wheel left/right.
 * Smaller values = slower panning.
 */
export const PAN_SPEED = 0.002;

/**
 * Epsilon value for double-click selection boundary expansion.
 * Selection bounds are expanded by this amount to ensure full zone/block coverage.
 */
export const SELECTION_EPSILON = 0.0000001;

/**
 * Initial base zoom level for camera.
 * This is the starting zoom before X/Y multiplier adjustments.
 */
export const INITIAL_BASE_ZOOM = 1.75;

/**
 * Initial Y offset for camera positioning (in world space units).
 * Camera is positioned to show top of trace with this offset.
 */
export const INITIAL_CAMERA_Y_OFFSET = 0.5;

// =============================================================================
// TIMELINE RENDERING
// =============================================================================

/**
 * Target spacing in pixels between tick marks.
 * Timeline algorithm calculates intervals to achieve approximately this spacing.
 */
export const TIMELINE_TICK_SPACING = 120;

/**
 * Target spacing in pixels between time labels.
 * Labels are spaced independently from ticks to avoid overcrowding.
 */
export const TIMELINE_LABEL_SPACING = 180;

/**
 * Minimum tick interval in milliseconds (world space units).
 * Represents 1 nanosecond precision limit. Intervals below this are clamped.
 */
export const MIN_TICK_INTERVAL = 0.000001;

/**
 * Floating-point tolerance factor for tick interval matching.
 * Used to determine if a time position matches a major/minor/tiny tick interval.
 */
export const TICK_INTERVAL_TOLERANCE = 0.01;

/**
 * Floating-point tolerance factor for tick bounds checking.
 * Ticks within this multiple of tiny interval from TIME_RANGE bounds are included.
 */
export const TICK_BOUNDS_TOLERANCE = 0.1;

/**
 * Viewport margin in pixels for tick culling.
 * Ticks within this distance of viewport edges are still rendered.
 */
export const TICK_VIEWPORT_MARGIN = 10;

/** Top offset in pixels for major ticks. */
export const TICK_MAJOR_TOP = 16;

/** Height in pixels for major ticks. */
export const TICK_MAJOR_HEIGHT = 12;

/** Top offset in pixels for medium ticks. */
export const TICK_MEDIUM_TOP = 19;

/** Height in pixels for medium ticks. */
export const TICK_MEDIUM_HEIGHT = 9;

/** Top offset in pixels for minor ticks. */
export const TICK_MINOR_TOP = 22;

/** Height in pixels for minor ticks. */
export const TICK_MINOR_HEIGHT = 6;

/** Top offset in pixels for tiny ticks. */
export const TICK_TINY_TOP = 25;

/** Height in pixels for tiny ticks. */
export const TICK_TINY_HEIGHT = 3;

/** Opacity for all timeline ticks (0.0-1.0). */
export const TICK_OPACITY = 0.9;

/**
 * Threshold interval in milliseconds for selecting seconds unit.
 * Intervals >= this value display times in seconds.
 */
export const TIME_UNIT_SECONDS_THRESHOLD = 1000.0;

/**
 * Threshold interval in milliseconds for selecting milliseconds unit.
 * Intervals >= this value (but < seconds threshold) display times in milliseconds.
 */
export const TIME_UNIT_MILLISECONDS_THRESHOLD = 1.0;

/**
 * Threshold interval in milliseconds for selecting microseconds unit.
 * Intervals >= this value (but < milliseconds threshold) display times in microseconds.
 */
export const TIME_UNIT_MICROSECONDS_THRESHOLD = 0.001;

// =============================================================================
// COLORS & VISUAL STYLING
// =============================================================================

/**
 * Base zone fill color brightness multiplier.
 * Zone base colors are multiplied by this factor for the fill.
 */
export const ZONE_FILL_BRIGHTNESS = 1.0;

/** Amount of the source event hue retained after neutral desaturation. */
export const ZONE_COLOR_SATURATION = 0.95;

/** White mixed into source event colors to produce Unreal-style pastel fills. */
export const ZONE_PASTEL_MIX = 0.18;

/** Text color drawn over the light pastel zone fills. */
export const ZONE_LABEL_COLOR = '#171717';

/**
 * Hover zone fill color RGB values (scaled by brightness).
 * Used when hovering over a zone. [R, G, B] in 0-1 range.
 */
export const ZONE_HOVER_COLOR_R = 0.48;
export const ZONE_HOVER_COLOR_G = 0.72;
export const ZONE_HOVER_COLOR_B = 0.84;

/**
 * Hover zone fill brightness multiplier.
 * Hover color is multiplied by this factor.
 */
export const ZONE_HOVER_BRIGHTNESS = 1.0;

/**
 * Selection highlight brightness multiplier.
 * Zones fully within selection are brightened by this factor.
 */
export const SELECTION_BRIGHTNESS_BOOST = 1.18;

/**
 * Zone outline color brightness multiplier.
 * Zone base colors are multiplied by this factor for the outline.
 */
export const ZONE_OUTLINE_BRIGHTNESS = 0.5;

/**
 * Hover zone outline color RGB values (scaled by brightness).
 * Used for zone outlines when hovering. [R, G, B] in 0-1 range.
 */
export const ZONE_HOVER_OUTLINE_COLOR_R = 0.64;
export const ZONE_HOVER_OUTLINE_COLOR_G = 0.82;
export const ZONE_HOVER_OUTLINE_COLOR_B = 0.92;

/**
 * Hover zone outline brightness multiplier.
 * Hover outline color is multiplied by this factor.
 */
export const ZONE_HOVER_OUTLINE_BRIGHTNESS = 1.0;

/**
 * Y-axis zoom threshold for disabling adaptive outlines.
 * Below this zoom level, outlines are disabled for performance.
 */
export const OUTLINE_DISABLE_ZOOM_THRESHOLD = 0.5;

/**
 * Outline thickness multiplier for fwidth() calculation.
 * Determines how many pixels wide the adaptive outline should be.
 */
export const OUTLINE_THICKNESS_MULTIPLIER = 1.0;

/**
 * Block border base color RGB values.
 * Default border color when not hovered. [R, G, B] in 0-1 range.
 */
export const BLOCK_BORDER_COLOR_R = 0.39;
export const BLOCK_BORDER_COLOR_G = 0.39;
export const BLOCK_BORDER_COLOR_B = 0.39;

/**
 * Block border hover color RGB values (scaled by brightness).
 * Used for block borders when hovering. [R, G, B] in 0-1 range.
 */
export const BLOCK_BORDER_HOVER_COLOR_R = 0.48;
export const BLOCK_BORDER_HOVER_COLOR_G = 0.72;
export const BLOCK_BORDER_HOVER_COLOR_B = 0.84;

/**
 * Block border hover brightness multiplier.
 * Hover border color is multiplied by this factor.
 */
export const BLOCK_BORDER_HOVER_BRIGHTNESS = 1.0;

/**
 * Block border selection brightness additive boost.
 * This value is added to RGB components for selected block borders.
 */
export const BLOCK_BORDER_SELECTION_BOOST = 0.12;

/**
 * Block border opacity (alpha channel, 0.0-1.0).
 * Allows partial transparency for block borders.
 */
export const BLOCK_BORDER_OPACITY = 0.65;

/**
 * Lane background color RGB values.
 * SM lane backgrounds. [R, G, B] in 0-1 range.
 */
export const LANE_BG_COLOR_R = 0.227;
export const LANE_BG_COLOR_G = 0.231;
export const LANE_BG_COLOR_B = 0.235;

/**
 * Block lane background color RGB values.
 * Block lane backgrounds (lighter than lanes). [R, G, B] in 0-1 range.
 */
export const BLOCK_LANE_BG_COLOR_R = 0.243;
export const BLOCK_LANE_BG_COLOR_G = 0.247;
export const BLOCK_LANE_BG_COLOR_B = 0.251;

/**
 * Block background color RGB values.
 * Individual block backgrounds (darker than block lanes). [R, G, B] in 0-1 range.
 */
export const BLOCK_BG_COLOR_R = 0.216;
export const BLOCK_BG_COLOR_G = 0.220;
export const BLOCK_BG_COLOR_B = 0.224;

/**
 * Full-screen background color RGB values.
 * Canvas background behind all trace elements. [R, G, B] in 0-1 range.
 */
export const CANVAS_BG_COLOR_R = 0.157;
export const CANVAS_BG_COLOR_G = 0.161;
export const CANVAS_BG_COLOR_B = 0.165;

/**
 * Render pass clear color RGB values.
 * Used to clear the framebuffer at the start of each frame. [R, G, B] in 0-1 range.
 */
export const CLEAR_COLOR_R = 0.157;
export const CLEAR_COLOR_G = 0.161;
export const CLEAR_COLOR_B = 0.165;

// =============================================================================
// TEXT RENDERING
// =============================================================================

/** Font size in pixels for all text labels (blocks, zones, timeline). */
export const LABEL_FONT_SIZE = 10;

/** Labels disappear when vertical scaling would make the font smaller. */
export const MIN_LABEL_FONT_SIZE = 1;

/** Vertical zoom below which all trace labels are hidden. */
export const MIN_LABEL_ZOOM_Y = 0.5;

/**
 * Font family for all text labels.
 * UI sans-serif stack for compact, readable trace labels.
 */
export const LABEL_FONT_FAMILY = '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';

// =============================================================================
// TIMING & PERFORMANCE
// =============================================================================

/**
 * Delay in milliseconds before showing loading overlay.
 * Provides visual feedback after file selection but before blocking parse.
 */
export const LOADING_OVERLAY_DELAY = 50;

/**
 * Delay in milliseconds for yielding to browser during initialization.
 * Allows UI updates to render between long-running operations.
 */
export const BROWSER_YIELD_DELAY = 0;

/**
 * Maximum kernel name length for stats display.
 * Longer names are truncated with ellipsis.
 */
/**
 * Number of spaces for FPS padding in stats display.
 * Ensures consistent alignment in monospace stats text.
 */
export const FPS_PADDING_WIDTH = 3;

/**
 * Precision threshold in nanoseconds for decimal formatting.
 * Durations below this are shown with 2 decimal places, otherwise as integers.
 */
export const TIME_DECIMAL_THRESHOLD = 10;

/**
 * Number of decimal places for sub-threshold time formatting.
 * Used when displaying very short durations.
 */
export const TIME_DECIMAL_PLACES = 2;

// =============================================================================
// GPU & MEMORY
// =============================================================================

/**
 * Minimum buffer size in bytes for GPU storage buffers.
 * Some buffers are padded to this minimum to satisfy WebGPU requirements.
 */
export const MIN_GPU_BUFFER_SIZE = 16;

/**
 * Uniform buffer size in bytes for main uniforms.
 * Contains view-projection matrix, hover/selection state, camera double-single, scales.
 * Layout: 64-byte matrix followed by camera, interaction, scale, and viewport
 * values. WGSL struct alignment rounds the allocation to 128 bytes.
 */
export const UNIFORM_BUFFER_SIZE = 128;

/**
 * Uniform buffer size in bytes for background uniforms.
 * Contains camera double-single, scales, time range dual float, and world height.
 * Layout: 32 (8 floats: camera_x_high, camera_x_low, camera_y, scale_x, scale_y, timeRange_high, timeRange_low, worldHeight)
 */
export const BACKGROUND_UNIFORM_BUFFER_SIZE = 32;

/**
 * Number of floats per zone in GPU storage buffer.
 * Layout: 3 vec4s = 12 floats (x_high, x_low, y, width, height, r, g, b, id, pad, pad, pad)
 */
export const ZONE_BUFFER_FLOATS = 12;

/**
 * Number of floats per block in GPU storage buffer.
 * Layout: 2 vec4s = 8 floats (startX_high, startX_low, y, width, height, pad, pad, pad)
 */
export const BLOCK_BUFFER_FLOATS = 8;

/**
 * Number of floats per lane in GPU storage buffer.
 * Layout: 2 vec4s = 8 floats (y, height, padding, padding,
 * start_high, start_low, end_high, end_low)
 */
export const LANE_BUFFER_FLOATS = 8;

/**
 * Number of floats per block lane in GPU storage buffer.
 * Layout: 2 vec4s = 8 floats (y, height, padding, padding,
 * start_high, start_low, end_high, end_low)
 */
export const BLOCK_LANE_BUFFER_FLOATS = 8;

/**
 * Number of vertices per quad (two triangles).
 * All geometry is rendered as instanced quads.
 */
export const VERTICES_PER_QUAD = 6;

// =============================================================================
// CONVERSION FACTORS
// =============================================================================

/**
 * Conversion factor from nanoseconds to milliseconds.
 * Multiply nanoseconds by this to get milliseconds (world space units).
 */
export const NS_TO_MS = 1 / 1_000_000;

/**
 * Conversion factor from milliseconds to nanoseconds.
 * Multiply milliseconds by this to get nanoseconds (display units).
 */
export const MS_TO_NS = 1_000_000;

/**
 * Conversion factor from milliseconds to seconds.
 * Used for time label formatting.
 */
export const MS_TO_SECONDS = 1 / 1000;

/**
 * Conversion factor from milliseconds to microseconds.
 * Used for time label formatting.
 */
export const MS_TO_MICROSECONDS = 1000;

// =============================================================================
// FILE FORMAT
// =============================================================================

/** Length in bytes of magic number in .nanotrace format ("nanotrace\0"). */
export const MAGIC_NUMBER_LENGTH = 10;

/** Expected format version number for .nanotrace files. */
export const EXPECTED_FORMAT_VERSION = 1;

/** Compression mode value for uncompressed data. */
export const COMPRESSION_MODE_NONE = 0;

/** Compression mode value for deflate-compressed data. */
export const COMPRESSION_MODE_DEFLATE = 1;
