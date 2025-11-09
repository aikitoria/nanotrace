/**
 * Camera system for viewport control and coordinate transformations.
 *
 * Provides independent X/Y zoom control for specialized timeline visualization:
 * - X-axis zoom: Timeline (scroll wheel) - Range: 0.001x to 1,000,000x
 * - Y-axis zoom: Vertical (Shift+scroll or Ctrl+scroll) - Range: 0.01x to 2.0x
 *
 * Coordinate spaces:
 * - World space: Time in milliseconds (X), vertical layout units (Y)
 * - NDC space: Normalized device coordinates [-1, 1]
 * - Screen space: Pixel coordinates [0, width] × [0, height]
 */

import { WorldPosition } from './types.js';
import { INITIAL_BASE_ZOOM, INITIAL_CAMERA_Y_OFFSET } from './constants.js';

/**
 * Manages viewport transformations with independent X/Y zoom and pan.
 *
 * The camera uses a base zoom multiplied by an X-axis multiplier to achieve
 * independent control. Pan state is tracked for right-click drag interactions.
 */
export class Camera {
    x: number;                       // Camera X position in world space (negative of viewed region center)
    y: number;                       // Camera Y position in world space
    zoom: number;                    // Base zoom level (affects Y, and X via multiplier)
    xZoomMultiplier: number;         // Additional X-axis zoom multiplier (for timeline)
    isDragging: boolean;             // Right-click drag state
    lastX: number;                   // Last mouse X position during drag
    lastY: number;                   // Last mouse Y position during drag
    timeRange: number;               // Total time range of loaded trace (for reset)

    // Preallocated buffer to avoid per-frame allocations (GC pressure reduction)
    private viewProjMatrix = new Float32Array(16);

    /**
     * Creates camera centered on the trace with initial auto-zoom.
     *
     * Initial position centers the view horizontally and positions vertically
     * to show the top of the trace (world origin is at bottom).
     */
    constructor(worldHeight: number, timeRange: number) {
        this.x = -timeRange / 2;                      // Center horizontally on time range
        this.y = -worldHeight + INITIAL_CAMERA_Y_OFFSET;  // Position to show top of trace
        this.zoom = INITIAL_BASE_ZOOM;                // Base zoom (will be adjusted for fit)
        this.xZoomMultiplier = 1.0;                   // 1:1 initially (adjusted in initVisualization)
        this.isDragging = false;
        this.lastX = 0;
        this.lastY = 0;
        this.timeRange = timeRange;
    }

    /** Computed X-axis zoom (base zoom * multiplier). */
    get zoomX(): number {
        return this.zoom * this.xZoomMultiplier;
    }

    /** Computed Y-axis zoom (base zoom only). */
    get zoomY(): number {
        return this.zoom;
    }

    /**
     * Converts screen coordinates to world space coordinates.
     *
     * Transformation pipeline:
     * 1. Screen space [0, width] × [0, height]
     * 2. NDC space [-1, 1] × [-1, 1] (Y-axis flipped)
     * 3. View space (undo zoom and aspect)
     * 4. World space (undo camera translation)
     */
    screenToWorld(screenX: number, screenY: number, canvas: HTMLCanvasElement): WorldPosition {
        const rect = canvas.getBoundingClientRect();
        // Screen to NDC (normalized device coordinates)
        const ndcX = (screenX / rect.width) * 2 - 1;
        const ndcY = -((screenY / rect.height) * 2 - 1);  // Flip Y-axis
        const aspect = rect.width / rect.height;
        return {
            // NDC to world: undo zoom, aspect, and camera translation
            x: (ndcX / this.zoomX * aspect) - this.x,
            y: (ndcY / this.zoomY) - this.y
        };
    }

    /**
     * Generates view-projection matrix for GPU shaders.
     *
     * Returns a 4x4 column-major matrix combining:
     * - Scale: Independent X/Y zoom (accounting for aspect ratio)
     * - Translation: Camera position
     *
     * This matrix transforms world coordinates directly to NDC space.
     */
    getViewProjectionMatrix(aspect: number): Float32Array {
        const scaleX = this.zoomX / aspect;
        const scaleY = this.zoomY;

        // Write into preallocated buffer to avoid per-frame allocations
        // Column-major 4x4 matrix: scale + translation
        const m = this.viewProjMatrix;
        m[0] = scaleX; m[1] = 0; m[2] = 0; m[3] = 0;                    // Column 0: X scale
        m[4] = 0; m[5] = scaleY; m[6] = 0; m[7] = 0;                    // Column 1: Y scale
        m[8] = 0; m[9] = 0; m[10] = 1; m[11] = 0;                       // Column 2: Z (unused, identity)
        m[12] = this.x * scaleX; m[13] = this.y * scaleY; m[14] = 0; m[15] = 1;  // Column 3: Translation

        return m;
    }

    /**
     * Splits a double-precision (f64) value into two single-precision (f32) components.
     *
     * Uses the "double-single" representation for emulated double precision in shaders:
     * - high: The f32 representation of the value (captures most significant bits)
     * - low: The residual difference (value - high), capturing remaining precision
     *
     * Reconstruction: value ≈ high + low (with f64 precision)
     *
     * @param value - The f64 number to split
     * @returns [high, low] - Two f32 values that together represent the original f64
     */
    static splitDouble(value: number): [number, number] {
        const high = Math.fround(value);       // Convert to f32 (loses precision)
        const low = Math.fround(value - high); // Capture lost precision as f32
        return [high, low];
    }

    /**
     * Returns camera X position as double-single pair for high-precision GPU calculations.
     *
     * The camera position needs double precision in shaders to avoid floating-point
     * precision loss when transforming world coordinates at extreme zoom levels.
     *
     * @returns [x_high, x_low] - Camera X position split into two f32 components
     */
    getCameraXDoubleSingle(): [number, number] {
        return Camera.splitDouble(this.x);
    }
}
