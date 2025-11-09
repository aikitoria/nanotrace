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

    /**
     * Creates camera centered on the trace with initial auto-zoom.
     *
     * Initial position centers the view horizontally and positions vertically
     * to show the top of the trace (world origin is at bottom).
     */
    constructor(worldHeight: number, timeRange: number) {
        this.x = -timeRange / 2;             // Center horizontally on time range
        this.y = -worldHeight + 0.5;         // Position to show top of trace
        this.zoom = 2.0;                     // Base zoom (will be adjusted for fit)
        this.xZoomMultiplier = 1.0;          // 1:1 initially (adjusted in initVisualization)
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
        // Column-major 4x4 matrix: scale + translation
        return new Float32Array([
            scaleX, 0, 0, 0,                           // Column 0: X scale
            0, scaleY, 0, 0,                           // Column 1: Y scale
            0, 0, 1, 0,                                // Column 2: Z (unused, identity)
            this.x * scaleX, this.y * scaleY, 0, 1     // Column 3: Translation
        ]);
    }
}
