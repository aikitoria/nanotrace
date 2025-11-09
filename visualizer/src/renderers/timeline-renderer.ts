/**
 * Timeline UI renderer with adaptive tick marks and time labels.
 *
 * Features:
 * - Hierarchical tick marks (4 levels: major, medium, minor, tiny)
 * - Power-of-10 intervals (1, 2, 5, 10, 20, 50, 100, ...)
 * - Adaptive spacing based on zoom (120px ticks, 180px labels)
 * - Automatic unit selection (s, ms, μs, ns)
 * - Whole number formatting with comma separators
 * - Minimum precision of 1ns
 *
 * Tick hierarchy:
 * - Tiny: 3px height, top offset 25px
 * - Minor: 6px height, top offset 22px
 * - Medium: 9px height, top offset 19px
 * - Major: 12px height, top offset 16px (with labels)
 *
 * Fixed 30px bar at top with frosted glass effect.
 */

import { Camera } from '../utils/camera.js';
import { TimelineIntervals } from '../utils/types.js';

/**
 * Manages timeline bar with adaptive tick marks and time labels.
 *
 * Calculates appropriate intervals based on zoom level and renders
 * hierarchical tick marks with formatted time labels.
 */
export class TimelineRenderer {
    private timelineContainer: HTMLElement;
    private canvas: HTMLCanvasElement;
    private camera: Camera;

    // DOM element pools to avoid per-frame allocations (GC pressure reduction)
    private tickPool: HTMLElement[] = [];
    private labelPool: HTMLElement[] = [];
    private activeTickCount = 0;
    private activeLabelCount = 0;

    /**
     * Creates timeline renderer with references to container, canvas, and camera.
     * Canvas is used for screen-to-world coordinate transformations.
     */
    constructor(timelineContainer: HTMLElement, canvas: HTMLCanvasElement, camera: Camera) {
        this.timelineContainer = timelineContainer;
        this.canvas = canvas;
        this.camera = camera;
    }

    /** Updates camera reference when visualization is reinitialized. */
    updateCamera(camera: Camera): void {
        this.camera = camera;
    }

    /**
     * Gets a tick element from the pool or creates a new one if pool is exhausted.
     * Reuses existing DOM elements to avoid per-frame allocations.
     */
    private getTickElement(): HTMLElement {
        let tick: HTMLElement;
        if (this.activeTickCount < this.tickPool.length) {
            // Reuse existing element from pool
            tick = this.tickPool[this.activeTickCount];
            tick.style.display = '';  // Make visible
        } else {
            // Pool exhausted, create new element and add to pool
            tick = document.createElement('div');
            tick.className = 'timeline-tick';
            this.tickPool.push(tick);
            this.timelineContainer.appendChild(tick);
        }
        this.activeTickCount++;
        return tick;
    }

    /**
     * Gets a label element from the pool or creates a new one if pool is exhausted.
     * Reuses existing DOM elements to avoid per-frame allocations.
     */
    private getLabelElement(): HTMLElement {
        let label: HTMLElement;
        if (this.activeLabelCount < this.labelPool.length) {
            // Reuse existing element from pool
            label = this.labelPool[this.activeLabelCount];
            label.style.display = '';  // Make visible
        } else {
            // Pool exhausted, create new element and add to pool
            label = document.createElement('div');
            label.className = 'timeline-label';
            this.timelineContainer.appendChild(label);
            this.labelPool.push(label);
        }
        this.activeLabelCount++;
        return label;
    }

    /**
     * Hides unused elements from the pools instead of destroying them.
     * Called at the end of updateTimeline after all visible elements have been reused.
     */
    private hideUnusedElements(): void {
        // Hide unused ticks
        for (let i = this.activeTickCount; i < this.tickPool.length; i++) {
            this.tickPool[i].style.display = 'none';
        }
        // Hide unused labels
        for (let i = this.activeLabelCount; i < this.labelPool.length; i++) {
            this.labelPool[i].style.display = 'none';
        }
    }

    /**
     * Calculates adaptive tick intervals based on current zoom level.
     *
     * Uses power-of-10 intervals (1, 2, 5, 10, 20, 50, ...) to ensure tick marks
     * appear at round numbers. Target spacing:
     * - Tick marks: Every ~120 pixels
     * - Labels: Every ~180 pixels (independent of ticks)
     *
     * Returns 4-level hierarchy: tiny (10x density), minor (5x), major (1x), label.
     * All intervals clamped to minimum of 1ns (0.000001 ms in world space).
     */
    private calculateTimelineInterval(): TimelineIntervals {
        const rect = this.canvas.getBoundingClientRect();

        const worldWidth = (rect.width / rect.height) * 2 / this.camera.zoomX;
        const worldPerPixel = worldWidth / rect.width;

        const tickTargetPixels = 120;
        const tickTargetWorldInterval = tickTargetPixels * worldPerPixel;
        const tickOrderOfMagnitude = Math.pow(10, Math.floor(Math.log10(tickTargetWorldInterval)));
        let tickBaseInterval: number;
        if (tickTargetWorldInterval / tickOrderOfMagnitude < 2) {
            tickBaseInterval = tickOrderOfMagnitude;
        } else if (tickTargetWorldInterval / tickOrderOfMagnitude < 5) {
            tickBaseInterval = 2 * tickOrderOfMagnitude;
        } else {
            tickBaseInterval = 5 * tickOrderOfMagnitude;
        }
        let tickInterval = Math.pow(10, Math.ceil(Math.log10(tickBaseInterval)));

        const minTickInterval = 0.000001;
        tickInterval = Math.max(tickInterval, minTickInterval);

        const labelTargetPixels = 180;
        const labelTargetWorldInterval = labelTargetPixels * worldPerPixel;
        const labelOrderOfMagnitude = Math.pow(10, Math.floor(Math.log10(labelTargetWorldInterval)));
        let labelInterval: number;
        if (labelTargetWorldInterval / labelOrderOfMagnitude < 2) {
            labelInterval = labelOrderOfMagnitude;
        } else if (labelTargetWorldInterval / labelOrderOfMagnitude < 5) {
            labelInterval = 2 * labelOrderOfMagnitude;
        } else {
            labelInterval = 5 * labelOrderOfMagnitude;
        }
        labelInterval = Math.max(labelInterval, minTickInterval);

        return {
            major: tickInterval,
            minor: tickInterval / 5,
            tiny: tickInterval / 10,
            label: labelInterval
        };
    }

    /**
     * Formats time value with appropriate unit based on interval magnitude.
     *
     * Unit selection (based on interval size):
     * - >= 1000ms → seconds (s)
     * - >= 1ms → milliseconds (ms)
     * - >= 0.001ms → microseconds (μs)
     * - < 0.001ms → nanoseconds (ns)
     *
     * Always rounds to whole numbers with comma separators for readability.
     */
    private formatTimeLabel(time: number, interval: number): string {
        let value: number, unit: string;

        const absInterval = Math.abs(interval);

        if (absInterval >= 1000.0) {
            value = time / 1000;
            unit = 's';
        } else if (absInterval >= 1.0) {
            value = time;
            unit = 'ms';
        } else if (absInterval >= 0.001) {
            value = time * 1000;
            unit = 'μs';
        } else {
            value = time * 1000000;
            unit = 'ns';
        }

        return Math.round(value).toLocaleString() + ' ' + unit;
    }

    /**
     * Regenerates timeline tick marks and labels based on current viewport.
     *
     * Process:
     * 1. Calculate adaptive intervals based on zoom
     * 2. Clear existing timeline DOM elements
     * 3. Iterate through tiny tick intervals (highest density)
     * 4. Determine tick hierarchy (major/medium/minor/tiny) using floating-point tolerance
     * 5. Transform world coordinates to screen space
     * 6. Create DOM elements for ticks (with hierarchy-specific styling)
     * 7. Add formatted labels at major tick positions
     *
     * Tick styling (height/offset from top):
     * - Major: 12px height, 16px offset (with labels)
     * - Medium: 9px height, 19px offset
     * - Minor: 6px height, 22px offset
     * - Tiny: 3px height, 25px offset
     *
     * Uses physical pixel widths (1 or 2 device pixels) for crisp rendering.
     */
    updateTimeline(TIME_RANGE: number): void {
        const rect = this.canvas.getBoundingClientRect();
        const aspect = rect.width / rect.height;

        const worldLeft = this.camera.screenToWorld(0, 0, this.canvas).x;
        const worldRight = this.camera.screenToWorld(rect.width, 0, this.canvas).x;

        const intervals = this.calculateTimelineInterval();

        // Reset active counts to reuse elements from the pool
        this.activeTickCount = 0;
        this.activeLabelCount = 0;

        const startTiny = Math.floor(worldLeft / intervals.tiny) * intervals.tiny;
        const endTiny = Math.ceil(worldRight / intervals.tiny) * intervals.tiny;

        const numTicks = Math.round((endTiny - startTiny) / intervals.tiny);

        for (let i = 0; i <= numTicks; i++) {
            const time = startTiny + i * intervals.tiny;

            const fudge = intervals.tiny * 0.1;
            if (time < -fudge || time > TIME_RANGE + fudge || time < worldLeft - intervals.tiny || time > worldRight + intervals.tiny) continue;

            const majorIndex = Math.round(time / intervals.major);
            const mediumIndex = Math.round(time / (intervals.major / 2));
            const minorIndex = Math.round(time / intervals.minor);
            const labelIndex = Math.round(time / intervals.label);

            const isMajor = Math.abs(time - majorIndex * intervals.major) < intervals.tiny * 0.01;
            const isMedium = Math.abs(time - mediumIndex * (intervals.major / 2)) < intervals.tiny * 0.01;
            const isMinor = Math.abs(time - minorIndex * intervals.minor) < intervals.tiny * 0.01;
            const isLabel = Math.abs(time - labelIndex * intervals.label) < intervals.tiny * 0.01;

            const screenX = Math.round(((time + this.camera.x) * this.camera.zoomX / aspect + 1) * rect.width / 2);

            if (screenX < -10 || screenX > rect.width + 10) continue;

            // Get tick element from pool (reuse existing or create new)
            const tick = this.getTickElement();
            tick.style.left = `${screenX}px`;

            const physicalPixelWidth1 = `${1 / devicePixelRatio}px`;
            const physicalPixelWidth2 = `${2 / devicePixelRatio}px`;

            if (isMajor) {
                tick.style.top = '16px';
                tick.style.width = physicalPixelWidth2;
                tick.style.height = '12px';
                tick.style.opacity = '0.9';
            } else if (isMedium && !isMajor) {
                tick.style.top = '19px';
                tick.style.width = physicalPixelWidth1;
                tick.style.height = '9px';
                tick.style.opacity = '0.9';
            } else if (isMinor) {
                tick.style.top = '22px';
                tick.style.width = physicalPixelWidth1;
                tick.style.height = '6px';
                tick.style.opacity = '0.9';
            } else {
                tick.style.top = '25px';
                tick.style.width = physicalPixelWidth1;
                tick.style.height = '3px';
                tick.style.opacity = '0.9';
            }

            if (isLabel) {
                // Get label element from pool (reuse existing or create new)
                const label = this.getLabelElement();
                label.textContent = this.formatTimeLabel(time, intervals.label);
                label.style.left = `${screenX}px`;
                label.style.transform = 'translateX(-50%)';
            }
        }

        // Hide all unused elements from the pools
        this.hideUnusedElements();
    }
}
