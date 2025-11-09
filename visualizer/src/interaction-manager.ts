/**
 * Manages user interaction with the visualizer (mouse hover, selection).
 *
 * Key responsibilities:
 * - Hit detection via hierarchical binary search (O(log n) performance)
 * - Hover state tracking and tooltip display
 * - Selection region management (drag selection, double-click snap)
 * - Coordinate transformation (screen space to world space)
 *
 * Uses camera to convert screen coordinates to world coordinates for hit testing.
 */

import { Camera } from './utils/camera.js';
import {
    Lane,
    Block,
    BlockLane,
    FormatDescriptor,
    FindZoneResult,
    LANE_EDGE_PADDING,
    SUBLANE_HEIGHT,
    SUBLANE_PADDING,
    BLOCK_EDGE_PADDING,
    BLOCK_LANE_PADDING
} from './utils/types.js';

/**
 * Handles all mouse interaction logic for the trace visualizer.
 *
 * Provides efficient hit detection using hierarchical binary search:
 * Lane → Block Lane → Block → Sublane → Zone (each level uses binary search when applicable)
 */
export class InteractionManager {
    private camera: Camera;
    private canvas: HTMLCanvasElement;

    // UI elements for hover and selection feedback
    private tooltip: HTMLElement;
    private selectionRegion: HTMLElement;
    private selectionLineStart: HTMLElement;
    private selectionLineEnd: HTMLElement;
    private selectionLabel: HTMLElement;

    // Hover state (IDs are passed to shaders for highlighting)
    private hoveredZoneId: number = -1;
    private hoveredBlockId: number = -1;

    // Selection state (world space X coordinates)
    private isSelecting: boolean = false;
    private selectionStartWorldX: number = 0;
    private selectionEndWorldX: number = 0;
    private hasSelection: boolean = false;

    /**
     * Creates interaction manager with references to camera and UI elements.
     * Camera is used for screen-to-world coordinate transformations.
     */
    constructor(
        camera: Camera,
        canvas: HTMLCanvasElement,
        tooltip: HTMLElement,
        selectionRegion: HTMLElement,
        selectionLineStart: HTMLElement,
        selectionLineEnd: HTMLElement,
        selectionLabel: HTMLElement
    ) {
        this.camera = camera;
        this.canvas = canvas;
        this.tooltip = tooltip;
        this.selectionRegion = selectionRegion;
        this.selectionLineStart = selectionLineStart;
        this.selectionLineEnd = selectionLineEnd;
        this.selectionLabel = selectionLabel;
    }

    /** Updates camera reference when visualization is reinitialized. */
    updateCamera(camera: Camera): void {
        this.camera = camera;
    }

    /** Returns ID of currently hovered zone (-1 if none). Passed to shaders for highlighting. */
    getHoveredZoneId(): number {
        return this.hoveredZoneId;
    }

    /** Returns ID of currently hovered block (-1 if none). Passed to shaders for border highlighting. */
    getHoveredBlockId(): number {
        return this.hoveredBlockId;
    }

    /** Returns true if user is actively dragging a selection. */
    isCurrentlySelecting(): boolean {
        return this.isSelecting;
    }

    /** Returns true if a completed selection exists. */
    hasActiveSelection(): boolean {
        return this.hasSelection;
    }

    /** Begins a new selection at the given world X coordinate. */
    startSelection(worldX: number): void {
        this.isSelecting = true;
        this.selectionStartWorldX = worldX;
        this.selectionEndWorldX = worldX;
    }

    /** Updates the end position of the selection during drag. */
    updateSelectionEnd(worldX: number): void {
        this.selectionEndWorldX = worldX;
    }

    /** Finalizes the selection (called on mouseup if distance threshold met). */
    endSelection(): void {
        this.isSelecting = false;
        this.hasSelection = true;
    }

    /** Hides all selection UI elements. */
    hideSelectionUI(): void {
        this.selectionRegion.style.display = 'none';
        this.selectionLineStart.style.display = 'none';
        this.selectionLineEnd.style.display = 'none';
        this.selectionLabel.style.display = 'none';
    }

    /** Clears selection state and hides UI. */
    clearSelection(): void {
        this.hasSelection = false;
        this.isSelecting = false;
        this.hideSelectionUI();
    }

    /**
     * Performs hierarchical hit detection to find zone under cursor.
     *
     * Search hierarchy with optimizations:
     * 1. Lane: Linear search (typically <150 lanes, sorted by Y)
     * 2. Block Lane: Linear search within lane (typically 1-4 block lanes)
     * 3. Block: Binary search by X coordinate (sorted by startX)
     * 4. Sublane: Direct index calculation from Y coordinate
     * 5. Zone: Binary search by X coordinate (sorted by startX)
     *
     * Returns the deepest match (zone if found, else block, else null).
     * Overall complexity: O(log n) where n is zones per block lane.
     */
    findZoneAtPosition(screenX: number, screenY: number, lanes: Lane[], blocks: Block[]): FindZoneResult {
        const worldPos = this.camera.screenToWorld(screenX, screenY, this.canvas);

        let foundLane: Lane | null = null;
        for (const lane of lanes) {
            if (worldPos.y >= lane.y && worldPos.y <= lane.y + lane.height) {
                foundLane = lane;
                break;
            }
        }
        if (!foundLane) return { zone: null, block: null, blockIndex: -1 };

        let foundBlockLane: BlockLane | null = null;
        let blockLaneY = foundLane.y + LANE_EDGE_PADDING;
        for (const blockLane of foundLane.blockLanes) {
            if (worldPos.y >= blockLaneY && worldPos.y < blockLaneY + blockLane.height) {
                foundBlockLane = blockLane;
                break;
            }
            blockLaneY += blockLane.height + BLOCK_LANE_PADDING;
        }
        if (!foundBlockLane) return { zone: null, block: null, blockIndex: -1 };

        let foundBlock: Block | null = null;
        let foundBlockIndex = -1;
        let left = 0;
        let right = foundBlockLane.blocks.length - 1;
        while (left <= right) {
            const mid = Math.floor((left + right) / 2);
            const block = foundBlockLane.blocks[mid];
            if (worldPos.x >= block.startX && worldPos.x < block.endX &&
                worldPos.y >= block.y && worldPos.y < block.y + block.height) {
                foundBlock = block;
                foundBlockIndex = blocks.indexOf(block);
                break;
            } else if (worldPos.x < block.startX) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        if (!foundBlock) return { zone: null, block: null, blockIndex: -1 };

        const relativeY = (foundBlock.y + foundBlock.height - BLOCK_EDGE_PADDING) - worldPos.y;
        const sublaneIdx = Math.floor(relativeY / (SUBLANE_HEIGHT + SUBLANE_PADDING));
        if (sublaneIdx < 0 || sublaneIdx >= foundBlock.sublanes.length) {
            return { zone: null, block: foundBlock, blockIndex: foundBlockIndex };
        }

        const zones = foundBlock.sublanes[sublaneIdx];
        left = 0;
        right = zones.length - 1;
        while (left <= right) {
            const mid = Math.floor((left + right) / 2);
            const zone = zones[mid];
            if (worldPos.x >= zone.startX && worldPos.x < zone.endX) {
                return { zone, block: foundBlock, blockIndex: foundBlockIndex };
            } else if (worldPos.x < zone.startX) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        return { zone: null, block: foundBlock, blockIndex: foundBlockIndex };
    }

    /**
     * Updates hover state and tooltip display based on cursor position.
     *
     * Performs hit detection, updates hover IDs (passed to shaders for highlighting),
     * and displays tooltip with formatted zone information:
     * - Zone name (from format descriptor)
     * - Hierarchy (warp/sublane / block)
     * - Timing (start, end, duration in nanoseconds)
     *
     * Tooltip appears offset 10px right and down from cursor.
     */
    updateHover(
        screenX: number,
        screenY: number,
        lanes: Lane[],
        blocks: Block[],
        formatDescriptors: FormatDescriptor[],
        formatString: (formatDescId: number, params: number[]) => string
    ): void {
        const result = this.findZoneAtPosition(screenX, screenY, lanes, blocks);

        // Update hover state for shader highlighting
        this.hoveredBlockId = result.blockIndex;

        if (result.zone) {
            this.hoveredZoneId = result.zone.id;

            // Convert world coordinates (milliseconds) to nanoseconds for display
            const startNs = Math.round(result.zone.startX * 1000000);
            const endNs = Math.round(result.zone.endX * 1000000);
            const durNs = endNs - startNs;

            // Format names using format descriptors with parameter substitution
            const zoneName = formatDescriptors.length > 0
                ? formatString(result.zone.formatDescId, result.zone.params)
                : `Zone #${result.zone.id}`;

            const blockName = (formatDescriptors.length > 0 && result.block)
                ? formatString(result.block.formatDescId, result.block.params)
                : `Block ${result.zone.blockIdx}`;

            const warpName = (formatDescriptors.length > 0 && result.zone.warpFormatDescId !== undefined)
                ? formatString(result.zone.warpFormatDescId, result.zone.warpParams)
                : `Sublane ${result.zone.sublaneIdx}`;

            // Display hierarchical tooltip: Zone / Warp / Block / Timing
            this.tooltip.innerHTML = `
                ${zoneName}<br>
                ${warpName} / ${blockName}<br>
                Start: ${startNs.toLocaleString()} ns<br>
                End: ${endNs.toLocaleString()} ns<br>
                Len: ${durNs.toLocaleString()} ns
            `;
            this.tooltip.style.left = `${screenX + 10}px`;
            this.tooltip.style.top = `${screenY + 10}px`;
            this.tooltip.classList.add('visible');
        } else {
            this.hoveredZoneId = -1;
            this.tooltip.classList.remove('visible');
        }
    }

    /**
     * Updates selection UI elements (region, lines, label) based on current selection bounds.
     *
     * Transforms world-space selection coordinates to screen space for rendering.
     * Only displays UI if width > 1px. Label shows start/end/duration in nanoseconds
     * with appropriate precision (decimal for sub-10ns, integer with commas otherwise).
     */
    updateSelection(): void {
        if (!this.isSelecting && !this.hasSelection) return;

        const rect = this.canvas.getBoundingClientRect();
        const aspect = rect.width / rect.height;

        const worldLeftX = Math.min(this.selectionStartWorldX, this.selectionEndWorldX);
        const worldRightX = Math.max(this.selectionStartWorldX, this.selectionEndWorldX);

        const ndcLeft = (worldLeftX + this.camera.x) * this.camera.zoomX / aspect;
        const ndcRight = (worldRightX + this.camera.x) * this.camera.zoomX / aspect;

        // Check for invalid values
        if (!isFinite(ndcLeft) || !isFinite(ndcRight)) {
            this.hideSelectionUI();
            return;
        }

        const screenLeft = (ndcLeft + 1) * rect.width / 2;
        const screenRight = (ndcRight + 1) * rect.width / 2;

        const width = screenRight - screenLeft;

        if (width > 1) {
            // Show full selection region with two lines
            this.selectionRegion.style.left = `${screenLeft}px`;
            this.selectionRegion.style.width = `${width}px`;
            this.selectionRegion.style.display = 'block';

            this.selectionLineStart.style.left = `${screenLeft}px`;
            this.selectionLineStart.style.display = 'block';

            this.selectionLineEnd.style.left = `${screenRight}px`;
            this.selectionLineEnd.style.display = 'block';

            const startNs = worldLeftX * 1000000;
            const endNs = worldRightX * 1000000;
            const durNs = endNs - startNs;

            let startText: string, endText: string, durText: string;
            if (durNs < 10) {
                startText = startNs.toFixed(2);
                endText = endNs.toFixed(2);
                durText = durNs.toFixed(2);
            } else {
                startText = Math.round(startNs).toLocaleString();
                endText = Math.round(endNs).toLocaleString();
                durText = Math.round(durNs).toLocaleString();
            }

            this.selectionLabel.textContent =
                `Start: ${startText} ns\n` +
                `End: ${endText} ns\n` +
                `Len: ${durText} ns`;

            this.selectionLabel.style.left = `${screenLeft + 4}px`;
            this.selectionLabel.style.display = 'block';
        } else {
            // Collapse to a single line when too narrow
            const centerX = (screenLeft + screenRight) / 2;

            // Hide the region and end line, show only start line at center
            this.selectionRegion.style.display = 'none';
            this.selectionLineEnd.style.display = 'none';

            this.selectionLineStart.style.left = `${centerX}px`;
            this.selectionLineStart.style.display = 'block';

            const startNs = worldLeftX * 1000000;
            const endNs = worldRightX * 1000000;
            const durNs = endNs - startNs;

            let startText: string, endText: string, durText: string;
            if (durNs < 10) {
                startText = startNs.toFixed(2);
                endText = endNs.toFixed(2);
                durText = durNs.toFixed(2);
            } else {
                startText = Math.round(startNs).toLocaleString();
                endText = Math.round(endNs).toLocaleString();
                durText = Math.round(durNs).toLocaleString();
            }

            this.selectionLabel.textContent =
                `Start: ${startText} ns\n` +
                `End: ${endText} ns\n` +
                `Len: ${durText} ns`;

            this.selectionLabel.style.left = `${centerX + 4}px`;
            this.selectionLabel.style.display = 'block';
        }
    }

    /**
     * Returns normalized selection bounds in world space.
     * Always returns start <= end regardless of drag direction.
     * Used for shader uniform updates and selection validation.
     */
    getSelectionBounds(): { start: number; end: number } {
        return {
            start: Math.min(this.selectionStartWorldX, this.selectionEndWorldX),
            end: Math.max(this.selectionStartWorldX, this.selectionEndWorldX)
        };
    }
}
