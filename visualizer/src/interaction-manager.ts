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
    HierarchyData,
    FindZoneResult
} from './utils/types.js';
import {
    binarySearchBlocksIndirect,
    binarySearchZones
} from './utils/soa-helpers.js';
import {
    LANE_EDGE_PADDING,
    SUBLANE_HEIGHT,
    SUBLANE_PADDING,
    BLOCK_EDGE_PADDING,
    BLOCK_LANE_PADDING,
    TOOLTIP_OFFSET_X,
    TOOLTIP_OFFSET_Y,
    SELECTION_LABEL_OFFSET,
    MS_TO_NS,
    TIME_DECIMAL_THRESHOLD,
    TIME_DECIMAL_PLACES
} from './utils/constants.js';

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
     * Performs hierarchical hit detection to find zone under cursor (SoA version).
     *
     * Search hierarchy with optimizations:
     * 1. Lane: Linear search (typically <150 lanes, sorted by Y)
     * 2. Block Lane: Linear search within lane (typically 1-4 block lanes)
     * 3. Block: Binary search by X coordinate using indirection (sorted by startX)
     * 4. Sublane: Direct index calculation from Y coordinate
     * 5. Zone: Binary search by X coordinate (sorted by blockIdx, sublaneIdx, startX)
     *
     * Returns zone and block indices (-1 if not found).
     * Overall complexity: O(log n) where n is zones per block lane.
     */
    findZoneAtPosition(screenX: number, screenY: number, hierarchy: HierarchyData): FindZoneResult {
        const worldPos = this.camera.screenToWorld(screenX, screenY, this.canvas);
        const { lanes, blockLanes, blocks, zones } = hierarchy;

        // Convert world X from milliseconds to nanoseconds for comparison with SoA data
        const worldXNs = worldPos.x * MS_TO_NS;

        // 1. Find lane (linear search)
        let foundLaneIdx = -1;
        for (let i = 0; i < lanes.count; i++) {
            if (worldPos.y >= lanes.ys[i] && worldPos.y <= lanes.ys[i] + lanes.heights[i]) {
                foundLaneIdx = i;
                break;
            }
        }
        if (foundLaneIdx === -1) return { zoneIdx: -1, blockIdx: -1 };

        // 2. Find block lane (linear search within lane's block lanes)
        const blockLaneStart = lanes.blockLanesStartIndices[foundLaneIdx];
        const blockLaneEnd = lanes.blockLanesEndIndices[foundLaneIdx];
        let foundBlockLaneIdx = -1;
        let blockLaneY = lanes.ys[foundLaneIdx] + LANE_EDGE_PADDING;

        for (let blIdx = blockLaneStart; blIdx < blockLaneEnd; blIdx++) {
            if (worldPos.y >= blockLaneY && worldPos.y < blockLaneY + blockLanes.heights[blIdx]) {
                foundBlockLaneIdx = blIdx;
                break;
            }
            blockLaneY += blockLanes.heights[blIdx];
            if (blIdx < blockLaneEnd - 1) {
                blockLaneY += BLOCK_LANE_PADDING;
            }
        }
        if (foundBlockLaneIdx === -1) return { zoneIdx: -1, blockIdx: -1 };

        // 3. Find block (binary search using indirection)
        const offset = blockLanes.blockIndicesOffsets[foundBlockLaneIdx];
        const count = blockLanes.blockIndicesCounts[foundBlockLaneIdx];
        const blockIdx = binarySearchBlocksIndirect(
            blocks,
            blockLanes.blockIndices,
            offset,
            count,
            worldXNs,
            worldPos.y
        );
        if (blockIdx === -1) return { zoneIdx: -1, blockIdx: -1 };

        // 4. Calculate sublane index from Y position
        const relativeY = (blocks.ys[blockIdx] + blocks.heights[blockIdx] - BLOCK_EDGE_PADDING) - worldPos.y;
        const sublaneIdx = Math.floor(relativeY / (SUBLANE_HEIGHT + SUBLANE_PADDING));
        if (sublaneIdx < 0 || sublaneIdx >= blocks.sublanesCounts[blockIdx]) {
            return { zoneIdx: -1, blockIdx };
        }

        // 5. Find zone (binary search within block's zone range, filtered by sublane)
        const zoneStart = blocks.zonesStartIndices[blockIdx];
        const zoneEnd = blocks.zonesEndIndices[blockIdx];
        const zoneIdx = binarySearchZones(
            zones,
            zoneStart,
            zoneEnd,
            worldXNs,
            sublaneIdx
        );

        return { zoneIdx, blockIdx };
    }

    /**
     * Updates hover state and tooltip display based on cursor position (SoA version).
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
        hierarchy: HierarchyData,
        formatString: (formatDescId: number, params: number[]) => string,
        formatTrackString: (formatDescId: number, laneId: number, params: number[]) => string,
        formatBlockString: (formatDescId: number, blockId: number, clusterId: number) => string
    ): void {
        const result = this.findZoneAtPosition(screenX, screenY, hierarchy);
        const { zones, blocks, tracks, formatDescriptors } = hierarchy;

        // Update hover state for shader highlighting
        this.hoveredBlockId = result.blockIdx;

        if (result.zoneIdx !== -1) {
            this.hoveredZoneId = result.zoneIdx;

            // Zone times are already in nanoseconds (SoA storage)
            const startNs = zones.startsX[result.zoneIdx];
            const endNs = zones.endsX[result.zoneIdx];
            const durNs = endNs - startNs;

            // Get zone params from pool (NO allocation!)
            const zoneParamsOffset = zones.paramsOffsets[result.zoneIdx];
            const zoneParamsCount = zones.paramsCounts[result.zoneIdx];
            const zoneParams: number[] = [];
            for (let i = 0; i < zoneParamsCount; i++) {
                zoneParams.push(zones.paramsPool[zoneParamsOffset + i]);
            }

            // Format zone name
            const zoneName = formatDescriptors.length > 0
                ? formatString(zones.formatDescIds[result.zoneIdx], zoneParams)
                : `Zone #${result.zoneIdx}`;

            // Format block name
            const blockName = (formatDescriptors.length > 0 && result.blockIdx !== -1)
                ? formatBlockString(
                    blocks.formatDescIds[result.blockIdx],
                    blocks.gridIds[result.blockIdx],
                    blocks.clusterIds[result.blockIdx]
                  )
                : `Block ${result.blockIdx}`;

            // Get track info for warp name
            const trackIdx = zones.trackIndices[result.zoneIdx];
            const trackParamsOffset = tracks.paramsOffsets[trackIdx];
            const trackParamsCount = tracks.paramsCounts[trackIdx];
            const trackParams: number[] = [];
            for (let i = 0; i < trackParamsCount; i++) {
                trackParams.push(tracks.paramsPool[trackParamsOffset + i]);
            }

            const warpName = formatDescriptors.length > 0
                ? formatTrackString(
                    tracks.formatDescIds[trackIdx],
                    tracks.sublaneIndices[trackIdx],
                    trackParams
                  )
                : `Sublane ${zones.sublaneIndices[result.zoneIdx]}`;

            // Display hierarchical tooltip: Zone / Warp / Block / Timing
            this.tooltip.innerHTML = `
                ${zoneName}<br>
                ${warpName} / ${blockName}<br>
                Start: ${startNs.toLocaleString()} ns<br>
                End: ${endNs.toLocaleString()} ns<br>
                Len: ${durNs.toLocaleString()} ns
            `;
            this.tooltip.style.left = `${screenX + TOOLTIP_OFFSET_X}px`;
            this.tooltip.style.top = `${screenY + TOOLTIP_OFFSET_Y}px`;
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

            const startNs = worldLeftX * MS_TO_NS;
            const endNs = worldRightX * MS_TO_NS;
            const durNs = endNs - startNs;

            let startText: string, endText: string, durText: string;
            if (durNs < TIME_DECIMAL_THRESHOLD) {
                startText = startNs.toFixed(TIME_DECIMAL_PLACES);
                endText = endNs.toFixed(TIME_DECIMAL_PLACES);
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

            this.selectionLabel.style.left = `${screenLeft + SELECTION_LABEL_OFFSET}px`;
            this.selectionLabel.style.display = 'block';
        } else {
            // Collapse to a single line when too narrow
            const centerX = (screenLeft + screenRight) / 2;

            // Hide the region and end line, show only start line at center
            this.selectionRegion.style.display = 'none';
            this.selectionLineEnd.style.display = 'none';

            this.selectionLineStart.style.left = `${centerX}px`;
            this.selectionLineStart.style.display = 'block';

            const startNs = worldLeftX * MS_TO_NS;
            const endNs = worldRightX * MS_TO_NS;
            const durNs = endNs - startNs;

            let startText: string, endText: string, durText: string;
            if (durNs < TIME_DECIMAL_THRESHOLD) {
                startText = startNs.toFixed(TIME_DECIMAL_PLACES);
                endText = endNs.toFixed(TIME_DECIMAL_PLACES);
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

            this.selectionLabel.style.left = `${centerX + SELECTION_LABEL_OFFSET}px`;
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
