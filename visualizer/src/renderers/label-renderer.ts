/**
 * 2D canvas text renderer for block and zone labels.
 *
 * Renders formatted names and durations on top of WebGPU visualization.
 * Implements performance culling:
 * - Viewport culling: Only renders visible elements
 * - Size culling: Only renders when zoomed in enough for readability
 * - Hierarchical max-width tracking: Pre-computes widest zones per block lane
 *
 * Label thresholds:
 * - Block labels: Min 100px width, min 12px padding height
 * - Zone labels: Min 100px width, min 15px height
 *
 * Labels are clipped using fillText maxWidth parameter to fit available space.
 * Uses 10px monospace font (Consolas/Monaco) for consistent readability.
 */

import { Camera } from '../utils/camera.js';
import {
    Block,
    BlockLane,
    Lane,
    FormatDescriptor
} from '../utils/types.js';
import {
    SUBLANE_HEIGHT,
    BLOCK_EDGE_PADDING,
    LABEL_COLOR,
    SM_LABEL_WIDTH,
    MIN_BLOCK_LABEL_WIDTH,
    MIN_BLOCK_LABEL_PADDING_HEIGHT,
    MIN_ZONE_LABEL_WIDTH,
    MIN_ZONE_LABEL_HEIGHT,
    BLOCK_LABEL_PADDING_X,
    BLOCK_LABEL_PADDING_Y,
    ZONE_LABEL_PADDING_X,
    ZONE_LABEL_PADDING_Y,
    LABEL_CLIP_MARGIN,
    LABEL_FONT_SIZE,
    LABEL_FONT_FAMILY,
    MS_TO_NS
} from '../utils/constants.js';

/**
 * Renders text labels for blocks and zones using Canvas 2D API.
 *
 * Overlays on top of WebGPU rendering with viewport culling for performance.
 * Labels show formatted names (from format descriptors) and durations.
 */
export class LabelRenderer {
    private labelCtx: CanvasRenderingContext2D;
    private canvas: HTMLCanvasElement;
    private camera: Camera;

    // Cached flattened blocks array to avoid per-frame allocations (GC pressure reduction)
    private cachedBlocks: Block[] = [];

    /**
     * Creates label renderer with 2D canvas context, canvas, and camera.
     * Canvas is used for coordinate transformations and viewport queries.
     */
    constructor(labelCtx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, camera: Camera) {
        this.labelCtx = labelCtx;
        this.canvas = canvas;
        this.camera = camera;
    }

    /** Updates camera reference when visualization is reinitialized. */
    updateCamera(camera: Camera): void {
        this.camera = camera;
    }

    /**
     * Updates cached blocks array when trace data changes.
     * Call this after loading new trace data to rebuild the cache.
     */
    updateBlocksCache(lanes: Lane[]): void {
        this.cachedBlocks = [];
        for (const lane of lanes) {
            for (const blockLane of lane.blockLanes) {
                for (const block of blockLane.blocks) {
                    this.cachedBlocks.push(block);
                }
            }
        }
    }

    /**
     * Renders block labels with performance culling.
     *
     * Culling strategy (hierarchical early exit):
     * 1. Global padding height check: Skip all labels if zoom too far out
     * 2. Hierarchical max-width check: Pre-computed widest block per visible block lane
     * 3. Per-block viewport culling: Only render visible blocks
     * 4. Per-block size check: Only render if block is wide/tall enough for readability
     *
     * Minimum thresholds:
     * - Block width: 100px
     * - Padding height: 12px (space above block for label)
     *
     * Labels show: "Block Name (duration ns)" in gray (#a8a8a8)
     * Long labels are clipped using fillText maxWidth to fit available space.
     */
    renderBlockLabels(
        blocks: Block[],
        blockLanes: BlockLane[],
        formatDescriptors: FormatDescriptor[],
        formatBlockString: (formatDescId: number, blockId: number, clusterId: number) => string
    ): void {
        const rect = this.canvas.getBoundingClientRect();
        const aspect = rect.width / rect.height;

        const minWidth = MIN_BLOCK_LABEL_WIDTH;
        const minPaddingHeight = MIN_BLOCK_LABEL_PADDING_HEIGHT;
        const smLabelWidth = SM_LABEL_WIDTH;

        const minScreenPaddingHeight = BLOCK_EDGE_PADDING * this.camera.zoomY * (rect.height / 2);
        if (minScreenPaddingHeight < minPaddingHeight) {
            return;
        }

        const worldLeft = this.camera.screenToWorld(0, 0, this.canvas).x;
        const worldRight = this.camera.screenToWorld(rect.width, 0, this.canvas).x;
        const worldTop = this.camera.screenToWorld(0, 0, this.canvas).y;
        const worldBottom = this.camera.screenToWorld(0, rect.height, this.canvas).y;

        let maxVisibleBlockWidth = 0;
        for (const blockLane of blockLanes) {
            const blockLaneTop = blockLane.y + blockLane.height;
            const blockLaneBottom = blockLane.y;
            if (blockLaneBottom <= worldTop && blockLaneTop >= worldBottom) {
                maxVisibleBlockWidth = Math.max(maxVisibleBlockWidth, blockLane.maxBlockWidth);
            }
        }

        if (maxVisibleBlockWidth > 0) {
            const maxScreenWidth = maxVisibleBlockWidth * this.camera.zoomX * (rect.width / 2) / aspect;
            if (maxScreenWidth < minWidth) {
                return;
            }
        }

        this.labelCtx.font = `${LABEL_FONT_SIZE}px ${LABEL_FONT_FAMILY}`;
        this.labelCtx.textAlign = 'left';
        this.labelCtx.textBaseline = 'top';
        this.labelCtx.fillStyle = LABEL_COLOR;

        for (const block of blocks) {
            if (block.endX < worldLeft || block.startX > worldRight) {
                continue;
            }

            const blockTop = block.y + block.height;
            const blockBottom = block.y;
            if (blockBottom > worldTop || blockTop < worldBottom) {
                continue;
            }

            const ndcLeft = (block.startX + this.camera.x) * this.camera.zoomX / aspect;
            const ndcRight = (block.endX + this.camera.x) * this.camera.zoomX / aspect;

            const blockTopY = block.y + block.height;
            const ndcTop = (blockTopY + this.camera.y) * this.camera.zoomY;

            const sublanesTopY = block.y + block.height - BLOCK_EDGE_PADDING;
            const ndcSublanesTop = (sublanesTopY + this.camera.y) * this.camera.zoomY;

            const screenLeft = (ndcLeft + 1) * rect.width / 2;
            const screenRight = (ndcRight + 1) * rect.width / 2;
            const screenTop = rect.height / 2 - (ndcTop * rect.height / 2);
            const screenSublanesTop = rect.height / 2 - (ndcSublanesTop * rect.height / 2);

            const screenPaddingHeight = screenSublanesTop - screenTop;

            const visibleLeft = Math.max(screenLeft, smLabelWidth);
            const visibleWidth = screenRight - visibleLeft;

            if (visibleWidth >= minWidth && screenPaddingHeight >= minPaddingHeight) {
                const labelX = visibleLeft + BLOCK_LABEL_PADDING_X;
                const labelY = screenTop + BLOCK_LABEL_PADDING_Y;
                const durationNs = Math.round((block.endX - block.startX) * MS_TO_NS);

                const blockName = formatDescriptors.length > 0
                    ? formatBlockString(block.formatDescId, block.blockId, block.clusterId)
                    : `Block #${block.id}`;

                // Use maxWidth parameter to clip long labels to fit available space
                this.labelCtx.fillText(`${blockName} (${durationNs.toLocaleString()} ns)`, labelX, labelY, visibleWidth - LABEL_CLIP_MARGIN);
            }
        }
    }

    /**
     * Renders zone labels with performance culling.
     *
     * Called every frame. Clears canvas and renders both block and zone labels.
     *
     * Culling strategy (hierarchical early exit):
     * 1. Global sublane height check: Skip if zoom too far out (< 15px height)
     * 2. Hierarchical max-width check: Pre-computed widest zone per visible block lane
     * 3. Per-lane/block lane/block/zone viewport culling: Nested checks at each level
     * 4. Per-zone size check: Only render if zone is wide/tall enough for readability
     *
     * Minimum thresholds:
     * - Zone width: 100px
     * - Zone height: 15px
     *
     * Labels show: "Zone Name (duration ns)" in light gray (#d4d4d4)
     * Long labels are clipped using fillText maxWidth to fit available space.
     * Font: 10px Consolas/Monaco monospace with device pixel ratio scaling
     */
    renderZoneLabels(
        lanes: Lane[],
        blockLanes: BlockLane[],
        formatDescriptors: FormatDescriptor[],
        formatString: (formatDescId: number, params: number[]) => string,
        formatBlockString: (formatDescId: number, blockId: number, clusterId: number) => string
    ): void {
        const rect = this.canvas.getBoundingClientRect();
        const aspect = rect.width / rect.height;

        this.labelCtx.clearRect(0, 0, this.labelCtx.canvas.width, this.labelCtx.canvas.height);

        this.labelCtx.save();
        this.labelCtx.scale(devicePixelRatio, devicePixelRatio);

        // Use cached blocks array to avoid per-frame allocations
        this.renderBlockLabels(this.cachedBlocks, blockLanes, formatDescriptors, formatBlockString);

        const minWidth = MIN_ZONE_LABEL_WIDTH;
        const minHeight = MIN_ZONE_LABEL_HEIGHT;
        const smLabelWidth = SM_LABEL_WIDTH;

        const maxZoneHeight = SUBLANE_HEIGHT;
        const minScreenHeight = maxZoneHeight * this.camera.zoomY * (rect.height / 2);
        if (minScreenHeight < minHeight) {
            this.labelCtx.restore();
            return;
        }

        const worldTop = this.camera.screenToWorld(0, 0, this.canvas).y;
        const worldBottom = this.camera.screenToWorld(0, rect.height, this.canvas).y;

        let maxVisibleZoneWidth = 0;
        for (const blockLane of blockLanes) {
            const blockLaneTop = blockLane.y + blockLane.height;
            const blockLaneBottom = blockLane.y;
            if (blockLaneBottom <= worldTop && blockLaneTop >= worldBottom) {
                maxVisibleZoneWidth = Math.max(maxVisibleZoneWidth, blockLane.maxZoneWidth);
            }
        }

        if (maxVisibleZoneWidth > 0) {
            const maxScreenWidth = maxVisibleZoneWidth * this.camera.zoomX * (rect.width / 2) / aspect;
            if (maxScreenWidth < minWidth) {
                this.labelCtx.restore();
                return;
            }
        }

        this.labelCtx.font = `${LABEL_FONT_SIZE}px ${LABEL_FONT_FAMILY}`;
        this.labelCtx.textAlign = 'left';
        this.labelCtx.textBaseline = 'middle';
        this.labelCtx.fillStyle = LABEL_COLOR;

        const worldLeft = this.camera.screenToWorld(0, 0, this.canvas).x;
        const worldRight = this.camera.screenToWorld(rect.width, 0, this.canvas).x;

        for (const lane of lanes) {
            if (lane.blockLanes.length === 0) {
                continue;
            }

            const laneTopY = lane.y + lane.height;
            const laneBottomY = lane.y;
            const ndcTopY = (laneTopY + this.camera.y) * this.camera.zoomY;
            const ndcBottomY = (laneBottomY + this.camera.y) * this.camera.zoomY;
            const screenTopY = rect.height / 2 - (ndcTopY * rect.height / 2);
            const screenBottomY = rect.height / 2 - (ndcBottomY * rect.height / 2);

            if (screenBottomY < -20 || screenTopY > rect.height + 20) {
                continue;
            }

            for (const blockLane of lane.blockLanes) {
                const blockLaneTopY = blockLane.y + blockLane.height;
                const blockLaneBottomY = blockLane.y;
                const ndcBlockLaneTopY = (blockLaneTopY + this.camera.y) * this.camera.zoomY;
                const ndcBlockLaneBottomY = (blockLaneBottomY + this.camera.y) * this.camera.zoomY;
                const screenBlockLaneTopY = rect.height / 2 - (ndcBlockLaneTopY * rect.height / 2);
                const screenBlockLaneBottomY = rect.height / 2 - (ndcBlockLaneBottomY * rect.height / 2);

                if (screenBlockLaneBottomY < -20 || screenBlockLaneTopY > rect.height + 20) {
                    continue;
                }

                for (const block of blockLane.blocks) {
                    if (block.endX < worldLeft || block.startX > worldRight) {
                        continue;
                    }

                    for (const sublane of block.sublanes) {
                        for (const zone of sublane) {
                            if (zone.endX < worldLeft || zone.startX > worldRight) {
                                continue;
                            }

                            const worldZoneLeft = zone.startX;
                            const worldZoneRight = zone.endX;
                            const worldZoneTop = zone.y + zone.height / 2;
                            const worldZoneBottom = zone.y - zone.height / 2;

                            const ndcLeft = (worldZoneLeft + this.camera.x) * this.camera.zoomX / aspect;
                            const ndcRight = (worldZoneRight + this.camera.x) * this.camera.zoomX / aspect;
                            const ndcTop = (worldZoneTop + this.camera.y) * this.camera.zoomY;
                            const ndcBottom = (worldZoneBottom + this.camera.y) * this.camera.zoomY;

                            const screenLeft = (ndcLeft + 1) * rect.width / 2;
                            const screenRight = (ndcRight + 1) * rect.width / 2;
                            const screenTop = rect.height / 2 - (ndcTop * rect.height / 2);
                            const screenBottom = rect.height / 2 - (ndcBottom * rect.height / 2);

                            const screenHeight = screenBottom - screenTop;

                            const visibleLeft = Math.max(screenLeft, smLabelWidth);
                            const visibleWidth = screenRight - visibleLeft;

                            if (visibleWidth >= minWidth && screenHeight >= minHeight) {
                                const labelX = visibleLeft + ZONE_LABEL_PADDING_X;
                                const labelY = screenTop + ZONE_LABEL_PADDING_Y;
                                const durationNs = Math.round((zone.endX - zone.startX) * MS_TO_NS);

                                const zoneName = formatDescriptors.length > 0
                                    ? formatString(zone.formatDescId, zone.params)
                                    : `#${zone.id}`;

                                // Use maxWidth parameter to clip long labels to fit available space
                                this.labelCtx.fillText(`${zoneName} (${durationNs.toLocaleString()} ns)`, labelX, labelY, visibleWidth - LABEL_CLIP_MARGIN);
                            }
                        }
                    }
                }
            }
        }

        this.labelCtx.restore();
    }

}
