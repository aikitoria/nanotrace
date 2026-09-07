/**
 * High-performance 2D canvas label renderer.
 *
 * Labels are indexed by duration and time. Duration is equivalent to the
 * horizontal zoom at which a label becomes wide enough to draw, so a frame
 * only visits events that can produce visible text at its current zoom.
 */

import { Camera } from '../utils/camera.js';
import { displayLabel, snapTextPixel } from '../utils/text-layout.js';
import { HierarchyData } from '../utils/types.js';
import { formatDuration, NS_TO_MS } from '../utils/soa-helpers.js';
import {
    LANE_EDGE_PADDING,
    SEMANTIC_GROUP_GAP,
    SUBLANE_HEIGHT,
    LABEL_COLOR,
    MIN_BLOCK_LABEL_WIDTH,
    MIN_ZONE_LABEL_WIDTH,
    MIN_ZONE_LABEL_HEIGHT,
    BLOCK_LABEL_PADDING_X,
    ZONE_LABEL_PADDING_X,
    LABEL_CLIP_MARGIN,
    LABEL_FONT_SIZE,
    MIN_LABEL_FONT_SIZE,
    MIN_LABEL_ZOOM_Y,
    LABEL_FONT_FAMILY,
    ZONE_LABEL_COLOR
} from '../utils/constants.js';

const DURATION_BUCKET_COUNT = 54;
const TIME_BUCKET_COUNT = 4096;
const ELLIPSIS = '...';

type CandidateVisitor = (index: number) => void;

/** A compact CSR index over logarithmic duration and linear time buckets. */
class TemporalWidthIndex {
    private starts: Float64Array;
    private ends: Float64Array;
    private totalDurationNs: number;
    private timeBucketScale: number;
    private offsets: Uint32Array;
    private indices: Uint32Array;
    private activeDurationBuckets: number[] = [];
    private maxDurationByBucket = new Float64Array(
        DURATION_BUCKET_COUNT);
    private maxDurationNs = 0;

    constructor(
        starts: Float64Array,
        ends: Float64Array,
        count: number,
        totalDurationNs: number
    ) {
        this.starts = starts;
        this.ends = ends;
        this.totalDurationNs = Math.max(1, totalDurationNs);
        this.timeBucketScale = TIME_BUCKET_COUNT / this.totalDurationNs;

        const cellCount = DURATION_BUCKET_COUNT * TIME_BUCKET_COUNT;
        const counts = new Uint32Array(cellCount);
        const durationCounts = new Uint32Array(DURATION_BUCKET_COUNT);
        for (let index = 0; index < count; index++) {
            const duration = Math.max(1, ends[index] - starts[index]);
            const durationBucket = this.durationBucket(duration);
            const timeBucket = this.timeBucket(starts[index]);
            counts[durationBucket * TIME_BUCKET_COUNT + timeBucket]++;
            durationCounts[durationBucket]++;
            this.maxDurationByBucket[durationBucket] = Math.max(
                this.maxDurationByBucket[durationBucket], duration);
            this.maxDurationNs = Math.max(this.maxDurationNs, duration);
        }

        this.offsets = new Uint32Array(cellCount + 1);
        for (let cell = 0; cell < cellCount; cell++) {
            this.offsets[cell + 1] = this.offsets[cell] + counts[cell];
        }

        const cursors = this.offsets.slice(0, cellCount);
        this.indices = new Uint32Array(count);
        for (let index = 0; index < count; index++) {
            const duration = Math.max(1, ends[index] - starts[index]);
            const durationBucket = this.durationBucket(duration);
            const timeBucket = this.timeBucket(starts[index]);
            const cell = durationBucket * TIME_BUCKET_COUNT + timeBucket;
            this.indices[cursors[cell]++] = index;
        }

        for (let bucket = 0; bucket < DURATION_BUCKET_COUNT; bucket++) {
            if (durationCounts[bucket] !== 0) {
                this.activeDurationBuckets.push(bucket);
            }
        }
    }

    visitCandidates(
        minDurationNs: number,
        visibleStartNs: number,
        visibleEndNs: number,
        visitor: CandidateVisitor
    ): void {
        if (minDurationNs > this.maxDurationNs
            || visibleEndNs < 0
            || visibleStartNs > this.totalDurationNs) {
            return;
        }

        const clampedEnd = Math.min(
            this.totalDurationNs, Math.max(0, visibleEndNs));
        const firstDurationBucket = this.durationBucket(
            Math.max(1, minDurationNs));

        for (const durationBucket of this.activeDurationBuckets) {
            if (durationBucket < firstDurationBucket) continue;

            const earliestStart = Math.max(
                0,
                visibleStartNs
                    - this.maxDurationByBucket[durationBucket]);
            const firstTimeBucket = this.timeBucket(earliestStart);
            const lastTimeBucket = this.timeBucket(clampedEnd);
            const bucketBase = durationBucket * TIME_BUCKET_COUNT;

            for (let timeBucket = firstTimeBucket;
                timeBucket <= lastTimeBucket; timeBucket++) {
                const cell = bucketBase + timeBucket;
                const endOffset = this.offsets[cell + 1];
                for (let offset = this.offsets[cell];
                    offset < endOffset; offset++) {
                    const index = this.indices[offset];
                    const start = this.starts[index];
                    const end = this.ends[index];
                    if (end < visibleStartNs
                        || start > visibleEndNs
                        || end - start < minDurationNs) {
                        continue;
                    }
                    visitor(index);
                }
            }
        }
    }

    private durationBucket(duration: number): number {
        return Math.min(
            DURATION_BUCKET_COUNT - 1,
            Math.max(0, Math.floor(Math.log2(duration))));
    }

    private timeBucket(timestamp: number): number {
        return Math.min(
            TIME_BUCKET_COUNT - 1,
            Math.max(0, Math.floor(timestamp * this.timeBucketScale)));
    }
}

export class LabelRenderer {
    private labelCtx: CanvasRenderingContext2D;
    private camera: Camera;
    private hierarchy: HierarchyData;
    private zoneIndex: TemporalWidthIndex;
    private blockIndex: TemporalWidthIndex;
    private overlayIndex: TemporalWidthIndex;

    private trackLabelWidth = 0;
    private textPixelRatio = 1;
    private textBaselineOffset = 0;
    private hoveredSemanticRange = -1;
    private dirty = true;
    private lastCameraX = Number.NaN;
    private lastCameraY = Number.NaN;
    private lastZoomX = Number.NaN;
    private lastZoomY = Number.NaN;
    private lastCanvasWidth = 0;
    private lastCanvasHeight = 0;
    private lastDevicePixelRatio = 0;

    constructor(
        labelCtx: CanvasRenderingContext2D,
        camera: Camera,
        hierarchy: HierarchyData
    ) {
        this.labelCtx = labelCtx;
        this.camera = camera;
        this.hierarchy = hierarchy;
        this.overlayIndex = new TemporalWidthIndex(
            hierarchy.overlays.startsX, hierarchy.overlays.endsX,
            hierarchy.overlays.count, hierarchy.totalDurationNs);
        this.zoneIndex = new TemporalWidthIndex(
            hierarchy.zones.startsX,
            hierarchy.zones.endsX,
            hierarchy.zones.count,
            hierarchy.totalDurationNs);
        this.blockIndex = new TemporalWidthIndex(
            hierarchy.blocks.startsX,
            hierarchy.blocks.endsX,
            hierarchy.blocks.count,
            hierarchy.totalDurationNs);
    }

    setTrackLabelWidth(width: number): void {
        if (this.trackLabelWidth === width) return;
        this.trackLabelWidth = width;
        this.invalidate();
    }

    updateCamera(camera: Camera): void {
        this.camera = camera;
        this.invalidate();
    }

    setHoveredSemanticRange(index: number): void {
        if (this.hoveredSemanticRange === index) return;
        this.hoveredSemanticRange = index;
        this.invalidate();
    }

    private semanticGroupBounds(rowVisible: Uint8Array, yScale: number, yOffset: number) {
        const ranges = this.hierarchy.overlays;
        const tops = new Float64Array(ranges.groupRows.length).fill(Infinity);
        const bottoms = new Float64Array(ranges.groupRows.length).fill(-Infinity);
        ranges.groupRows.forEach((rows, group) => {
            for (const row of rows) {
                if (!rowVisible[row]) continue;
                const lane = this.hierarchy.smAccelerator.getLaneIndex(row);
                if (lane === undefined) continue;
                // Lane positions already include row compaction.
                const bottom = yOffset - this.hierarchy.lanes.ys[lane] * yScale;
                const top = bottom - this.hierarchy.lanes.heights[lane] * yScale;
                tops[group] = Math.min(tops[group], top);
                bottoms[group] = Math.max(bottoms[group], bottom);
            }
        });
        return {tops, bottoms};
    }

    findSemanticRangeAtPosition(x: number, y: number, rowVisible: Uint8Array): {index: number; header: boolean} {
        const width = this.labelCtx.canvas.width / devicePixelRatio;
        const height = this.labelCtx.canvas.height / devicePixelRatio;
        if (x < this.trackLabelWidth || x >= width || y < 0 || y >= height)
            return {index: -1, header: false};
        const nsToPixels = NS_TO_MS * this.camera.zoomX * height / 2;
        const xOffset = this.camera.x * this.camera.zoomX * height / 2 + width / 2;
        const yScale = this.camera.zoomY * height / 2;
        const yOffset = height / 2 - this.camera.y * yScale;
        const {tops, bottoms} = this.semanticGroupBounds(rowVisible, yScale, yOffset);
        const ns = (x - xOffset) / nsToPixels;
        let hit = -1;
        this.overlayIndex.visitCandidates(0, ns, ns, index => {
            const group = this.hierarchy.overlays.groupIndices[index];
            if (ns >= this.hierarchy.overlays.endsX[index] || y < tops[group] - SEMANTIC_GROUP_GAP * yScale || y >= bottoms[group]) return;
            if (hit !== -1) throw new Error('Invalid overlapping semantic ranges in trace');
            hit = index;
        });
        const headerBottom = hit < 0 ? 0 : tops[this.hierarchy.overlays.groupIndices[hit]];
        return {index: hit, header: hit >= 0 && y < headerBottom};
    }

    invalidate(): void {
        this.dirty = true;
    }

    render(
        rowOffsets: Float32Array,
        rowVisible: Uint8Array,
        zoneVisibility: Uint32Array
    ): void {
        const dpr = devicePixelRatio;
        const canvasWidth = this.labelCtx.canvas.width;
        const canvasHeight = this.labelCtx.canvas.height;
        const zoomX = this.camera.zoomX;
        const zoomY = this.camera.zoomY;

        if (!this.dirty
            && this.lastCameraX === this.camera.x
            && this.lastCameraY === this.camera.y
            && this.lastZoomX === zoomX
            && this.lastZoomY === zoomY
            && this.lastCanvasWidth === canvasWidth
            && this.lastCanvasHeight === canvasHeight
            && this.lastDevicePixelRatio === dpr) {
            return;
        }

        this.dirty = false;
        this.lastCameraX = this.camera.x;
        this.lastCameraY = this.camera.y;
        this.lastZoomX = zoomX;
        this.lastZoomY = zoomY;
        this.lastCanvasWidth = canvasWidth;
        this.lastCanvasHeight = canvasHeight;
        this.lastDevicePixelRatio = dpr;

        this.labelCtx.clearRect(0, 0, canvasWidth, canvasHeight);
        if (canvasWidth === 0 || canvasHeight === 0) {
            return;
        }

        const width = canvasWidth / dpr;
        const height = canvasHeight / dpr;
        const aspect = width / height;
        const nanosecondsToPixels = NS_TO_MS * zoomX
            * (width / 2) / aspect;
        if (nanosecondsToPixels <= 0) return;

        const xOffset = (this.camera.x * zoomX / aspect + 1)
            * width / 2;
        const yScale = zoomY * height / 2;
        const yOffset = height / 2 - this.camera.y * yScale;
        const visibleStartNs = Math.max(
            0, (this.trackLabelWidth - xOffset) / nanosecondsToPixels);
        const visibleEndNs = Math.min(
            this.hierarchy.totalDurationNs,
            (width - xOffset) / nanosecondsToPixels);
        if (visibleEndNs < visibleStartNs) return;

        const zoneScreenHeight = SUBLANE_HEIGHT * yScale;
        const fontSize = snapTextPixel(LABEL_FONT_SIZE * zoneScreenHeight
            / MIN_ZONE_LABEL_HEIGHT, dpr);

        this.labelCtx.save();
        this.labelCtx.scale(dpr, dpr);
        this.labelCtx.font = `${fontSize}px ${LABEL_FONT_FAMILY}`;
        this.labelCtx.textAlign = 'left';
        this.labelCtx.textBaseline = 'alphabetic';
        this.textPixelRatio = dpr;
        const metrics = this.labelCtx.measureText('Mg');
        const ascent = metrics.fontBoundingBoxAscent ?? fontSize * 0.8;
        const descent = metrics.fontBoundingBoxDescent ?? fontSize * 0.2;
        this.textBaselineOffset = (ascent - descent) / 2;
        const fontScale = fontSize / LABEL_FONT_SIZE;

        const drawLabels = zoomY >= MIN_LABEL_ZOOM_Y && fontSize >= MIN_LABEL_FONT_SIZE;
        if (drawLabels) {
            this.renderBlockLabels(
                rowOffsets, rowVisible, zoneVisibility,
                nanosecondsToPixels, xOffset, yScale, yOffset,
                visibleStartNs, visibleEndNs, height,
                fontScale);
            this.renderZoneLabels(
                rowOffsets, rowVisible, zoneVisibility,
                nanosecondsToPixels, xOffset, yScale, yOffset,
                visibleStartNs, visibleEndNs, height,
                zoneScreenHeight, fontSize, fontScale);
        }
        this.renderSemanticRanges(rowVisible,
            nanosecondsToPixels, xOffset, yScale, yOffset, width, height, fontScale, drawLabels);
        this.labelCtx.restore();
    }

    private renderSemanticRanges(
        rowVisible: Uint8Array,
        nanosecondsToPixels: number, xOffset: number,
        yScale: number, yOffset: number, width: number, height: number, fontScale: number, drawLabels: boolean
    ): void {
        // Labels occupy the reserved gap between groups. The range line sits
        // on the group's outer frame; row and zone heights stay unchanged.
        const ctx = this.labelCtx;
        const ranges = this.hierarchy.overlays;
        const {tops: groupTops, bottoms: groupBottoms} = this.semanticGroupBounds(rowVisible, yScale, yOffset);
        const gapHeight = (SEMANTIC_GROUP_GAP - LANE_EDGE_PADDING) * yScale;
        const horizontalPadding = ZONE_LABEL_PADDING_X * fontScale;
        const visibleStart = Math.max(0, (this.trackLabelWidth - xOffset) / nanosecondsToPixels);
        const visibleEnd = Math.min(this.hierarchy.totalDurationNs, (width - xOffset) / nanosecondsToPixels);
        this.overlayIndex.visitCandidates(1 / nanosecondsToPixels, visibleStart, visibleEnd, index => {
            const group = ranges.groupIndices[index];
            const trackTop = groupTops[group];
            if (groupBottoms[group] < 0 || trackTop > height) return;
            const top = trackTop - gapHeight;
            const left = Math.max(this.trackLabelWidth, ranges.startsX[index] * nanosecondsToPixels + xOffset);
            const right = Math.min(width, ranges.endsX[index] * nanosecondsToPixels + xOffset);
            if (right <= left) return;
            ctx.save();
            ctx.beginPath();
            ctx.rect(left, top, right - left, groupBottoms[group] - top);
            ctx.clip();
            ctx.fillStyle = `#${ranges.colors[index].toString(16).padStart(6, '0')}`;
            ctx.globalAlpha = 1;
            ctx.fillRect(left, trackTop, right - left, 2);
            const label = drawLabels && right - left >= MIN_ZONE_LABEL_WIDTH
                ? this.fitLabel(displayLabel(ranges.labels[index]), right - left - horizontalPadding
                    - LABEL_CLIP_MARGIN * fontScale) : null;
            if (label !== null) {
                this.drawText(label, left + horizontalPadding, top + gapHeight / 2);
            }
            ctx.restore();
        });
        const index = this.hoveredSemanticRange;
        if (index >= 0 && index < ranges.count) {
            const group = ranges.groupIndices[index];
            const top = Math.max(0, groupTops[group]);
            const rowHeight = Math.min(height, groupBottoms[group]) - top;
            if (rowHeight <= 0) return;
            const left = Math.max(this.trackLabelWidth, ranges.startsX[index] * nanosecondsToPixels + xOffset);
            const right = Math.min(width, ranges.endsX[index] * nanosecondsToPixels + xOffset);
            if (right <= left) return;
            ctx.save();
            ctx.beginPath();
            ctx.rect(left, top, right - left, rowHeight);
            ctx.clip();
            ctx.fillStyle = `#${ranges.colors[index].toString(16).padStart(6, '0')}`;
            ctx.globalAlpha = 0.13;
            ctx.fillRect(left, top, right - left, rowHeight);
            ctx.globalAlpha = 0.9;
            ctx.fillRect(left, top, right - left, 2);
            ctx.restore();
        }
    }

    private renderBlockLabels(
        rowOffsets: Float32Array,
        rowVisible: Uint8Array,
        zoneVisibility: Uint32Array,
        nanosecondsToPixels: number,
        xOffset: number,
        yScale: number,
        yOffset: number,
        visibleStartNs: number,
        visibleEndNs: number,
        height: number,
        fontScale: number
    ): void {
        const blocks = this.hierarchy.blocks;
        const minDurationNs = MIN_BLOCK_LABEL_WIDTH
            / nanosecondsToPixels;
        const horizontalPadding = BLOCK_LABEL_PADDING_X * fontScale;
        this.labelCtx.fillStyle = LABEL_COLOR;

        this.blockIndex.visitCandidates(
            minDurationNs, visibleStartNs, visibleEndNs, blockIndex => {
                const rowIndex = blocks.smIndices[blockIndex];
                const firstZone = blocks.zonesStartIndices[blockIndex];
                if (rowVisible[rowIndex] === 0
                    || zoneVisibility[firstZone] === 0) {
                    return;
                }

                const rowOffset = rowOffsets[rowIndex];
                const blockTop = blocks.ys[blockIndex] + rowOffset
                    + blocks.heights[blockIndex];
                const screenTop = yOffset - blockTop * yScale;
                const headerHeight = blocks.headerHeights[blockIndex]
                    * yScale;
                if (headerHeight < MIN_LABEL_FONT_SIZE
                    || screenTop + headerHeight < 0
                    || screenTop > height) {
                    return;
                }

                const screenLeft = blocks.startsX[blockIndex]
                    * nanosecondsToPixels + xOffset;
                const screenRight = blocks.endsX[blockIndex]
                    * nanosecondsToPixels + xOffset;
                const visibleLeft = Math.max(
                    screenLeft, this.trackLabelWidth);
                const visibleWidth = screenRight - visibleLeft;
                if (visibleWidth < MIN_BLOCK_LABEL_WIDTH) return;

                const name = this.formatBlockName(blockIndex);
                const duration = blocks.endsX[blockIndex]
                    - blocks.startsX[blockIndex];
                const label = this.fitLabel(
                    `${name} (${formatDuration(duration)})`,
                    visibleWidth - horizontalPadding
                        - LABEL_CLIP_MARGIN * fontScale);
                if (label !== null) {
                    this.drawText(
                        label,
                        visibleLeft + horizontalPadding,
                        screenTop + headerHeight / 2);
                }
            });
    }

    private renderZoneLabels(
        rowOffsets: Float32Array,
        rowVisible: Uint8Array,
        zoneVisibility: Uint32Array,
        nanosecondsToPixels: number,
        xOffset: number,
        yScale: number,
        yOffset: number,
        visibleStartNs: number,
        visibleEndNs: number,
        height: number,
        zoneScreenHeight: number,
        fontSize: number,
        fontScale: number
    ): void {
        const zones = this.hierarchy.zones;
        const minRenderableWidth = Math.min(
            MIN_ZONE_LABEL_WIDTH, fontSize);
        const minDurationNs = minRenderableWidth / nanosecondsToPixels;
        const horizontalPadding = ZONE_LABEL_PADDING_X * fontScale;
        this.labelCtx.fillStyle = ZONE_LABEL_COLOR;

        this.zoneIndex.visitCandidates(
            minDurationNs, visibleStartNs, visibleEndNs, zoneIndex => {
                if (zoneVisibility[zoneIndex] === 0) return;
                const rowIndex = zones.smIndices[zoneIndex];
                if (rowVisible[rowIndex] === 0) return;

                const screenCenterY = yOffset
                    - (zones.ys[zoneIndex] + rowOffsets[rowIndex]) * yScale;
                if (screenCenterY + zoneScreenHeight / 2 < 0
                    || screenCenterY - zoneScreenHeight / 2 > height) {
                    return;
                }

                const screenLeft = zones.startsX[zoneIndex]
                    * nanosecondsToPixels + xOffset;
                const screenRight = zones.endsX[zoneIndex]
                    * nanosecondsToPixels + xOffset;
                const visibleLeft = Math.max(
                    screenLeft, this.trackLabelWidth);
                const visibleWidth = screenRight - visibleLeft;
                const fullLabel = visibleWidth >= MIN_ZONE_LABEL_WIDTH;

                if (!fullLabel) {
                    if (zones.hasChildren[zoneIndex] !== 0
                        && visibleWidth >= fontSize
                        && zoneScreenHeight >= fontSize) {
                        this.drawText(
                            zones.expanded[zoneIndex] !== 0
                                ? '\u25be' : '\u25b8',
                            visibleLeft + horizontalPadding,
                            screenCenterY);
                    }
                    return;
                }

                const name = this.formatZoneName(zoneIndex);
                const duration = zones.endsX[zoneIndex]
                    - zones.startsX[zoneIndex];
                const disclosure = zones.hasChildren[zoneIndex] !== 0
                    ? (zones.expanded[zoneIndex] !== 0
                        ? '\u25be ' : '\u25b8 ')
                    : '';
                const label = this.fitLabel(
                    `${disclosure}${name} (${formatDuration(duration)})`,
                    visibleWidth - horizontalPadding
                        - LABEL_CLIP_MARGIN * fontScale);
                if (label !== null) {
                    this.drawText(
                        label, visibleLeft + horizontalPadding,
                        screenCenterY);
                }
            });
    }

    private drawText(text: string, x: number, centerY: number): void {
        this.labelCtx.fillText(text,
            snapTextPixel(x, this.textPixelRatio),
            snapTextPixel(centerY + this.textBaselineOffset, this.textPixelRatio));
    }

    private formatZoneName(zoneIndex: number): string {
        const zones = this.hierarchy.zones;
        let result = this.hierarchy.formatDescriptors[
            zones.formatDescIds[zoneIndex]]?.labelString ?? 'Event';
        const parameterOffset = zones.paramsOffsets[zoneIndex];
        const parameterCount = zones.paramsCounts[zoneIndex];
        for (let parameterIndex = 0;
            parameterIndex < parameterCount; parameterIndex++) {
            result = result.replace(
                `{${parameterIndex}}`,
                zones.paramsPool[
                    parameterOffset + parameterIndex].toString());
        }
        return displayLabel(result);
    }

    private formatBlockName(blockIndex: number): string {
        const blocks = this.hierarchy.blocks;
        let result = this.hierarchy.formatDescriptors[
            blocks.formatDescIds[blockIndex]]?.labelString ?? 'Block';
        result = result.replace(
            '{blockLinear}', blocks.gridIds[blockIndex].toString());
        result = result.replace(
            '{clusterLinear}', blocks.clusterIds[blockIndex].toString());
        return displayLabel(result);
    }

    /** Returns the longest exact prefix that fits without scaling the text. */
    private fitLabel(text: string, maxWidth: number): string | null {
        if (maxWidth <= 0) return null;
        if (this.labelCtx.measureText(text).width <= maxWidth) return text;
        if (this.labelCtx.measureText(ELLIPSIS).width > maxWidth) return null;

        let low = 0;
        let high = text.length;
        while (low < high) {
            const middle = Math.ceil((low + high) / 2);
            const candidate = `${text.slice(0, middle)}${ELLIPSIS}`;
            if (this.labelCtx.measureText(candidate).width <= maxWidth) {
                low = middle;
            } else {
                high = middle - 1;
            }
        }

        return `${text.slice(0, low)}${ELLIPSIS}`;
    }
}
