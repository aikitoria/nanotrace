#!/usr/bin/env node

import * as fs from 'node:fs';
import * as path from 'node:path';

import {
    buildHierarchy,
    formatTooltipString,
    formatTrackTooltipString,
    formatBlockTooltipString,
    parseTraceFile,
    projectTraceData,
    type ParsedTraceData
} from '../src/utils/file-loader.js';
import { binarySearchZones } from '../src/utils/soa-helpers.js';
import {
    AggregateSelectionStatistics
} from '../src/utils/selection-stats.js';

import { Camera } from '../src/utils/camera.js';
import { InteractionManager } from '../src/interaction-manager.js';
import { LabelRenderer } from '../src/renderers/label-renderer.js';
import { semanticGroupTopRows, semanticRangeColor } from '../src/utils/semantic-layout.js';
import { TraceOverlays, type HierarchyData } from '../src/utils/types.js';
import { LABEL_FONT_SIZE, SUBLANE_HEIGHT, MIN_ZONE_LABEL_HEIGHT, SEMANTIC_GROUP_GAP, LANE_PADDING, BLOCK_LANE_PADDING, LANE_EDGE_PADDING } from '../src/utils/constants.js';

function validateOverlayRendering(): void {
    const draws: Array<{text: string; font: number; y: number}> = [];
    const lines: number[] = [];
    const fills: Array<{x: number; y: number; width: number; height: number; color: string; alpha: number}> = [];
    let font = '';
    const ctx = {
        canvas: {width: 800, height: 600},
        get font() { return font; }, set font(value: string) { font = value; },
        clearRect() { draws.length = 0; lines.length = 0; fills.length = 0; },
        save() {}, restore() {}, scale() {}, beginPath() {}, rect() {}, clip() {},
        fillStyle: '', globalAlpha: 1,
        fillRect(x: number, y: number, width: number, height: number) {
            if (height === 2) lines.push(y);
            fills.push({x, y, width, height, color: ctx.fillStyle as string, alpha: ctx.globalAlpha});
        },
        measureText(text: string) { return {width: text.length * parseFloat(font) / 2}; },
        fillText(text: string, _x: number, y: number) { draws.push({text, font: parseFloat(font), y}); }
    } as unknown as CanvasRenderingContext2D;
    const overlays = new TraceOverlays(2, [[0]]);
    overlays.startsX.set([100000, 400000]);
    overlays.endsX.set([400000, 800000]);
    overlays.labels[0] = 'Slot 0 Attention Layer 12';
    overlays.labels[1] = 'Slot 1 MoE Layer 12';
    overlays.colors.set([0x4080c0, 0xc08040]);
    const empty = {count: 0, startsX: new Float64Array(), endsX: new Float64Array()};
    const hierarchy = {
        overlays, totalDurationNs: 1000000, zones: empty, blocks: empty,
        smAccelerator: {getLaneIndex() { return 0; }},
        lanes: {ys: new Float32Array([0]), heights: new Float32Array([0.02])}
    } as unknown as HierarchyData;
    const camera = new Camera(0, 1);
    camera.x = camera.y = 0;
    camera.zoom = 5;
    camera.xZoomMultiplier = 0.2;
    Object.defineProperty(globalThis, 'devicePixelRatio', {value: 1, configurable: true});
    const renderer = new LabelRenderer(ctx, camera, hierarchy);
    const rows = new Float32Array(1);
    const visible = new Uint8Array([1]);
    const render = () => renderer.render(rows, visible, new Uint32Array());
    render();
    const zoneSize = LABEL_FONT_SIZE * SUBLANE_HEIGHT * camera.zoomY * 300 / MIN_ZONE_LABEL_HEIGHT;
    if (draws.length !== 2 || draws.some(draw => Math.abs(draw.font - zoneSize * 2 / 3) > 1e-8))
        throw new Error('Non-hover semantic labels must use two-thirds zone font size');
    if (draws.some(draw => draw.y + draw.font / 2 >= 270))
        throw new Error('Semantic labels must stay above the tracks');
    if (lines.length !== 2 || lines.some(y => Math.abs(y - 270) > 0.001))
        throw new Error('Semantic highlight lines must align with the outer group frame');
    const expectedLabelY = 270 - (SEMANTIC_GROUP_GAP - LANE_EDGE_PADDING) * camera.zoomY * 300 + draws[0].font / 2 + 4;
    if (draws.some(draw => Math.abs(draw.y - expectedLabelY) > 0.001))
        throw new Error('Region labels must be lowered by four screen pixels');
    if (fills.length !== 2 || fills.some(fill => fill.height !== 2 || fill.alpha !== 1))
        throw new Error('Non-hover regions must draw only opaque horizontal lines');
    const first = renderer.findSemanticRangeAtPosition(500, draws[0].y, visible);
    const second = renderer.findSemanticRangeAtPosition(520, 290, visible);
    if (first.index !== 0 || !first.header || second.index !== 1 || second.header)
        throw new Error('Range hover must use disjoint half-open intervals and identify its header');
    renderer.setHoveredSemanticRange(first.index);
    render();
    if (draws.length !== 2)
        throw new Error('Range hover must not draw a separate canvas popup');
    rows[0] = 1000;
    renderer.invalidate();
    render();
    if (draws.length !== 2 || renderer.findSemanticRangeAtPosition(500, 280, visible).index !== 0)
        throw new Error('Compacted row offsets must not move semantic highlights or hit regions');
    renderer.setHoveredSemanticRange(-1);
    visible[0] = 0;
    renderer.invalidate();
    render();
    if (Number(draws.length) !== 0 || renderer.findSemanticRangeAtPosition(500, 280, visible).index !== -1)
        throw new Error('Hidden tracks must hide semantic highlights and hit regions');
    visible[0] = 1;
    overlays.startsX[1] = 200000;
    const invalid = new LabelRenderer(ctx, camera, hierarchy);
    let rejected = false;
    try { invalid.findSemanticRangeAtPosition(500, 290, visible); } catch { rejected = true; }
    if (!rejected) throw new Error('Invalid overlapping ranges must not be resolved by picking a winner');
    const groups = new TraceOverlays(0, [[0, 1], [2, 3]]);
    if (String(semanticGroupTopRows(4, groups)) !== '1,0,1,0'
        || String(semanticGroupTopRows(4, groups, new Uint8Array([0, 1, 1, 0]))) !== '0,1,1,0'
        || String(semanticGroupTopRows(4, groups, new Uint8Array(4))) !== '0,0,0,0')
        throw new Error('Group spacing must follow the first visible row and disappear for hidden groups');
    const colors = new Set<number>();
    for (const slot of [0, 1]) for (const stage of ['Attention', 'Expert', 'Merge']) {
        const color = semanticRangeColor(`Slot ${slot} ${stage} Layer 3`, 0);
        colors.add(color);
        if (color !== semanticRangeColor(`Slot ${slot} ${stage} Layer 77`, 1))
            throw new Error('Semantic colors must be stable across model layers');
    }
    if (colors.size !== 6) throw new Error('The two banks must use six distinct slot/stage colors');
}

function validateSharedTooltip(hierarchy: HierarchyData): void {
    if (!hierarchy.overlays.count || !hierarchy.zones.count) return;
    const tooltip = {innerHTML: '', style: {left: '', top: ''},
        classList: {add() {}, remove() {}}} as unknown as HTMLElement;
    const canvas = {style: {cursor: ''}} as unknown as HTMLCanvasElement;
    const manager = new InteractionManager(new Camera(0, 1), canvas, tooltip,
        tooltip, tooltip, tooltip, tooltip, () => {});
    const formatter = (id: number) => hierarchy.formatDescriptors[id].labelString;
    manager.findZoneAtPosition = () => ({zoneIdx: -1, blockIdx: -1});
    const range = manager.updateHover(500, 280, hierarchy, formatter, formatter, formatter, 0);
    if (range !== 0 || !tooltip.innerHTML.includes(hierarchy.overlays.labels[0])
        || tooltip.innerHTML.includes('Start:') || tooltip.innerHTML.includes('End:')
        || !tooltip.innerHTML.includes('Duration:'))
        throw new Error('Semantic hover must show duration without absolute start/end times');
    const left = tooltip.style.left, top = tooltip.style.top;
    manager.findZoneAtPosition = () => ({zoneIdx: 0, blockIdx: hierarchy.zones.blockIndices[0]});
    if (manager.updateHover(500, 280, hierarchy, formatter, formatter, formatter, 0) !== 0
        || tooltip.style.left !== left || tooltip.style.top !== top
        || tooltip.innerHTML.includes('Start:') || tooltip.innerHTML.includes('End:')
        || !tooltip.innerHTML.includes('Duration:'))
        throw new Error('Zone details must retain tooltip priority while highlighting their semantic region');
    if (manager.updateHover(500, 280, hierarchy, formatter, formatter, formatter, 0, true) !== 0)
        throw new Error('The semantic header must select its range through the same tooltip');
}

function validateZoneHover(hierarchy: HierarchyData): void {
    const element = {style: {}, classList: {add() {}, remove() {}},
        getBoundingClientRect() { return {left: 0, top: 0, width: 1000, height: 600}; }} as unknown as HTMLElement;
    const camera = new Camera(0, 1);
    camera.zoom = 5;
    camera.xZoomMultiplier = 1;
    const renderer = new LabelRenderer({canvas: {width: 1000, height: 600}} as CanvasRenderingContext2D, camera, hierarchy);
    const manager = new InteractionManager(camera, element as HTMLCanvasElement,
        element, element, element, element, element, () => {});
    const visible = new Uint8Array(hierarchy.lanes.count).fill(1);
    const offsets = new Float32Array(hierarchy.lanes.count);
    manager.updateLayout(offsets, visible, Uint32Array.from(hierarchy.zones.parentEventIds, parent => parent === 0n ? 1 : 0));
    const {lanes, blockLanes, blocks, zones} = hierarchy;
    let checks = 0;
    const originalYs = lanes.ys.slice();
    visible.fill(0);
    for (let zone = 0; zone < zones.count; zone++)
        if (zones.parentEventIds[zone] === 0n) visible[zones.smIndices[zone]] = 1;
    const semanticTops = semanticGroupTopRows(lanes.count, hierarchy.overlays, visible);
    let currentY = 0;
    for (let row = lanes.count - 1; row >= 0; row--) {
        offsets[row] = currentY - originalYs[row];
        lanes.ys[row] = currentY;
        if (visible[row]) currentY += lanes.heights[row] + LANE_PADDING + BLOCK_LANE_PADDING
            + (semanticTops[row] ? SEMANTIC_GROUP_GAP : 0);
    }
    for (let row = 0; row < lanes.count; row++) {
        for (let lane = lanes.blockLanesStartIndices[row]; lane < lanes.blockLanesEndIndices[row]; lane++) {
            const offset = blockLanes.blockIndicesOffsets[lane];
            const count = blockLanes.blockIndicesCounts[lane];
            for (const index of new Set([0, Math.floor(count / 2), count - 1])) {
                if (index < 0 || index >= count) continue;
                const block = blockLanes.blockIndices[offset + index];
                const start = blocks.zonesStartIndices[block], end = blocks.zonesEndIndices[block];
                for (const zone of new Set(Array.from({length: 33}, (_, i) => start + Math.floor((end - start - 1) * i / 32)))) {
                    if (zone < start || zone >= end || zones.parentEventIds[zone] !== 0n || zones.endsX[zone] <= zones.startsX[zone]) continue;
                    camera.x = -(zones.startsX[zone] + zones.endsX[zone]) / 2e6;
                    camera.y = -(zones.ys[zone] + offsets[row]);
                    const semantic = renderer.findSemanticRangeAtPosition(500, 300, visible);
                    if (semantic.header) throw new Error(`Semantic header stole zone hover on row ${row}`);
                    const hit = manager.findZoneAtPosition(500, 300, hierarchy);
                    if (hit.zoneIdx !== zone)
                        throw new Error(`Zone hover missed row ${row} (${lanes.names[row]}), block ${block}, zone ${zone}: ${JSON.stringify(hit)}`);
                    const highlighted = manager.updateHover(500, 300, hierarchy,
                        (id, params) => formatTooltipString(hierarchy.formatDescriptors, id, params),
                        (id, lane, params) => formatTrackTooltipString(hierarchy.formatDescriptors, id, lane, params),
                        (id, block, cluster) => formatBlockTooltipString(hierarchy.formatDescriptors, id, block, cluster),
                        semantic.index, semantic.header);
                    if (highlighted !== semantic.index || manager.getHoveredZoneId() !== zone)
                        throw new Error(`Full hover failed on row ${row}, zone ${zone}`);
                    checks++;
                }
            }
        }
    }
    lanes.ys.set(originalYs);
    console.log(`Actual zone hover checks: ${checks}`);
}

function validateTrackOrder(trace: ParsedTraceData): void {
    let encounteredCpuTrack = false;

    for (const hierarchy of trace.trackHierarchies) {
        const gpuTrack = hierarchy.some(node => node.kind === 2);
        if (!gpuTrack) encounteredCpuTrack = true;
        if (gpuTrack && encounteredCpuTrack) {
            throw new Error('GPU track appears below a CPU track');
        }
    }
}

function validateProjection(trace: ParsedTraceData): void {
    if (trace.tracks.count === 0 || trace.zones.count === 0) {
        throw new Error('Trace has no visible events');
    }
    if (trace.trackNames.length !== trace.trackHierarchies.length
        || trace.trackNames.length !== trace.trackDepths.length) {
        throw new Error('Track metadata arrays have different lengths');
    }

    for (let trackIndex = 0;
        trackIndex < trace.tracks.count; trackIndex++) {
        if (trace.tracks.formatDescIds[trackIndex]
            >= trace.formatDescriptors.length) {
            throw new Error(`Track ${trackIndex} has an invalid format`);
        }
    }
    for (let zoneIndex = 0;
        zoneIndex < trace.zones.count; zoneIndex++) {
        if (!Number.isInteger(trace.zones.eventSpecIds[zoneIndex])) {
            throw new Error(`Event ${zoneIndex} has an invalid specification`);
        }
        if (trace.zones.formatDescIds[zoneIndex]
            >= trace.formatDescriptors.length) {
            throw new Error(`Event ${zoneIndex} has an invalid format`);
        }
    }
    for (let blockIndex = 0;
        blockIndex < trace.blocks.count; blockIndex++) {
        if (trace.blocks.formatDescIds[blockIndex]
            >= trace.formatDescriptors.length) {
            throw new Error(`Block ${blockIndex} has an invalid format`);
        }
    }

    for (let blockIndex = 0;
        blockIndex < trace.blocks.count; blockIndex++) {
        const sublaneEnds: number[] = [];
        const zoneStart = trace.blocks.zonesStartIndices[blockIndex];
        const zoneEnd = trace.blocks.zonesEndIndices[blockIndex];
        const hoverSampleStride = Math.max(
            1, Math.floor((zoneEnd - zoneStart) / 32));

        for (let zoneIndex = zoneStart;
            zoneIndex < zoneEnd; zoneIndex++) {
            const start = trace.zones.startsX[zoneIndex];
            const end = trace.zones.endsX[zoneIndex];
            const sublane = trace.zones.sublaneIndices[zoneIndex];
            const row = trace.zones.smIndices[zoneIndex];
            const track = trace.zones.trackIndices[zoneIndex];

            if (!Number.isFinite(start) || !Number.isFinite(end)
                || end < start) {
                throw new Error(
                    `Event ${zoneIndex} has invalid bounds [${start}, ${end}]`);
            }
            if (row >= trace.trackNames.length
                || track >= trace.tracks.count) {
                throw new Error(
                    `Event ${zoneIndex} references an unknown row or track`);
            }
            if ((sublaneEnds[sublane] ?? Number.NEGATIVE_INFINITY) > start) {
                throw new Error(
                    `Block ${blockIndex} has overlapping zones on sublane `
                    + `${sublane}`);
            }
            sublaneEnds[sublane] = end;

            if ((zoneIndex - zoneStart) % hoverSampleStride === 0) {
                const midpoint = (start + end) / 2;
                const foundZone = binarySearchZones(
                    trace.zones, zoneStart, zoneEnd, midpoint, sublane);
                if (foundZone !== zoneIndex) {
                    throw new Error(
                        `Hover lookup returned zone ${foundZone} instead of `
                        + `${zoneIndex}`);
                }
            }
        }
    }

    for (const bookmark of trace.bookmarks) {
        if (!Number.isFinite(bookmark.timestampNs)
            || bookmark.timestampNs < 0
            || bookmark.label.length === 0) {
            throw new Error('Trace has an invalid bookmark');
        }
    }
}

function validateSelectionStatistics(trace: ParsedTraceData): void {
    const zoneVisibility = new Uint32Array(trace.zones.count);
    zoneVisibility.fill(1);
    const rowSelected = new Uint8Array(trace.trackNames.length);
    rowSelected.fill(1);
    const rowOffsets = new Float32Array(trace.trackNames.length);
    const rowIsGpu = new Uint8Array(trace.trackNames.length);
    for (let rowIndex = 0;
        rowIndex < trace.trackHierarchies.length; rowIndex++) {
        rowIsGpu[rowIndex] = trace.trackHierarchies[rowIndex]
            .some(node => node.kind === 2) ? 1 : 0;
    }

    const statistics = AggregateSelectionStatistics(
        trace.zones,
        zoneVisibility,
        rowSelected,
        rowOffsets,
        rowIsGpu,
        Number.NEGATIVE_INFINITY,
        Number.POSITIVE_INFINITY,
        Number.NEGATIVE_INFINITY,
        Number.POSITIVE_INFINITY
    );
    const countedZones = [...statistics.cpu, ...statistics.gpu]
        .reduce((count, statistic) => count + statistic.count, 0);
    if (countedZones !== trace.zones.count) {
        throw new Error(
            `Selection statistics counted ${countedZones} of `
            + `${trace.zones.count} zones`);
    }

    if (trace.zones.count === 0) return;
    rowSelected.fill(0);
    const selectedRow = trace.zones.smIndices[0];
    rowSelected[selectedRow] = 1;
    const rowStatistics = AggregateSelectionStatistics(
        trace.zones,
        zoneVisibility,
        rowSelected,
        rowOffsets,
        rowIsGpu,
        Number.NEGATIVE_INFINITY,
        Number.POSITIVE_INFINITY,
        Number.NEGATIVE_INFINITY,
        Number.POSITIVE_INFINITY
    );
    const countedRowZones = [...rowStatistics.cpu, ...rowStatistics.gpu]
        .reduce((count, statistic) => count + statistic.count, 0);
    let expectedRowZones = 0;
    for (let zoneIndex = 0;
        zoneIndex < trace.zones.count; zoneIndex++) {
        if (trace.zones.smIndices[zoneIndex] === selectedRow) {
            expectedRowZones++;
        }
    }
    if (countedRowZones !== expectedRowZones) {
        throw new Error(
            `Row-scoped selection counted ${countedRowZones} of `
            + `${expectedRowZones} zones`);
    }

    const selectedY = trace.zones.ys[0];
    const sublaneStatistics = AggregateSelectionStatistics(
        trace.zones,
        zoneVisibility,
        rowSelected,
        rowOffsets,
        rowIsGpu,
        Number.NEGATIVE_INFINITY,
        Number.POSITIVE_INFINITY,
        selectedY,
        selectedY
    );
    const countedSublaneZones = [
        ...sublaneStatistics.cpu,
        ...sublaneStatistics.gpu
    ].reduce((count, statistic) => count + statistic.count, 0);
    let expectedSublaneZones = 0;
    for (let zoneIndex = 0;
        zoneIndex < trace.zones.count; zoneIndex++) {
        if (trace.zones.smIndices[zoneIndex] === selectedRow
            && trace.zones.ys[zoneIndex] === selectedY) {
            expectedSublaneZones++;
        }
    }
    if (countedSublaneZones !== expectedSublaneZones) {
        throw new Error(
            `Sublane-scoped selection counted ${countedSublaneZones} of `
            + `${expectedSublaneZones} zones`);
    }
}

function validateEventParents(trace: ParsedTraceData): Set<bigint> {
    const zonesByEventId = new Map<bigint, number>();
    const expandableEventIds = new Set<bigint>();

    for (let zoneIndex = 0; zoneIndex < trace.zones.count; zoneIndex++) {
        zonesByEventId.set(trace.zones.eventIds[zoneIndex], zoneIndex);
        if (trace.zones.hasChildren[zoneIndex] !== 0) {
            expandableEventIds.add(trace.zones.eventIds[zoneIndex]);
        }
    }

    for (let zoneIndex = 0; zoneIndex < trace.zones.count; zoneIndex++) {
        const parentEventId = trace.zones.parentEventIds[zoneIndex];
        if (parentEventId === 0n) continue;

        const parentZoneIndex = zonesByEventId.get(parentEventId);
        if (parentZoneIndex === undefined) {
            throw new Error(
                `Event ${zoneIndex} references unknown parent ${parentEventId}`);
        }
        if (trace.zones.startsX[zoneIndex]
                < trace.zones.startsX[parentZoneIndex]
            || trace.zones.endsX[zoneIndex]
                > trace.zones.endsX[parentZoneIndex]) {
            throw new Error(
                `Event ${zoneIndex} lies outside parent ${parentEventId}`);
        }
    }

    return expandableEventIds;
}

function gpuTrackIds(trace: ParsedTraceData): Set<bigint> {
    const ids = new Set<bigint>();

    for (const hierarchy of trace.trackHierarchies) {
        const gpu = hierarchy.find(node => node.kind === 2);
        if (gpu) ids.add(gpu.id);
    }
    return ids;
}

function validateOverlays(full: ParsedTraceData, projected: ParsedTraceData): void {
    const annotations = full.trackHierarchies.some(hierarchy => hierarchy.some(node => node.kind === 9));
    if (projected.trackHierarchies.some(hierarchy => hierarchy.some(node => node.kind === 9)))
        throw new Error('An annotation allocated a visible track');
    if (annotations && !projected.overlays?.count)
        throw new Error('Annotations were lost during projection');
    if (full.kernelName === 'CPU semantic metadata') {
        if (projected.overlays?.count !== 1 || projected.overlays.groupRows[0].length !== 2)
            throw new Error('One CUDA-thread envelope must span both rank and worker rows');
        if (!projected.formatDescriptors.some(format => format.labelString === 'SyntheticCompute'))
            throw new Error('Component namespace was not hidden in display labels');
    }
    const ranges = projected.overlays;
    let sourceCount = 0;
    for (let zone = 0; zone < full.zones.count; ++zone)
        if (full.zones.semanticFormatIds[zone] || full.trackHierarchies[full.zones.trackIndices[zone]].some(node => node.kind === 9)) sourceCount++;
    if ((ranges?.count ?? 0) !== sourceCount)
        throw new Error('Semantic ranges were replicated across CUDA streams or dropped');
    const byGroup: number[][] = Array.from({length: ranges?.groupRows.length ?? 0}, () => []);
    for (let index = 0; index < (ranges?.count ?? 0); ++index)
        byGroup[ranges!.groupIndices[index]].push(index);
    for (const group of byGroup) {
        group.sort((a, b) => ranges!.startsX[a] - ranges!.startsX[b]);
        for (let i = 1; i < group.length; ++i)
            if (ranges!.startsX[group[i]] < ranges!.endsX[group[i - 1]])
                throw new Error(`Overlapping emitted semantic ranges: ${ranges!.labels[group[i - 1]]} / ${ranges!.labels[group[i]]}`);
    }
    for (let index = 0; index < (ranges?.count ?? 0); ++index) {
        const group = ranges!.groupRows[ranges!.groupIndices[index]];
        const row = group?.[0];
        if (row === undefined || row >= projected.trackNames.length || ranges!.endsX[index] <= ranges!.startsX[index] || !ranges!.labels[index])
            throw new Error('Invalid semantic overlay');
        if (!projected.trackHierarchies[row].some(node => node.kind === 2 || node.kind === 1))
            throw new Error('Semantic overlay attached to a non-CPU/GPU row');
    }
}

async function validateNanotrace(filename: string): Promise<void> {
    const contents = fs.readFileSync(filename);
    const file = new File([contents], path.basename(filename));
    const fullTrace = await parseTraceFile(file);
    const expandableEventIds = validateEventParents(fullTrace);
    const gpuIds = gpuTrackIds(fullTrace);
    const collapsed = projectTraceData(
        fullTrace, new Set<bigint>());
    const expanded = projectTraceData(
        fullTrace, gpuIds);

    validateOverlays(fullTrace, collapsed);
    validateOverlays(fullTrace, expanded);
    validateTrackOrder(collapsed);
    validateProjection(collapsed);
    validateSelectionStatistics(collapsed);
    validateTrackOrder(expanded);
    validateProjection(expanded);
    validateSelectionStatistics(expanded);

    if (collapsed.tracks.count !== expanded.tracks.count
        || collapsed.zones.count !== expanded.zones.count
        || collapsed.blocks.count !== expanded.blocks.count) {
        throw new Error('GPU expansion changed the materialized topology');
    }

    const hierarchy = buildHierarchy(
        expanded.kernelName,
        [expanded.gridDimX, expanded.gridDimY, expanded.gridDimZ],
        [expanded.clusterDimX, expanded.clusterDimY, expanded.clusterDimZ],
        expanded.formatDescriptors,
        expanded.tracks,
        expanded.zones,
        expanded.blocks,
        expanded.trackNames,
        expanded.trackDepths,
        expanded.bookmarks,
        expanded.overlays
    );
    if (!Number.isFinite(hierarchy.totalDurationNs)
        || hierarchy.totalDurationNs <= 0) {
        throw new Error('Trace has an invalid total duration');
    }

    validateSharedTooltip(hierarchy);
    validateZoneHover(hierarchy);
    const semanticTops = semanticGroupTopRows(hierarchy.lanes.count, hierarchy.overlays);
    for (let row = 0; row < hierarchy.lanes.count; ++row) {
        if (!semanticTops[row]) continue;
        const top = hierarchy.lanes.ys[row] + hierarchy.lanes.heights[row];
        const above = row === 0 ? hierarchy.worldHeight : hierarchy.lanes.ys[row - 1];
        if (above - top < SEMANTIC_GROUP_GAP - 0.0001)
            throw new Error('Semantic label space must be reserved outside the group frame');
    }
    console.log(`Session: ${expanded.kernelName}`);
    console.log(`Rows: ${expanded.trackNames.length}`);
    console.log(`Tracks: ${expanded.tracks.count}`);
    console.log(`Events: ${expanded.zones.count}`);
    console.log(`Overlays: ${expanded.overlays?.count ?? 0}`);
    console.log(`Bookmarks: ${expanded.bookmarks.length}`);
    console.log(`Expandable events: ${expandableEventIds.size}`);
    console.log(`Duration: ${hierarchy.totalDurationNs} ns`);
    console.log('Trace and all materialized projections validated');
}

const filename = process.argv[2];
if (!filename) {
    console.error('Usage: npm run validate -- <trace_file.nanotrace>');
    process.exit(1);
}

try {
    validateOverlayRendering();
    await validateNanotrace(filename);
} catch (error) {
    console.error((error as Error).message);
    process.exit(1);
}
