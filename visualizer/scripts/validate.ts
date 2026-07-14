#!/usr/bin/env node

import * as fs from 'node:fs';
import * as path from 'node:path';

import {
    buildHierarchy,
    parseTraceFile,
    projectTraceData,
    type ParsedTraceData
} from '../src/utils/file-loader.js';
import { binarySearchZones } from '../src/utils/soa-helpers.js';

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

async function validateNanotrace(filename: string): Promise<void> {
    const contents = fs.readFileSync(filename);
    const file = new File([contents], path.basename(filename));
    const fullTrace = await parseTraceFile(file);
    const expandableEventIds = validateEventParents(fullTrace);
    const gpuIds = gpuTrackIds(fullTrace);
    const collapsed = projectTraceData(
        fullTrace, expandableEventIds, new Set<bigint>());
    const expanded = projectTraceData(
        fullTrace, expandableEventIds, gpuIds);

    validateTrackOrder(collapsed);
    validateProjection(collapsed);
    validateTrackOrder(expanded);
    validateProjection(expanded);

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
        expanded.trackDepths
    );
    if (!Number.isFinite(hierarchy.totalDurationNs)
        || hierarchy.totalDurationNs <= 0) {
        throw new Error('Trace has an invalid total duration');
    }

    console.log(`Session: ${expanded.kernelName}`);
    console.log(`Rows: ${expanded.trackNames.length}`);
    console.log(`Tracks: ${expanded.tracks.count}`);
    console.log(`Events: ${expanded.zones.count}`);
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
    await validateNanotrace(filename);
} catch (error) {
    console.error((error as Error).message);
    process.exit(1);
}
