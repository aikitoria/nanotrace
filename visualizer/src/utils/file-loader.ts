/**
 * Binary trace file parser and visualization hierarchy builder.
 *
 * Key functions:
 * - parseTraceFile(): Parses binary .nanotrace format
 * - buildHierarchy(): Transforms flat trace data into nested rendering hierarchy
 * - formatString(): Substitutes parameters into format descriptor templates
 *
 * Binary format (.nanotrace):
 * - Magic: "nanotrace\0" + version (uint8) + compression (uint8)
 * - [If compressed, remaining data is deflate-compressed]
 * - Kernel name (string)
 * - Format descriptors (templates for names)
 * - Block descriptors (SM assignment + parameters)
 * - Warp tracks (per-warp event sequences)
 * - All integers are little-endian
 * - Times are in nanoseconds, converted to milliseconds for rendering
 */

import {
    FormatDescriptor,
    BlockDescriptor,
    EventData,
    WarpTrack,
    Zone,
    Block,
    BlockLane,
    Lane
} from './types.js';
import {
    SUBLANE_HEIGHT,
    LANE_PADDING,
    SUBLANE_PADDING,
    LANE_EDGE_PADDING,
    BLOCK_LANE_PADDING,
    BLOCK_EDGE_PADDING,
    MAGIC_NUMBER_LENGTH,
    EXPECTED_FORMAT_VERSION,
    COMPRESSION_MODE_DEFLATE,
    NS_TO_MS
} from './constants.js';

/** Raw data extracted from binary trace file. */
export interface ParsedTraceData {
    kernelName: string;
    gridDimX: number;
    gridDimY: number;
    gridDimZ: number;
    clusterDimX: number;
    clusterDimY: number;
    clusterDimZ: number;
    formatDescriptors: FormatDescriptor[];
    blockDescriptors: BlockDescriptor[];
    tracks: WarpTrack[];
}

/** Visualization hierarchy built from parsed trace data. */
export interface HierarchyData {
    zones: Zone[];                   // Flattened array of all zones (for GPU upload)
    blocks: Block[];                 // Flattened array of all blocks
    blockLanes: BlockLane[];         // Flattened array of all block lanes
    lanes: Lane[];                   // Top-level SM lanes
    worldHeight: number;             // Total vertical extent of visualization
    timeRange: number;               // Maximum end time across all events (milliseconds)
}

/**
 * 20-color palette for zone rendering.
 * Colors are mapped by format descriptor ID (modulo 20).
 * RGB values in [0-1] range, optimized for dark theme with good contrast.
 */
const COLOR_PALETTE: [number, number, number][] = [
    [0.30, 0.58, 0.90],
    [0.90, 0.35, 0.35],
    [0.40, 0.85, 0.50],
    [0.95, 0.65, 0.25],
    [0.70, 0.45, 0.90],
    [0.30, 0.78, 0.78],
    [0.95, 0.85, 0.30],
    [0.90, 0.50, 0.70],
    [0.50, 0.75, 0.40],
    [0.85, 0.40, 0.55],
    [0.45, 0.70, 0.90],
    [0.75, 0.55, 0.35],
    [0.55, 0.45, 0.85],
    [0.35, 0.82, 0.65],
    [0.92, 0.60, 0.35],
    [0.60, 0.82, 0.70],
    [0.80, 0.45, 0.80],
    [0.55, 0.88, 0.55],
    [0.88, 0.70, 0.45],
    [0.45, 0.60, 0.82],
];

/**
 * Decompresses deflate-compressed data using browser's native DecompressionStream.
 * Returns an ArrayBuffer containing the decompressed data.
 */
async function decompressData(compressedData: Uint8Array): Promise<ArrayBuffer> {
    const stream = new Blob([compressedData as BlobPart]).stream();
    const decompressedStream = stream.pipeThrough(new DecompressionStream('deflate'));
    const decompressedBlob = await new Response(decompressedStream).blob();
    return await decompressedBlob.arrayBuffer();
}

/**
 * Parses binary .nanotrace file into structured trace data.
 *
 * Format structure:
 * - Header: Magic "nanotrace\0" + version (uint8) + compression (uint8)
 * - Kernel name (length-prefixed string)
 * - Counts: format descriptors, blocks, tracks, total events
 * - Format descriptors: reusable string templates
 * - Block descriptors: SM assignments and parameters
 * - Warp tracks: per-warp event sequences with timing
 *
 * All multi-byte integers are little-endian.
 * Times are stored in nanoseconds.
 * If compression mode is 1, data after compression mode is deflate-compressed (includes kernel name).
 */
export async function parseTraceFile(
    file: File,
    onProgress?: (message: string) => void
): Promise<ParsedTraceData> {
    performance.mark('parseTraceFile:start');

    let buffer = await file.arrayBuffer();
    performance.mark('parseTraceFile:arrayBuffer');

    let view = new DataView(buffer);
    let offset = 0;

    /**
     * Helper: Reads length-prefixed UTF-8 string from binary buffer.
     * Format: uint16 length (little-endian) + UTF-8 bytes
     * Returns decoded string and updated offset for sequential parsing.
     */
    const readString = (view: DataView, offset: number): { str: string; newOffset: number } => {
        const length = view.getUint16(offset, true);
        offset += 2;
        const bytes = new Uint8Array(view.buffer, offset, length);
        const str = new TextDecoder().decode(bytes);
        return { str, newOffset: offset + length };
    };

    const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 9));
    if (magic !== "nanotrace") {
        throw new Error(`Invalid magic number: ${magic}`);
    }
    offset += MAGIC_NUMBER_LENGTH;

    const formatVersion = view.getUint8(offset);
    offset += 1;
    if (formatVersion !== EXPECTED_FORMAT_VERSION) {
        throw new Error(`Unsupported format version: ${formatVersion}`);
    }

    const compressionMode = view.getUint8(offset);
    offset += 1;
    if (compressionMode !== 0 && compressionMode !== 1) {
        throw new Error(`Unsupported compression mode: ${compressionMode}`);
    }

    // Decompress the remaining data if compression is enabled
    // Compression includes kernel name and everything after
    if (compressionMode === COMPRESSION_MODE_DEFLATE) {
        performance.mark('parseTraceFile:decompress:start');
        if (onProgress) onProgress('Decompressing trace data...');
        console.log(`Decompressing trace data...`);
        const compressedData = new Uint8Array(buffer, offset);
        buffer = await decompressData(compressedData);
        view = new DataView(buffer);
        offset = 0;
        performance.mark('parseTraceFile:decompress:end');
        performance.measure('Decompress', 'parseTraceFile:decompress:start', 'parseTraceFile:decompress:end');
        if (onProgress) onProgress('Parsing trace data...');
    }

    const { str: kernelName, newOffset: o1 } = readString(view, offset);
    offset = o1;

    // Read grid dimensions
    const gridDimX = view.getUint32(offset, true); offset += 4;
    const gridDimY = view.getUint32(offset, true); offset += 4;
    const gridDimZ = view.getUint32(offset, true); offset += 4;

    // Read cluster dimensions
    const clusterDimX = view.getUint32(offset, true); offset += 4;
    const clusterDimY = view.getUint32(offset, true); offset += 4;
    const clusterDimZ = view.getUint32(offset, true); offset += 4;

    const formatDescCount = view.getUint32(offset, true); offset += 4;
    const blockDescCount = view.getUint32(offset, true); offset += 4;
    const trackCount = view.getUint32(offset, true); offset += 4;
    const totalEventCount = Number(view.getBigUint64(offset, true)); offset += 8;

    console.log(`Parsing ${file.name}:`);
    console.log(`  Kernel: ${kernelName}`);
    console.log(`  Grid dimensions: (${gridDimX}, ${gridDimY}, ${gridDimZ})`);
    console.log(`  Cluster dimensions: (${clusterDimX}, ${clusterDimY}, ${clusterDimZ})`);
    console.log(`  Compression: ${compressionMode === COMPRESSION_MODE_DEFLATE ? 'deflate' : 'none'}`);
    console.log(`  Format descriptors: ${formatDescCount}`);
    console.log(`  Blocks: ${blockDescCount}`);
    console.log(`  Tracks: ${trackCount}`);
    console.log(`  Total events: ${totalEventCount}`);

    performance.mark('parseTraceFile:formatDesc:start');
    const formatDescriptors: FormatDescriptor[] = [];
    for (let i = 0; i < formatDescCount; i++) {
        const { str: formatString, newOffset: o2 } = readString(view, offset);
        offset = o2;
        const placeholderCount = view.getUint8(offset); offset += 1;
        formatDescriptors.push({ formatString, placeholderCount });
    }
    performance.mark('parseTraceFile:formatDesc:end');
    performance.measure('Parse Format Descriptors', 'parseTraceFile:formatDesc:start', 'parseTraceFile:formatDesc:end');

    performance.mark('parseTraceFile:blockDesc:start');
    const blockDescriptors: BlockDescriptor[] = [];
    for (let i = 0; i < blockDescCount; i++) {
        const smId = view.getUint16(offset, true); offset += 2;
        const formatDescId = view.getUint16(offset, true); offset += 2;
        const paramCount = formatDescriptors[formatDescId].placeholderCount;
        const params: number[] = [];
        for (let j = 0; j < paramCount; j++) {
            params.push(view.getUint32(offset, true)); offset += 4;
        }
        blockDescriptors.push({ smId, formatDescId, params });
    }
    performance.mark('parseTraceFile:blockDesc:end');
    performance.measure('Parse Block Descriptors', 'parseTraceFile:blockDesc:start', 'parseTraceFile:blockDesc:end');

    performance.mark('parseTraceFile:tracks:start');
    const tracks: WarpTrack[] = [];
    for (let i = 0; i < trackCount; i++) {
        const blockDescId = view.getUint32(offset, true); offset += 4;
        const formatDescId = view.getUint16(offset, true); offset += 2;
        const warpParamCount = formatDescriptors[formatDescId].placeholderCount;
        const params: number[] = [];
        for (let j = 0; j < warpParamCount; j++) {
            params.push(view.getUint32(offset, true)); offset += 4;
        }
        const eventCount = view.getUint32(offset, true); offset += 4;
        const events: EventData[] = [];
        for (let j = 0; j < eventCount; j++) {
            const timeOffset = view.getUint32(offset, true); offset += 4;
            const duration = view.getUint32(offset, true); offset += 4;
            const eventFormatDescId = view.getUint16(offset, true); offset += 2;
            const eventParamCount = formatDescriptors[eventFormatDescId].placeholderCount;
            const eventParams: number[] = [];
            for (let k = 0; k < eventParamCount; k++) {
                eventParams.push(view.getUint32(offset, true)); offset += 4;
            }
            events.push({ timeOffset, duration, formatDescId: eventFormatDescId, params: eventParams });
        }
        tracks.push({ blockDescId, formatDescId, params, events });
    }
    performance.mark('parseTraceFile:tracks:end');
    performance.measure('Parse Event Tracks', 'parseTraceFile:tracks:start', 'parseTraceFile:tracks:end');

    performance.mark('parseTraceFile:end');
    performance.measure('Parse Trace File (Total)', 'parseTraceFile:start', 'parseTraceFile:end');

    return {
        kernelName,
        gridDimX,
        gridDimY,
        gridDimZ,
        clusterDimX,
        clusterDimY,
        clusterDimZ,
        formatDescriptors,
        blockDescriptors,
        tracks
    };
}

/**
 * Builds visualization hierarchy from parsed trace data.
 *
 * Process:
 * 1. Assign colors to format descriptors (modulo 20-color palette)
 * 2. Group blocks by SM, create zones from warp track events
 * 3. Assign blocks to block lanes (non-overlapping horizontal grouping)
 * 4. Calculate Y positions bottom-up (world origin at bottom)
 * 5. Convert times from nanoseconds to milliseconds
 *
 * Hierarchy: Lane (SM) → BlockLane → Block → Sublane → Zone
 * All zones are flattened into a single array for GPU upload.
 */
export function buildHierarchy(
    formatDescriptors: FormatDescriptor[],
    blockDescriptors: BlockDescriptor[],
    tracks: WarpTrack[]
): HierarchyData {
    performance.mark('buildHierarchy:start');

    // Map format descriptor IDs to colors from the 20-color palette
    const formatToColor = new Map<number, [number, number, number]>();
    formatDescriptors.forEach((_desc, idx) => {
        const color = COLOR_PALETTE[idx % COLOR_PALETTE.length];
        formatToColor.set(idx, color);
    });

    const maxSmId = blockDescriptors.reduce((max, desc) => Math.max(max, desc.smId), -1);
    const numLanes = maxSmId + 1;

    const lanes: Lane[] = [];
    for (let i = 0; i < numLanes; i++) {
        lanes.push({ index: i, blockLanes: [], height: 0, width: 0, y: 0 });
    }

    const blocksBySM = new Map<number, Array<{ descIndex: number; desc: BlockDescriptor }>>();
    blockDescriptors.forEach((desc, idx) => {
        if (!blocksBySM.has(desc.smId)) {
            blocksBySM.set(desc.smId, []);
        }
        blocksBySM.get(desc.smId)!.push({ descIndex: idx, desc });
    });

    let globalBlockIdx = 0;
    let globalBlockLaneIdx = 0;
    let globalZoneIdx = 0;

    const blocks: Block[] = [];
    const blockLanes: BlockLane[] = [];
    const zones: Zone[] = [];

    const tracksByBlock = new Map<number, WarpTrack[]>();
    tracks.forEach(track => {
        if (!tracksByBlock.has(track.blockDescId)) {
            tracksByBlock.set(track.blockDescId, []);
        }
        tracksByBlock.get(track.blockDescId)!.push(track);
    });

    blocksBySM.forEach((smBlocks, smId) => {
        const lane = lanes[smId];

        const blockObjects: Block[] = [];
        smBlocks.forEach(({ descIndex, desc }) => {
            const blockTracksForBlock = tracksByBlock.get(descIndex) || [];

            let minTime = Infinity;
            let maxTime = 0;
            blockTracksForBlock.forEach(track => {
                track.events.forEach(event => {
                    minTime = Math.min(minTime, event.timeOffset);
                    maxTime = Math.max(maxTime, event.timeOffset + event.duration);
                });
            });

            const startX = minTime * NS_TO_MS;
            const endX = maxTime * NS_TO_MS;

            const sublanes: Zone[][] = blockTracksForBlock.map(track => {
                return track.events.map(event => {
                    const zoneStartX = event.timeOffset * NS_TO_MS;
                    const zoneEndX = (event.timeOffset + event.duration) * NS_TO_MS;
                    const baseColor = formatToColor.get(event.formatDescId)!;

                    const zone: Zone = {
                        id: globalZoneIdx++,
                        startX: zoneStartX,
                        endX: zoneEndX,
                        width: zoneEndX - zoneStartX,
                        height: SUBLANE_HEIGHT,
                        r: baseColor[0],
                        g: baseColor[1],
                        b: baseColor[2],
                        formatDescId: event.formatDescId,
                        params: event.params,
                        warpFormatDescId: track.formatDescId,
                        warpParams: track.params,
                        laneIdx: smId,
                        blockLaneIdx: -1,
                        blockIdx: -1,
                        sublaneIdx: -1,
                        x: 0,
                        y: 0,
                    };
                    zones.push(zone);
                    return zone;
                });
            });

            const maxZoneWidth = Math.max(...sublanes.flat().map(z => z.width), 0);

            const block: Block = {
                id: globalBlockIdx++,
                startX,
                endX,
                width: endX - startX,
                height: BLOCK_EDGE_PADDING + sublanes.length * SUBLANE_HEIGHT + (sublanes.length - 1) * SUBLANE_PADDING,
                sublanes,
                numSublanes: sublanes.length,
                maxZoneWidth,
                formatDescId: desc.formatDescId,
                params: desc.params,
                laneIdx: smId,
                blockLaneIdx: -1,
                x: 0,
                y: 0,
            };
            blocks.push(block);
            blockObjects.push(block);
        });

        const blockLanesForSM: BlockLane[] = [];
        blockObjects.forEach(block => {
            let placed = false;
            for (const blockLane of blockLanesForSM) {
                const overlaps = blockLane.blocks.some(b =>
                    !(block.endX <= b.startX || block.startX >= b.endX)
                );
                if (!overlaps) {
                    blockLane.blocks.push(block);
                    placed = true;
                    break;
                }
            }
            if (!placed) {
                blockLanesForSM.push({
                    id: globalBlockLaneIdx++,
                    blocks: [block],
                    height: 0,
                    width: 0,
                    maxBlockWidth: 0,
                    maxZoneWidth: 0,
                    laneIdx: smId,
                    y: 0,
                });
            }
        });

        blockLanesForSM.forEach((blockLane, _blIdx) => {
            blockLane.blocks.forEach(block => {
                block.blockLaneIdx = blockLane.id;
                block.sublanes.forEach((sublane, slIdx) => {
                    sublane.forEach(zone => {
                        zone.blockLaneIdx = blockLane.id;
                        zone.blockIdx = block.id;
                        zone.sublaneIdx = slIdx;
                    });
                });
            });

            const maxBlockHeight = Math.max(...blockLane.blocks.map(b => b.height), 0);
            blockLane.height = maxBlockHeight;
            blockLane.width = Math.max(...blockLane.blocks.map(b => b.endX), 0);
            blockLane.maxBlockWidth = Math.max(...blockLane.blocks.map(b => b.width), 0);
            blockLane.maxZoneWidth = Math.max(...blockLane.blocks.map(b => b.maxZoneWidth), 0);

            blockLanes.push(blockLane);
            lane.blockLanes.push(blockLane);
        });

        let laneHeight = 2 * LANE_EDGE_PADDING;
        for (let i = 0; i < lane.blockLanes.length; i++) {
            laneHeight += lane.blockLanes[i].height;
            if (i < lane.blockLanes.length - 1) {
                laneHeight += BLOCK_LANE_PADDING;
            }
        }
        lane.height = laneHeight;
        lane.width = Math.max(...lane.blockLanes.map(bl => bl.width), 0);
    });

    const maxLaneWidth = Math.max(...lanes.map(l => l.width), 0);
    const timeRange = maxLaneWidth > 0 ? maxLaneWidth : 1.0;

    let currentY = 0;
    for (let i = lanes.length - 1; i >= 0; i--) {
        const lane = lanes[i];

        if (lane.blockLanes.length === 0) {
            lane.y = 0;
            lane.height = 0;
            continue;
        }

        lane.y = currentY;
        let blockLaneY = currentY + LANE_EDGE_PADDING;
        lane.blockLanes.forEach((blockLane, blIdx) => {
            blockLane.y = blockLaneY;

            blockLane.blocks.forEach(block => {
                block.y = blockLaneY + (blockLane.height - block.height);
                block.x = block.startX + block.width / 2;

                block.sublanes.forEach((sublane, slIdx) => {
                    const sublaneY = block.y + block.height - BLOCK_EDGE_PADDING - slIdx * (SUBLANE_HEIGHT + SUBLANE_PADDING) - SUBLANE_HEIGHT / 2;
                    sublane.forEach(zone => {
                        zone.y = sublaneY;
                        zone.x = zone.startX + zone.width / 2;
                    });
                });
            });

            blockLaneY += blockLane.height;
            if (blIdx < lane.blockLanes.length - 1) {
                blockLaneY += BLOCK_LANE_PADDING;
            }
        });

        currentY = blockLaneY + LANE_EDGE_PADDING + LANE_PADDING;
    }

    const worldHeight = currentY;

    performance.mark('buildHierarchy:end');
    performance.measure('Build Hierarchy', 'buildHierarchy:start', 'buildHierarchy:end');

    return { zones, blocks, blockLanes, lanes, worldHeight, timeRange };
}

/**
 * Substitutes parameters into format descriptor template string.
 * Replaces placeholders {0}, {1}, etc. with corresponding parameter values.
 * Example: "Warp {0}" with params [5] → "Warp 5"
 */
export function formatString(formatDescriptors: FormatDescriptor[], formatDescId: number, params: number[]): string {
    const desc = formatDescriptors[formatDescId];
    let result = desc.formatString;
    for (let i = 0; i < params.length; i++) {
        result = result.replace(`{${i}}`, params[i].toString());
    }
    return result;
}
