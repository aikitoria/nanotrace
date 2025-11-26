/**
 * Binary trace file parser and visualization hierarchy builder (SoA version).
 *
 * Key functions:
 * - parseTraceFile(): Parses binary .nanotrace format (unchanged)
 * - buildHierarchy(): Transforms flat trace data into SoA hierarchy (complete rewrite)
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
 * - Times are in nanoseconds (NO conversion to milliseconds!)
 */

import {
    FormatDescriptor,
    HierarchyData,
    TracksSoA,
    ZonesSoA,
    BlocksSoA,
    BlockLanesSoA,
    LanesSoA,
    SMAccelerator
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
    COMPRESSION_MODE_DEFLATE
} from './constants.js';

/** Raw data extracted from binary trace file (SoA version). */
export interface ParsedTraceData {
    kernelName: string;
    gridDimX: number;
    gridDimY: number;
    gridDimZ: number;
    clusterDimX: number;
    clusterDimY: number;
    clusterDimZ: number;
    formatDescriptors: FormatDescriptor[];

    // SoA structures (pre-populated)
    tracks: TracksSoA;
    zones: ZonesSoA;
    blocks: BlocksSoA;
}

/**
 * 20-color palette for zone rendering.
 * Colors are mapped by format descriptor ID (modulo 20).
 * RGB values in [0-255] range for packed byte storage.
 */
const COLOR_PALETTE: [number, number, number][] = [
    [77, 148, 230],   // [0.30, 0.58, 0.90] * 255
    [230, 89, 89],    // [0.90, 0.35, 0.35] * 255
    [102, 217, 128],  // [0.40, 0.85, 0.50] * 255
    [242, 166, 64],   // [0.95, 0.65, 0.25] * 255
    [179, 115, 230],  // [0.70, 0.45, 0.90] * 255
    [77, 199, 199],   // [0.30, 0.78, 0.78] * 255
    [242, 217, 77],   // [0.95, 0.85, 0.30] * 255
    [230, 128, 179],  // [0.90, 0.50, 0.70] * 255
    [128, 191, 102],  // [0.50, 0.75, 0.40] * 255
    [217, 102, 140],  // [0.85, 0.40, 0.55] * 255
    [115, 179, 230],  // [0.45, 0.70, 0.90] * 255
    [191, 140, 89],   // [0.75, 0.55, 0.35] * 255
    [140, 115, 217],  // [0.55, 0.45, 0.85] * 255
    [89, 209, 166],   // [0.35, 0.82, 0.65] * 255
    [235, 153, 89],   // [0.92, 0.60, 0.35] * 255
    [153, 209, 179],  // [0.60, 0.82, 0.70] * 255
    [204, 115, 204],  // [0.80, 0.45, 0.80] * 255
    [140, 224, 140],  // [0.55, 0.88, 0.55] * 255
    [224, 179, 115],  // [0.88, 0.70, 0.45] * 255
    [115, 153, 209],  // [0.45, 0.60, 0.82] * 255
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
 * Parses binary .nanotrace file directly into SoA structures.
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
 * Parses directly into TypedArrays - NO intermediate JavaScript objects!
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

    // Parse format descriptors (small count, OK to use objects)
    performance.mark('parseTraceFile:formatDesc:start');
    const formatDescriptors: FormatDescriptor[] = [];
    for (let i = 0; i < formatDescCount; i++) {
        const { str: labelString, newOffset: o2 } = readString(view, offset);
        offset = o2;
        const { str: tooltipString, newOffset: o3 } = readString(view, offset);
        offset = o3;
        const placeholderCount = view.getUint8(offset); offset += 1;
        formatDescriptors.push({ labelString, tooltipString, placeholderCount });
    }
    performance.mark('parseTraceFile:formatDesc:end');
    performance.measure('Parse Format Descriptors', 'parseTraceFile:formatDesc:start', 'parseTraceFile:formatDesc:end');

    // Map format descriptor IDs to colors
    const formatToColor = new Map<number, [number, number, number]>();
    formatDescriptors.forEach((_desc, idx) => {
        const color = COLOR_PALETTE[idx % COLOR_PALETTE.length];
        formatToColor.set(idx, color);
    });

    // PRE-SCAN: Count params to allocate exact sizes
    performance.mark('parseTraceFile:prescan:start');
    const prescanOffset = offset;  // Save current position
    let zoneCount = 0;
    let trackParamsCount = 0;
    let zoneParamsCount = 0;

    // Skip blocks (no params)
    offset += blockDescCount * 12;  // 4+4+2+2 bytes per block

    // Count track and zone params
    for (let i = 0; i < trackCount; i++) {
        offset += 4;  // blockDescId
        const formatDescId = view.getUint16(offset, true); offset += 2;
        offset += 4;  // laneId
        const trackParamCount = formatDescriptors[formatDescId].placeholderCount;
        trackParamsCount += trackParamCount;
        offset += trackParamCount * 4;

        const eventCount = view.getUint32(offset, true); offset += 4;
        zoneCount += eventCount;

        for (let j = 0; j < eventCount; j++) {
            offset += 8;  // timeOffset + duration
            const eventFormatDescId = view.getUint16(offset, true); offset += 2;
            const eventParamCount = formatDescriptors[eventFormatDescId].placeholderCount;
            zoneParamsCount += eventParamCount;
            offset += eventParamCount * 4;
        }
    }

    performance.mark('parseTraceFile:prescan:end');
    performance.measure('Pre-scan', 'parseTraceFile:prescan:start', 'parseTraceFile:prescan:end');

    console.log(`  Zones (events): ${zoneCount}`);
    console.log(`  Track params: ${trackParamsCount}`);
    console.log(`  Zone params: ${zoneParamsCount}`);

    // Allocate all SoA structures
    performance.mark('parseTraceFile:allocate:start');

    const tracksSoA = new TracksSoA();
    tracksSoA.count = trackCount;
    tracksSoA.formatDescIds = new Uint16Array(trackCount);
    tracksSoA.sublaneIndices = new Uint8Array(trackCount);
    tracksSoA.paramsOffsets = new Uint32Array(trackCount);
    tracksSoA.paramsCounts = new Uint8Array(trackCount);
    tracksSoA.blockIndices = new Uint32Array(trackCount);
    tracksSoA.paramsPool = new Uint32Array(trackParamsCount);

    const zonesSoA = new ZonesSoA();
    zonesSoA.count = zoneCount;
    zonesSoA.startsX = new Uint32Array(zoneCount);
    zonesSoA.endsX = new Uint32Array(zoneCount);
    zonesSoA.ys = new Float32Array(zoneCount);  // Filled later by buildHierarchy
    zonesSoA.colors = new Uint8Array(zoneCount * 3);
    zonesSoA.formatDescIds = new Uint16Array(zoneCount);
    zonesSoA.paramsOffsets = new Uint32Array(zoneCount);
    zonesSoA.paramsCounts = new Uint8Array(zoneCount);
    zonesSoA.trackIndices = new Uint32Array(zoneCount);
    zonesSoA.smIndices = new Uint8Array(zoneCount);
    zonesSoA.blockIndices = new Uint32Array(zoneCount);
    zonesSoA.sublaneIndices = new Uint8Array(zoneCount);
    zonesSoA.paramsPool = new Uint32Array(zoneParamsCount);

    const blocksSoA = new BlocksSoA();
    blocksSoA.count = blockDescCount;
    blocksSoA.startsX = new Uint32Array(blockDescCount);
    blocksSoA.endsX = new Uint32Array(blockDescCount);
    blocksSoA.ys = new Float32Array(blockDescCount);  // Filled later by buildHierarchy
    blocksSoA.heights = new Float32Array(blockDescCount);
    blocksSoA.sublanesCounts = new Uint8Array(blockDescCount);
    blocksSoA.sublanesMaxWidths = new Uint32Array(blockDescCount);
    blocksSoA.formatDescIds = new Uint16Array(blockDescCount);
    blocksSoA.gridIds = new Uint32Array(blockDescCount);
    blocksSoA.clusterIds = new Uint32Array(blockDescCount);
    blocksSoA.smIndices = new Uint8Array(blockDescCount);
    blocksSoA.blockLaneIndices = new Uint16Array(blockDescCount);  // Filled later by buildHierarchy
    blocksSoA.zonesStartIndices = new Uint32Array(blockDescCount).fill(0xFFFFFFFF);
    blocksSoA.zonesEndIndices = new Uint32Array(blockDescCount);
    blocksSoA.tracksStartIndices = new Uint32Array(blockDescCount).fill(0xFFFFFFFF);
    blocksSoA.tracksEndIndices = new Uint32Array(blockDescCount);

    performance.mark('parseTraceFile:allocate:end');
    performance.measure('Allocate SoA', 'parseTraceFile:allocate:start', 'parseTraceFile:allocate:end');

    // Reset offset to start of block descriptors
    offset = prescanOffset;

    // Parse blocks directly into SoA
    performance.mark('parseTraceFile:blocks:start');
    for (let i = 0; i < blockDescCount; i++) {
        blocksSoA.gridIds[i] = view.getUint32(offset, true); offset += 4;
        blocksSoA.clusterIds[i] = view.getUint32(offset, true); offset += 4;
        blocksSoA.smIndices[i] = view.getUint8(offset); offset += 2;  // uint16, only use lower byte
        blocksSoA.formatDescIds[i] = view.getUint16(offset, true); offset += 2;

        // Initialize bounds (will be computed from zones)
        blocksSoA.startsX[i] = 0xFFFFFFFF;  // Infinity
        blocksSoA.endsX[i] = 0;
        blocksSoA.sublanesMaxWidths[i] = 0;
    }
    performance.mark('parseTraceFile:blocks:end');
    performance.measure('Parse Blocks', 'parseTraceFile:blocks:start', 'parseTraceFile:blocks:end');

    // Parse tracks and zones directly into SoA
    performance.mark('parseTraceFile:tracks:start');
    let trackIdx = 0;
    let zoneIdx = 0;
    let trackParamsPoolIdx = 0;
    let zoneParamsPoolIdx = 0;

    for (let i = 0; i < trackCount; i++) {
        const blockDescId = view.getUint32(offset, true); offset += 4;
        const formatDescId = view.getUint16(offset, true); offset += 2;
        const sublaneIdx = view.getUint32(offset, true); offset += 4;

        // Store track data
        tracksSoA.formatDescIds[trackIdx] = formatDescId;
        tracksSoA.sublaneIndices[trackIdx] = sublaneIdx;
        tracksSoA.blockIndices[trackIdx] = blockDescId;
        tracksSoA.paramsOffsets[trackIdx] = trackParamsPoolIdx;

        const trackParamCount = formatDescriptors[formatDescId].placeholderCount;
        tracksSoA.paramsCounts[trackIdx] = trackParamCount;
        for (let j = 0; j < trackParamCount; j++) {
            tracksSoA.paramsPool[trackParamsPoolIdx++] = view.getUint32(offset, true);
            offset += 4;
        }

        // Update block's track range (only set start on first track for this block)
        if (blocksSoA.tracksStartIndices[blockDescId] === 0xFFFFFFFF) {
            blocksSoA.tracksStartIndices[blockDescId] = trackIdx;
        }

        const zoneStartForTrack = zoneIdx;
        const eventCount = view.getUint32(offset, true); offset += 4;

        for (let j = 0; j < eventCount; j++) {
            const timeOffset = view.getUint32(offset, true); offset += 4;
            const duration = view.getUint32(offset, true); offset += 4;
            const eventFormatDescId = view.getUint16(offset, true); offset += 2;

            const color = formatToColor.get(eventFormatDescId)!;

            // Store zone data
            zonesSoA.startsX[zoneIdx] = timeOffset;
            zonesSoA.endsX[zoneIdx] = timeOffset + duration;
            zonesSoA.colors[zoneIdx * 3 + 0] = color[0];
            zonesSoA.colors[zoneIdx * 3 + 1] = color[1];
            zonesSoA.colors[zoneIdx * 3 + 2] = color[2];
            zonesSoA.formatDescIds[zoneIdx] = eventFormatDescId;
            zonesSoA.paramsOffsets[zoneIdx] = zoneParamsPoolIdx;
            zonesSoA.trackIndices[zoneIdx] = trackIdx;
            zonesSoA.smIndices[zoneIdx] = blocksSoA.smIndices[blockDescId];
            zonesSoA.blockIndices[zoneIdx] = blockDescId;
            zonesSoA.sublaneIndices[zoneIdx] = sublaneIdx;

            const eventParamCount = formatDescriptors[eventFormatDescId].placeholderCount;
            zonesSoA.paramsCounts[zoneIdx] = eventParamCount;
            for (let k = 0; k < eventParamCount; k++) {
                zonesSoA.paramsPool[zoneParamsPoolIdx++] = view.getUint32(offset, true);
                offset += 4;
            }

            // Update block bounds
            if (timeOffset < blocksSoA.startsX[blockDescId]) {
                blocksSoA.startsX[blockDescId] = timeOffset;
            }
            if (timeOffset + duration > blocksSoA.endsX[blockDescId]) {
                blocksSoA.endsX[blockDescId] = timeOffset + duration;
            }
            if (duration > blocksSoA.sublanesMaxWidths[blockDescId]) {
                blocksSoA.sublanesMaxWidths[blockDescId] = duration;
            }

            zoneIdx++;
        }

        // Update block's zone range (only set start on first track with events for this block)
        if (zoneIdx > zoneStartForTrack) {
            if (blocksSoA.zonesStartIndices[blockDescId] === 0xFFFFFFFF) {
                blocksSoA.zonesStartIndices[blockDescId] = zoneStartForTrack;
            }
            blocksSoA.zonesEndIndices[blockDescId] = zoneIdx;
        }

        blocksSoA.tracksEndIndices[blockDescId] = trackIdx + 1;
        blocksSoA.sublanesCounts[blockDescId]++;

        trackIdx++;
    }

    // Calculate block heights
    for (let i = 0; i < blockDescCount; i++) {
        const numSublanes = blocksSoA.sublanesCounts[i];
        blocksSoA.heights[i] = BLOCK_EDGE_PADDING +
                              numSublanes * SUBLANE_HEIGHT +
                              Math.max(0, numSublanes - 1) * SUBLANE_PADDING;

        // Handle blocks with no zones
        if (blocksSoA.startsX[i] === 0xFFFFFFFF) {
            blocksSoA.startsX[i] = 0;
        }
    }

    performance.mark('parseTraceFile:tracks:end');
    performance.measure('Parse Tracks & Zones', 'parseTraceFile:tracks:start', 'parseTraceFile:tracks:end');

    performance.mark('parseTraceFile:end');
    performance.measure('Parse Trace File (Total)', 'parseTraceFile:start', 'parseTraceFile:end');

    console.log(`Parsed into SoA:`);
    console.log(`  Tracks: ${tracksSoA.count}`);
    console.log(`  Zones: ${zonesSoA.count}`);
    console.log(`  Blocks: ${blocksSoA.count}`);

    return {
        kernelName,
        gridDimX,
        gridDimY,
        gridDimZ,
        clusterDimX,
        clusterDimY,
        clusterDimZ,
        formatDescriptors,
        tracks: tracksSoA,
        zones: zonesSoA,
        blocks: blocksSoA
    };
}


/**
 * Groups blocks into non-overlapping lanes for a single SM.
 * Returns array of block index arrays (each inner array is one block lane).
 */
function groupIntoNonOverlappingLanes(
    blockIndices: number[],
    blocks: BlocksSoA
): number[][] {
    // Sort blocks by startX (COPY indices, don't modify original!)
    const sorted = blockIndices.slice().sort((a, b) =>
        blocks.startsX[a] - blocks.startsX[b]
    );

    const lanes: Array<{blockIndices: number[], endTime: number}> = [];

    for (const blockIdx of sorted) {
        const startX = blocks.startsX[blockIdx];
        const endX = blocks.endsX[blockIdx];

        // Find lane where this block doesn't overlap
        let assignedLane = lanes.find(lane => startX >= lane.endTime);

        if (!assignedLane) {
            assignedLane = {blockIndices: [], endTime: 0};
            lanes.push(assignedLane);
        }

        assignedLane.blockIndices.push(blockIdx);
        assignedLane.endTime = endX;
    }

    return lanes.map(lane => lane.blockIndices);
}

/**
 * Builds hierarchical acceleration structures from already-populated SoA data.
 *
 * Process:
 * 1. Group blocks by SM
 * 2. Build block lanes with indirection arrays (non-overlapping blocks)
 * 3. Build lanes
 * 4. Calculate Y positions bottom-up
 *
 * Input SoA structures are already populated by parseTraceFile().
 * This function only builds the hierarchy on top of them.
 */
export function buildHierarchy(
    kernelName: string,
    gridDims: [number, number, number],
    clusterDims: [number, number, number],
    formatDescriptors: FormatDescriptor[],
    tracks: TracksSoA,
    zones: ZonesSoA,
    blocks: BlocksSoA
): HierarchyData {
    performance.mark('buildHierarchy:start');

    console.log(`Building hierarchy from SoA:`);
    console.log(`  Blocks: ${blocks.count}`);
    console.log(`  Tracks: ${tracks.count}`);
    console.log(`  Zones: ${zones.count}`);

    // Group blocks by SM
    performance.mark('buildHierarchy:groupSM:start');
    const blocksBySM = new Map<number, number[]>();
    for (let i = 0; i < blocks.count; i++) {
        const smId = blocks.smIndices[i];
        if (!blocksBySM.has(smId)) {
            blocksBySM.set(smId, []);
        }
        blocksBySM.get(smId)!.push(i);
    }
    performance.mark('buildHierarchy:groupSM:end');
    performance.measure('Group Blocks by SM', 'buildHierarchy:groupSM:start', 'buildHierarchy:groupSM:end');

    // Build block lanes with indirection
    performance.mark('buildHierarchy:blockLanes:start');

    const uniqueSMs = Array.from(blocksBySM.keys()).sort((a, b) => a - b);
    const numLanes = uniqueSMs.length;

    // Temporary structure to collect block lane groups
    const blockLaneGroups: Array<{smId: number, blockIndices: number[]}> = [];

    uniqueSMs.forEach(smId => {
        const blockIndicesForSM = blocksBySM.get(smId)!;
        const groups = groupIntoNonOverlappingLanes(blockIndicesForSM, blocks);

        groups.forEach(blockIndices => {
            blockLaneGroups.push({smId, blockIndices});
        });
    });

    // Allocate BlockLanesSoA
    const blockLanesSoA = new BlockLanesSoA();
    blockLanesSoA.count = blockLaneGroups.length;
    blockLanesSoA.ys = new Float32Array(blockLaneGroups.length);
    blockLanesSoA.heights = new Float32Array(blockLaneGroups.length);
    blockLanesSoA.widths = new Uint32Array(blockLaneGroups.length);
    blockLanesSoA.maxBlockWidths = new Uint32Array(blockLaneGroups.length);
    blockLanesSoA.maxZoneWidths = new Uint32Array(blockLaneGroups.length);
    blockLanesSoA.smIndices = new Uint8Array(blockLaneGroups.length);

    // Count total block references for indirection array
    let totalBlockRefs = 0;
    blockLaneGroups.forEach(group => {
        totalBlockRefs += group.blockIndices.length;
    });

    blockLanesSoA.blockIndices = new Uint32Array(totalBlockRefs);
    blockLanesSoA.blockIndicesOffsets = new Uint32Array(blockLaneGroups.length);
    blockLanesSoA.blockIndicesCounts = new Uint16Array(blockLaneGroups.length);

    // Populate block lanes and indirection
    let blockIndicesOffset = 0;
    blockLaneGroups.forEach((group, blIdx) => {
        blockLanesSoA.smIndices[blIdx] = group.smId;
        blockLanesSoA.blockIndicesOffsets[blIdx] = blockIndicesOffset;
        blockLanesSoA.blockIndicesCounts[blIdx] = group.blockIndices.length;

        let maxBlockHeight = 0;
        let maxBlockWidth = 0;
        let maxZoneWidth = 0;
        let rightmostTime = 0;

        for (const blockIdx of group.blockIndices) {
            blockLanesSoA.blockIndices[blockIndicesOffset++] = blockIdx;

            // Update block's block lane index
            blocks.blockLaneIndices[blockIdx] = blIdx;

            // Calculate stats
            maxBlockHeight = Math.max(maxBlockHeight, blocks.heights[blockIdx]);
            maxBlockWidth = Math.max(maxBlockWidth, blocks.endsX[blockIdx] - blocks.startsX[blockIdx]);
            maxZoneWidth = Math.max(maxZoneWidth, blocks.sublanesMaxWidths[blockIdx]);
            rightmostTime = Math.max(rightmostTime, blocks.endsX[blockIdx]);
        }

        blockLanesSoA.heights[blIdx] = maxBlockHeight;
        blockLanesSoA.maxBlockWidths[blIdx] = maxBlockWidth;
        blockLanesSoA.maxZoneWidths[blIdx] = maxZoneWidth;
        blockLanesSoA.widths[blIdx] = rightmostTime;
    });

    performance.mark('buildHierarchy:blockLanes:end');
    performance.measure('Build Block Lanes', 'buildHierarchy:blockLanes:start', 'buildHierarchy:blockLanes:end');

    // Build lanes
    performance.mark('buildHierarchy:lanes:start');

    const lanesSoA = new LanesSoA();
    lanesSoA.count = numLanes;
    lanesSoA.smIndices = new Uint8Array(numLanes);
    lanesSoA.ys = new Float32Array(numLanes);
    lanesSoA.heights = new Float32Array(numLanes);
    lanesSoA.widths = new Uint32Array(numLanes);
    lanesSoA.blockLanesStartIndices = new Uint32Array(numLanes);
    lanesSoA.blockLanesEndIndices = new Uint32Array(numLanes);

    // Map SM to block lanes
    const blockLanesBySM = new Map<number, number[]>();
    blockLaneGroups.forEach((group, blIdx) => {
        if (!blockLanesBySM.has(group.smId)) {
            blockLanesBySM.set(group.smId, []);
        }
        blockLanesBySM.get(group.smId)!.push(blIdx);
    });

    uniqueSMs.forEach((smId, laneIdx) => {
        lanesSoA.smIndices[laneIdx] = smId;

        const blockLaneIndices = blockLanesBySM.get(smId)!;
        lanesSoA.blockLanesStartIndices[laneIdx] = blockLaneIndices[0];
        lanesSoA.blockLanesEndIndices[laneIdx] = blockLaneIndices[blockLaneIndices.length - 1] + 1;

        // Calculate lane dimensions
        let laneHeight = 2 * LANE_EDGE_PADDING;
        let maxWidth = 0;

        blockLaneIndices.forEach((blIdx, i) => {
            laneHeight += blockLanesSoA.heights[blIdx];
            if (i < blockLaneIndices.length - 1) {
                laneHeight += BLOCK_LANE_PADDING;
            }
            maxWidth = Math.max(maxWidth, blockLanesSoA.widths[blIdx]);
        });

        lanesSoA.heights[laneIdx] = laneHeight;
        lanesSoA.widths[laneIdx] = maxWidth;
    });

    performance.mark('buildHierarchy:lanes:end');
    performance.measure('Build Lanes', 'buildHierarchy:lanes:start', 'buildHierarchy:lanes:end');

    // Calculate Y positions (bottom-up)
    performance.mark('buildHierarchy:ypositions:start');

    let currentY = 0;
    for (let i = lanesSoA.count - 1; i >= 0; i--) {
        lanesSoA.ys[i] = currentY;

        const blockLaneStart = lanesSoA.blockLanesStartIndices[i];
        const blockLaneEnd = lanesSoA.blockLanesEndIndices[i];

        let blockLaneY = currentY + LANE_EDGE_PADDING;

        for (let blIdx = blockLaneStart; blIdx < blockLaneEnd; blIdx++) {
            blockLanesSoA.ys[blIdx] = blockLaneY;

            // Position blocks in this block lane
            const offset = blockLanesSoA.blockIndicesOffsets[blIdx];
            const count = blockLanesSoA.blockIndicesCounts[blIdx];

            for (let j = 0; j < count; j++) {
                const blockIdx = blockLanesSoA.blockIndices[offset + j];
                blocks.ys[blockIdx] = blockLaneY + (blockLanesSoA.heights[blIdx] - blocks.heights[blockIdx]);

                // Position zones in this block
                const zoneStart = blocks.zonesStartIndices[blockIdx];
                const zoneEnd = blocks.zonesEndIndices[blockIdx];

                for (let zIdx = zoneStart; zIdx < zoneEnd; zIdx++) {
                    const sublaneIdx = zones.sublaneIndices[zIdx];
                    const sublaneY = blocks.ys[blockIdx] + blocks.heights[blockIdx] -
                                    BLOCK_EDGE_PADDING - sublaneIdx * (SUBLANE_HEIGHT + SUBLANE_PADDING) -
                                    SUBLANE_HEIGHT / 2;
                    zones.ys[zIdx] = sublaneY;
                }
            }

            blockLaneY += blockLanesSoA.heights[blIdx];
            if (blIdx < blockLaneEnd - 1) {
                blockLaneY += BLOCK_LANE_PADDING;
            }
        }

        currentY = blockLaneY + LANE_EDGE_PADDING + LANE_PADDING;
    }

    const worldHeight = currentY;
    const totalDurationNs = Math.max(...Array.from(lanesSoA.widths), 0);

    performance.mark('buildHierarchy:ypositions:end');
    performance.measure('Calculate Y Positions', 'buildHierarchy:ypositions:start', 'buildHierarchy:ypositions:end');

    // Create SM accelerator
    const smAccelerator = new SMAccelerator(lanesSoA);

    performance.mark('buildHierarchy:end');
    performance.measure('Build Hierarchy (Total)', 'buildHierarchy:start', 'buildHierarchy:end');

    console.log(`Hierarchy built:`);
    console.log(`  Lanes: ${lanesSoA.count}`);
    console.log(`  Block lanes: ${blockLanesSoA.count}`);
    console.log(`  World height: ${worldHeight.toFixed(2)}`);
    console.log(`  Total duration: ${totalDurationNs} ns`);

    return {
        tracks,
        zones,
        blocks,
        blockLanes: blockLanesSoA,
        lanes: lanesSoA,
        smAccelerator,
        worldHeight,
        totalDurationNs,
        formatDescriptors,
        kernelName,
        gridDims,
        clusterDims
    };
}

/**
 * Substitutes parameters into format descriptor label string.
 * Replaces placeholders {0}, {1}, etc. with corresponding parameter values.
 * Example: "Warp {0}" with params [5] → "Warp 5"
 * Uses the short label string (not tooltip).
 */
export function formatString(formatDescriptors: FormatDescriptor[], formatDescId: number, params: number[]): string {
    const desc = formatDescriptors[formatDescId];
    let result = desc.labelString;
    for (let i = 0; i < params.length; i++) {
        result = result.replace(`{${i}}`, params[i].toString());
    }
    return result;
}

/**
 * Substitutes parameters into format descriptor tooltip string.
 * Replaces placeholders {0}, {1}, etc. with corresponding parameter values.
 * Example: "Warp {0} on SM {1}" with params [5, 3] → "Warp 5 on SM 3"
 * Uses the full tooltip string.
 */
export function formatTooltipString(formatDescriptors: FormatDescriptor[], formatDescId: number, params: number[]): string {
    const desc = formatDescriptors[formatDescId];
    let result = desc.tooltipString;
    for (let i = 0; i < params.length; i++) {
        result = result.replace(`{${i}}`, params[i].toString());
    }
    return result;
}

/**
 * Substitutes track parameters and {lane} placeholder into format descriptor label string.
 * Replaces {lane} with sublaneIndex and {0}, {1}, etc. with parameter values.
 * Example: "Warp {lane}" with sublaneIndex=5 → "Warp 5"
 * Uses the short label string (not tooltip).
 */
export function formatTrackString(formatDescriptors: FormatDescriptor[], formatDescId: number, sublaneIndex: number, params: number[]): string {
    const desc = formatDescriptors[formatDescId];
    let result = desc.labelString;

    // Replace {lane} placeholder with sublaneIndex
    result = result.replace('{lane}', sublaneIndex.toString());

    // Replace numbered placeholders
    for (let i = 0; i < params.length; i++) {
        result = result.replace(`{${i}}`, params[i].toString());
    }
    return result;
}

/**
 * Substitutes track parameters and {lane} placeholder into format descriptor tooltip string.
 * Replaces {lane} with sublaneIndex and {0}, {1}, etc. with parameter values.
 * Example: "Warp {lane} on SM" with sublaneIndex=5 → "Warp 5 on SM"
 * Uses the full tooltip string.
 */
export function formatTrackTooltipString(formatDescriptors: FormatDescriptor[], formatDescId: number, sublaneIndex: number, params: number[]): string {
    const desc = formatDescriptors[formatDescId];
    let result = desc.tooltipString;

    // Replace {lane} placeholder with sublaneIndex
    result = result.replace('{lane}', sublaneIndex.toString());

    // Replace numbered placeholders
    for (let i = 0; i < params.length; i++) {
        result = result.replace(`{${i}}`, params[i].toString());
    }
    return result;
}

/**
 * Formats block descriptor label string with special placeholders.
 * Supports: {blockLinear}, {blockX}, {blockY}, {blockZ},
 *           {clusterLinear}, {clusterX}, {clusterY}, {clusterZ}
 *
 * Block coordinates are computed from linear block ID using row-major layout.
 * Uses the short label string (not tooltip).
 */
export function formatBlockString(
    formatDescriptors: FormatDescriptor[],
    formatDescId: number,
    blockId: number,
    clusterId: number,
    gridDimX: number,
    gridDimY: number,
    _gridDimZ: number,
    clusterDimX: number,
    clusterDimY: number,
    clusterDimZ: number
): string {
    const desc = formatDescriptors[formatDescId];
    let result = desc.labelString;

    // Compute block coordinates (row-major: x + y*dimX + z*dimX*dimY)
    const blockX = blockId % gridDimX;
    const blockY = Math.floor(blockId / gridDimX) % gridDimY;
    const blockZ = Math.floor(blockId / (gridDimX * gridDimY));

    // Compute cluster coordinates (row-major, if using clusters)
    const clusterX = clusterDimX > 0 ? (clusterId % clusterDimX) : 0;
    const clusterY = clusterDimY > 0 ? (Math.floor(clusterId / clusterDimX) % clusterDimY) : 0;
    const clusterZ = clusterDimZ > 0 ? Math.floor(clusterId / (clusterDimX * clusterDimY)) : 0;

    // Replace special placeholders
    result = result.replace('{blockLinear}', blockId.toString());
    result = result.replace('{blockX}', blockX.toString());
    result = result.replace('{blockY}', blockY.toString());
    result = result.replace('{blockZ}', blockZ.toString());
    result = result.replace('{clusterLinear}', clusterId.toString());
    result = result.replace('{clusterX}', clusterX.toString());
    result = result.replace('{clusterY}', clusterY.toString());
    result = result.replace('{clusterZ}', clusterZ.toString());

    return result;
}

/**
 * Formats block descriptor tooltip string with special placeholders.
 * Supports: {blockLinear}, {blockX}, {blockY}, {blockZ},
 *           {clusterLinear}, {clusterX}, {clusterY}, {clusterZ}
 *
 * Block coordinates are computed from linear block ID using row-major layout.
 * Uses the full tooltip string.
 */
export function formatBlockTooltipString(
    formatDescriptors: FormatDescriptor[],
    formatDescId: number,
    blockId: number,
    clusterId: number,
    gridDimX: number,
    gridDimY: number,
    _gridDimZ: number,
    clusterDimX: number,
    clusterDimY: number,
    clusterDimZ: number
): string {
    const desc = formatDescriptors[formatDescId];
    let result = desc.tooltipString;

    // Compute block coordinates (row-major: x + y*dimX + z*dimX*dimY)
    const blockX = blockId % gridDimX;
    const blockY = Math.floor(blockId / gridDimX) % gridDimY;
    const blockZ = Math.floor(blockId / (gridDimX * gridDimY));

    // Compute cluster coordinates (row-major, if using clusters)
    const clusterX = clusterDimX > 0 ? (clusterId % clusterDimX) : 0;
    const clusterY = clusterDimY > 0 ? (Math.floor(clusterId / clusterDimX) % clusterDimY) : 0;
    const clusterZ = clusterDimZ > 0 ? Math.floor(clusterId / (clusterDimX * clusterDimY)) : 0;

    // Replace special placeholders
    result = result.replace('{blockLinear}', blockId.toString());
    result = result.replace('{blockX}', blockX.toString());
    result = result.replace('{blockY}', blockY.toString());
    result = result.replace('{blockZ}', blockZ.toString());
    result = result.replace('{clusterLinear}', clusterId.toString());
    result = result.replace('{clusterX}', clusterX.toString());
    result = result.replace('{clusterY}', clusterY.toString());
    result = result.replace('{clusterZ}', clusterZ.toString());

    return result;
}
