import {
    BlocksSoA,
    BlockLanesSoA,
    FormatDescriptor,
    HierarchyData,
    LanesSoA,
    SMAccelerator,
    TraceBookmark,
    TracksSoA,
    ZonesSoA
} from './types.js';
import {
    BLOCK_EDGE_PADDING,
    BLOCK_LANE_PADDING,
    LANE_EDGE_PADDING,
    LANE_PADDING,
    SUBLANE_HEIGHT,
    SUBLANE_PADDING
} from './constants.js';

const MAGIC = 'NTRACE4';
const FILE_HEADER_SIZE = 32;
const CHUNK_HEADER_SIZE = 24;
const MIN_EVENT_DURATION_NS = 1;
const EVENT_KIND_BOOKMARK = 1;

enum ChunkType {
    Session = 1,
    Strings = 2,
    Clocks = 3,
    ClockSnapshots = 4,
    Tracks = 5,
    Events = 6,
    Arguments = 7,
    EventFormats = 8
}

interface ClockRecord {
    id: number;
    frequencyHz: bigint;
}

interface ClockSnapshotRecord {
    sourceClock: number;
    referenceClock: number;
    sourceTimestamp: bigint;
    referenceTimestamp: bigint;
}

interface TrackRecord {
    id: bigint;
    parentId: bigint;
    clockId: number;
    nameId: number;
    kind: number;
    sortOrder: number;
    sourceId: bigint;
}

interface EventRecord {
    id: bigint;
    parentId: bigint;
    trackId: bigint;
    timestamp: bigint;
    duration: bigint;
    correlationId: bigint;
    nameId: number;
    firstArgument: number;
    argumentCount: number;
    color: number;
    kind: number;
}

interface ArgumentRecord {
    nameId: number;
    kind: number;
    value: bigint;
}

interface EventFormatRecord {
    labelId: number;
    tooltipId: number;
    parameterCount: number;
}

export interface ParsedTraceData {
    kernelName: string;
    gridDimX: number;
    gridDimY: number;
    gridDimZ: number;
    clusterDimX: number;
    clusterDimY: number;
    clusterDimZ: number;
    formatDescriptors: FormatDescriptor[];
    tracks: TracksSoA;
    zones: ZonesSoA;
    blocks: BlocksSoA;
    trackNames: string[];
    trackTooltips: string[];
    trackDepths: number[];
    trackHierarchies: TraceTrackNode[][];
    trackDisclosureKeys: string[];
    trackExpanded: boolean[];
    trackExpansionGroupIds: bigint[];
    trackExpansionModes: TrackExpansionMode[];
    bookmarks: TraceBookmark[];
}

export enum TrackExpansionMode {
    Always,
    Collapsed,
    Expanded
}

export interface TraceTrackNode {
    id: bigint;
    parentId: bigint;
    name: string;
    kind: number;
    sortOrder: number;
    sourceId: bigint;
}

interface ProjectionIndex {
    rootZonesByTrack: Uint32Array[];
    parentZoneIndices: Int32Array;
    childOffsets: Uint32Array;
    childZoneIndices: Uint32Array;
    expandableEventZoneIndices: Map<bigint, number>;
    streamCountsByGpu: Map<bigint, Set<bigint>>;
}

const projectionIndices = new WeakMap<ParsedTraceData, ProjectionIndex>();

function buildProjectionIndex(source: ParsedTraceData): ProjectionIndex {
    const eventZoneIndices = new Map<bigint, number>();
    for (let zoneIndex = 0; zoneIndex < source.zones.count; zoneIndex++) {
        eventZoneIndices.set(source.zones.eventIds[zoneIndex], zoneIndex);
    }

    const rootZonesByTrack = new Array<number[]>(source.tracks.count);
    for (let trackIndex = 0;
        trackIndex < source.tracks.count; trackIndex++) {
        rootZonesByTrack[trackIndex] = [];
    }
    const parentZoneIndices = new Int32Array(source.zones.count);
    parentZoneIndices.fill(-1);
    const childOffsets = new Uint32Array(source.zones.count + 1);
    const parentEventIds = new Set<bigint>();

    for (let zoneIndex = 0; zoneIndex < source.zones.count; zoneIndex++) {
        const parentEventId = source.zones.parentEventIds[zoneIndex];
        if (parentEventId === 0n) {
            rootZonesByTrack[source.zones.trackIndices[zoneIndex]].push(
                zoneIndex);
            continue;
        }

        parentEventIds.add(parentEventId);
        const parentZoneIndex = eventZoneIndices.get(parentEventId);
        if (parentZoneIndex !== undefined) {
            parentZoneIndices[zoneIndex] = parentZoneIndex;
            childOffsets[parentZoneIndex + 1]++;
        }
    }

    const expandableEventZoneIndices = new Map<bigint, number>();
    for (const parentEventId of parentEventIds) {
        const parentZoneIndex = eventZoneIndices.get(parentEventId);
        if (parentZoneIndex !== undefined) {
            expandableEventZoneIndices.set(parentEventId, parentZoneIndex);
        }
    }
    eventZoneIndices.clear();
    for (let zoneIndex = 1; zoneIndex < childOffsets.length; zoneIndex++) {
        childOffsets[zoneIndex] += childOffsets[zoneIndex - 1];
    }

    const childZoneIndices = new Uint32Array(
        childOffsets[source.zones.count]);
    const nextChildOffsets = childOffsets.slice(0, source.zones.count);
    for (let zoneIndex = 0; zoneIndex < source.zones.count; zoneIndex++) {
        const parentZoneIndex = parentZoneIndices[zoneIndex];
        if (parentZoneIndex >= 0) {
            childZoneIndices[nextChildOffsets[parentZoneIndex]++] = zoneIndex;
        }
    }

    const streamCountsByGpu = new Map<bigint, Set<bigint>>();
    for (let trackIndex = 0; trackIndex < source.tracks.count; trackIndex++) {
        const hierarchy = source.trackHierarchies[trackIndex];
        const gpuNode = hierarchy.find(node => node.kind === 2);
        const streamNode = hierarchy.find(node => node.kind === 3);
        if (!gpuNode || !streamNode) continue;

        const streams = streamCountsByGpu.get(gpuNode.id)
            ?? new Set<bigint>();
        streams.add(streamNode.id);
        streamCountsByGpu.set(gpuNode.id, streams);
    }

    return {
        rootZonesByTrack: rootZonesByTrack.map(
            zoneIndices => Uint32Array.from(zoneIndices)),
        parentZoneIndices,
        childOffsets,
        childZoneIndices,
        expandableEventZoneIndices,
        streamCountsByGpu
    };
}

function getProjectionIndex(source: ParsedTraceData): ProjectionIndex {
    const cached = projectionIndices.get(source);
    if (cached) return cached;

    const index = buildProjectionIndex(source);
    projectionIndices.set(source, index);
    return index;
}

class Reader {
    constructor(
        private readonly view: DataView,
        public offset: number
    ) {}

    u8(): number {
        const value = this.view.getUint8(this.offset);
        this.offset += 1;
        return value;
    }

    u16(): number {
        const value = this.view.getUint16(this.offset, true);
        this.offset += 2;
        return value;
    }

    u32(): number {
        const value = this.view.getUint32(this.offset, true);
        this.offset += 4;
        return value;
    }

    i32(): number {
        const value = this.view.getInt32(this.offset, true);
        this.offset += 4;
        return value;
    }

    u64(): bigint {
        const value = this.view.getBigUint64(this.offset, true);
        this.offset += 8;
        return value;
    }

    varUint(): bigint {
        let value = 0n;
        let shift = 0n;

        for (let byteIndex = 0; byteIndex < 10; byteIndex++) {
            const byte = this.u8();
            value |= BigInt(byte & 0x7f) << shift;

            if ((byte & 0x80) === 0) return value;
            shift += 7n;
        }

        throw new Error('Varint exceeds 64 bits');
    }

    string(): string {
        const length = this.u32();
        const bytes = new Uint8Array(
            this.view.buffer, this.view.byteOffset + this.offset, length);
        this.offset += length;
        return new TextDecoder().decode(bytes);
    }
}

const COLOR_PALETTE: [number, number, number][] = [
    [77, 148, 230], [230, 89, 89], [102, 217, 128], [242, 166, 64],
    [179, 115, 230], [77, 199, 199], [242, 217, 77], [230, 128, 179],
    [128, 191, 102], [217, 102, 140], [115, 179, 230], [191, 140, 89],
    [140, 115, 217], [89, 209, 166], [235, 153, 89], [153, 209, 179]
];

function unpackColor(color: number, fallbackIndex: number): [number, number, number] {
    if (color === 0) {
        return COLOR_PALETTE[fallbackIndex % COLOR_PALETTE.length];
    }

    return [
        (color >> 16) & 0xff,
        (color >> 8) & 0xff,
        color & 0xff
    ];
}

function buildTrackPath(
    track: TrackRecord,
    tracksById: Map<bigint, TrackRecord>,
    strings: string[]
): string {
    const names: string[] = [];
    let current: TrackRecord | undefined = track;

    while (current) {
        names.push(strings[current.nameId] ?? `Track ${current.id}`);
        current = current.parentId === 0n
            ? undefined : tracksById.get(current.parentId);
    }

    names.reverse();
    return names.join(' / ');
}

function isGpuTrack(
    track: TrackRecord,
    tracksById: Map<bigint, TrackRecord>
): boolean {
    let current: TrackRecord | undefined = track;

    while (current) {
        if (current.kind === 2) return true;
        current = current.parentId === 0n
            ? undefined : tracksById.get(current.parentId);
    }

    return false;
}

function assignIntervalSublane(
    sublaneEnds: number[], start: number, end: number
): number {
    for (let sublane = 0; sublane < sublaneEnds.length; sublane++) {
        if (sublaneEnds[sublane] <= start) {
            sublaneEnds[sublane] = end;
            return sublane;
        }
    }

    sublaneEnds.push(end);
    return sublaneEnds.length - 1;
}

function buildTrackHierarchy(
    track: TrackRecord,
    tracksById: Map<bigint, TrackRecord>,
    strings: string[]
): TraceTrackNode[] {
    const hierarchy: TraceTrackNode[] = [];
    let current: TrackRecord | undefined = track;

    while (current) {
        hierarchy.push({
            id: current.id,
            parentId: current.parentId,
            name: strings[current.nameId] ?? `Track ${current.id}`,
            kind: current.kind,
            sortOrder: current.sortOrder,
            sourceId: current.sourceId
        });
        current = current.parentId === 0n
            ? undefined : tracksById.get(current.parentId);
    }

    hierarchy.reverse();
    return hierarchy;
}

function buildCompressedTrackLabel(
    track: TrackRecord,
    tracksById: Map<bigint, TrackRecord>,
    visibleTrackIds: Set<bigint>,
    strings: string[]
): string {
    const gpuStreamKind = 3;
    const gpuDeviceKind = 2;
    const parent = track.parentId === 0n
        ? undefined : tracksById.get(track.parentId);
    if (track.kind === gpuStreamKind && parent?.kind === gpuDeviceKind) {
        let visibleStreamCount = 0;
        for (const candidate of tracksById.values()) {
            if (candidate.kind === gpuStreamKind
                && candidate.parentId === parent.id
                && visibleTrackIds.has(candidate.id)) {
                visibleStreamCount++;
            }
        }

        if (visibleStreamCount === 1) {
            return strings[parent.nameId] ?? `GPU ${parent.sourceId}`;
        }
    }

    const names: string[] = [];
    let current: TrackRecord | undefined = track;

    while (current) {
        names.push(strings[current.nameId] ?? `Track ${current.id}`);
        const parent: TrackRecord | undefined = current.parentId === 0n
            ? undefined : tracksById.get(current.parentId);
        if (parent && visibleTrackIds.has(parent.id)) {
            break;
        }
        current = parent;
    }

    names.reverse();
    return names.join(' / ');
}

function visibleTrackDepth(
    track: TrackRecord,
    tracksById: Map<bigint, TrackRecord>,
    visibleTrackIds: Set<bigint>
): number {
    let depth = 0;
    let current = track.parentId === 0n
        ? undefined : tracksById.get(track.parentId);

    while (current) {
        if (visibleTrackIds.has(current.id)) {
            depth++;
        }
        current = current.parentId === 0n
            ? undefined : tracksById.get(current.parentId);
    }

    return depth;
}

function mapTimestamp(
    clockId: number,
    timestamp: bigint,
    referenceClock: number,
    clocks: Map<number, ClockRecord>,
    snapshots: Map<number, ClockSnapshotRecord[]>,
    visitedClocks: Set<number> = new Set<number>()
): bigint {
    if (clockId === referenceClock) {
        return timestamp;
    }

    if (visitedClocks.has(clockId)) {
        throw new Error(`Clock correlation cycle at clock ${clockId}`);
    }
    visitedClocks.add(clockId);

    const clockSnapshots = snapshots.get(clockId);
    if (!clockSnapshots || clockSnapshots.length === 0) {
        return timestamp;
    }

    const first = clockSnapshots[0];
    let parentTimestamp: bigint;
    if (clockSnapshots.length === 1) {
        const frequency = clocks.get(clockId)?.frequencyHz ?? 1_000_000_000n;
        parentTimestamp = first.referenceTimestamp
            + (timestamp - first.sourceTimestamp) * 1_000_000_000n / frequency;
    } else {
        let left = first;
        let right = clockSnapshots[clockSnapshots.length - 1];
        for (let i = 1; i < clockSnapshots.length; i++) {
            if (timestamp <= clockSnapshots[i].sourceTimestamp) {
                left = clockSnapshots[i - 1];
                right = clockSnapshots[i];
                break;
            }
        }

        const sourceDelta = right.sourceTimestamp - left.sourceTimestamp;
        parentTimestamp = sourceDelta === 0n
            ? left.referenceTimestamp
            : left.referenceTimestamp
                + (timestamp - left.sourceTimestamp)
                    * (right.referenceTimestamp - left.referenceTimestamp)
                    / sourceDelta;
    }

    return mapTimestamp(
        first.referenceClock, parentTimestamp, referenceClock,
        clocks, snapshots, visitedClocks);
}

export async function parseTraceFile(
    file: File,
    onProgress?: (message: string) => void
): Promise<ParsedTraceData> {
    if (onProgress) onProgress('Parsing unified trace...');
    const fileBuffer = await file.arrayBuffer();
    const headerView = new DataView(fileBuffer);
    const magic = new TextDecoder().decode(new Uint8Array(fileBuffer, 0, 7));
    if (magic !== MAGIC) {
        throw new Error(`Unsupported trace magic: ${magic}`);
    }

    const header = new Reader(headerView, 8);
    const majorVersion = header.u16();
    header.u16();
    const endianness = header.u8();
    const fileFlags = header.u8();
    header.u16();
    const uncompressedBodySize = Number(header.u64());
    const storedBodySize = Number(header.u64());
    if (majorVersion !== 4 || endianness !== 1) {
        throw new Error(`Unsupported nanotrace format ${majorVersion}`);
    }
    if ((fileFlags & ~1) !== 0) {
        throw new Error(`Unsupported nanotrace flags ${fileFlags}`);
    }
    if (storedBodySize !== fileBuffer.byteLength - FILE_HEADER_SIZE) {
        throw new Error('Trace body size does not match the file header');
    }

    let buffer: ArrayBuffer;
    let offset: number;
    if ((fileFlags & 1) !== 0) {
        if (onProgress) onProgress('Decompressing trace...');
        const compressedBody = fileBuffer.slice(FILE_HEADER_SIZE);
        const decompressedStream = new Blob([compressedBody]).stream()
            .pipeThrough(new DecompressionStream('deflate'));
        buffer = await new Response(decompressedStream).arrayBuffer();
        if (buffer.byteLength !== uncompressedBodySize) {
            throw new Error('Decompressed trace size does not match the header');
        }
        offset = 0;
    } else {
        buffer = fileBuffer;
        offset = FILE_HEADER_SIZE;
        if (uncompressedBodySize !== storedBodySize) {
            throw new Error('Uncompressed trace body sizes do not match');
        }
    }

    const view = new DataView(buffer);

    let sessionName = 'Nanotrace session';
    let referenceClock = 1;
    const strings: string[] = [];
    const clocks = new Map<number, ClockRecord>();
    const clockSnapshots: ClockSnapshotRecord[] = [];
    const trackRecords: TrackRecord[] = [];
    const eventRecords: EventRecord[] = [];
    const argumentRecords: ArgumentRecord[] = [];
    const eventFormatRecords: EventFormatRecord[] = [];
    while (offset + CHUNK_HEADER_SIZE <= buffer.byteLength) {
        const chunkHeader = new Reader(view, offset);
        const chunkType = chunkHeader.u32();
        chunkHeader.u32();
        const byteSize = Number(chunkHeader.u64());
        const count = Number(chunkHeader.u64());
        const payloadOffset = offset + CHUNK_HEADER_SIZE;
        const reader = new Reader(view, payloadOffset);

        if (chunkType === ChunkType.Session) {
            sessionName = reader.string();
            referenceClock = reader.u32();
            reader.u32();
        } else if (chunkType === ChunkType.Strings) {
            for (let i = 0; i < count; i++) strings.push(reader.string());
        } else if (chunkType === ChunkType.Clocks) {
            for (let i = 0; i < count; i++) {
                const id = reader.u32();
                reader.u8();
                reader.u8();
                reader.u16();
                reader.u32();
                reader.u32();
                clocks.set(id, { id, frequencyHz: reader.u64() });
            }
        } else if (chunkType === ChunkType.ClockSnapshots) {
            for (let i = 0; i < count; i++) {
                clockSnapshots.push({
                    sourceClock: reader.u32(),
                    referenceClock: reader.u32(),
                    sourceTimestamp: reader.u64(),
                    referenceTimestamp: reader.u64()
                });
                reader.u64();
            }
        } else if (chunkType === ChunkType.Tracks) {
            for (let i = 0; i < count; i++) {
                const id = reader.u64();
                const parentId = reader.u64();
                const clockId = reader.u32();
                const nameId = reader.u32();
                const kind = reader.u8();
                reader.u8();
                reader.u16();
                const sortOrder = reader.i32();
                reader.u32();
                const sourceId = reader.u64();
                trackRecords.push({
                    id, parentId, clockId, nameId, kind, sortOrder, sourceId
                });
            }
        } else if (chunkType === ChunkType.Events) {
            const eventCount = Number(reader.varUint());
            const previousTimestamps = new Map<bigint, bigint>();

            for (let eventIndex = 0; eventIndex < eventCount; eventIndex++) {
                const trackId = reader.varUint();
                const encodedTimestampDelta = reader.varUint();
                const timestampDelta = (encodedTimestampDelta >> 1n)
                    ^ -(encodedTimestampDelta & 1n);
                const timestamp = (previousTimestamps.get(trackId) ?? 0n)
                    + timestampDelta;

                if (timestamp < 0n) {
                    throw new Error('Event timestamp underflow');
                }

                previousTimestamps.set(trackId, timestamp);
                const duration = reader.varUint();
                const nameId = Number(reader.varUint());
                const flags = reader.u8();
                const kind = flags & 0x03;
                const parentId = (flags & 0x04) !== 0
                    ? reader.varUint() : 0n;
                const correlationId = (flags & 0x08) !== 0
                    ? reader.varUint() : 0n;
                const firstArgument = (flags & 0x10) !== 0
                    ? Number(reader.varUint()) : 0;
                const argumentCount = (flags & 0x10) !== 0
                    ? Number(reader.varUint()) : 0;
                const color = (flags & 0x20) !== 0
                    ? Number(reader.varUint()) : 0;

                eventRecords.push({
                    id: BigInt(eventRecords.length + 1), parentId, trackId,
                    timestamp, duration, correlationId, nameId, firstArgument,
                    argumentCount, color, kind
                });
            }

            if (eventCount !== count) {
                throw new Error(
                    `Event chunk declared ${count} records but contained `
                    + `${eventCount}`);
            }
        } else if (chunkType === ChunkType.Arguments) {
            for (let i = 0; i < count; i++) {
                const nameId = Number(reader.varUint());
                const kind = reader.u8();
                let value: bigint;

                if (kind === 1) {
                    const encoded = reader.varUint();
                    const signed = (encoded >> 1n) ^ -(encoded & 1n);
                    value = BigInt.asUintN(64, signed);
                } else if (kind === 2) {
                    value = reader.u64();
                } else {
                    value = reader.varUint();
                }

                argumentRecords.push({ nameId, kind, value });
            }
        } else if (chunkType === ChunkType.EventFormats) {
            for (let i = 0; i < count; i++) {
                eventFormatRecords.push({
                    labelId: Number(reader.varUint()),
                    tooltipId: Number(reader.varUint()),
                    parameterCount: reader.u8()
                });
            }
        }

        offset = payloadOffset + byteSize;
    }

    const snapshotsByClock = new Map<number, ClockSnapshotRecord[]>();
    for (const snapshot of clockSnapshots) {
        const records = snapshotsByClock.get(snapshot.sourceClock) ?? [];
        records.push(snapshot);
        snapshotsByClock.set(snapshot.sourceClock, records);
    }
    for (const records of snapshotsByClock.values()) {
        records.sort((a, b) => a.sourceTimestamp < b.sourceTimestamp ? -1 : 1);
    }

    const tracksById = new Map<bigint, TrackRecord>();
    for (const track of trackRecords) tracksById.set(track.id, track);
    const eventsByTrack = new Map<bigint, EventRecord[]>();
    let origin: bigint | undefined;
    for (const event of eventRecords) {
        const track = tracksById.get(event.trackId);
        if (!track) continue;
        const mapped = mapTimestamp(
            track.clockId, event.timestamp, referenceClock, clocks, snapshotsByClock);
        if (origin === undefined || mapped < origin) origin = mapped;
        if (event.kind !== EVENT_KIND_BOOKMARK) {
            const records = eventsByTrack.get(event.trackId) ?? [];
            records.push(event);
            eventsByTrack.set(event.trackId, records);
        }
    }
    const traceOrigin = origin ?? 0n;

    const naturalOrder = new Intl.Collator(undefined, { numeric: true });
    const visibleTracks = trackRecords
        .filter(track => (eventsByTrack.get(track.id)?.length ?? 0) > 0)
        .sort((a, b) => {
            const gpuOrder = Number(isGpuTrack(b, tracksById))
                - Number(isGpuTrack(a, tracksById));
            if (gpuOrder !== 0) return gpuOrder;

            const pathA = buildTrackPath(a, tracksById, strings);
            const pathB = buildTrackPath(b, tracksById, strings);
            return naturalOrder.compare(pathA, pathB)
                || a.sortOrder - b.sortOrder;
        });
    const visibleTrackIds = new Set(visibleTracks.map(track => track.id));
    const trackNames = visibleTracks.map(track => buildCompressedTrackLabel(
        track, tracksById, visibleTrackIds, strings));
    const trackTooltips = visibleTracks.map(
        track => buildTrackPath(track, tracksById, strings));
    const trackDepths = visibleTracks.map(track => visibleTrackDepth(
        track, tracksById, visibleTrackIds));
    const trackHierarchies = visibleTracks.map(track => buildTrackHierarchy(
        track, tracksById, strings));

    const eventNameIds = new Map<number, number>();
    const eventFormatsByLabel = new Map<number, EventFormatRecord>();
    for (const format of eventFormatRecords) {
        eventFormatsByLabel.set(format.labelId, format);
    }
    const formatDescriptors: FormatDescriptor[] = [];
    const trackFormatIds: number[] = [];
    for (let i = 0; i < trackNames.length; i++) {
        trackFormatIds.push(formatDescriptors.length);
        formatDescriptors.push({
            labelString: trackNames[i],
            tooltipString: trackTooltips[i],
            placeholderCount: 0
        });
    }
    for (const event of eventRecords) {
        if (!eventNameIds.has(event.nameId)) {
            const name = strings[event.nameId] ?? `Event ${event.nameId}`;
            const metadata = eventFormatsByLabel.get(event.nameId);
            eventNameIds.set(event.nameId, formatDescriptors.length);
            formatDescriptors.push({
                labelString: name,
                tooltipString: metadata
                    ? strings[metadata.tooltipId] ?? name : name,
                placeholderCount: metadata?.parameterCount ?? 0
            });
        }
    }

    const bookmarks: TraceBookmark[] = [];
    for (const event of eventRecords) {
        if (event.kind !== EVENT_KIND_BOOKMARK) continue;

        const track = tracksById.get(event.trackId);
        if (!track) continue;
        const mappedTimestamp = mapTimestamp(
            track.clockId, event.timestamp,
            referenceClock, clocks, snapshotsByClock);
        bookmarks.push({
            timestampNs: Number(mappedTimestamp - traceOrigin),
            label: strings[event.nameId] ?? `Bookmark ${event.nameId}`
        });
    }
    bookmarks.sort((first, second) =>
        first.timestampNs - second.timestampNs);

    let zoneCount = 0;
    for (const track of visibleTracks) zoneCount += eventsByTrack.get(track.id)!.length;
    const tracks = new TracksSoA();
    tracks.count = visibleTracks.length;
    tracks.formatDescIds = new Uint16Array(tracks.count);
    tracks.sublaneIndices = new Uint8Array(tracks.count);
    tracks.paramsOffsets = new Uint32Array(tracks.count);
    tracks.paramsCounts = new Uint8Array(tracks.count);
    tracks.blockIndices = new Uint32Array(tracks.count);
    tracks.paramsPool = new Uint32Array(0);

    const zones = new ZonesSoA();
    zones.count = zoneCount;
    zones.eventIds = new BigUint64Array(zoneCount);
    zones.parentEventIds = new BigUint64Array(zoneCount);
    zones.hasChildren = new Uint8Array(zoneCount);
    zones.expanded = new Uint8Array(zoneCount);
    zones.disclosureKeys = new Array<string>(zoneCount);
    zones.details = new Array<string>(zoneCount);
    zones.startsX = new Float64Array(zoneCount);
    zones.endsX = new Float64Array(zoneCount);
    zones.ys = new Float32Array(zoneCount);
    zones.colors = new Uint8Array(zoneCount * 3);
    zones.formatDescIds = new Uint16Array(zoneCount);
    zones.paramsOffsets = new Uint32Array(zoneCount);
    zones.paramsCounts = new Uint8Array(zoneCount);
    zones.trackIndices = new Uint32Array(zoneCount);
    zones.smIndices = new Uint32Array(zoneCount);
    zones.blockIndices = new Uint32Array(zoneCount);
    zones.sublaneIndices = new Uint8Array(zoneCount);
    let zoneParameterCount = 0;
    for (const event of eventRecords) {
        if (event.kind === EVENT_KIND_BOOKMARK) continue;
        const descriptorId = eventNameIds.get(event.nameId);
        zoneParameterCount += descriptorId === undefined ? 0
            : Math.min(formatDescriptors[descriptorId].placeholderCount,
                event.argumentCount);
    }
    zones.paramsPool = new Uint32Array(zoneParameterCount);

    const parentEventIds = new Set<bigint>();
    for (const event of eventRecords) {
        if (event.kind === EVENT_KIND_BOOKMARK) continue;
        if (event.parentId !== 0n) parentEventIds.add(event.parentId);
    }

    const blocks = new BlocksSoA();
    blocks.count = visibleTracks.length;
    blocks.startsX = new Float64Array(blocks.count);
    blocks.endsX = new Float64Array(blocks.count);
    blocks.ys = new Float32Array(blocks.count);
    blocks.heights = new Float32Array(blocks.count);
    blocks.headerHeights = new Float32Array(blocks.count);
    blocks.sublanesCounts = new Uint8Array(blocks.count);
    blocks.sublanesMaxWidths = new Float64Array(blocks.count);
    blocks.formatDescIds = new Uint16Array(blocks.count);
    blocks.gridIds = new Uint32Array(blocks.count);
    blocks.clusterIds = new Uint32Array(blocks.count);
    blocks.smIndices = new Uint32Array(blocks.count);
    blocks.blockLaneIndices = new Uint32Array(blocks.count);
    blocks.zonesStartIndices = new Uint32Array(blocks.count);
    blocks.zonesEndIndices = new Uint32Array(blocks.count);
    blocks.tracksStartIndices = new Uint32Array(blocks.count);
    blocks.tracksEndIndices = new Uint32Array(blocks.count);

    let zoneIndex = 0;
    let zoneParameterIndex = 0;
    visibleTracks.forEach((track, trackIndex) => {
        const trackEvents = eventsByTrack.get(track.id)!;
        trackEvents.sort((a, b) => a.timestamp < b.timestamp ? -1 : 1);
        tracks.formatDescIds[trackIndex] = trackFormatIds[trackIndex];
        tracks.blockIndices[trackIndex] = trackIndex;
        blocks.startsX[trackIndex] = Number.MAX_SAFE_INTEGER;
        blocks.endsX[trackIndex] = 0;
        blocks.formatDescIds[trackIndex] = trackFormatIds[trackIndex];
        blocks.gridIds[trackIndex] = trackIndex;
        blocks.smIndices[trackIndex] = trackIndex;
        blocks.zonesStartIndices[trackIndex] = zoneIndex;
        blocks.tracksStartIndices[trackIndex] = trackIndex;
        blocks.tracksEndIndices[trackIndex] = trackIndex + 1;
        const sublaneEnds: number[] = [];

        for (const event of trackEvents) {
            const mappedStart = mapTimestamp(
                track.clockId, event.timestamp,
                referenceClock, clocks, snapshotsByClock);
            const mappedEnd = mapTimestamp(
                track.clockId, event.timestamp + event.duration,
                referenceClock, clocks, snapshotsByClock);
            const start = Number(mappedStart - traceOrigin);
            const duration = Math.max(
                Number(mappedEnd - mappedStart), MIN_EVENT_DURATION_NS);
            zones.startsX[zoneIndex] = start;
            zones.endsX[zoneIndex] = start + duration;
            zones.eventIds[zoneIndex] = event.id;
            zones.parentEventIds[zoneIndex] = event.parentId;
            zones.hasChildren[zoneIndex] = parentEventIds.has(event.id) ? 1 : 0;
            zones.disclosureKeys[zoneIndex] = parentEventIds.has(event.id)
                ? `event:${event.id}` : '';
            const details: string[] = [];
            const descriptorId = eventNameIds.get(event.nameId)!;
            const placeholderCount =
                formatDescriptors[descriptorId].placeholderCount;
            zones.paramsOffsets[zoneIndex] = zoneParameterIndex;
            for (let argumentIndex = 0;
                argumentIndex < event.argumentCount; argumentIndex++) {
                const argument = argumentRecords[
                    event.firstArgument + argumentIndex];
                if (!argument) continue;

                const argumentName = strings[argument.nameId]
                    ?? `argument_${argumentIndex}`;
                if (argumentName === 'kernel_signature'
                    || argumentName === 'graph_id'
                    || argumentName === 'graph_node_id'
                    || argumentName === 'context_id') {
                    continue;
                }

                let argumentValue: string;
                if (argument.kind === 1) {
                    argumentValue = BigInt.asIntN(64, argument.value).toString();
                } else if (argument.kind === 2) {
                    const floating = new ArrayBuffer(8);
                    new DataView(floating).setBigUint64(
                        0, argument.value, true);
                    argumentValue = new DataView(floating)
                        .getFloat64(0, true).toString();
                } else if (argument.kind === 3) {
                    argumentValue = strings[Number(argument.value)]
                        ?? `<string ${argument.value}>`;
                } else {
                    argumentValue = argument.value.toString();
                }
                details.push(`${argumentName}: ${argumentValue}`);
                if (argumentIndex < placeholderCount && argument.kind !== 3) {
                    zones.paramsPool[zoneParameterIndex++] = Number(
                        BigInt.asUintN(32, argument.value));
                    zones.paramsCounts[zoneIndex]++;
                }
            }
            zones.details[zoneIndex] = details.join('\n');
            zones.formatDescIds[zoneIndex] = descriptorId;
            zones.trackIndices[zoneIndex] = trackIndex;
            zones.smIndices[zoneIndex] = trackIndex;
            zones.blockIndices[zoneIndex] = trackIndex;
            zones.sublaneIndices[zoneIndex] = assignIntervalSublane(
                sublaneEnds, start, start + duration);
            const color = unpackColor(event.color, event.nameId);
            zones.colors[zoneIndex * 3] = color[0];
            zones.colors[zoneIndex * 3 + 1] = color[1];
            zones.colors[zoneIndex * 3 + 2] = color[2];
            blocks.startsX[trackIndex] = Math.min(blocks.startsX[trackIndex], start);
            blocks.endsX[trackIndex] = Math.max(
                blocks.endsX[trackIndex], start + duration);
            blocks.sublanesMaxWidths[trackIndex] = Math.max(
                blocks.sublanesMaxWidths[trackIndex], duration);
            zoneIndex++;
        }

        blocks.sublanesCounts[trackIndex] = sublaneEnds.length;
        blocks.zonesEndIndices[trackIndex] = zoneIndex;
        blocks.heights[trackIndex] = BLOCK_EDGE_PADDING
            + sublaneEnds.length * SUBLANE_HEIGHT
            + Math.max(0, sublaneEnds.length - 1) * SUBLANE_PADDING;
    });

    return {
        kernelName: sessionName,
        gridDimX: 0,
        gridDimY: 0,
        gridDimZ: 0,
        clusterDimX: 0,
        clusterDimY: 0,
        clusterDimZ: 0,
        formatDescriptors,
        tracks,
        zones,
        blocks,
        trackNames,
        trackTooltips,
        trackDepths,
        trackHierarchies,
        trackDisclosureKeys: new Array<string>(visibleTracks.length).fill(''),
        trackExpanded: new Array<boolean>(visibleTracks.length).fill(false),
        trackExpansionGroupIds: new Array<bigint>(visibleTracks.length).fill(0n),
        trackExpansionModes: new Array<TrackExpansionMode>(
            visibleTracks.length).fill(TrackExpansionMode.Always),
        bookmarks
    };
}

export function projectTraceData(
    source: ParsedTraceData,
    expandedEventIds: ReadonlySet<bigint>,
    expandedTrackIds: ReadonlySet<bigint> = new Set<bigint>()
): ParsedTraceData {
    interface ProjectedRow {
        sourceTrackIndex: number;
        sourceZoneIndices: number[];
        name: string;
        tooltip: string;
        depth: number;
        hierarchy: TraceTrackNode[];
        disclosureKey: string;
        trackDisclosureKey: string;
        trackExpanded: boolean;
        expansionGroupId: bigint;
        expansionMode: TrackExpansionMode;
        requiresSort: boolean;
        physicalHierarchy: boolean;
    }

    const rows: ProjectedRow[] = [];
    const groupedRows = new Map<string, ProjectedRow>();
    const projectionIndex = getProjectionIndex(source);
    const streamCountsByGpu = projectionIndex.streamCountsByGpu;
    const visibleZonesByTrack = new Array<number[] | undefined>(
        source.tracks.count);
    for (let trackIndex = 0;
        trackIndex < source.tracks.count; trackIndex++) {
        const rootZones = projectionIndex.rootZonesByTrack[trackIndex];
        if (rootZones.length > 0) {
            visibleZonesByTrack[trackIndex] = Array.from(rootZones);
        }
    }

    const visibility = new Map<number, boolean>();
    const isZoneVisible = (zoneIndex: number): boolean => {
        const cached = visibility.get(zoneIndex);
        if (cached !== undefined) return cached;

        const parentZoneIndex = projectionIndex.parentZoneIndices[zoneIndex];
        const visible = parentZoneIndex < 0
            ? source.zones.parentEventIds[zoneIndex] === 0n
            : expandedEventIds.has(source.zones.eventIds[parentZoneIndex])
                && isZoneVisible(parentZoneIndex);
        visibility.set(zoneIndex, visible);
        return visible;
    };

    const changedTrackIndices = new Set<number>();
    for (const expandedEventId of expandedEventIds) {
        const parentZoneIndex =
            projectionIndex.expandableEventZoneIndices.get(expandedEventId);
        if (parentZoneIndex === undefined
            || !isZoneVisible(parentZoneIndex)) continue;

        const childStart = projectionIndex.childOffsets[parentZoneIndex];
        const childEnd = projectionIndex.childOffsets[parentZoneIndex + 1];
        for (let childOffset = childStart;
            childOffset < childEnd; childOffset++) {
            const childZoneIndex =
                projectionIndex.childZoneIndices[childOffset];
            const trackIndex = source.zones.trackIndices[childZoneIndex];
            let visibleZones = visibleZonesByTrack[trackIndex];
            if (!visibleZones) {
                visibleZones = [];
                visibleZonesByTrack[trackIndex] = visibleZones;
            }
            visibleZones.push(childZoneIndex);
            changedTrackIndices.add(trackIndex);
        }
    }

    for (const trackIndex of changedTrackIndices) {
        visibleZonesByTrack[trackIndex]?.sort(
            (first, second) => first - second);
    }

    const AddRow = (
        groupKey: string,
        trackIndex: number,
        visibleZones: number[],
        name: string,
        tooltip: string,
        depth: number,
        hierarchy: TraceTrackNode[],
        disclosureKey = '',
        trackDisclosureKey = '',
        trackExpanded = false,
        physicalHierarchy = false,
        expansionGroupId = 0n,
        expansionMode = TrackExpansionMode.Always
    ): void => {
        const existing = groupedRows.get(groupKey);
        if (existing) {
            for (const zoneIndex of visibleZones) {
                existing.sourceZoneIndices.push(zoneIndex);
            }
            existing.requiresSort = true;
            return;
        }

        const row: ProjectedRow = {
            sourceTrackIndex: trackIndex,
            sourceZoneIndices: [...visibleZones],
            name,
            tooltip,
            depth,
            hierarchy,
            disclosureKey,
            trackDisclosureKey,
            trackExpanded,
            expansionGroupId,
            expansionMode,
            requiresSort: false,
            physicalHierarchy
        };
        groupedRows.set(groupKey, row);
        rows.push(row);
    };

    for (let trackIndex = 0; trackIndex < source.tracks.count; trackIndex++) {
        const visibleZones = visibleZonesByTrack[trackIndex];
        if (!visibleZones || visibleZones.length === 0) continue;

        const parentEventId = source.zones.parentEventIds[visibleZones[0]];
        const hierarchy = source.trackHierarchies[trackIndex];
        const smNode = hierarchy.find(node => node.kind === 5);
        const gpuNode = hierarchy.find(node => node.kind === 2);
        const streamNode = hierarchy.find(node => node.kind === 3);

        if (parentEventId === 0n && gpuNode && streamNode
            && (streamCountsByGpu.get(gpuNode.id)?.size ?? 0) > 1) {
            const trackDisclosureKey = `track:${gpuNode.id}`;
            const trackExpanded = expandedTrackIds.has(gpuNode.id);
            AddRow(
                `gpu:${gpuNode.id}`,
                trackIndex,
                visibleZones,
                gpuNode.name,
                gpuNode.name,
                0,
                hierarchy.slice(0, hierarchy.indexOf(gpuNode) + 1),
                '',
                trackDisclosureKey,
                trackExpanded,
                false,
                gpuNode.id,
                TrackExpansionMode.Collapsed);
            AddRow(
                `stream:${streamNode.id}`,
                trackIndex,
                visibleZones,
                streamNode.name,
                `${gpuNode.name} / ${streamNode.name}`,
                1,
                hierarchy,
                '',
                '',
                trackExpanded,
                false,
                gpuNode.id,
                TrackExpansionMode.Expanded);
        } else if (parentEventId !== 0n && smNode) {
            const smTooltip = hierarchy
                .slice(0, hierarchy.indexOf(smNode) + 1)
                .map(node => node.name).join(' / ');
            AddRow(
                `event:${parentEventId}:sm:${smNode.id}`,
                trackIndex,
                visibleZones,
                smNode.name,
                smTooltip,
                1,
                hierarchy,
                '',
                '',
                false,
                true);
        } else {
            AddRow(
                `source:${trackIndex}`,
                trackIndex,
                visibleZones,
                source.trackNames[trackIndex],
                source.trackTooltips[trackIndex],
                source.trackDepths[trackIndex],
                hierarchy);
        }
    }

    const originalRowOrder = new Map<ProjectedRow, number>();
    for (let rowIndex = 0; rowIndex < rows.length; rowIndex++) {
        originalRowOrder.set(rows[rowIndex], rowIndex);
    }
    const streamOrder = new Map<bigint, number>();
    for (const row of rows) {
        const streamNode = row.hierarchy.find(node => node.kind === 3);
        if (streamNode && !streamOrder.has(streamNode.id)) {
            streamOrder.set(streamNode.id, streamOrder.size);
        }
    }
    rows.sort((first, second) => {
        const firstGpu = first.hierarchy.find(node => node.kind === 2);
        const secondGpu = second.hierarchy.find(node => node.kind === 2);
        if (firstGpu && secondGpu && firstGpu.id !== secondGpu.id) {
            return firstGpu.sourceId < secondGpu.sourceId ? -1 : 1;
        }
        if (firstGpu && !secondGpu) return -1;
        if (!firstGpu && secondGpu) return 1;
        if (!firstGpu || !secondGpu) {
            return (originalRowOrder.get(first) ?? 0)
                - (originalRowOrder.get(second) ?? 0);
        }

        if (first.expansionMode === TrackExpansionMode.Collapsed) return -1;
        if (second.expansionMode === TrackExpansionMode.Collapsed) return 1;

        const firstStream = first.hierarchy.find(node => node.kind === 3);
        const secondStream = second.hierarchy.find(node => node.kind === 3);
        const streamDifference = (streamOrder.get(firstStream?.id ?? 0n) ?? 0)
            - (streamOrder.get(secondStream?.id ?? 0n) ?? 0);
        if (streamDifference !== 0) return streamDifference;
        if (first.physicalHierarchy !== second.physicalHierarchy) {
            return first.physicalHierarchy ? 1 : -1;
        }
        return (originalRowOrder.get(first) ?? 0)
            - (originalRowOrder.get(second) ?? 0);
    });

    interface ProjectedBlockLayout {
        rowIndex: number;
        zoneIndices: number[];
        sublaneIndices: number[];
        sourceTrackIndices: number[];
        formatDescId: number;
        sourceId: number;
    }

    interface PhysicalBlockGroup {
        node: TraceTrackNode;
        zonesByTrack: Map<number, number[]>;
    }

    const projectedBlocks: ProjectedBlockLayout[] = [];
    const formatDescriptors = [...source.formatDescriptors];
    const physicalBlockFormatIds = new Map<string, number>();

    const physicalBlockFormatId = (name: string): number => {
        const existing = physicalBlockFormatIds.get(name);
        if (existing !== undefined) return existing;

        const id = formatDescriptors.length;
        formatDescriptors.push({
            labelString: name,
            tooltipString: name,
            placeholderCount: 0
        });
        physicalBlockFormatIds.set(name, id);
        return id;
    };

    for (let rowIndex = 0; rowIndex < rows.length; rowIndex++) {
        const row = rows[rowIndex];

        if (!row.physicalHierarchy) {
            if (row.requiresSort) {
                row.sourceZoneIndices.sort((first, second) =>
                    source.zones.startsX[first] - source.zones.startsX[second]);
            }

            const sublaneEnds: number[] = [];
            const sublaneBySourceZone = new Map<number, number>();
            for (const sourceZoneIndex of row.sourceZoneIndices) {
                const sublane = assignIntervalSublane(
                    sublaneEnds,
                    source.zones.startsX[sourceZoneIndex],
                    source.zones.endsX[sourceZoneIndex]);
                if (sublane > 255) {
                    throw new Error('Track nesting exceeds 256 sublanes');
                }
                sublaneBySourceZone.set(sourceZoneIndex, sublane);
            }

            if (sublaneEnds.length > 1) {
                row.sourceZoneIndices.sort((first, second) =>
                    (sublaneBySourceZone.get(first) ?? 0)
                        - (sublaneBySourceZone.get(second) ?? 0)
                    || source.zones.startsX[first]
                        - source.zones.startsX[second]);
            }

            projectedBlocks.push({
                rowIndex,
                zoneIndices: row.sourceZoneIndices,
                sublaneIndices: row.sourceZoneIndices.map(
                    zoneIndex => sublaneBySourceZone.get(zoneIndex) ?? 0),
                sourceTrackIndices: new Array<number>(sublaneEnds.length)
                    .fill(row.sourceTrackIndex),
                formatDescId:
                    source.blocks.formatDescIds[row.sourceTrackIndex],
                sourceId: source.blocks.gridIds[row.sourceTrackIndex]
            });
            continue;
        }

        const blocksById = new Map<string, PhysicalBlockGroup>();
        for (const sourceZoneIndex of row.sourceZoneIndices) {
            const sourceTrackIndex =
                source.zones.trackIndices[sourceZoneIndex];
            const hierarchy = source.trackHierarchies[sourceTrackIndex];
            const blockNode = hierarchy.find(node => node.kind === 6);
            if (!blockNode) continue;

            const parentEventId =
                source.zones.parentEventIds[sourceZoneIndex];
            const blockKey = `${parentEventId}:${blockNode.id}`;
            let block = blocksById.get(blockKey);
            if (!block) {
                block = {
                    node: blockNode,
                    zonesByTrack: new Map()
                };
                blocksById.set(blockKey, block);
            }

            const trackZones = block.zonesByTrack.get(sourceTrackIndex) ?? [];
            trackZones.push(sourceZoneIndex);
            block.zonesByTrack.set(sourceTrackIndex, trackZones);
        }

        const physicalBlocks = Array.from(blocksById.values());
        physicalBlocks.sort((first, second) => {
            const firstZone = first.zonesByTrack.values().next().value?.[0];
            const secondZone = second.zonesByTrack.values().next().value?.[0];
            return source.zones.startsX[firstZone ?? 0]
                - source.zones.startsX[secondZone ?? 0];
        });

        for (const block of physicalBlocks) {
            const zoneIndices: number[] = [];
            const sublaneBySourceZone = new Map<number, number>();
            const sourceTrackIndices: number[] = [];
            const trackGroups = Array.from(block.zonesByTrack.entries());
            trackGroups.sort((first, second) => {
                const firstHierarchy = source.trackHierarchies[first[0]];
                const secondHierarchy = source.trackHierarchies[second[0]];
                return firstHierarchy[firstHierarchy.length - 1].sourceId <
                    secondHierarchy[secondHierarchy.length - 1].sourceId ? -1 : 1;
            });

            let sublaneOffset = 0;
            for (const [sourceTrackIndex, trackZones] of trackGroups) {
                trackZones.sort((first, second) =>
                    source.zones.startsX[first] - source.zones.startsX[second]);
                const sublaneEnds: number[] = [];

                for (const sourceZoneIndex of trackZones) {
                    const localSublane = assignIntervalSublane(
                        sublaneEnds,
                        source.zones.startsX[sourceZoneIndex],
                        source.zones.endsX[sourceZoneIndex]);
                    const sublane = sublaneOffset + localSublane;
                    if (sublane > 255) {
                        throw new Error('Block exceeds 256 sublanes');
                    }
                    sublaneBySourceZone.set(sourceZoneIndex, sublane);
                    zoneIndices.push(sourceZoneIndex);
                }

                for (let i = 0; i < sublaneEnds.length; i++) {
                    sourceTrackIndices.push(sourceTrackIndex);
                }
                sublaneOffset += sublaneEnds.length;
            }

            zoneIndices.sort((first, second) =>
                (sublaneBySourceZone.get(first) ?? 0)
                    - (sublaneBySourceZone.get(second) ?? 0)
                || source.zones.startsX[first] - source.zones.startsX[second]);
            projectedBlocks.push({
                rowIndex,
                zoneIndices,
                sublaneIndices: zoneIndices.map(
                    zoneIndex => sublaneBySourceZone.get(zoneIndex) ?? 0),
                sourceTrackIndices,
                formatDescId: physicalBlockFormatId(block.node.name),
                sourceId: Number(block.node.sourceId)
            });
        }
    }

    const zoneCount = projectedBlocks.reduce(
        (count, block) => count + block.zoneIndices.length, 0);
    const trackCount = projectedBlocks.reduce(
        (count, block) => count + block.sourceTrackIndices.length, 0);

    const tracks = new TracksSoA();
    tracks.count = trackCount;
    tracks.formatDescIds = new Uint16Array(tracks.count);
    tracks.sublaneIndices = new Uint8Array(tracks.count);
    tracks.paramsOffsets = new Uint32Array(tracks.count);
    tracks.paramsCounts = new Uint8Array(tracks.count);
    tracks.blockIndices = new Uint32Array(tracks.count);
    tracks.paramsPool = source.tracks.paramsPool;

    const zones = new ZonesSoA();
    zones.count = zoneCount;
    zones.eventIds = new BigUint64Array(zoneCount);
    zones.parentEventIds = new BigUint64Array(zoneCount);
    zones.hasChildren = new Uint8Array(zoneCount);
    zones.expanded = new Uint8Array(zoneCount);
    zones.disclosureKeys = new Array<string>(zoneCount);
    zones.details = new Array<string>(zoneCount);
    zones.startsX = new Float64Array(zoneCount);
    zones.endsX = new Float64Array(zoneCount);
    zones.ys = new Float32Array(zoneCount);
    zones.colors = new Uint8Array(zoneCount * 3);
    zones.formatDescIds = new Uint16Array(zoneCount);
    zones.paramsOffsets = new Uint32Array(zoneCount);
    zones.paramsCounts = new Uint8Array(zoneCount);
    zones.trackIndices = new Uint32Array(zoneCount);
    zones.smIndices = new Uint32Array(zoneCount);
    zones.blockIndices = new Uint32Array(zoneCount);
    zones.sublaneIndices = new Uint8Array(zoneCount);
    zones.paramsPool = source.zones.paramsPool;

    const blocks = new BlocksSoA();
    blocks.count = projectedBlocks.length;
    blocks.startsX = new Float64Array(blocks.count);
    blocks.endsX = new Float64Array(blocks.count);
    blocks.ys = new Float32Array(blocks.count);
    blocks.heights = new Float32Array(blocks.count);
    blocks.headerHeights = new Float32Array(blocks.count);
    blocks.sublanesCounts = new Uint8Array(blocks.count);
    blocks.sublanesMaxWidths = new Float64Array(blocks.count);
    blocks.formatDescIds = new Uint16Array(blocks.count);
    blocks.gridIds = new Uint32Array(blocks.count);
    blocks.clusterIds = new Uint32Array(blocks.count);
    blocks.smIndices = new Uint32Array(blocks.count);
    blocks.blockLaneIndices = new Uint32Array(blocks.count);
    blocks.zonesStartIndices = new Uint32Array(blocks.count);
    blocks.zonesEndIndices = new Uint32Array(blocks.count);
    blocks.tracksStartIndices = new Uint32Array(blocks.count);
    blocks.tracksEndIndices = new Uint32Array(blocks.count);

    let targetTrackIndex = 0;
    let targetZoneIndex = 0;
    for (let blockIndex = 0;
        blockIndex < projectedBlocks.length; blockIndex++) {
        const projectedBlock = projectedBlocks[blockIndex];
        const projectedRow = rows[projectedBlock.rowIndex];
        const blockTrackStart = targetTrackIndex;

        for (let sublane = 0;
            sublane < projectedBlock.sourceTrackIndices.length; sublane++) {
            const sourceTrackIndex =
                projectedBlock.sourceTrackIndices[sublane];
            tracks.formatDescIds[targetTrackIndex] =
                source.tracks.formatDescIds[sourceTrackIndex];
            tracks.sublaneIndices[targetTrackIndex] = sublane;
            tracks.paramsOffsets[targetTrackIndex] =
                source.tracks.paramsOffsets[sourceTrackIndex];
            tracks.paramsCounts[targetTrackIndex] =
                source.tracks.paramsCounts[sourceTrackIndex];
            tracks.blockIndices[targetTrackIndex] = blockIndex;
            targetTrackIndex++;
        }

        blocks.startsX[blockIndex] = Number.MAX_SAFE_INTEGER;
        blocks.headerHeights[blockIndex] = projectedRow.physicalHierarchy
            ? SUBLANE_HEIGHT : 0;
        blocks.formatDescIds[blockIndex] = projectedBlock.formatDescId;
        blocks.gridIds[blockIndex] = projectedBlock.sourceId;
        blocks.smIndices[blockIndex] = projectedBlock.rowIndex;
        blocks.zonesStartIndices[blockIndex] = targetZoneIndex;
        blocks.tracksStartIndices[blockIndex] = blockTrackStart;
        blocks.tracksEndIndices[blockIndex] = targetTrackIndex;

        for (let blockZoneIndex = 0;
            blockZoneIndex < projectedBlock.zoneIndices.length;
            blockZoneIndex++) {
            const sourceZoneIndex = projectedBlock.zoneIndices[blockZoneIndex];
            zones.eventIds[targetZoneIndex] =
                source.zones.eventIds[sourceZoneIndex];
            zones.parentEventIds[targetZoneIndex] =
                source.zones.parentEventIds[sourceZoneIndex];
            if (projectedRow.disclosureKey) {
                zones.hasChildren[targetZoneIndex] = 1;
                zones.disclosureKeys[targetZoneIndex] =
                    projectedRow.disclosureKey;
                const trackId = BigInt(
                    projectedRow.disclosureKey.slice('track:'.length));
                zones.expanded[targetZoneIndex] =
                    expandedTrackIds.has(trackId) ? 1 : 0;
            } else {
                zones.hasChildren[targetZoneIndex] =
                    source.zones.hasChildren[sourceZoneIndex];
                zones.disclosureKeys[targetZoneIndex] =
                    source.zones.disclosureKeys[sourceZoneIndex];
                zones.expanded[targetZoneIndex] = expandedEventIds.has(
                    source.zones.eventIds[sourceZoneIndex]) ? 1 : 0;
            }
            zones.details[targetZoneIndex] =
                source.zones.details[sourceZoneIndex];
            zones.startsX[targetZoneIndex] =
                source.zones.startsX[sourceZoneIndex];
            zones.endsX[targetZoneIndex] =
                source.zones.endsX[sourceZoneIndex];
            zones.formatDescIds[targetZoneIndex] =
                source.zones.formatDescIds[sourceZoneIndex];
            zones.paramsOffsets[targetZoneIndex] =
                source.zones.paramsOffsets[sourceZoneIndex];
            zones.paramsCounts[targetZoneIndex] =
                source.zones.paramsCounts[sourceZoneIndex];
            const sublane = projectedBlock.sublaneIndices[blockZoneIndex];
            zones.trackIndices[targetZoneIndex] = blockTrackStart + sublane;
            zones.smIndices[targetZoneIndex] = projectedBlock.rowIndex;
            zones.blockIndices[targetZoneIndex] = blockIndex;
            zones.sublaneIndices[targetZoneIndex] = sublane;
            zones.colors[targetZoneIndex * 3] =
                source.zones.colors[sourceZoneIndex * 3];
            zones.colors[targetZoneIndex * 3 + 1] =
                source.zones.colors[sourceZoneIndex * 3 + 1];
            zones.colors[targetZoneIndex * 3 + 2] =
                source.zones.colors[sourceZoneIndex * 3 + 2];

            const duration = zones.endsX[targetZoneIndex]
                - zones.startsX[targetZoneIndex];
            blocks.startsX[blockIndex] = Math.min(
                blocks.startsX[blockIndex], zones.startsX[targetZoneIndex]);
            blocks.endsX[blockIndex] = Math.max(
                blocks.endsX[blockIndex], zones.endsX[targetZoneIndex]);
            blocks.sublanesMaxWidths[blockIndex] = Math.max(
                blocks.sublanesMaxWidths[blockIndex], duration);
            targetZoneIndex++;
        }

        const sublaneCount = projectedBlock.sourceTrackIndices.length;
        blocks.sublanesCounts[blockIndex] = sublaneCount;
        blocks.zonesEndIndices[blockIndex] = targetZoneIndex;
        blocks.heights[blockIndex] = blocks.headerHeights[blockIndex]
            + sublaneCount * SUBLANE_HEIGHT
            + Math.max(0, sublaneCount - 1) * SUBLANE_PADDING;
    }

    return {
        kernelName: source.kernelName,
        gridDimX: source.gridDimX,
        gridDimY: source.gridDimY,
        gridDimZ: source.gridDimZ,
        clusterDimX: source.clusterDimX,
        clusterDimY: source.clusterDimY,
        clusterDimZ: source.clusterDimZ,
        formatDescriptors,
        tracks,
        zones,
        blocks,
        trackNames: rows.map(row => row.name),
        trackTooltips: rows.map(row => row.tooltip),
        trackDepths: rows.map(row => row.depth),
        trackHierarchies: rows.map(row => row.hierarchy),
        trackDisclosureKeys: rows.map(row => row.trackDisclosureKey),
        trackExpanded: rows.map(row => row.trackExpanded),
        trackExpansionGroupIds: rows.map(row => row.expansionGroupId),
        trackExpansionModes: rows.map(row => row.expansionMode),
        bookmarks: source.bookmarks
    };
}

function groupNonOverlappingBlocks(
    blockIndices: number[], blocks: BlocksSoA
): number[][] {
    const sorted = [...blockIndices].sort((first, second) =>
        blocks.startsX[first] - blocks.startsX[second]);
    const groups: Array<{ indices: number[]; end: number }> = [];

    for (const blockIndex of sorted) {
        const start = blocks.startsX[blockIndex];
        let group = groups.find(candidate => start >= candidate.end);
        if (!group) {
            group = { indices: [], end: Number.NEGATIVE_INFINITY };
            groups.push(group);
        }
        group.indices.push(blockIndex);
        group.end = blocks.endsX[blockIndex];
    }

    return groups.map(group => group.indices);
}

export function buildHierarchy(
    kernelName: string,
    gridDims: [number, number, number],
    clusterDims: [number, number, number],
    formatDescriptors: FormatDescriptor[],
    tracks: TracksSoA,
    zones: ZonesSoA,
    blocks: BlocksSoA,
    trackNames: string[] = [],
    trackDepths: number[] = [],
    bookmarks: TraceBookmark[] = []
): HierarchyData {
    const blocksByRow = new Map<number, number[]>();
    for (let blockIndex = 0; blockIndex < blocks.count; blockIndex++) {
        const rowIndex = blocks.smIndices[blockIndex];
        const rowBlocks = blocksByRow.get(rowIndex) ?? [];
        rowBlocks.push(blockIndex);
        blocksByRow.set(rowIndex, rowBlocks);
    }

    const blockLaneGroups: Array<{
        rowIndex: number;
        blockIndices: number[];
    }> = [];
    const blockLanesByRow: number[][] = new Array(trackNames.length);
    for (let rowIndex = 0; rowIndex < trackNames.length; rowIndex++) {
        blockLanesByRow[rowIndex] = [];
        const rowBlocks = blocksByRow.get(rowIndex) ?? [];
        for (const blockIndices of groupNonOverlappingBlocks(rowBlocks, blocks)) {
            blockLanesByRow[rowIndex].push(blockLaneGroups.length);
            blockLaneGroups.push({ rowIndex, blockIndices });
        }
    }

    const blockLanes = new BlockLanesSoA();
    blockLanes.count = blockLaneGroups.length;
    blockLanes.ys = new Float32Array(blockLanes.count);
    blockLanes.heights = new Float32Array(blockLanes.count);
    blockLanes.startsX = new Float64Array(blockLanes.count);
    blockLanes.widths = new Float64Array(blockLanes.count);
    blockLanes.maxBlockWidths = new Float64Array(blockLanes.count);
    blockLanes.maxZoneWidths = new Float64Array(blockLanes.count);
    blockLanes.smIndices = new Uint32Array(blockLanes.count);
    blockLanes.blockIndices = new Uint32Array(blocks.count);
    blockLanes.blockIndicesOffsets = new Uint32Array(blockLanes.count);
    blockLanes.blockIndicesCounts = new Uint16Array(blockLanes.count);

    let blockReferenceIndex = 0;
    for (let blockLaneIndex = 0;
        blockLaneIndex < blockLaneGroups.length; blockLaneIndex++) {
        const group = blockLaneGroups[blockLaneIndex];
        blockLanes.smIndices[blockLaneIndex] = group.rowIndex;
        blockLanes.blockIndicesOffsets[blockLaneIndex] = blockReferenceIndex;
        blockLanes.blockIndicesCounts[blockLaneIndex] = group.blockIndices.length;
        blockLanes.startsX[blockLaneIndex] = Number.MAX_SAFE_INTEGER;

        for (const blockIndex of group.blockIndices) {
            blockLanes.blockIndices[blockReferenceIndex++] = blockIndex;
            blocks.blockLaneIndices[blockIndex] = blockLaneIndex;
            blockLanes.heights[blockLaneIndex] = Math.max(
                blockLanes.heights[blockLaneIndex], blocks.heights[blockIndex]);
            blockLanes.startsX[blockLaneIndex] = Math.min(
                blockLanes.startsX[blockLaneIndex], blocks.startsX[blockIndex]);
            blockLanes.widths[blockLaneIndex] = Math.max(
                blockLanes.widths[blockLaneIndex], blocks.endsX[blockIndex]);
            blockLanes.maxBlockWidths[blockLaneIndex] = Math.max(
                blockLanes.maxBlockWidths[blockLaneIndex],
                blocks.endsX[blockIndex] - blocks.startsX[blockIndex]);
            blockLanes.maxZoneWidths[blockLaneIndex] = Math.max(
                blockLanes.maxZoneWidths[blockLaneIndex],
                blocks.sublanesMaxWidths[blockIndex]);
        }
    }

    const lanes = new LanesSoA();
    lanes.count = trackNames.length;
    lanes.smIndices = new Uint32Array(lanes.count);
    lanes.depths = new Uint8Array(lanes.count);
    lanes.ys = new Float32Array(lanes.count);
    lanes.heights = new Float32Array(lanes.count);
    lanes.startsX = new Float64Array(lanes.count);
    lanes.widths = new Float64Array(lanes.count);
    lanes.blockLanesStartIndices = new Uint32Array(lanes.count);
    lanes.blockLanesEndIndices = new Uint32Array(lanes.count);
    lanes.names = trackNames;

    let totalDurationNs = 0;
    for (let rowIndex = 0; rowIndex < lanes.count; rowIndex++) {
        const rowBlockLanes = blockLanesByRow[rowIndex];
        lanes.smIndices[rowIndex] = rowIndex;
        lanes.depths[rowIndex] = trackDepths[rowIndex] ?? 0;
        lanes.startsX[rowIndex] = Number.MAX_SAFE_INTEGER;
        lanes.blockLanesStartIndices[rowIndex] = rowBlockLanes[0] ?? 0;
        lanes.blockLanesEndIndices[rowIndex] = rowBlockLanes.length === 0
            ? lanes.blockLanesStartIndices[rowIndex]
            : rowBlockLanes[rowBlockLanes.length - 1] + 1;
        let height = 2 * LANE_EDGE_PADDING;

        for (let i = 0; i < rowBlockLanes.length; i++) {
            const blockLaneIndex = rowBlockLanes[i];
            height += blockLanes.heights[blockLaneIndex];
            if (i + 1 < rowBlockLanes.length) height += BLOCK_LANE_PADDING;
            lanes.startsX[rowIndex] = Math.min(
                lanes.startsX[rowIndex], blockLanes.startsX[blockLaneIndex]);
            lanes.widths[rowIndex] = Math.max(
                lanes.widths[rowIndex], blockLanes.widths[blockLaneIndex]);
        }

        if (lanes.startsX[rowIndex] === Number.MAX_SAFE_INTEGER) {
            lanes.startsX[rowIndex] = 0;
        }
        lanes.heights[rowIndex] = height;
        totalDurationNs = Math.max(totalDurationNs, lanes.widths[rowIndex]);
    }
    for (const bookmark of bookmarks) {
        totalDurationNs = Math.max(totalDurationNs, bookmark.timestampNs);
    }

    let currentY = 0;
    for (let rowIndex = lanes.count - 1; rowIndex >= 0; rowIndex--) {
        lanes.ys[rowIndex] = currentY;
        let blockLaneY = currentY + LANE_EDGE_PADDING;
        const blockLaneStart = lanes.blockLanesStartIndices[rowIndex];
        const blockLaneEnd = lanes.blockLanesEndIndices[rowIndex];

        for (let blockLaneIndex = blockLaneStart;
            blockLaneIndex < blockLaneEnd; blockLaneIndex++) {
            blockLanes.ys[blockLaneIndex] = blockLaneY;
            const offset = blockLanes.blockIndicesOffsets[blockLaneIndex];
            const count = blockLanes.blockIndicesCounts[blockLaneIndex];

            for (let i = 0; i < count; i++) {
                const blockIndex = blockLanes.blockIndices[offset + i];
                blocks.ys[blockIndex] = blockLaneY
                    + blockLanes.heights[blockLaneIndex]
                    - blocks.heights[blockIndex];

                for (let zoneIndex = blocks.zonesStartIndices[blockIndex];
                    zoneIndex < blocks.zonesEndIndices[blockIndex]; zoneIndex++) {
                    zones.ys[zoneIndex] = blocks.ys[blockIndex]
                        + blocks.heights[blockIndex]
                        - blocks.headerHeights[blockIndex]
                        - SUBLANE_HEIGHT / 2
                        - zones.sublaneIndices[zoneIndex]
                            * (SUBLANE_HEIGHT + SUBLANE_PADDING);
                }
            }

            blockLaneY += blockLanes.heights[blockLaneIndex];
            if (blockLaneIndex + 1 < blockLaneEnd) {
                blockLaneY += BLOCK_LANE_PADDING;
            }
        }

        const rowPadding = LANE_PADDING + BLOCK_LANE_PADDING;
        const startsExpandedGroup = rowIndex > 0
            && (trackDepths[rowIndex] ?? 0) === 0
            && (trackDepths[rowIndex - 1] ?? 0) > 0;
        currentY += lanes.heights[rowIndex] + rowPadding;
        if (startsExpandedGroup) {
            currentY += rowPadding * 2;
        }
    }

    for (let i = 0; i < lanes.count; i++) {
        if (lanes.depths[i] === 0) {
            lanes.startsX[i] = 0;
            lanes.widths[i] = totalDurationNs;
        }
    }

    return {
        tracks,
        zones,
        blocks,
        blockLanes,
        lanes,
        smAccelerator: new SMAccelerator(lanes),
        worldHeight: currentY,
        totalDurationNs,
        formatDescriptors,
        kernelName,
        gridDims,
        clusterDims,
        bookmarks
    };
}

export function formatString(
    formatDescriptors: FormatDescriptor[], formatDescId: number, params: number[]
): string {
    let result = formatDescriptors[formatDescId]?.labelString ?? 'Event';
    params.forEach((value, index) => {
        result = result.replace(`{${index}}`, value.toString());
    });
    return result;
}

export function formatTooltipString(
    formatDescriptors: FormatDescriptor[], formatDescId: number, params: number[]
): string {
    let result = formatDescriptors[formatDescId]?.tooltipString ?? 'Event';
    params.forEach((value, index) => {
        result = result.replace(`{${index}}`, value.toString());
    });
    return result;
}

export function formatTrackString(
    formatDescriptors: FormatDescriptor[], formatDescId: number,
    laneId: number, params: number[]
): string {
    return formatString(formatDescriptors, formatDescId, params)
        .replace('{lane}', laneId.toString());
}

export function formatTrackTooltipString(
    formatDescriptors: FormatDescriptor[], formatDescId: number,
    laneId: number, params: number[]
): string {
    return formatTooltipString(formatDescriptors, formatDescId, params)
        .replace('{lane}', laneId.toString());
}

export function formatBlockString(
    formatDescriptors: FormatDescriptor[], formatDescId: number,
    blockId: number, clusterId: number
): string {
    return formatString(formatDescriptors, formatDescId, [])
        .replace('{blockLinear}', blockId.toString())
        .replace('{clusterLinear}', clusterId.toString());
}

export function formatBlockTooltipString(
    formatDescriptors: FormatDescriptor[], formatDescId: number,
    blockId: number, clusterId: number
): string {
    return formatTooltipString(formatDescriptors, formatDescId, [])
        .replace('{blockLinear}', blockId.toString())
        .replace('{clusterLinear}', clusterId.toString());
}
