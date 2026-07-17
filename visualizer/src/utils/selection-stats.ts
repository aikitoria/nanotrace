import { ZonesSoA } from './types.js';

export interface ZoneStatistics {
    eventSpecId: number;
    displayFormatDescId: number;
    count: number;
    totalNs: number;
    averageNs: number;
}

export interface SelectionStatistics {
    cpu: ZoneStatistics[];
    gpu: ZoneStatistics[];
}

interface DurationBucket {
    eventSpecId: number;
    displayFormatDescId: number;
    count: number;
    totalNs: number;
}

function SummarizeBuckets(buckets: Map<number, DurationBucket>): ZoneStatistics[] {
    const statistics: ZoneStatistics[] = [];

    for (const bucket of buckets.values()) {
        statistics.push({
            eventSpecId: bucket.eventSpecId,
            displayFormatDescId: bucket.displayFormatDescId,
            count: bucket.count,
            totalNs: bucket.totalNs,
            averageNs: bucket.totalNs / bucket.count
        });
    }

    statistics.sort((first, second) =>
        second.totalNs - first.totalNs
        || second.count - first.count
        || first.eventSpecId - second.eventSpecId);
    return statistics;
}

/** Aggregates zones whose time span and rendered sublane center lie inside the selection. */
export function AggregateSelectionStatistics(
    zones: ZonesSoA,
    zoneVisibility: Uint32Array,
    rowSelected: Uint8Array,
    rowOffsets: Float32Array,
    rowIsGpu: Uint8Array,
    selectionStartNs: number,
    selectionEndNs: number,
    selectionBottom: number,
    selectionTop: number
): SelectionStatistics {
    const cpuBuckets = new Map<number, DurationBucket>();
    const gpuBuckets = new Map<number, DurationBucket>();

    for (let zoneIndex = 0; zoneIndex < zones.count; zoneIndex++) {
        const rowIndex = zones.smIndices[zoneIndex];
        const zoneY = zones.ys[zoneIndex] + rowOffsets[rowIndex];
        if (zoneVisibility[zoneIndex] === 0 || rowSelected[rowIndex] === 0
            || zones.startsX[zoneIndex] < selectionStartNs
            || zones.endsX[zoneIndex] > selectionEndNs
            || zoneY < selectionBottom || zoneY > selectionTop) {
            continue;
        }

        const eventSpecId = zones.eventSpecIds[zoneIndex];
        const buckets = rowIsGpu[rowIndex] !== 0 ? gpuBuckets : cpuBuckets;
        let bucket = buckets.get(eventSpecId);
        if (!bucket) {
            bucket = {
                eventSpecId,
                displayFormatDescId: zones.formatDescIds[zoneIndex],
                count: 0,
                totalNs: 0
            };
            buckets.set(eventSpecId, bucket);
        }

        const durationNs = zones.endsX[zoneIndex] - zones.startsX[zoneIndex];
        bucket.count++;
        bucket.totalNs += durationNs;
    }

    return {
        cpu: SummarizeBuckets(cpuBuckets),
        gpu: SummarizeBuckets(gpuBuckets)
    };
}
