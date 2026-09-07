import { ZonesSoA, BlocksSoA } from './types.js';

export { NS_TO_MS, MS_TO_NS } from './constants.js';

/**
 * Binary search for a block using indirection array.
 * Searches blocks within a block lane for given time and Y position.
 * Returns block index in BlocksSoA, or -1 if not found.
 */
export function binarySearchBlocksIndirect(
    blocks: BlocksSoA,
    blockIndices: Uint32Array,  // Indirection array (from BlockLanesSoA)
    offset: number,             // Start offset in indirection array
    count: number,              // Number of blocks to search
    timeNs: number,             // Time in nanoseconds
    y: number                   // Y position in world space
): number {
    let left = 0;
    let right = count - 1;

    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        const blockIdx = blockIndices[offset + mid];

        if (timeNs >= blocks.startsX[blockIdx] &&
            timeNs < blocks.endsX[blockIdx] &&
            y >= blocks.ys[blockIdx] &&
            y < blocks.ys[blockIdx] + blocks.heights[blockIdx]) {
            return blockIdx;
        }

        if (timeNs < blocks.startsX[blockIdx]) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    return -1;
}

/**
 * Binary search for a zone within a block and sublane.
 * Zones are sorted by (blockIdx, sublaneIdx, startX).
 * Returns zone index in ZonesSoA, or -1 if not found.
 */
export function binarySearchZones(
    zones: ZonesSoA,
    start: number,              // Start index in ZonesSoA
    end: number,                // End index in ZonesSoA (exclusive)
    timeNs: number,             // Time in nanoseconds
    sublaneIdx: number          // Sublane to filter by
): number {
    let left = start;
    let right = end - 1;

    while (left <= right) {
        const mid = Math.floor((left + right) / 2);

        // Filter by sublane (zones sorted by blockIdx, sublaneIdx, startX)
        if (zones.sublaneIndices[mid] !== sublaneIdx) {
            if (zones.sublaneIndices[mid] < sublaneIdx) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
            continue;
        }

        if (timeNs >= zones.startsX[mid] && timeNs < zones.endsX[mid]) {
            return mid;
        }

        if (timeNs < zones.startsX[mid]) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    return -1;
}

const durationFormatter = new Intl.NumberFormat(undefined, {
    maximumFractionDigits: 3
});

/** Formats a time span using the largest practical unit. */
export function formatDuration(ns: number): string {
    const magnitude = Math.abs(ns);
    if (magnitude >= 1e9) {
        return `${durationFormatter.format(ns / 1e9)} s`;
    }
    if (magnitude >= 1e6) {
        return `${durationFormatter.format(ns / 1e6)} ms`;
    }
    if (magnitude >= 1e3) {
        return `${durationFormatter.format(ns / 1e3)} μs`;
    }
    return `${durationFormatter.format(ns)} ns`;
}
