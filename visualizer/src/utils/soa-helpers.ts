/**
 * Helper functions for working with Structure of Arrays (SoA) data structures.
 * Provides convenient accessors and utilities for the TypedArray-based memory layout.
 */

import { ZonesSoA, BlocksSoA } from './types';

// ============================================================================
// Constants
// ============================================================================

/** Conversion factor: nanoseconds to milliseconds */
export const NS_TO_MS = 1e-6;

/** Conversion factor: milliseconds to nanoseconds */
export const MS_TO_NS = 1e6;

// ============================================================================
// Direct Pool Access Helpers
// ============================================================================

/**
 * Format a zone's label string using its params pool.
 * Consumers should call formatString() directly with pool access:
 *
 * const offset = zones.paramsOffsets[idx];
 * const count = zones.paramsCounts[idx];
 * for (let i = 0; i < count; i++) {
 *   const param = zones.paramsPool[offset + i];
 *   // use param...
 * }
 *
 * DO NOT create temporary arrays - work directly with pools!
 */

// ============================================================================
// Bounds Helpers
// ============================================================================

/**
 * Get zone width (duration) in nanoseconds.
 * Computed on-the-fly from start/end times.
 */
export function getZoneWidth(zones: ZonesSoA, idx: number): number {
    return zones.endsX[idx] - zones.startsX[idx];
}

/**
 * Get block width (duration) in nanoseconds.
 * Computed on-the-fly from start/end times.
 */
export function getBlockWidth(blocks: BlocksSoA, idx: number): number {
    return blocks.endsX[idx] - blocks.startsX[idx];
}

/**
 * Get zone center X position in nanoseconds.
 * Computed on-the-fly (not cached in SoA).
 */
export function getZoneCenterX(zones: ZonesSoA, idx: number): number {
    return (zones.startsX[idx] + zones.endsX[idx]) / 2;
}

/**
 * Get block center X position in nanoseconds.
 * Computed on-the-fly (not cached in SoA).
 */
export function getBlockCenterX(blocks: BlocksSoA, idx: number): number {
    return (blocks.startsX[idx] + blocks.endsX[idx]) / 2;
}

// ============================================================================
// Color Unpacking
// ============================================================================

/**
 * Get zone color as [r, g, b] tuple (0-255 range).
 * Colors are packed as bytes in the SoA for memory efficiency.
 */
export function getZoneColor(zones: ZonesSoA, idx: number): [number, number, number] {
    return [
        zones.colors[idx * 3 + 0],
        zones.colors[idx * 3 + 1],
        zones.colors[idx * 3 + 2]
    ];
}

// ============================================================================
// Block Lane Indirection Access
// ============================================================================

/**
 * Iterate blocks in a block lane using indirection.
 * Consumers should access the indirection array directly:
 *
 * const offset = blockLanes.blockIndicesOffsets[blIdx];
 * const count = blockLanes.blockIndicesCounts[blIdx];
 * for (let i = 0; i < count; i++) {
 *   const blockIdx = blockLanes.blockIndices[offset + i];
 *   // use blocks.startsX[blockIdx], etc...
 * }
 *
 * DO NOT create temporary arrays - use direct indexing!
 */

// ============================================================================
// Binary Search
// ============================================================================

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
        const blockIdx = blockIndices[offset + mid];  // INDIRECTION

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

// ============================================================================
// Time Formatting
// ============================================================================

/**
 * Format nanoseconds as human-readable time string.
 * Automatically chooses appropriate units (s/ms/μs/ns).
 */
export function formatTime(ns: number): string {
    if (ns >= 1e9) return `${(ns / 1e9).toFixed(2)} s`;
    if (ns >= 1e6) return `${(ns / 1e6).toFixed(2)} ms`;
    if (ns >= 1e3) return `${(ns / 1e3).toFixed(2)} μs`;
    return `${ns.toFixed(0)} ns`;
}

/**
 * Format nanoseconds as locale string with units.
 * Uses toLocaleString() for thousands separators.
 */
export function formatTimeLocale(ns: number): string {
    if (ns >= 1e9) return `${(ns / 1e9).toLocaleString(undefined, { maximumFractionDigits: 2 })} s`;
    if (ns >= 1e6) return `${(ns / 1e6).toLocaleString(undefined, { maximumFractionDigits: 2 })} ms`;
    if (ns >= 1e3) return `${(ns / 1e3).toLocaleString(undefined, { maximumFractionDigits: 2 })} μs`;
    return `${ns.toLocaleString()} ns`;
}
