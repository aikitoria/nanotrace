#!/usr/bin/env tsx
/**
 * Validate a .nanotrace file by parsing it and printing its structure.
 *
 * Usage:
 *   npm run validate <file.nanotrace>
 */

import * as fs from 'fs';
import * as zlib from 'zlib';

interface FormatDescriptor {
    formatString: string;
    paramCount: number;
}

function validateNanotrace(filename: string): boolean {
    console.log(`Validating ${filename}...`);

    const fileBuffer = fs.readFileSync(filename);
    let buffer = fileBuffer;
    let offset = 0;

    // Helper to read string
    const readString = (): string => {
        const length = buffer.readUInt16LE(offset);
        offset += 2;
        const str = buffer.toString('utf-8', offset, offset + length);
        offset += length;
        return str;
    };

    // Read magic number
    const magic = buffer.toString('binary', 0, 9);
    if (magic !== 'nanotrace') {
        console.log(`ERROR: Invalid magic number: ${magic}`);
        return false;
    }
    console.log(`✓ Magic number: nanotrace`);
    offset = 10;

    // Read format version
    const version = buffer.readUInt8(offset);
    offset += 1;
    console.log(`✓ Format version: ${version}`);

    // Read compression mode
    const compressionMode = buffer.readUInt8(offset);
    offset += 1;
    console.log(`✓ Compression mode: ${compressionMode === 1 ? 'deflate' : 'none'}`);

    // Decompress if needed
    if (compressionMode === 1) {
        const compressedData = buffer.subarray(offset);
        buffer = zlib.inflateSync(compressedData);
        offset = 0;
        console.log(`✓ Decompressed ${compressedData.length} → ${buffer.length} bytes`);
    }

    // Read kernel name
    const kernelName = readString();
    console.log(`✓ Kernel name: ${kernelName}`);

    // Read grid dimensions
    const gridDimX = buffer.readUInt32LE(offset); offset += 4;
    const gridDimY = buffer.readUInt32LE(offset); offset += 4;
    const gridDimZ = buffer.readUInt32LE(offset); offset += 4;
    console.log(`✓ Grid dimensions: (${gridDimX}, ${gridDimY}, ${gridDimZ})`);

    // Read cluster dimensions
    const clusterDimX = buffer.readUInt32LE(offset); offset += 4;
    const clusterDimY = buffer.readUInt32LE(offset); offset += 4;
    const clusterDimZ = buffer.readUInt32LE(offset); offset += 4;
    const usingClusters = clusterDimX > 0 || clusterDimY > 0 || clusterDimZ > 0;
    console.log(`✓ Cluster dimensions: (${clusterDimX}, ${clusterDimY}, ${clusterDimZ})${usingClusters ? ' [ENABLED]' : ''}`);

    // Read counts
    const formatDescCount = buffer.readUInt32LE(offset); offset += 4;
    const blockDescCount = buffer.readUInt32LE(offset); offset += 4;
    const trackCount = buffer.readUInt32LE(offset); offset += 4;
    const totalEventCount = Number(buffer.readBigUInt64LE(offset)); offset += 8;

    console.log(`✓ Format descriptors: ${formatDescCount}`);
    console.log(`✓ Block descriptors: ${blockDescCount}`);
    console.log(`✓ Event tracks: ${trackCount}`);
    console.log(`✓ Total events: ${totalEventCount}`);

    // Read format descriptors
    console.log('\nFormat Descriptors:');
    const formatDescriptors: FormatDescriptor[] = [];
    for (let i = 0; i < formatDescCount; i++) {
        const formatString = readString();
        const paramCount = buffer.readUInt8(offset); offset += 1;
        formatDescriptors.push({ formatString, paramCount });
        console.log(`  [${i}] ${formatString} (${paramCount} params)`);
    }

    // Read block descriptors
    console.log(`\nBlock Descriptors: (${blockDescCount} total)`);
    const smIds = new Set<number>();
    for (let i = 0; i < Math.min(5, blockDescCount); i++) {
        const smId = buffer.readUInt16LE(offset); offset += 2;
        const formatId = buffer.readUInt16LE(offset); offset += 2;
        smIds.add(smId);

        const paramCount = formatDescriptors[formatId].paramCount;
        const params: number[] = [];
        for (let j = 0; j < paramCount; j++) {
            params.push(buffer.readUInt32LE(offset)); offset += 4;
        }

        const fmtStr = formatDescriptors[formatId].formatString;
        console.log(`  [${i}] SM ${smId}: ${fmtStr} with params [${params.join(', ')}]`);
    }

    // Skip remaining block descriptors
    for (let i = 5; i < blockDescCount; i++) {
        const smId = buffer.readUInt16LE(offset); offset += 2;
        const formatId = buffer.readUInt16LE(offset); offset += 2;
        smIds.add(smId);
        const paramCount = formatDescriptors[formatId].paramCount;
        offset += paramCount * 4; // Skip params
    }

    if (blockDescCount > 5) {
        console.log(`  ... (${blockDescCount - 5} more)`);
    }
    console.log(`  Unique SMs: [${Array.from(smIds).sort((a, b) => a - b).join(', ')}]`);

    // Read event tracks
    console.log(`\nEvent Tracks: (${trackCount} total)`);
    let totalEventsRead = 0;

    for (let i = 0; i < Math.min(3, trackCount); i++) {
        const blockDescId = buffer.readUInt32LE(offset); offset += 4;
        const formatId = buffer.readUInt16LE(offset); offset += 2;

        // Read track params
        const trackParamCount = formatDescriptors[formatId].paramCount;
        const trackParams: number[] = [];
        for (let j = 0; j < trackParamCount; j++) {
            trackParams.push(buffer.readUInt32LE(offset)); offset += 4;
        }

        const eventCount = buffer.readUInt32LE(offset); offset += 4;
        totalEventsRead += eventCount;

        const trackFmt = formatDescriptors[formatId].formatString;
        console.log(`  [${i}] Block ${blockDescId}, ${trackFmt} with params [${trackParams.join(', ')}]: ${eventCount} events`);

        // Read events
        for (let j = 0; j < Math.min(2, eventCount); j++) {
            const timeOffset = buffer.readUInt32LE(offset); offset += 4;
            const duration = buffer.readUInt32LE(offset); offset += 4;
            const eventFormatId = buffer.readUInt16LE(offset); offset += 2;

            const eventParamCount = formatDescriptors[eventFormatId].paramCount;
            const eventParams: number[] = [];
            for (let k = 0; k < eventParamCount; k++) {
                eventParams.push(buffer.readUInt32LE(offset)); offset += 4;
            }

            const eventFmt = formatDescriptors[eventFormatId].formatString;
            console.log(`    Event ${j}: t=${timeOffset}ns, dur=${duration}ns, ${eventFmt} with params [${eventParams.join(', ')}]`);
        }

        // Skip remaining events in this track
        for (let j = 2; j < eventCount; j++) {
            offset += 8; // time + duration
            const eventFormatId = buffer.readUInt16LE(offset); offset += 2;
            const eventParamCount = formatDescriptors[eventFormatId].paramCount;
            offset += eventParamCount * 4; // params
        }

        if (eventCount > 2) {
            console.log(`    ... (${eventCount - 2} more events)`);
        }
    }

    // Skip remaining tracks
    for (let i = 3; i < trackCount; i++) {
        offset += 4; // block desc id
        const formatId = buffer.readUInt16LE(offset); offset += 2;
        const trackParamCount = formatDescriptors[formatId].paramCount;
        offset += trackParamCount * 4;
        const eventCount = buffer.readUInt32LE(offset); offset += 4;
        totalEventsRead += eventCount;

        for (let j = 0; j < eventCount; j++) {
            offset += 8; // time + duration
            const eventFormatId = buffer.readUInt16LE(offset); offset += 2;
            const eventParamCount = formatDescriptors[eventFormatId].paramCount;
            offset += eventParamCount * 4;
        }
    }

    if (trackCount > 3) {
        console.log(`  ... (${trackCount - 3} more)`);
    }

    // Check if we read all events
    if (totalEventsRead === totalEventCount) {
        console.log(`\n✓ All ${totalEventCount} events accounted for`);
    } else {
        console.log(`\n✗ Event count mismatch: expected ${totalEventCount}, read ${totalEventsRead}`);
        return false;
    }

    // Check if we're at end of buffer
    const remaining = buffer.length - offset;
    if (remaining === 0) {
        console.log('✓ Reached end of file cleanly');
    } else {
        console.log(`✗ ${remaining} unexpected bytes at end of file`);
        return false;
    }

    console.log('\n✅ File is valid!');
    return true;
}

const filename = process.argv[2];
if (!filename) {
    console.error('Usage: tsx validate.ts <trace_file.nanotrace>');
    process.exit(1);
}

const success = validateNanotrace(filename);
process.exit(success ? 0 : 1);
