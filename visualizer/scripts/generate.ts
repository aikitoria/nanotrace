#!/usr/bin/env tsx
/**
 * Generate .nanotrace files for testing and samples.
 *
 * Usage:
 *   npm run generate:minimal
 *   npm run generate:small
 *   npm run generate:large
 */

import * as fs from 'fs';
import * as path from 'path';
import * as zlib from 'zlib';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Ensure public directory exists
const publicDir = path.join(__dirname, '../public');
if (!fs.existsSync(publicDir)) {
    fs.mkdirSync(publicDir, { recursive: true });
}

class BinaryWriter {
    private buffer: Buffer;
    private offset: number;

    constructor(initialSize = 1024 * 1024) {
        this.buffer = Buffer.allocUnsafe(initialSize);
        this.offset = 0;
    }

    private ensureCapacity(bytes: number) {
        if (this.offset + bytes > this.buffer.length) {
            const newBuffer = Buffer.allocUnsafe(Math.max(this.buffer.length * 2, this.offset + bytes));
            this.buffer.copy(newBuffer);
            this.buffer = newBuffer;
        }
    }

    writeUint8(value: number) {
        this.ensureCapacity(1);
        this.buffer.writeUInt8(value, this.offset);
        this.offset += 1;
    }

    writeUint16(value: number) {
        this.ensureCapacity(2);
        this.buffer.writeUInt16LE(value, this.offset);
        this.offset += 2;
    }

    writeUint32(value: number) {
        this.ensureCapacity(4);
        this.buffer.writeUInt32LE(value, this.offset);
        this.offset += 4;
    }

    writeUint64(value: bigint) {
        this.ensureCapacity(8);
        this.buffer.writeBigUInt64LE(value, this.offset);
        this.offset += 8;
    }

    writeString(str: string) {
        const encoded = Buffer.from(str, 'utf-8');
        this.writeUint16(encoded.length);
        this.ensureCapacity(encoded.length);
        encoded.copy(this.buffer, this.offset);
        this.offset += encoded.length;
    }

    getBuffer(): Buffer {
        return this.buffer.subarray(0, this.offset);
    }

    reset() {
        this.offset = 0;
    }
}

interface BlockDescriptor {
    blockId: number;
    clusterId: number;
    smId: number;
    formatDescId: number;
}

interface Event {
    time: number;
    duration: number;
    formatDescId: number;
    params: number[];
}

interface Track {
    blockIdx: number;
    trackId: number;
    events: Event[];
}

function seededRandom(seed: number): () => number {
    let state = seed;
    return () => {
        state = (state * 1664525 + 1013904223) | 0;
        return (state >>> 0) / 4294967296;
    };
}

function randomInt(rand: () => number, min: number, max: number): number {
    return Math.floor(rand() * (max - min + 1)) + min;
}

function randomChoice<T>(rand: () => number, arr: T[]): T {
    return arr[Math.floor(rand() * arr.length)];
}

function weightedChoice(rand: () => number, choices: number[], weights: number[]): number {
    const total = weights.reduce((a, b) => a + b, 0);
    let r = rand() * total;
    for (let i = 0; i < choices.length; i++) {
        r -= weights[i];
        if (r <= 0) return choices[i];
    }
    return choices[choices.length - 1];
}

function generateMinimal() {
    const outputPath = path.join(__dirname, '../public/minimal.nanotrace');

    console.log(`Generating ${outputPath}...`);
    console.log(`  Blocks: 1`);
    console.log(`  Tracks: 1`);
    console.log(`  Events: 2`);

    const writer = new BinaryWriter(1024);

    // Magic + version + compression
    writer.buffer.write('nanotrace\x00', 0, 10, 'binary');
    writer.offset = 10;
    writer.writeUint8(1);  // version
    writer.writeUint8(0);  // no compression

    // Kernel name
    writer.writeString('MinimalKernel');
    // Grid dimensions
    writer.writeUint32(1);  // grid X
    writer.writeUint32(1);  // grid Y
    writer.writeUint32(1);  // grid Z
    // Cluster dimensions
    writer.writeUint32(0);  // cluster X
    writer.writeUint32(0);  // cluster Y
    writer.writeUint32(0);  // cluster Z
    writer.writeUint32(3);  // format descriptor count
    writer.writeUint32(1);  // block descriptor count
    writer.writeUint32(1);  // track count
    writer.writeUint64(2n); // total event count

    // Format descriptors (label + tooltip + param count)
    writer.writeString('Block {blockLinear}');  // label
    writer.writeString('Block {blockLinear} on SM');  // tooltip
    writer.writeUint8(0);  // param count
    writer.writeString('Track {lane}');  // label
    writer.writeString('Track {lane}');  // tooltip
    writer.writeUint8(0);  // param count
    writer.writeString('Event {0}');  // label
    writer.writeString('Event {0}');  // tooltip
    writer.writeUint8(1);  // param count

    // Block descriptor
    writer.writeUint32(0);  // block ID
    writer.writeUint32(0);  // cluster ID
    writer.writeUint16(0);  // SM ID
    writer.writeUint16(0);  // format desc ID

    // Track
    writer.writeUint32(0);  // block descriptor ID
    writer.writeUint16(1);  // format descriptor ID
    writer.writeUint32(0);  // lane ID (for {lane} placeholder)
    // No track parameters (format has 0 params)
    writer.writeUint32(2);  // event count

    // Events
    writer.writeUint32(0);
    writer.writeUint32(1000);
    writer.writeUint16(2);
    writer.writeUint32(0);

    writer.writeUint32(1000);
    writer.writeUint32(1000);
    writer.writeUint16(2);
    writer.writeUint32(1);

    fs.writeFileSync(outputPath, writer.getBuffer());
    console.log(`\nGenerated ${outputPath} (${writer.offset} bytes)\n`);
}

function generateRandom(small: boolean) {
    const outputPath = path.join(__dirname, small ? '../public/random_small.nanotrace' : '../public/random.nanotrace');

    const config = small ? {
        numLanes: 16,
        blockLanesPerLaneMin: 1,
        blockLanesPerLaneMax: 3,
        blocksPerBlockLaneMin: 20,
        blocksPerBlockLaneMax: 50,
        tracksPerBlockMin: 3,
        tracksPerBlockMax: 6,
        eventsPerTrackMin: 8,
        eventsPerTrackMax: 15,
    } : {
        numLanes: 148,
        blockLanesPerLaneMin: 1,
        blockLanesPerLaneMax: 4,
        blocksPerBlockLaneMin: 100,
        blocksPerBlockLaneMax: 500,
        tracksPerBlockMin: 4,
        tracksPerBlockMax: 8,
        eventsPerTrackMin: 10,
        eventsPerTrackMax: 20,
    };

    const rand = seededRandom(42);
    const blockDescriptors: BlockDescriptor[] = [];
    const tracks: Track[] = [];
    let totalEvents = 0;

    const ZONE_MIN_DURATION = 100;
    const ZONE_MAX_DURATION = 3000;
    const ZONE_GAP = 10;
    const BLOCK_PADDING = 50;

    for (let laneIdx = 0; laneIdx < config.numLanes; laneIdx++) {
        const numBlockLanes = randomInt(rand, config.blockLanesPerLaneMin, config.blockLanesPerLaneMax);

        for (let blIdx = 0; blIdx < numBlockLanes; blIdx++) {
            const numBlocks = randomInt(rand, config.blocksPerBlockLaneMin, config.blocksPerBlockLaneMax);
            let blockLaneTime = 0;

            for (let blockNum = 0; blockNum < numBlocks; blockNum++) {
                const numTracks = randomInt(rand, config.tracksPerBlockMin, config.tracksPerBlockMax);
                const blockStartTime = blockLaneTime;
                let blockEndTime = blockLaneTime;

                const blockIdx = blockDescriptors.length;
                blockDescriptors.push({
                    blockId: blockIdx,
                    clusterId: 0,
                    smId: laneIdx,
                    formatDescId: 0
                });

                for (let trackId = 0; trackId < numTracks; trackId++) {
                    const numEvents = randomInt(rand, config.eventsPerTrackMin, config.eventsPerTrackMax);
                    const events: Event[] = [];
                    let currentTime = blockStartTime;

                    for (let eventIdx = 0; eventIdx < numEvents; eventIdx++) {
                        const duration = randomInt(rand, ZONE_MIN_DURATION, ZONE_MAX_DURATION);
                        const eventType = weightedChoice(rand, [0, 1, 2, 3], [40, 30, 20, 10]);

                        let formatDescId: number;
                        let params: number[];

                        if (eventType === 3) {
                            formatDescId = 6;
                            const tileSizes = [8, 16, 32, 64];
                            params = [
                                randomChoice(rand, tileSizes),
                                randomChoice(rand, tileSizes)
                            ];
                        } else {
                            formatDescId = 3 + eventType;
                            params = [eventType === 0 || eventType === 1 ? randomInt(rand, 0, 1023) : randomInt(rand, 0, 15)];
                        }

                        events.push({
                            time: currentTime,
                            duration,
                            formatDescId,
                            params
                        });

                        currentTime += duration + ZONE_GAP;
                        totalEvents++;
                    }

                    blockEndTime = Math.max(blockEndTime, currentTime);
                    tracks.push({
                        blockIdx,
                        trackId,
                        events
                    });
                }

                blockLaneTime = blockEndTime + BLOCK_PADDING;
            }
        }
    }

    console.log(`Generating ${outputPath}...`);
    console.log(`  Lanes (SMs): ${config.numLanes}`);
    console.log(`  Blocks: ${blockDescriptors.length}`);
    console.log(`  Event tracks: ${tracks.length}`);
    console.log(`  Total events: ${totalEvents}`);

    const header = new BinaryWriter(1024);
    header.buffer.write('nanotrace\x00', 0, 10, 'binary');
    header.offset = 10;
    header.writeUint8(1);  // version
    header.writeUint8(1);  // deflate compression

    const data = new BinaryWriter(small ? 10 * 1024 * 1024 : 200 * 1024 * 1024);
    data.writeString(small ? 'SmallRandomKernel' : 'RandomKernel');
    // Grid dimensions (must accommodate all block IDs)
    data.writeUint32(blockDescriptors.length);  // grid X
    data.writeUint32(1);  // grid Y
    data.writeUint32(1);  // grid Z
    // Cluster dimensions
    data.writeUint32(0);  // cluster X
    data.writeUint32(0);  // cluster Y
    data.writeUint32(0);  // cluster Z
    data.writeUint32(7);
    data.writeUint32(blockDescriptors.length);
    data.writeUint32(tracks.length);
    data.writeUint64(BigInt(totalEvents));

    // Format descriptors (label + tooltip + param count)
    data.writeString('Block {blockLinear}');  // label
    data.writeString('Block {blockLinear} on SM');  // tooltip
    data.writeUint8(0);  // param count
    data.writeString('Track {lane}');  // label
    data.writeString('Track {lane}');  // tooltip
    data.writeUint8(0);  // param count
    data.writeString('Event {0}');  // label
    data.writeString('Event {0}');  // tooltip
    data.writeUint8(1);  // param count
    data.writeString('Load {0}');  // label
    data.writeString('Load from address {0}');  // tooltip
    data.writeUint8(1);  // param count
    data.writeString('Store {0}');  // label
    data.writeString('Store to address {0}');  // tooltip
    data.writeUint8(1);  // param count
    data.writeString('Compute {0}');  // label
    data.writeString('Compute iteration {0}');  // tooltip
    data.writeUint8(1);  // param count
    data.writeString('Tile {0}x{1}');  // label
    data.writeString('Tile operation {0}×{1}');  // tooltip
    data.writeUint8(2);  // param count

    // Block descriptors
    for (const block of blockDescriptors) {
        data.writeUint32(block.blockId);
        data.writeUint32(block.clusterId);
        data.writeUint16(block.smId);
        data.writeUint16(block.formatDescId);
    }

    // Tracks
    for (const track of tracks) {
        data.writeUint32(track.blockIdx);  // Block descriptor ID
        data.writeUint16(1);  // Format descriptor ID
        data.writeUint32(track.trackId);  // Lane ID (for {lane} placeholder)
        // No track parameters (format has 0 params)
        data.writeUint32(track.events.length);  // Event count

        for (const event of track.events) {
            data.writeUint32(event.time);
            data.writeUint32(event.duration);
            data.writeUint16(event.formatDescId);
            for (const param of event.params) {
                data.writeUint32(param);
            }
        }
    }

    const compressed = zlib.deflateSync(data.getBuffer(), { level: 6 });
    const output = Buffer.concat([header.getBuffer(), compressed]);

    fs.writeFileSync(outputPath, output);
    console.log(`\nGenerated ${outputPath} (${output.length.toLocaleString()} bytes)`);
    console.log(`Compression: deflate (zlib level 6)\n`);
}

const command = process.argv[2];
switch (command) {
    case 'minimal':
        generateMinimal();
        break;
    case 'small':
        generateRandom(true);
        break;
    case 'large':
        generateRandom(false);
        break;
    default:
        console.error('Usage: tsx generate.ts [minimal|small|large]');
        process.exit(1);
}
