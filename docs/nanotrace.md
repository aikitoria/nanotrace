# nanotrace Binary Format Specification

## Overview

The `.nanotrace` format is a binary file format for storing GPU kernel execution traces with high performance and minimal overhead.

## Data Types

| Type      | Size     | Description                        |
|-----------|----------|------------------------------------|
| `uint8`   | 1 byte   | Unsigned 8-bit integer             |
| `uint16`  | 2 bytes  | Unsigned 16-bit integer            |
| `uint32`  | 4 bytes  | Unsigned 32-bit integer            |
| `uint64`  | 8 bytes  | Unsigned 64-bit integer            |
| `string`  | Variable | uint16 length prefix + UTF-8 bytes |

All multi-byte integers are stored in **little-endian** byte order.

## File Structure

```
┌─────────────────────────────────┐
│ File Header                     │
├─────────────────────────────────┤
│ Format Descriptors              │
│  (repeated N times)             │
├─────────────────────────────────┤
│ Block Descriptors               │
│  (repeated M times)             │
├─────────────────────────────────┤
│ Event Tracks                    │
│  (repeated K times)             │
│   └─ Event Descriptors          │
│      (repeated per track)       │
└─────────────────────────────────┘
```

## Detailed Format

### 1. File Header

| Field                    | Type        | Description                                                   |
|--------------------------|-------------|---------------------------------------------------------------|
| Magic                    | `uint8[10]` | Magic identifier: ASCII "nanotrace" + null terminator         |
| Format Version           | `uint8`     | File format version number (current: 1)                       |
| Compression Mode         | `uint8`     | Compression flag: 0 = uncompressed, 1 = deflate               |
| Kernel Name              | `string`    | Name of the profiled kernel                                   |
| Format Descriptor Count  | `uint32`    | Number of format descriptors (N)                              |
| Block Descriptor Count   | `uint32`    | Number of block descriptors (M)                               |
| Track Count              | `uint32`    | Number of event tracks (K)                                    |
| Total Event Count        | `uint64`    | Total number of events across all tracks (for pre-allocation) |

**Size:** 10 + 1 + 1 + (2 + kernel_name_length) + 4 + 4 + 4 + 8 = 34 + kernel_name_length bytes

**Compression:** When compression mode is 1 (deflate), all data after the Compression Mode field is compressed using deflate/zlib compression. This includes the kernel name, counts, format descriptors, block descriptors, and event tracks. The decompressed data should be parsed as if it were the uncompressed remainder of the file.

### 2. Format Descriptors

Each format descriptor describes a reusable format string template.

**Repeated N times** (where N = Format Descriptor Count):

| Field              | Type      | Description                                       |
|--------------------|-----------|---------------------------------------------------|
| Format String      | `string`  | Format template (e.g., "Tile {0}x{1}")            |
| Placeholder Count  | `uint8`   | Number of `{N}` placeholders in the format string |

**Size per descriptor:** 2 + format_string_length + 1 bytes

### 3. Block Descriptors

Each block descriptor represents a thread block execution on a specific SM.

**Repeated M times** (where M = Block Descriptor Count):

| Field                 | Type        | Description                                                                  |
|-----------------------|-------------|------------------------------------------------------------------------------|
| SM ID                 | `uint16`    | Streaming Multiprocessor ID (0-65535)                                        |
| Format Descriptor ID  | `uint16`    | Index into format descriptors array                                          |
| Format Parameters     | `uint32[]`  | Parameter values (count = placeholder count of referenced format descriptor) |

**Size per descriptor:** 2 + 2 + (placeholder_count × 4) bytes

**Note:** To parse format parameters, look up the Format Descriptor at the specified ID and read as many `uint32` values as its placeholder count indicates.

### 4. Event Tracks

Each track contains a sequence of events for a specific execution unit (e.g., warp, thread) within a block.

**Repeated K times** (where K = Track Count):

| Field                 | Type        | Description                                                                  |
|-----------------------|-------------|------------------------------------------------------------------------------|
| Block Descriptor ID   | `uint32`    | Index into block descriptors array                                           |
| Format Descriptor ID  | `uint16`    | Index into format descriptors array (for track/sublane name)                 |
| Format Parameters     | `uint32[]`  | Parameter values (count = placeholder count of referenced format descriptor) |
| Event Count           | `uint32`    | Number of events in this track                                               |

Followed by **Event Count** event descriptors:

| Field                 | Type        | Description                                                                  |
|-----------------------|-------------|------------------------------------------------------------------------------|
| Time Offset           | `uint32`    | Nanoseconds since kernel start                                               |
| Duration              | `uint32`    | Event duration in nanoseconds                                                |
| Format Descriptor ID  | `uint16`    | Index into format descriptors array                                          |
| Format Parameters     | `uint32[]`  | Parameter values (count = placeholder count of referenced format descriptor) |

**Size per track:** 4 + 2 + (track_placeholder_count × 4) + 4 + Σ(8 + 2 + (event_placeholder_count × 4)) bytes for each event

## Parsing Order and Requirements

1. **Read File Header** - Validate magic number, extract counts
2. **Read All Format Descriptors** - Store placeholder counts for later reference
3. **Read All Block Descriptors** - Use format descriptor placeholder counts to determine parameter array sizes
4. **Read All Event Tracks** - Use format descriptor placeholder counts to determine parameter array sizes for each event

**Critical:** Format descriptors must be fully parsed before reading block descriptors or events, as their placeholder counts determine how many parameters to read.

## Example: Simple Trace

### Scenario
- Kernel: "MyKernel"
- 3 Format Descriptors:
  - FD[0]: "Block {0}" (1 placeholder)
  - FD[1]: "Warp {0}" (1 placeholder)
  - FD[2]: "Load {0}" (1 placeholder)
- 1 Block Descriptor:
  - SM 0, uses FD[0], param=42
- 1 Event Track:
  - Belongs to Block 0, uses FD[1], param=7, has 2 events:
    - Event 0: t=1000ns, dur=50ns, FD[2], param=128
    - Event 1: t=1100ns, dur=75ns, FD[2], param=256

### Binary Layout (Hex)

```
Offset  Hex                                         Description
------  ---                                         -----------
0x0000  6E 61 6E 6F 74 72 61 63 65 00               Magic: "nanotrace\0"
0x000A  01                                          Format version: 1
0x000B  00                                          Compression mode: 0 (uncompressed)
0x000C  08 00                                       Kernel name length: 8
0x000E  4D 79 4B 65 72 6E 65 6C                     Kernel name: "MyKernel"
0x0016  03 00 00 00                                 Format descriptor count: 3
0x001A  01 00 00 00                                 Block descriptor count: 1
0x001E  01 00 00 00                                 Track count: 1
0x0022  02 00 00 00 00 00 00 00                     Total event count: 2

# Format Descriptor 0
0x002A  09 00                                       String length: 9
0x002C  42 6C 6F 63 6B 20 7B 30 7D                  "Block {0}"
0x0035  01                                          Placeholder count: 1

# Format Descriptor 1
0x0036  08 00                                       String length: 8
0x0038  57 61 72 70 20 7B 30 7D                     "Warp {0}"
0x0040  01                                          Placeholder count: 1

# Format Descriptor 2
0x0041  08 00                                       String length: 8
0x0043  4C 6F 61 64 20 7B 30 7D                     "Load {0}"
0x004B  01                                          Placeholder count: 1

# Block Descriptor 0
0x004C  00 00                                       SM ID: 0
0x004E  00 00                                       Format descriptor ID: 0
0x0050  2A 00 00 00                                 Param[0]: 42

# Event Track 0
0x0054  00 00 00 00                                 Block descriptor ID: 0
0x0058  01 00                                       Format descriptor ID: 1 (Warp {0})
0x005A  07 00 00 00                                 Param[0]: 7
0x005E  02 00 00 00                                 Event count: 2

# Event 0
0x0062  E8 03 00 00                                 Time: 1000ns
0x0066  32 00 00 00                                 Duration: 50ns
0x006A  02 00                                       Format descriptor ID: 2 (Load {0})
0x006C  80 00 00 00                                 Param[0]: 128

# Event 1
0x0070  4C 04 00 00                                 Time: 1100ns
0x0074  4B 00 00 00                                 Duration: 75ns
0x0078  02 00                                       Format descriptor ID: 2 (Load {0})
0x007A  00 01 00 00                                 Param[0]: 256
```

**Total file size:** 126 bytes (0x7E)

## Implementation Notes

### For C++/CUDA Writers

```cpp
// Pseudo-code structure
void writeNanotrace(const char* filename, bool useCompression = false) {
    FILE* f = fopen(filename, "wb");

    // Write header
    fwrite("nanotrace\0", 10, 1, f);
    writeUint8(f, 1);  // Format version
    writeUint8(f, useCompression ? 1 : 0);  // Compression mode

    // If compression enabled, write remaining data to buffer first
    std::vector<uint8_t> dataBuffer;
    FILE* out = useCompression ? createMemoryFile(&dataBuffer) : f;

    writeString(out, kernelName);
    writeUint32(out, formatDescriptors.size());
    writeUint32(out, blockDescriptors.size());
    writeUint32(out, eventTracks.size());
    writeUint64(out, totalEventCount);

    // Write format descriptors
    for (auto& fd : formatDescriptors) {
        writeString(out, fd.formatString);
        writeUint8(out, fd.placeholderCount);
    }

    // Write block descriptors
    for (auto& bd : blockDescriptors) {
        writeUint16(out, bd.smId);
        writeUint16(out, bd.formatDescriptorId);
        uint8_t paramCount = formatDescriptors[bd.formatDescriptorId].placeholderCount;
        for (int i = 0; i < paramCount; i++) {
            writeUint32(out, bd.params[i]);
        }
    }

    // Write event tracks
    for (auto& track : eventTracks) {
        writeUint32(out, track.blockDescriptorId);
        writeUint16(out, track.formatDescriptorId);
        uint8_t trackParamCount = formatDescriptors[track.formatDescriptorId].placeholderCount;
        for (int i = 0; i < trackParamCount; i++) {
            writeUint32(out, track.params[i]);
        }
        writeUint32(out, track.events.size());
        for (auto& event : track.events) {
            writeUint32(out, event.timeOffset);
            writeUint32(out, event.duration);
            writeUint16(out, event.formatDescriptorId);
            uint8_t paramCount = formatDescriptors[event.formatDescriptorId].placeholderCount;
            for (int i = 0; i < paramCount; i++) {
                writeUint32(out, event.params[i]);
            }
        }
    }

    // Compress and write data if compression enabled
    if (useCompression) {
        std::vector<uint8_t> compressed = zlibCompress(dataBuffer);
        fwrite(compressed.data(), 1, compressed.size(), f);
    }

    fclose(f);
}
```

### For JavaScript Readers

```javascript
// Pseudo-code structure
async function readNanotrace(file) {
    const buffer = await file.arrayBuffer();
    let view = new DataView(buffer);
    let offset = 0;

    // Read header
    const magic = new TextDecoder().decode(buffer.slice(0, 9));
    if (magic !== "nanotrace") throw new Error("Invalid magic number");
    offset += 10;

    const formatVersion = view.getUint8(offset); offset += 1;
    if (formatVersion !== 1) throw new Error(`Unsupported format version: ${formatVersion}`);

    const compressionMode = view.getUint8(offset); offset += 1;

    // Decompress if needed
    if (compressionMode === 1) {
        const compressedData = buffer.slice(offset);
        const decompressed = pako.inflate(new Uint8Array(compressedData));
        view = new DataView(decompressed.buffer);
        offset = 0;
    }

    const kernelName = readString(view, offset);
    offset += 2 + kernelName.length;

    const formatDescCount = view.getUint32(offset, true); offset += 4;
    const blockDescCount = view.getUint32(offset, true); offset += 4;
    const trackCount = view.getUint32(offset, true); offset += 4;
    const totalEventCount = view.getBigUint64(offset, true); offset += 8;

    // Read format descriptors
    const formatDescriptors = [];
    for (let i = 0; i < formatDescCount; i++) {
        const formatString = readString(view, offset);
        offset += 2 + formatString.length;
        const placeholderCount = view.getUint8(offset); offset += 1;
        formatDescriptors.push({ formatString, placeholderCount });
    }

    // Read block descriptors
    const blockDescriptors = [];
    for (let i = 0; i < blockDescCount; i++) {
        const smId = view.getUint16(offset, true); offset += 2;
        const formatDescId = view.getUint16(offset, true); offset += 2;
        const paramCount = formatDescriptors[formatDescId].placeholderCount;
        const params = [];
        for (let j = 0; j < paramCount; j++) {
            params.push(view.getUint32(offset, true)); offset += 4;
        }
        blockDescriptors.push({ smId, formatDescId, params });
    }

    // Read event tracks
    const tracks = [];
    for (let i = 0; i < trackCount; i++) {
        const blockDescId = view.getUint32(offset, true); offset += 4;
        const formatDescId = view.getUint16(offset, true); offset += 2;
        const trackParamCount = formatDescriptors[formatDescId].placeholderCount;
        const params = [];
        for (let j = 0; j < trackParamCount; j++) {
            params.push(view.getUint32(offset, true)); offset += 4;
        }
        const eventCount = view.getUint32(offset, true); offset += 4;
        const events = [];
        for (let j = 0; j < eventCount; j++) {
            const timeOffset = view.getUint32(offset, true); offset += 4;
            const duration = view.getUint32(offset, true); offset += 4;
            const eventFormatDescId = view.getUint16(offset, true); offset += 2;
            const eventParamCount = formatDescriptors[eventFormatDescId].placeholderCount;
            const eventParams = [];
            for (let k = 0; k < eventParamCount; k++) {
                eventParams.push(view.getUint32(offset, true)); offset += 4;
            }
            events.push({ timeOffset, duration, formatDescId: eventFormatDescId, params: eventParams });
        }
        tracks.push({ blockDescId, formatDescId, params, events });
    }

    return { kernelName, formatDescriptors, blockDescriptors, tracks };
}
```

## Design Rationale

1. **Magic Number** - File type identification and validation
2. **Compression Support** - Deflate compression for large traces (typically 50% size reduction)
3. **Format Descriptors** - Deduplicates repeated format strings, reduces file size
4. **Little Endian** - Matches x86/x64 CPU native byte order for zero-cost parsing
5. **uint32 for Time** - Supports up to ~4.3 seconds at nanosecond precision (sufficient for kernel traces)
6. **uint16 for Format IDs** - 65K unique format strings is far more than needed
7. **uint16 for SM ID** - Supports up to 65K SMs (future-proof for massive GPUs)
8. **No Padding/Alignment** - Minimal file size, parser handles unaligned reads

## Limitations

- Maximum kernel duration: ~4.3 seconds (uint32 nanoseconds)
- Maximum format string/kernel name: 65,535 bytes (uint16 length)
- Maximum format descriptors: 65,535 (uint16 IDs)
- Maximum blocks: 4,294,967,295 (uint32 IDs)
- Maximum SMs: 65,535 (uint16)
- Maximum placeholders per format: 255 (uint8)

All limits are far beyond practical use cases for GPU kernel profiling.
