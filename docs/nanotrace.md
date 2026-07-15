# Nanotrace v4 binary format

Nanotrace v4 stores CPU events, CUPTI HES kernel events, and explicit CUDA
intra-kernel events in one clock-correlated trace. All integers are
little-endian. The format is chunked so unknown chunk types can be skipped.
High-volume events and arguments use unsigned LEB128-style varints. The chunk
body is normally stored as a zlib-wrapped deflate stream.

Multi-byte fixed-width fields are little-endian. `varuint` means unsigned
LEB128. `ZigZag` maps a signed integer to a `varuint`. Chunks may appear in any
order, and readers must skip unknown chunk types using the payload size.

## File header

The fixed header is 32 bytes:

| Offset | Type | Value |
| --- | --- | --- |
| 0 | char[8] | `NTRACE4` followed by zero |
| 8 | u16 | major version, currently 4 |
| 10 | u16 | minor version, currently 0 |
| 12 | u8 | endianness, 1 for little-endian |
| 13 | u8 | flags; bit 0 means the chunk body is deflate-compressed |
| 14 | u16 | reserved |
| 16 | u64 | uncompressed chunk-body size |
| 24 | u64 | stored chunk-body size |

The file header is always uncompressed. If flag bit 0 is set, bytes after the
header are one zlib-wrapped deflate stream which expands to the concatenated
chunks. Otherwise the chunk body follows the header directly and both sizes
must match.

All currently undefined file-flag bits are reserved and must be zero.

## Chunk header

Every chunk starts with a 24-byte header:

| Type | Meaning |
| --- | --- |
| u32 | chunk type |
| u32 | chunk flags |
| u64 | payload size in bytes |
| u64 | record count |

Readers must advance by the payload size rather than deriving it from the
record count. Chunk flags are currently reserved and written as zero.

## Numeric enums

Clock kinds:

| Value | Kind |
| --- | --- |
| 0 | `MonotonicRaw` |
| 1 | `CpuTsc` |
| 2 | `Cupti` |
| 3 | `GpuGlobalTimer` |

Track kinds:

| Value | Kind |
| --- | --- |
| 0 | `Process` |
| 1 | `CpuThread` |
| 2 | `GpuDevice` |
| 3 | `GpuStream` |
| 4 | `Kernel` |
| 5 | `StreamingMultiprocessor` |
| 6 | `ThreadBlock` |
| 7 | `Warp` |
| 8 | `Generic` |

Event kinds occupy bits 0-1 of the event flags byte: 0 is `Slice`, 1 is
`Bookmark`, 2 is `Counter`, and 3 is `Flow`.

Argument kinds are 0 `Unsigned`, 1 `Signed`, 2 `Floating`, 3 `String`, and 4
`Boolean`.

## Chunk types

### 1: Session

One record containing a u32-length-prefixed UTF-8 session name, the u32
reference clock ID, and one reserved u32.

### 2: Strings

Each record is a u32 byte length followed by UTF-8 bytes. String ID is the
record index. ID zero is the empty string.

### 3: Clocks

Each 24-byte record contains:

| Type | Field |
| --- | --- |
| u32 | clock ID |
| u8 | clock kind |
| u8 | flags |
| u16 | reserved |
| u32 | name string ID |
| u32 | device ID |
| u64 | frequency in Hz |

Clock kinds currently include `CLOCK_MONOTONIC_RAW`, CUPTI hardware time, GPU
`%globaltimer`, and CPU TSC.

### 4: Clock snapshots

Each 32-byte record maps one clock into another:

| Type | Field |
| --- | --- |
| u32 | source clock ID |
| u32 | reference clock ID |
| u64 | source timestamp |
| u64 | reference timestamp |
| u64 | uncertainty in nanoseconds |

Clock mappings form a graph. A typical CUDA event follows:

    GPU %globaltimer -> CUPTI HES clock -> CLOCK_MONOTONIC_RAW

Readers must follow all mapping edges until reaching the session reference
clock. One snapshot maps using the source clock frequency. With two or more
snapshots, timestamps are linearly interpolated between surrounding snapshots;
timestamps outside the sampled range use the available endpoint span.

### 5: Tracks

Each 44-byte record contains:

| Type | Field |
| --- | --- |
| u64 | track ID |
| u64 | parent track ID, zero for none |
| u32 | clock ID |
| u32 | name string ID |
| u8 | track kind |
| u8 | flags |
| u16 | reserved |
| i32 | sibling sort order |
| u32 | reserved |
| u64 | source-specific ID |

Tracks form a hierarchy such as GPU, stream, kernel, SM, block, and lane.
Track IDs start at 1; zero means no track. The source-specific ID is interpreted
according to the track kind, such as a CUDA device, stream, SM, block, or lane
identifier.

### 6: Events

Events remain in session insertion order, so their IDs are implicit and start
at 1. This lets the writer encode registered source buffers directly without
sorting or copying their records into a session-wide event vector. The chunk
record count is the total event count, and its payload starts with that same
count as a varuint. Each event then contains:

| Type | Field |
| --- | --- |
| varuint | track ID |
| varuint | ZigZag timestamp delta from the preceding event on this track |
| varuint | duration in the track clock |
| varuint | name string ID |
| u8 | kind and optional-field flags |
| varuint? | parent event ID |
| varuint? | correlation ID |
| varuint? | first argument index |
| varuint? | argument count |
| varuint? | RGB color |

Flag bits 0-1 store the event kind. Bits 2-5 indicate parent, correlation,
arguments, and color respectively. Fields whose flag is clear are omitted.
The first timestamp on a track is a delta from zero. Event and parent IDs
preserve session-wide insertion order. Signed timestamp deltas allow a track
to contain clock samples whose timestamps are not monotonic without requiring
a save-time sort.

Event IDs start at 1 and are implicit from record order; zero means no event.
The RGB color is packed as `0xRRGGBB`.

### 7: Arguments

Each argument starts with a varint name string ID and a u8 argument kind.
Unsigned, string, and boolean values use an unsigned varint. Signed values use
ZigZag encoding followed by an unsigned varint. Floating-point values retain
their fixed eight-byte bit representation. String argument values are string
IDs.

The chunk record count is the total number of arguments. An event's first
argument index is zero-based into this chunk.

### 8: Event formats

Each record describes one reusable event label template:

| Type | Field |
| --- | --- |
| varuint | label string ID, also referenced by events as their name |
| varuint | tooltip template string ID |
| u8 | formatting parameter count |

Parameterized events store their values as the first arguments of each event.
Readers retain one descriptor per label template and substitute `{0}`, `{1}`,
and subsequent placeholders only when rendering. They must not expand each
parameter combination into a separate descriptor. Formatting parameters
written by the CUDA recorder are unsigned 32-bit values. The parameter count
must not exceed the event's argument count.

The label string ID is the same ID carried in the event's name field. At most
one format record may exist for a label string ID. If an event has no matching
format record, readers use its name as both label and tooltip and treat it as a
zero-parameter event. This keeps format chunk 8 optional for plain CPU and HES
events and for older v4 writers.

## CUDA lane buffer format

The device-side buffer is not the file format. Every traced lane starts with a
16-byte header:

| Word | Meaning |
| --- | --- |
| 0 | SM ID |
| 1 | final byte write offset from the tensor buffer base |
| 2 | low 32 bits of the final full `%globaltimer` anchor |
| 3 | high 32 bits of the final full `%globaltimer` anchor |

Event timestamps remain 32-bit `%globaltimer_lo` values. The host reconstructs
an absolute event time as:

    anchor - u32(u32(anchor) - event_low)

The anchor is read once by `finish_lane()`, after all lane events, so it does
not occupy a register or stack slot across the measured work. Reconstruction
is unambiguous while a lane trace spans less than one 32-bit timer wrap.
Durations use unsigned 32-bit subtraction and therefore handle one wrap.

Every lane row is padded to a 16-byte boundary because the header is committed
with one vectorized four-word store.

Rows are ordered by block and then lane. For `num_lanes` lanes per block, row
`block_id * num_lanes + lane_id` begins at that row index multiplied by the
aligned row stride.

### Static-lane event records

A static lane has one event format implied by the lane. Every event slot in a
tensor uses the tensor's maximum event width, so lane rows have a constant
stride. The meaningful prefix of each slot is:

| Parameter count | Active prefix | Words |
| --- | --- | --- |
| 0 | 2 words | start, end |
| 1-2 | 4 words | start, end, parameters, zero padding |
| 3-6 | 8 words | start, end, parameters, zero padding |

Start and end are low 32-bit `%globaltimer` values. Parameters are unsigned
32-bit values. The event format ID is supplied by the static lane metadata and
is not stored in each event.

### Dynamic-lane event records

Every dynamic event occupies eight words:

| Word | Meaning |
| --- | --- |
| 0 | start `%globaltimer_lo` |
| 1 | end `%globaltimer_lo` |
| 2 | event format ID in the low 16 bits |
| 3-7 | zero to five unsigned 32-bit parameters, then zero padding |

The final header write offset advances by the fixed event stride. Exceeding the
configured lane capacity is an error detected by the host parser; events are
not silently truncated.

## GPU capture lifecycle

`GpuTrace` must be constructed before CUDA driver initialization and before any
context exists. Internally, nanotrace follows this sequence:

1. Subscribe to CUPTI and register activity buffers.
2. Enable concurrent-kernel activity.
3. Request `CUPTI_ACTIVITY_ATTR_ENABLE_HES`.
4. Call `cuInit(0)` without creating a context.
5. Verify HES became active.
6. Let the application create CUDA contexts and complete setup or warm-up.
7. `GpuTrace::Begin()` starts capture, launches and synchronizes an internal
   priming kernel, discards priming records, and captures a clock snapshot.
8. Run the instrumented workload and copy its lane buffers into a
   `trace_writer`.
9. `GpuTrace::Write()` stops and flushes HES, matches the requested kernel,
   attaches the intra-kernel events below its HES event, and serializes the
   unified session.

The collector only enables concurrent-kernel activity. A forced CUPTI flush is
sufficient; `cudaDeviceReset()` is not required.
