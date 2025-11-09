#pragma once

#include <cstdint>

namespace nanotrace {

// Lane type enum
enum class lane_type : uint8_t {
    STATIC,   // Fixed format per lane (format_id not written per event)
    DYNAMIC   // Format_id written per event
};

// Macro to define trace types at compile time
#define NANOTRACE_DEFINE_TRACE_TYPE(name, label_str, tooltip_str, pcount, lane_usage) \
    struct name { \
        static constexpr uint16_t id = __COUNTER__; \
        static constexpr const char* label_string = label_str; \
        static constexpr const char* tooltip_string = tooltip_str; \
        static constexpr uint8_t param_count = pcount; \
        static constexpr nanotrace::lane_type usage = lane_usage; \
    }

// Macro to define block types at compile time
#define NANOTRACE_DEFINE_BLOCK_TYPE(name, label_str, tooltip_str) \
    struct name { \
        static constexpr uint16_t id = __COUNTER__; \
        static constexpr const char* label_string = label_str; \
        static constexpr const char* tooltip_string = tooltip_str; \
        static constexpr uint8_t param_count = 0; \
        static constexpr bool is_block_type = true; \
    }

// Macro to define track types at compile time
#define NANOTRACE_DEFINE_TRACK_TYPE(name, label_str, tooltip_str, pcount) \
    struct name { \
        static constexpr uint16_t id = __COUNTER__; \
        static constexpr const char* label_string = label_str; \
        static constexpr const char* tooltip_string = tooltip_str; \
        static constexpr uint8_t param_count = pcount; \
        static constexpr bool is_track_type = true; \
    }

// ============================================================================
// Tensor handles (passed to kernels)
// ============================================================================

template<uint32_t NumLanes, uint32_t MaxEventWidth>
struct static_tensor_handle {
    uint32_t* buffer;
    uint32_t row_stride;

    static constexpr uint32_t num_lanes = NumLanes;
    static constexpr uint32_t max_event_width = MaxEventWidth;
};

template<uint32_t NumLanes>
struct dynamic_tensor_handle {
    uint32_t* buffer;
    uint32_t row_stride;

    static constexpr uint32_t num_lanes = NumLanes;
    static constexpr uint32_t event_width = 8;
};

// ============================================================================
// Start token (opaque timestamp wrapper)
// ============================================================================

struct start_token {
    uint32_t time;
    __device__ __forceinline__ start_token(uint32_t t) : time(t) {}
};

// ============================================================================
// Lane contexts (minimal state: 2 uint32s)
// ============================================================================

template<uint32_t MaxEventWidth>
class lane_context_static {
private:
    uint32_t base_offset_;
    uint32_t write_offset_;
    bool enabled_;

public:
    static constexpr uint32_t max_event_width = MaxEventWidth;

    __device__ __forceinline__
    lane_context_static(uint32_t base_offset, bool enabled = true)
        : base_offset_(base_offset)
        , write_offset_(base_offset + MaxEventWidth)  // Skip first event slot for header
        , enabled_(enabled)
    {}

    __device__ __forceinline__ uint32_t write_offset() const {
        return write_offset_;
    }

    __device__ __forceinline__ uint32_t base_offset() const {
        return base_offset_;
    }

    __device__ __forceinline__ bool enabled() const {
        return enabled_;
    }

    __device__ __forceinline__ void advance() {
        write_offset_ += MaxEventWidth;
    }
};

class lane_context_dynamic {
private:
    uint32_t base_offset_;
    uint32_t write_offset_;
    bool enabled_;

public:
    static constexpr uint32_t event_width = 8;

    __device__ __forceinline__
    lane_context_dynamic(uint32_t base_offset, bool enabled = true)
        : base_offset_(base_offset)
        , write_offset_(base_offset + 8)  // Skip first event slot for header
        , enabled_(enabled)
    {}

    __device__ __forceinline__ uint32_t write_offset() const {
        return write_offset_;
    }

    __device__ __forceinline__ uint32_t base_offset() const {
        return base_offset_;
    }

    __device__ __forceinline__ bool enabled() const {
        return enabled_;
    }

    __device__ __forceinline__ void advance() {
        write_offset_ += 8;
    }
};

// ============================================================================
// Capture start timestamp
// ============================================================================

__device__ __forceinline__ start_token start() {
    uint32_t t;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(t));
    return start_token(t);
}

__device__ __forceinline__ start_token start_zero() {
    return start_token(0);
}

// ============================================================================
// Begin lane
// ============================================================================

template<uint32_t NumLanes, uint32_t MaxEventWidth>
__device__ __forceinline__ auto begin_lane(
    static_tensor_handle<NumLanes, MaxEventWidth> handle,
    uint32_t block_id,
    uint32_t lane_index,
    bool enabled = true)
{
    uint32_t base_offset = block_id * NumLanes * handle.row_stride + lane_index * handle.row_stride;
    return lane_context_static<MaxEventWidth>(base_offset, enabled);
}

template<uint32_t NumLanes>
__device__ __forceinline__ auto begin_lane_dynamic(
    dynamic_tensor_handle<NumLanes> handle,
    uint32_t block_id,
    uint32_t lane_index,
    bool enabled = true)
{
    uint32_t base_offset = block_id * NumLanes * handle.row_stride + lane_index * handle.row_stride;
    return lane_context_dynamic(base_offset, enabled);
}

// ============================================================================
// End functions (static lanes) - event width deduced from parameter count
// ============================================================================

// 0 params → width 2
template<uint32_t NumLanes, uint32_t MaxEventWidth>
__device__ __forceinline__ void end(
    start_token start,
    static_tensor_handle<NumLanes, MaxEventWidth> handle,
    lane_context_static<MaxEventWidth>& lane)
{
    if (!lane.enabled()) return;

    constexpr uint32_t event_width = 2;
    static_assert(event_width <= MaxEventWidth, "Event width exceeds lane max event width");

    uint32_t end_time;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(end_time));

    if (lane.write_offset() + event_width <= lane.base_offset() + handle.row_stride) {
        uint32_t* addr = handle.buffer + lane.write_offset();

        asm volatile("st.global.cs.v2.u32 [%0], {%1, %2};"
                     :: "l"(addr), "r"(start.time), "r"(end_time)
                     : "memory");

        lane.advance();
    }
}

// 1 param → width 4
template<uint32_t NumLanes, uint32_t MaxEventWidth>
__device__ __forceinline__ void end(
    start_token start,
    static_tensor_handle<NumLanes, MaxEventWidth> handle,
    lane_context_static<MaxEventWidth>& lane,
    uint32_t p0)
{
    if (!lane.enabled()) return;

    constexpr uint32_t event_width = 4;
    static_assert(event_width <= MaxEventWidth, "Event width exceeds lane max event width");

    uint32_t end_time;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(end_time));

    if (lane.write_offset() + event_width <= lane.base_offset() + handle.row_stride) {
        uint32_t* addr = handle.buffer + lane.write_offset();

        asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
                     :: "l"(addr), "r"(start.time), "r"(end_time),
                        "r"(p0), "r"(0u)
                     : "memory");

        lane.advance();
    }
}

// 2 params → width 4
template<uint32_t NumLanes, uint32_t MaxEventWidth>
__device__ __forceinline__ void end(
    start_token start,
    static_tensor_handle<NumLanes, MaxEventWidth> handle,
    lane_context_static<MaxEventWidth>& lane,
    uint32_t p0,
    uint32_t p1)
{
    if (!lane.enabled()) return;

    constexpr uint32_t event_width = 4;
    static_assert(event_width <= MaxEventWidth, "Event width exceeds lane max event width");

    uint32_t end_time;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(end_time));

    if (lane.write_offset() + event_width <= lane.base_offset() + handle.row_stride) {
        uint32_t* addr = handle.buffer + lane.write_offset();

        asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
                     :: "l"(addr), "r"(start.time), "r"(end_time),
                        "r"(p0), "r"(p1)
                     : "memory");

        lane.advance();
    }
}

// 3 params → width 8
template<uint32_t NumLanes, uint32_t MaxEventWidth>
__device__ __forceinline__ void end(
    start_token start,
    static_tensor_handle<NumLanes, MaxEventWidth> handle,
    lane_context_static<MaxEventWidth>& lane,
    uint32_t p0,
    uint32_t p1,
    uint32_t p2)
{
    if (!lane.enabled()) return;

    constexpr uint32_t event_width = 8;
    static_assert(event_width <= MaxEventWidth, "Event width exceeds lane max event width");

    uint32_t end_time;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(end_time));

    if (lane.write_offset() + event_width <= lane.base_offset() + handle.row_stride) {
        uint32_t* addr = handle.buffer + lane.write_offset();

        asm volatile("st.global.cs.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                     :: "l"(addr), "r"(start.time), "r"(end_time),
                        "r"(p0), "r"(p1), "r"(p2),
                        "r"(0u), "r"(0u), "r"(0u)
                     : "memory");

        lane.advance();
    }
}

// 4 params → width 8
template<uint32_t NumLanes, uint32_t MaxEventWidth>
__device__ __forceinline__ void end(
    start_token start,
    static_tensor_handle<NumLanes, MaxEventWidth> handle,
    lane_context_static<MaxEventWidth>& lane,
    uint32_t p0,
    uint32_t p1,
    uint32_t p2,
    uint32_t p3)
{
    if (!lane.enabled()) return;

    constexpr uint32_t event_width = 8;
    static_assert(event_width <= MaxEventWidth, "Event width exceeds lane max event width");

    uint32_t end_time;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(end_time));

    if (lane.write_offset() + event_width <= lane.base_offset() + handle.row_stride) {
        uint32_t* addr = handle.buffer + lane.write_offset();

        asm volatile("st.global.cs.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                     :: "l"(addr), "r"(start.time), "r"(end_time),
                        "r"(p0), "r"(p1), "r"(p2), "r"(p3),
                        "r"(0u), "r"(0u)
                     : "memory");

        lane.advance();
    }
}

// 5 params → width 8
template<uint32_t NumLanes, uint32_t MaxEventWidth>
__device__ __forceinline__ void end(
    start_token start,
    static_tensor_handle<NumLanes, MaxEventWidth> handle,
    lane_context_static<MaxEventWidth>& lane,
    uint32_t p0,
    uint32_t p1,
    uint32_t p2,
    uint32_t p3,
    uint32_t p4)
{
    if (!lane.enabled()) return;

    constexpr uint32_t event_width = 8;
    static_assert(event_width <= MaxEventWidth, "Event width exceeds lane max event width");

    uint32_t end_time;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(end_time));

    if (lane.write_offset() + event_width <= lane.base_offset() + handle.row_stride) {
        uint32_t* addr = handle.buffer + lane.write_offset();

        asm volatile("st.global.cs.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                     :: "l"(addr), "r"(start.time), "r"(end_time),
                        "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(p4),
                        "r"(0u)
                     : "memory");

        lane.advance();
    }
}

// 6 params → width 8
template<uint32_t NumLanes, uint32_t MaxEventWidth>
__device__ __forceinline__ void end(
    start_token start,
    static_tensor_handle<NumLanes, MaxEventWidth> handle,
    lane_context_static<MaxEventWidth>& lane,
    uint32_t p0,
    uint32_t p1,
    uint32_t p2,
    uint32_t p3,
    uint32_t p4,
    uint32_t p5)
{
    if (!lane.enabled()) return;

    constexpr uint32_t event_width = 8;
    static_assert(event_width <= MaxEventWidth, "Event width exceeds lane max event width");

    uint32_t end_time;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(end_time));

    if (lane.write_offset() + event_width <= lane.base_offset() + handle.row_stride) {
        uint32_t* addr = handle.buffer + lane.write_offset();

        asm volatile("st.global.cs.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                     :: "l"(addr), "r"(start.time), "r"(end_time),
                        "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(p4), "r"(p5)
                     : "memory");

        lane.advance();
    }
}

// ============================================================================
// End function (dynamic lanes) - variadic template
// ============================================================================

template<uint32_t NumLanes, typename TraceType, typename... Params>
__device__ __forceinline__ void end(
    start_token start,
    dynamic_tensor_handle<NumLanes> handle,
    lane_context_dynamic& lane,
    TraceType,
    Params... params)
{
    if (!lane.enabled()) return;

    static_assert(TraceType::usage == lane_type::DYNAMIC, "Trace type must be DYNAMIC");
    static_assert(TraceType::param_count == sizeof...(Params), "Parameter count mismatch");
    static_assert(sizeof...(Params) <= 5, "Maximum 5 parameters for dynamic events");

    uint32_t end_time;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(end_time));

    constexpr uint32_t event_width = 8;
    if (lane.write_offset() + event_width <= lane.base_offset() + handle.row_stride) {
        uint32_t* addr = handle.buffer + lane.write_offset();
        uint32_t p[5] = {0, 0, 0, 0, 0};

        // Copy parameters into array
        uint32_t idx = 0;
        ((p[idx++] = params), ...);

        asm volatile("st.global.cs.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                     :: "l"(addr), "r"(start.time), "r"(end_time),
                        "r"((uint32_t)TraceType::id),
                        "r"(p[0]), "r"(p[1]), "r"(p[2]), "r"(p[3]), "r"(p[4])
                     : "memory");

        lane.advance();
    }
}

// ============================================================================
// Finish lane - write header with SM ID and event count
// ============================================================================

template<uint32_t NumLanes, uint32_t MaxEventWidth>
__device__ __forceinline__ void finish_lane(
    static_tensor_handle<NumLanes, MaxEventWidth> handle,
    lane_context_static<MaxEventWidth>& lane)
{
    if (!lane.enabled()) return;

    uint32_t sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));

    // Compute event count from write_offset
    constexpr uint32_t max_event_width = MaxEventWidth;
    uint32_t event_count = (lane.write_offset() - lane.base_offset() - max_event_width) / max_event_width;

    uint32_t* header = handle.buffer + lane.base_offset();

    // Header: [sm_id, event_count]
    asm volatile("st.global.cs.v2.u32 [%0], {%1, %2};"
                 :: "l"(header),
                    "r"(sm_id), "r"(event_count)
                 : "memory");
}

template<uint32_t NumLanes>
__device__ __forceinline__ void finish_lane(
    dynamic_tensor_handle<NumLanes> handle,
    lane_context_dynamic& lane)
{
    if (!lane.enabled()) return;

    uint32_t sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));

    constexpr uint32_t event_width = 8;
    uint32_t event_count = (lane.write_offset() - lane.base_offset() - event_width) / event_width;

    uint32_t* header = handle.buffer + lane.base_offset();

    // Header: [sm_id, event_count]
    asm volatile("st.global.cs.v2.u32 [%0], {%1, %2};"
                 :: "l"(header),
                    "r"(sm_id), "r"(event_count)
                 : "memory");
}

} // namespace nanotrace
