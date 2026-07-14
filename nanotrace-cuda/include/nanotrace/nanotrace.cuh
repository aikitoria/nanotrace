#pragma once

#include <cstdint>

namespace nanotrace {

// Lane type enum (always defined for host code compatibility)
enum class lane_type : uint8_t {
    STATIC,   // Fixed format per lane (format_id not written per event)
    DYNAMIC   // Format_id written per event
};

// Format descriptor (always defined for host code compatibility)
struct format_descriptor {
    const char* label_string;
    const char* tooltip_string;
    uint16_t id;
    uint8_t param_count;
    const char* const* param_names;

    bool operator==(const format_descriptor& other) const {
        return id == other.id &&
               param_count == other.param_count &&
               label_string == other.label_string &&
               tooltip_string == other.tooltip_string &&
               param_names == other.param_names;
    }
};

} // namespace nanotrace

// Macros to define trace/block/track types - ALWAYS provide full definitions
// so host code can access ::id, ::descriptor, ::usage, etc.
#define NANOTRACE_DEFINE_TRACE_TYPE(name, label_str, tooltip_str, pcount, lane_usage) \
    struct name { \
        static constexpr uint16_t id = __COUNTER__; \
        static constexpr const char* label_string = label_str; \
        static constexpr const char* tooltip_string = tooltip_str; \
        static constexpr uint8_t param_count = pcount; \
        static constexpr nanotrace::lane_type usage = lane_usage; \
        static inline const nanotrace::format_descriptor descriptor = { \
            label_string, tooltip_string, id, param_count, nullptr \
        }; \
    }

#define NANOTRACE_DEFINE_TRACE_TYPE_WITH_PARAMETERS(name, label_str, tooltip_str, lane_usage, ...) \
    struct name { \
        static constexpr uint16_t id = __COUNTER__; \
        static constexpr const char* label_string = label_str; \
        static constexpr const char* tooltip_string = tooltip_str; \
        static constexpr const char* param_names[] = { __VA_ARGS__ }; \
        static constexpr uint8_t param_count = sizeof(param_names) / sizeof(param_names[0]); \
        static constexpr nanotrace::lane_type usage = lane_usage; \
        static inline const nanotrace::format_descriptor descriptor = { \
            label_string, tooltip_string, id, param_count, param_names \
        }; \
    }

#define NANOTRACE_DEFINE_BLOCK_TYPE(name, label_str, tooltip_str) \
    struct name { \
        static constexpr uint16_t id = __COUNTER__; \
        static constexpr const char* label_string = label_str; \
        static constexpr const char* tooltip_string = tooltip_str; \
        static constexpr uint8_t param_count = 0; \
        static constexpr bool is_block_type = true; \
        static inline const nanotrace::format_descriptor descriptor = { \
            label_string, tooltip_string, id, param_count, nullptr \
        }; \
    }

#define NANOTRACE_DEFINE_TRACK_TYPE(name, label_str, tooltip_str, pcount) \
    struct name { \
        static constexpr uint16_t id = __COUNTER__; \
        static constexpr const char* label_string = label_str; \
        static constexpr const char* tooltip_string = tooltip_str; \
        static constexpr uint8_t param_count = pcount; \
        static constexpr bool is_track_type = true; \
        static inline const nanotrace::format_descriptor descriptor = { \
            label_string, tooltip_string, id, param_count, nullptr \
        }; \
    }

#ifdef NANOTRACE_DISABLED

namespace nanotrace {

template<uint32_t NumLanes, uint32_t MaxEventWidth>
struct static_tensor_handle {
    uint8_t* buffer = nullptr;
    uint32_t row_stride_bytes = 0;
    static constexpr uint32_t num_lanes = NumLanes;
    static constexpr uint32_t max_event_width = MaxEventWidth;
};

template<uint32_t NumLanes>
struct dynamic_tensor_handle {
    uint8_t* buffer = nullptr;
    uint32_t row_stride_bytes = 0;
    static constexpr uint32_t num_lanes = NumLanes;
    static constexpr uint32_t event_width = 8;
};

struct start_token {
    uint32_t time = 0;
};

template<uint32_t MaxEventWidth>
class lane_context_static {
public:
    static constexpr uint32_t max_event_width = MaxEventWidth;
    __device__ __forceinline__ lane_context_static(uint32_t = 0, bool = true) {}
    __device__ __forceinline__ bool enabled() const { return false; }
    __device__ __forceinline__ void advance() {}
};

class lane_context_dynamic {
public:
    static constexpr uint32_t event_width = 8;
    __device__ __forceinline__ lane_context_dynamic(uint32_t = 0, bool = true) {}
    __device__ __forceinline__ bool enabled() const { return false; }
    __device__ __forceinline__ void advance() {}
};

__device__ __forceinline__ start_token start() { return start_token{}; }

__device__ __forceinline__ start_token start_zero() { return start_token{}; }

template<uint32_t NumLanes, uint32_t MaxEventWidth>
__device__ __forceinline__ lane_context_static<MaxEventWidth> begin_lane(
    const static_tensor_handle<NumLanes, MaxEventWidth>&,
    uint32_t,
    uint32_t,
    bool = true
) {
    return lane_context_static<MaxEventWidth>{};
}

template<uint32_t NumLanes>
__device__ __forceinline__ lane_context_dynamic begin_lane_dynamic(
    const dynamic_tensor_handle<NumLanes>&,
    uint32_t,
    uint32_t,
    bool = true
) {
    return lane_context_dynamic{};
}

template<typename... Args>
__device__ __forceinline__ void end(Args...) {}

template<typename Handle, typename Lane>
__device__ __forceinline__ void finish_lane(const Handle&, const Lane&) {}

} // namespace nanotrace

#else

namespace nanotrace {

// ============================================================================
// Tensor handles (passed to kernels)
// ============================================================================

template<uint32_t NumLanes, uint32_t MaxEventWidth>
struct static_tensor_handle {
    uint8_t* buffer;
    uint32_t row_stride_bytes;

    static constexpr uint32_t num_lanes = NumLanes;
    static constexpr uint32_t max_event_width = MaxEventWidth;
};

template<uint32_t NumLanes>
struct dynamic_tensor_handle {
    uint8_t* buffer;
    uint32_t row_stride_bytes;

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
    uint32_t base_offset_bytes_;
    uint32_t write_offset_bytes_;
    bool enabled_;

public:
    static constexpr uint32_t max_event_width = MaxEventWidth;
    static constexpr uint32_t max_event_width_bytes = MaxEventWidth << 2;
    static constexpr uint32_t header_bytes = 4 * sizeof(uint32_t);

    __device__ __forceinline__
    lane_context_static(uint32_t base_offset_bytes, bool enabled = true)
        : base_offset_bytes_(base_offset_bytes)
        , write_offset_bytes_(base_offset_bytes + header_bytes)
        , enabled_(enabled)
    {}

    __device__ __forceinline__ uint32_t write_offset_bytes() const {
        return write_offset_bytes_;
    }

    __device__ __forceinline__ uint32_t base_offset_bytes() const {
        return base_offset_bytes_;
    }

    __device__ __forceinline__ bool enabled() const {
        return enabled_;
    }

    __device__ __forceinline__ void advance() {
        write_offset_bytes_ += max_event_width_bytes;
    }
};

class lane_context_dynamic {
private:
    uint32_t base_offset_bytes_;
    uint32_t write_offset_bytes_;
    bool enabled_;

public:
    static constexpr uint32_t event_width = 8;
    static constexpr uint32_t event_width_bytes = 8 << 2;
    static constexpr uint32_t header_bytes = 4 * sizeof(uint32_t);

    __device__ __forceinline__
    lane_context_dynamic(uint32_t base_offset_bytes, bool enabled = true)
        : base_offset_bytes_(base_offset_bytes)
        , write_offset_bytes_(base_offset_bytes + header_bytes)
        , enabled_(enabled)
    {}

    __device__ __forceinline__ uint32_t write_offset_bytes() const {
        return write_offset_bytes_;
    }

    __device__ __forceinline__ uint32_t base_offset_bytes() const {
        return base_offset_bytes_;
    }

    __device__ __forceinline__ bool enabled() const {
        return enabled_;
    }

    __device__ __forceinline__ void advance() {
        write_offset_bytes_ += event_width_bytes;
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

__device__ __forceinline__ uint64_t read_global_timer() {
    uint64_t time;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time));
    return time;
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
    uint32_t base_offset_bytes = block_id * NumLanes * handle.row_stride_bytes + lane_index * handle.row_stride_bytes;
    return lane_context_static<MaxEventWidth>(base_offset_bytes, enabled);
}

template<uint32_t NumLanes>
__device__ __forceinline__ auto begin_lane_dynamic(
    dynamic_tensor_handle<NumLanes> handle,
    uint32_t block_id,
    uint32_t lane_index,
    bool enabled = true)
{
    uint32_t base_offset_bytes = block_id * NumLanes * handle.row_stride_bytes + lane_index * handle.row_stride_bytes;
    return lane_context_dynamic(base_offset_bytes, enabled);
}

// ============================================================================
// End functions (static lanes) - event width deduced from parameter count
// ============================================================================

// 0 params → width 2
template<uint32_t NumLanes, uint32_t MaxEventWidth, typename TraceType>
__device__ __forceinline__ void end(
    start_token start,
    static_tensor_handle<NumLanes, MaxEventWidth> handle,
    lane_context_static<MaxEventWidth>& lane,
    TraceType)
{
    if (!lane.enabled()) return;

    static_assert(TraceType::usage == lane_type::STATIC, "Trace type must be STATIC");
    static_assert(TraceType::param_count == 0, "Parameter count mismatch");

    constexpr uint32_t event_width = 2;
    static_assert(event_width <= MaxEventWidth, "Event width exceeds lane max event width");

    uint32_t end_time;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(end_time));

    if (lane.write_offset_bytes() < lane.base_offset_bytes() + handle.row_stride_bytes) {
        asm volatile("st.global.cs.v2.u32 [%0], {%1, %2};"
                     :: "l"(handle.buffer + lane.write_offset_bytes()),
                        "r"(start.time), "r"(end_time)
                     : "memory");
    }

    lane.advance();
}

// 1 param → width 4
template<uint32_t NumLanes, uint32_t MaxEventWidth, typename TraceType>
__device__ __forceinline__ void end(
    start_token start,
    static_tensor_handle<NumLanes, MaxEventWidth> handle,
    lane_context_static<MaxEventWidth>& lane,
    TraceType,
    uint32_t p0)
{
    if (!lane.enabled()) return;

    static_assert(TraceType::usage == lane_type::STATIC, "Trace type must be STATIC");
    static_assert(TraceType::param_count == 1, "Parameter count mismatch");

    constexpr uint32_t event_width = 4;
    static_assert(event_width <= MaxEventWidth, "Event width exceeds lane max event width");

    uint32_t end_time;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(end_time));

    if (lane.write_offset_bytes() < lane.base_offset_bytes() + handle.row_stride_bytes) {
        asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
                     :: "l"(handle.buffer + lane.write_offset_bytes()),
                        "r"(start.time), "r"(end_time),
                        "r"(p0), "r"(0u)
                     : "memory");
    }

    lane.advance();
}

// 2 params → width 4
template<uint32_t NumLanes, uint32_t MaxEventWidth, typename TraceType>
__device__ __forceinline__ void end(
    start_token start,
    static_tensor_handle<NumLanes, MaxEventWidth> handle,
    lane_context_static<MaxEventWidth>& lane,
    TraceType,
    uint32_t p0,
    uint32_t p1)
{
    if (!lane.enabled()) return;

    static_assert(TraceType::usage == lane_type::STATIC, "Trace type must be STATIC");
    static_assert(TraceType::param_count == 2, "Parameter count mismatch");

    constexpr uint32_t event_width = 4;
    static_assert(event_width <= MaxEventWidth, "Event width exceeds lane max event width");

    uint32_t end_time;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(end_time));

    if (lane.write_offset_bytes() < lane.base_offset_bytes() + handle.row_stride_bytes) {
        asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
                     :: "l"(handle.buffer + lane.write_offset_bytes()), "r"(start.time), "r"(end_time),
                        "r"(p0), "r"(p1)
                     : "memory");
    }

    lane.advance();
}

// 3 params → width 8
template<uint32_t NumLanes, uint32_t MaxEventWidth, typename TraceType>
__device__ __forceinline__ void end(
    start_token start,
    static_tensor_handle<NumLanes, MaxEventWidth> handle,
    lane_context_static<MaxEventWidth>& lane,
    TraceType,
    uint32_t p0,
    uint32_t p1,
    uint32_t p2)
{
    if (!lane.enabled()) return;

    static_assert(TraceType::usage == lane_type::STATIC, "Trace type must be STATIC");
    static_assert(TraceType::param_count == 3, "Parameter count mismatch");

    constexpr uint32_t event_width = 8;
    static_assert(event_width <= MaxEventWidth, "Event width exceeds lane max event width");

    uint32_t end_time;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(end_time));

    if (lane.write_offset_bytes() < lane.base_offset_bytes() + handle.row_stride_bytes) {

        asm volatile("st.global.cs.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                     :: "l"(handle.buffer + lane.write_offset_bytes()), "r"(start.time), "r"(end_time),
                        "r"(p0), "r"(p1), "r"(p2),
                        "r"(0u), "r"(0u), "r"(0u)
                     : "memory");
    }

    lane.advance();
}

// 4 params → width 8
template<uint32_t NumLanes, uint32_t MaxEventWidth, typename TraceType>
__device__ __forceinline__ void end(
    start_token start,
    static_tensor_handle<NumLanes, MaxEventWidth> handle,
    lane_context_static<MaxEventWidth>& lane,
    TraceType,
    uint32_t p0,
    uint32_t p1,
    uint32_t p2,
    uint32_t p3)
{
    if (!lane.enabled()) return;

    static_assert(TraceType::usage == lane_type::STATIC, "Trace type must be STATIC");
    static_assert(TraceType::param_count == 4, "Parameter count mismatch");

    constexpr uint32_t event_width = 8;
    static_assert(event_width <= MaxEventWidth, "Event width exceeds lane max event width");

    uint32_t end_time;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(end_time));

    if (lane.write_offset_bytes() < lane.base_offset_bytes() + handle.row_stride_bytes) {
        asm volatile("st.global.cs.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                     :: "l"(handle.buffer + lane.write_offset_bytes()), "r"(start.time), "r"(end_time),
                        "r"(p0), "r"(p1), "r"(p2), "r"(p3),
                        "r"(0u), "r"(0u)
                     : "memory");
    }

    lane.advance();
}

// 5 params → width 8
template<uint32_t NumLanes, uint32_t MaxEventWidth, typename TraceType>
__device__ __forceinline__ void end(
    start_token start,
    static_tensor_handle<NumLanes, MaxEventWidth> handle,
    lane_context_static<MaxEventWidth>& lane,
    TraceType,
    uint32_t p0,
    uint32_t p1,
    uint32_t p2,
    uint32_t p3,
    uint32_t p4)
{
    if (!lane.enabled()) return;

    static_assert(TraceType::usage == lane_type::STATIC, "Trace type must be STATIC");
    static_assert(TraceType::param_count == 5, "Parameter count mismatch");

    constexpr uint32_t event_width = 8;
    static_assert(event_width <= MaxEventWidth, "Event width exceeds lane max event width");

    uint32_t end_time;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(end_time));

    if (lane.write_offset_bytes() < lane.base_offset_bytes() + handle.row_stride_bytes) {
        asm volatile("st.global.cs.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                     :: "l"(handle.buffer + lane.write_offset_bytes()), "r"(start.time), "r"(end_time),
                        "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(p4),
                        "r"(0u)
                     : "memory");
    }

    lane.advance();
}

// 6 params → width 8
template<uint32_t NumLanes, uint32_t MaxEventWidth, typename TraceType>
__device__ __forceinline__ void end(
    start_token start,
    static_tensor_handle<NumLanes, MaxEventWidth> handle,
    lane_context_static<MaxEventWidth>& lane,
    TraceType,
    uint32_t p0,
    uint32_t p1,
    uint32_t p2,
    uint32_t p3,
    uint32_t p4,
    uint32_t p5)
{
    if (!lane.enabled()) return;

    static_assert(TraceType::usage == lane_type::STATIC, "Trace type must be STATIC");
    static_assert(TraceType::param_count == 6, "Parameter count mismatch");

    constexpr uint32_t event_width = 8;
    static_assert(event_width <= MaxEventWidth, "Event width exceeds lane max event width");

    uint32_t end_time;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(end_time));

    if (lane.write_offset_bytes() < lane.base_offset_bytes() + handle.row_stride_bytes) {
        asm volatile("st.global.cs.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                     :: "l"(handle.buffer + lane.write_offset_bytes()), "r"(start.time), "r"(end_time),
                        "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(p4), "r"(p5)
                     : "memory");
    }

    lane.advance();
}

// ============================================================================
// End function (dynamic lanes) - variadic template
// ============================================================================

// Dynamic end - 0 params
template<uint32_t NumLanes, typename TraceType>
__device__ __forceinline__ void end(
    start_token start,
    dynamic_tensor_handle<NumLanes> handle,
    lane_context_dynamic& lane,
    TraceType)
{
    if (!lane.enabled()) return;

    static_assert(TraceType::usage == lane_type::DYNAMIC, "Trace type must be DYNAMIC");
    static_assert(TraceType::param_count == 0, "Parameter count mismatch");

    uint32_t end_time;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(end_time));


    if (lane.write_offset_bytes() < lane.base_offset_bytes() + handle.row_stride_bytes) {
        asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
                     :: "l"(handle.buffer + lane.write_offset_bytes()),
                        "r"(start.time), "r"(end_time),
                        "r"((uint32_t)TraceType::id), "r"(0u)
                     : "memory");
    }

    lane.advance();
}

// Dynamic end - 1 param
template<uint32_t NumLanes, typename TraceType>
__device__ __forceinline__ void end(
    start_token start,
    dynamic_tensor_handle<NumLanes> handle,
    lane_context_dynamic& lane,
    TraceType,
    uint32_t p0)
{
    if (!lane.enabled()) return;

    static_assert(TraceType::usage == lane_type::DYNAMIC, "Trace type must be DYNAMIC");
    static_assert(TraceType::param_count == 1, "Parameter count mismatch");

    uint32_t end_time;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(end_time));


    if (lane.write_offset_bytes() < lane.base_offset_bytes() + handle.row_stride_bytes) {
        asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
                     :: "l"(handle.buffer + lane.write_offset_bytes()),
                        "r"(start.time), "r"(end_time),
                        "r"((uint32_t)TraceType::id), "r"(p0)
                     : "memory");
    }

    lane.advance();
}

// Dynamic end - 2 params
template<uint32_t NumLanes, typename TraceType>
__device__ __forceinline__ void end(
    start_token start,
    dynamic_tensor_handle<NumLanes> handle,
    lane_context_dynamic& lane,
    TraceType,
    uint32_t p0, uint32_t p1)
{
    if (!lane.enabled()) return;

    static_assert(TraceType::usage == lane_type::DYNAMIC, "Trace type must be DYNAMIC");
    static_assert(TraceType::param_count == 2, "Parameter count mismatch");

    uint32_t end_time;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(end_time));


    if (lane.write_offset_bytes() < lane.base_offset_bytes() + handle.row_stride_bytes) {
        asm volatile("st.global.cs.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                     :: "l"(handle.buffer + lane.write_offset_bytes()),
                        "r"(start.time), "r"(end_time),
                        "r"((uint32_t)TraceType::id),
                        "r"(p0), "r"(p1), "r"(0u), "r"(0u), "r"(0u)
                     : "memory");
    }

    lane.advance();
}

// Dynamic end - 3 params
template<uint32_t NumLanes, typename TraceType>
__device__ __forceinline__ void end(
    start_token start,
    dynamic_tensor_handle<NumLanes> handle,
    lane_context_dynamic& lane,
    TraceType,
    uint32_t p0, uint32_t p1, uint32_t p2)
{
    if (!lane.enabled()) return;

    static_assert(TraceType::usage == lane_type::DYNAMIC, "Trace type must be DYNAMIC");
    static_assert(TraceType::param_count == 3, "Parameter count mismatch");

    uint32_t end_time;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(end_time));


    if (lane.write_offset_bytes() < lane.base_offset_bytes() + handle.row_stride_bytes) {
        asm volatile("st.global.cs.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                     :: "l"(handle.buffer + lane.write_offset_bytes()),
                        "r"(start.time), "r"(end_time),
                        "r"((uint32_t)TraceType::id),
                        "r"(p0), "r"(p1), "r"(p2), "r"(0u), "r"(0u)
                     : "memory");
    }

    lane.advance();
}

// Dynamic end - 4 params
template<uint32_t NumLanes, typename TraceType>
__device__ __forceinline__ void end(
    start_token start,
    dynamic_tensor_handle<NumLanes> handle,
    lane_context_dynamic& lane,
    TraceType,
    uint32_t p0, uint32_t p1, uint32_t p2, uint32_t p3)
{
    if (!lane.enabled()) return;

    static_assert(TraceType::usage == lane_type::DYNAMIC, "Trace type must be DYNAMIC");
    static_assert(TraceType::param_count == 4, "Parameter count mismatch");

    uint32_t end_time;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(end_time));


    if (lane.write_offset_bytes() < lane.base_offset_bytes() + handle.row_stride_bytes) {
        asm volatile("st.global.cs.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                     :: "l"(handle.buffer + lane.write_offset_bytes()),
                        "r"(start.time), "r"(end_time),
                        "r"((uint32_t)TraceType::id),
                        "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(0u)
                     : "memory");
    }

    lane.advance();
}

// Dynamic end - 5 params
template<uint32_t NumLanes, typename TraceType>
__device__ __forceinline__ void end(
    start_token start,
    dynamic_tensor_handle<NumLanes> handle,
    lane_context_dynamic& lane,
    TraceType,
    uint32_t p0, uint32_t p1, uint32_t p2, uint32_t p3, uint32_t p4)
{
    if (!lane.enabled()) return;

    static_assert(TraceType::usage == lane_type::DYNAMIC, "Trace type must be DYNAMIC");
    static_assert(TraceType::param_count == 5, "Parameter count mismatch");

    uint32_t end_time;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(end_time));


    if (lane.write_offset_bytes() < lane.base_offset_bytes() + handle.row_stride_bytes) {
        asm volatile("st.global.cs.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                     :: "l"(handle.buffer + lane.write_offset_bytes()),
                        "r"(start.time), "r"(end_time),
                        "r"((uint32_t)TraceType::id),
                        "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(p4)
                     : "memory");
    }

    lane.advance();
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

    // Write raw byte offset - host will calculate event count later
    uint32_t write_offset_bytes = lane.write_offset_bytes();

    uint64_t anchor_time = read_global_timer();
    uint32_t anchor_low = static_cast<uint32_t>(anchor_time);
    uint32_t anchor_high = static_cast<uint32_t>(anchor_time >> 32);

    // Header: [sm_id, write_offset_bytes, anchor_low, anchor_high]
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
                 :: "l"(handle.buffer + lane.base_offset_bytes()),
                    "r"(sm_id), "r"(write_offset_bytes),
                    "r"(anchor_low), "r"(anchor_high)
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

    // Write raw byte offset - host will calculate event count later
    uint32_t write_offset_bytes = lane.write_offset_bytes();

    uint64_t anchor_time = read_global_timer();
    uint32_t anchor_low = static_cast<uint32_t>(anchor_time);
    uint32_t anchor_high = static_cast<uint32_t>(anchor_time >> 32);

    // Header: [sm_id, write_offset_bytes, anchor_low, anchor_high]
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
                 :: "l"(handle.buffer + lane.base_offset_bytes()),
                    "r"(sm_id), "r"(write_offset_bytes),
                    "r"(anchor_low), "r"(anchor_high)
                 : "memory");
}

} // namespace nanotrace

#endif
