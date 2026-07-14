#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <utility>
#include <cuda_runtime.h>

#include "nanotrace.cuh"
#include "nanotrace_session.h"

// format_descriptor is now defined in nanotrace.cuh for NANOTRACE_DISABLED compatibility

namespace nanotrace {

[[noreturn]] inline void Fail(const char* message)
{
    std::fprintf(stderr, "nanotrace fatal error: %s\n",
        message ? message : "unknown error");
    std::abort();
}

struct event_parent_interval {
    uint64_t start_ns;
    uint64_t end_ns;
    EventId event_id;
};

}

#ifdef NANOTRACE_DISABLED

namespace nanotrace {

// ============================================================================
// Disabled stub implementations - do nothing but maintain API
// ============================================================================

template<uint32_t NumLanes, typename... TraceTypes>
class static_trace_builder {
public:
    static constexpr uint32_t max_event_width = []() {
        uint32_t widths[] = {
            (TraceTypes::param_count == 0 ? 2 :
             TraceTypes::param_count <= 2 ? 4 : 8)...
        };
        uint32_t max_width = 0;
        for (uint32_t width : widths) {
            if (width > max_width) max_width = width;
        }
        return max_width;
    }();

    static_trace_builder(uint32_t, dim3, dim3 = dim3(0, 0, 0)) {}
    ~static_trace_builder() = default;

    static_trace_builder(const static_trace_builder&) = delete;
    static_trace_builder& operator=(const static_trace_builder&) = delete;
    static_trace_builder(static_trace_builder&&) = delete;
    static_trace_builder& operator=(static_trace_builder&&) = delete;

    static_tensor_handle<NumLanes, max_event_width> get_handle() const {
        return static_tensor_handle<NumLanes, max_event_width>{};
    }

    template<uint32_t LaneIndex>
    static constexpr uint16_t get_lane_format_id() { return 0; }

    void reset() {}

    template<typename TrackType>
    void set_track_type() {}

    template<typename TrackType>
    void set_track_type(uint32_t) {}

    uint32_t* d_buffer = nullptr;
    dim3 grid_dims{};
    dim3 cluster_dims{};
    uint32_t total_blocks = 0;
    uint32_t row_stride = 0;
    size_t buffer_size = 0;
    uint16_t default_track_format_id = UINT16_MAX;
    std::unordered_map<uint32_t, uint16_t> lane_track_format_ids;
    std::vector<format_descriptor> format_descriptors;
};

template<uint32_t NumLanes>
class dynamic_trace_builder {
public:
    static constexpr uint32_t event_width = 8;

    dynamic_trace_builder(uint32_t, dim3, dim3 = dim3(0, 0, 0)) {}
    ~dynamic_trace_builder() = default;

    dynamic_trace_builder(const dynamic_trace_builder&) = delete;
    dynamic_trace_builder& operator=(const dynamic_trace_builder&) = delete;
    dynamic_trace_builder(dynamic_trace_builder&&) = delete;
    dynamic_trace_builder& operator=(dynamic_trace_builder&&) = delete;

    dynamic_tensor_handle<NumLanes> get_handle() const {
        return dynamic_tensor_handle<NumLanes>{};
    }

    void reset() {}

    template<typename TrackType>
    void set_track_type() {}

    template<typename TrackType>
    void set_track_type(uint32_t) {}

    uint32_t* d_buffer = nullptr;
    dim3 grid_dims{};
    dim3 cluster_dims{};
    uint32_t total_blocks = 0;
    uint32_t row_stride = 0;
    size_t buffer_size = 0;
    uint16_t default_track_format_id = UINT16_MAX;
    std::unordered_map<uint32_t, uint16_t> lane_track_format_ids;
    std::vector<format_descriptor> format_descriptors;
};

class trace_writer {
public:
    trace_writer(const char*) {}
    ~trace_writer() = default;

    trace_writer(const trace_writer&) = delete;
    trace_writer& operator=(const trace_writer&) = delete;
    trace_writer(trace_writer&&) = delete;
    trace_writer& operator=(trace_writer&&) = delete;

    template<typename TraceType>
    void register_trace_type() {}

    template<typename BlockType>
    void set_block_type() {}

    template<uint32_t NumLanes, typename... TraceTypes>
    void add_tensor(const static_trace_builder<NumLanes, TraceTypes...>&) {}

    template<uint32_t NumLanes>
    void add_tensor(const dynamic_trace_builder<NumLanes>&) {}

    void write(const char*, bool = true) {
        fprintf(stderr, "Warning: No trace events found\n");
    }
};

} // namespace nanotrace

#else // NANOTRACE_DISABLED

namespace nanotrace {

// ============================================================================
// Static trace builder
// ============================================================================

template<uint32_t NumLanes, typename... TraceTypes>
class static_trace_builder {
public:
    static_assert(sizeof...(TraceTypes) == NumLanes,
                  "Must provide exactly one TraceType per lane");

    static_assert((... && (TraceTypes::usage == lane_type::STATIC)),
                  "All trace types must be STATIC");

    // Compute max event width at compile time
    static constexpr uint32_t max_event_width = []() {
        uint32_t widths[] = {
            (TraceTypes::param_count == 0 ? 2 :
             TraceTypes::param_count <= 2 ? 4 : 8)...
        };
        uint32_t max_w = 0;
        for (uint32_t w : widths) {
            if (w > max_w) max_w = w;
        }
        return max_w;
    }();

    static_trace_builder(uint32_t max_events_per_lane, dim3 grid_dims, dim3 cluster_dims = dim3(0, 0, 0))
    {
        this->grid_dims = grid_dims;
        this->cluster_dims = cluster_dims;
        total_blocks = grid_dims.x * grid_dims.y * grid_dims.z;
        constexpr uint32_t HEADER_WORDS = 4;
        constexpr uint32_t ROW_ALIGNMENT_WORDS = 4;
        uint32_t row_words =
            HEADER_WORDS + max_events_per_lane * max_event_width;
        row_stride = (row_words + ROW_ALIGNMENT_WORDS - 1)
            & ~(ROW_ALIGNMENT_WORDS - 1);

        // Check for uint32 byte offset overflow
        uint64_t row_stride_bytes = static_cast<uint64_t>(row_stride) * 4;
        uint64_t max_offset_bytes = static_cast<uint64_t>(total_blocks) * NumLanes * row_stride_bytes;
        if (max_offset_bytes > UINT32_MAX) {
            Fail("Tensor configuration exceeds uint32 byte offsets");
        }

        buffer_size = static_cast<size_t>(total_blocks) * NumLanes * row_stride * sizeof(uint32_t);
        cudaMalloc(&d_buffer, buffer_size);

        // Store trace type descriptors for auto-registration
        (format_descriptors.push_back(TraceTypes::descriptor), ...);
    }

    ~static_trace_builder() {
        if (d_buffer) cudaFree(d_buffer);
    }

    // Deleted copy/move to prevent double-free
    static_trace_builder(const static_trace_builder&) = delete;
    static_trace_builder& operator=(const static_trace_builder&) = delete;
    static_trace_builder(static_trace_builder&&) = delete;
    static_trace_builder& operator=(static_trace_builder&&) = delete;

    static_tensor_handle<NumLanes, max_event_width> get_handle() const {
        return {reinterpret_cast<uint8_t*>(d_buffer), row_stride << 2};
    }

    template<uint32_t LaneIndex>
    static constexpr uint16_t get_lane_format_id() {
        static_assert(LaneIndex < NumLanes, "Lane index out of bounds");
        return std::tuple_element_t<LaneIndex, std::tuple<TraceTypes...>>::id;
    }

    void reset() {
        cudaMemset(d_buffer, 0, buffer_size);
    }

    template<typename TrackType>
    void set_track_type() {
        static_assert(TrackType::is_track_type, "Must use NANOTRACE_DEFINE_TRACK_TYPE macro");
        default_track_format_id = TrackType::id;
        // Store format descriptor for auto-registration
        format_descriptors.push_back(TrackType::descriptor);
    }

    template<typename TrackType>
    void set_track_type(uint32_t lane) {
        static_assert(TrackType::is_track_type, "Must use NANOTRACE_DEFINE_TRACK_TYPE macro");
        lane_track_format_ids[lane] = TrackType::id;
        // Store format descriptor for auto-registration
        format_descriptors.push_back(TrackType::descriptor);
    }

    uint32_t* d_buffer;
    dim3 grid_dims;
    dim3 cluster_dims;
    uint32_t total_blocks;
    uint32_t row_stride;
    size_t buffer_size;
    uint16_t default_track_format_id = UINT16_MAX;
    std::unordered_map<uint32_t, uint16_t> lane_track_format_ids;
    std::vector<format_descriptor> format_descriptors;
};

// ============================================================================
// Dynamic trace builder
// ============================================================================

template<uint32_t NumLanes>
class dynamic_trace_builder {
public:
    static constexpr uint32_t event_width = 8;

    dynamic_trace_builder(uint32_t max_events_per_lane, dim3 grid_dims, dim3 cluster_dims = dim3(0, 0, 0))
    {
        this->grid_dims = grid_dims;
        this->cluster_dims = cluster_dims;
        total_blocks = grid_dims.x * grid_dims.y * grid_dims.z;
        constexpr uint32_t HEADER_WORDS = 4;
        constexpr uint32_t ROW_ALIGNMENT_WORDS = 4;
        uint32_t row_words = HEADER_WORDS + max_events_per_lane * event_width;
        row_stride = (row_words + ROW_ALIGNMENT_WORDS - 1)
            & ~(ROW_ALIGNMENT_WORDS - 1);

        // Check for uint32 byte offset overflow
        uint64_t row_stride_bytes = static_cast<uint64_t>(row_stride) * 4;
        uint64_t max_offset_bytes = static_cast<uint64_t>(total_blocks) * NumLanes * row_stride_bytes;
        if (max_offset_bytes > UINT32_MAX) {
            Fail("Tensor configuration exceeds uint32 byte offsets");
        }

        buffer_size = static_cast<size_t>(total_blocks) * NumLanes * row_stride * sizeof(uint32_t);
        cudaMalloc(&d_buffer, buffer_size);
    }

    ~dynamic_trace_builder() {
        if (d_buffer) cudaFree(d_buffer);
    }

    // Deleted copy/move to prevent double-free
    dynamic_trace_builder(const dynamic_trace_builder&) = delete;
    dynamic_trace_builder& operator=(const dynamic_trace_builder&) = delete;
    dynamic_trace_builder(dynamic_trace_builder&&) = delete;
    dynamic_trace_builder& operator=(dynamic_trace_builder&&) = delete;

    dynamic_tensor_handle<NumLanes> get_handle() const {
        return {reinterpret_cast<uint8_t*>(d_buffer), row_stride << 2};
    }

    void reset() {
        cudaMemset(d_buffer, 0, buffer_size);
    }

    template<typename TrackType>
    void set_track_type() {
        static_assert(TrackType::is_track_type, "Must use NANOTRACE_DEFINE_TRACK_TYPE macro");
        default_track_format_id = TrackType::id;
        // Store format descriptor for auto-registration
        format_descriptors.push_back(TrackType::descriptor);
    }

    template<typename TrackType>
    void set_track_type(uint32_t lane) {
        static_assert(TrackType::is_track_type, "Must use NANOTRACE_DEFINE_TRACK_TYPE macro");
        lane_track_format_ids[lane] = TrackType::id;
        // Store format descriptor for auto-registration
        format_descriptors.push_back(TrackType::descriptor);
    }

    uint32_t* d_buffer;
    dim3 grid_dims;
    dim3 cluster_dims;
    uint32_t total_blocks;
    uint32_t row_stride;
    size_t buffer_size;
    uint16_t default_track_format_id = UINT16_MAX;
    std::unordered_map<uint32_t, uint16_t> lane_track_format_ids;
    std::vector<format_descriptor> format_descriptors;
};

// ============================================================================
// Trace writer
// ============================================================================

class trace_writer {
public:
    trace_writer(const char* kernel_name);
    ~trace_writer();

    // Deleted copy/move
    trace_writer(const trace_writer&) = delete;
    trace_writer& operator=(const trace_writer&) = delete;
    trace_writer(trace_writer&&) = delete;
    trace_writer& operator=(trace_writer&&) = delete;

    template<typename TraceType>
    void register_trace_type();

    template<typename BlockType>
    void set_block_type();

    template<uint32_t NumLanes, typename... TraceTypes>
    void add_tensor(const static_trace_builder<NumLanes, TraceTypes...>& builder);

    template<uint32_t NumLanes>
    void add_tensor(const dynamic_trace_builder<NumLanes>& builder);

    bool AppendToSession(TraceSession& session, TrackId parent_track,
        ClockId gpu_clock, ClockId anchor_clock,
        uint64_t reference_anchor_ns,
        EventId parent_event = INVALID_EVENT_ID,
        uint64_t uncertainty_ns = 0,
        const std::vector<event_parent_interval>* parent_intervals = nullptr,
        uint32_t displayed_block_count = 0,
        bool parents_indexed_by_block = false);
    void write(const char* filename, bool compress = true);
    const char* KernelName() const { return kernel_name.c_str(); }

private:
    struct tensor_info {
        std::vector<uint32_t> host_buffer;  // Copied from device
        dim3 grid_dims;
        dim3 cluster_dims;
        uint32_t total_blocks;
        uint32_t num_lanes;
        uint32_t row_stride;
        uint32_t event_width;
        std::vector<uint16_t> lane_format_ids;  // 0xFFFF for dynamic lanes
        uint16_t default_track_format_id;  // Default track type for this tensor
        std::unordered_map<uint32_t, uint16_t> lane_track_format_ids;  // Per-lane overrides (relative to tensor)
    };

public:
    struct parsed_event {
        uint32_t block_id;
        uint32_t lane_id;
        uint64_t anchor_time;
        uint64_t time_offset;    // 64-bit after unwrap, ns from kernel start after conversion
        uint32_t duration;       // Nanoseconds
        uint16_t sm_id;
        uint16_t format_id;
        uint16_t track_format_id;
        uint8_t param_count;
        uint32_t params[6];      // Maximum 6 parameters (static lanes)
    };

private:
    std::string kernel_name;
    uint16_t default_block_format_id;
    std::vector<format_descriptor> formats;
    std::vector<tensor_info> tensors;

    std::vector<parsed_event> parse_all_events();
};

// ============================================================================
// Template implementations
// ============================================================================

template<typename TraceType>
void trace_writer::register_trace_type() {
    // Check for duplicate trace type ID
    for (const auto& fmt : formats) {
        if (fmt.id == TraceType::id) {
            Fail("Duplicate trace type ID");
        }
    }

    formats.push_back(TraceType::descriptor);
}

template<typename BlockType>
void trace_writer::set_block_type() {
    static_assert(BlockType::is_block_type, "Must use NANOTRACE_DEFINE_BLOCK_TYPE macro");

    // Check for duplicate block type ID
    for (const auto& fmt : formats) {
        if (fmt.id == BlockType::id) {
            Fail("Duplicate block type ID");
        }
    }

    default_block_format_id = BlockType::id;
    formats.push_back(BlockType::descriptor);
}

template<uint32_t NumLanes, typename... TraceTypes>
void trace_writer::add_tensor(const static_trace_builder<NumLanes, TraceTypes...>& builder) {
    // Extract lane format IDs for static tensor using index_sequence
    std::vector<uint16_t> format_ids;
    format_ids.reserve(NumLanes);

    [&]<size_t... Is>(std::index_sequence<Is...>) {
        (format_ids.push_back(builder.template get_lane_format_id<Is>()), ...);
    }(std::make_index_sequence<NumLanes>{});

    // Auto-register format descriptors from builder (trace types and track types)
    for (const auto& desc : builder.format_descriptors) {
        // Check if already registered or if ID collision
        bool found = false;
        for (const auto& fmt : formats) {
            if (fmt.id == desc.id) {
                // Check if it's the same type or a collision
                if (!(fmt == desc)) {
                    Fail("Duplicate format ID from __COUNTER__ collision");
                }
                found = true;
                break;
            }
        }
        // Auto-register if not found
        if (!found) {
            formats.push_back(desc);
        }
    }

    // Allocate host buffer and copy from device
    size_t buffer_size = builder.total_blocks * NumLanes * builder.row_stride;
    std::vector<uint32_t> host_buffer(buffer_size);

    cudaMemcpy(host_buffer.data(), builder.d_buffer,
               buffer_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    tensors.push_back({
        std::move(host_buffer),
        builder.grid_dims,
        builder.cluster_dims,
        builder.total_blocks,
        NumLanes,
        builder.row_stride,
        builder.max_event_width,
        std::move(format_ids),
        builder.default_track_format_id,
        builder.lane_track_format_ids
    });
}

template<uint32_t NumLanes>
void trace_writer::add_tensor(const dynamic_trace_builder<NumLanes>& builder) {
    // Dynamic tensor: all lanes have format_id = 0xFFFF
    std::vector<uint16_t> format_ids(NumLanes, 0xFFFF);

    // Auto-register format descriptors from builder (track types)
    for (const auto& desc : builder.format_descriptors) {
        // Check if already registered or if ID collision
        bool found = false;
        for (const auto& fmt : formats) {
            if (fmt.id == desc.id) {
                // Check if it's the same type or a collision
                if (!(fmt == desc)) {
                    Fail("Duplicate format ID from __COUNTER__ collision");
                }
                found = true;
                break;
            }
        }
        // Auto-register if not found
        if (!found) {
            formats.push_back(desc);
        }
    }

    // Allocate host buffer and copy from device
    size_t buffer_size = builder.total_blocks * NumLanes * builder.row_stride;
    std::vector<uint32_t> host_buffer(buffer_size);

    cudaMemcpy(host_buffer.data(), builder.d_buffer,
               buffer_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    tensors.push_back({
        std::move(host_buffer),
        builder.grid_dims,
        builder.cluster_dims,
        builder.total_blocks,
        NumLanes,
        builder.row_stride,
        8,  // event_width for dynamic
        std::move(format_ids),
        builder.default_track_format_id,
        builder.lane_track_format_ids
    });
}

} // namespace nanotrace

#endif // NANOTRACE_DISABLED
