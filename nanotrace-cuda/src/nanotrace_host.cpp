#include "nanotrace/nanotrace_host.h"
#include <cstdio>
#include <map>
#include <unordered_map>
#include <algorithm>

#ifdef NANOTRACE_WITH_MINIZ
#include "miniz.h"
#endif

#if !defined(NANOTRACE_DISABLED)

namespace nanotrace {

trace_writer::trace_writer(const char* kernel_name)
    : kernel_name(kernel_name)
    , default_block_format_id(UINT16_MAX)
{}

trace_writer::~trace_writer() = default;

// Parse all events from all tensors into a single vector
std::vector<trace_writer::parsed_event> trace_writer::parse_all_events() {
    std::vector<parsed_event> events;

    // Validate all tensors have same grid dimensions (required for lane stacking)
    if (!tensors.empty()) {
        dim3 reference_grid = tensors[0].grid_dims;
        for (size_t i = 1; i < tensors.size(); ++i) {
            if (tensors[i].grid_dims.x != reference_grid.x ||
                tensors[i].grid_dims.y != reference_grid.y ||
                tensors[i].grid_dims.z != reference_grid.z) {
                Fail("All tensors must have identical grid dimensions");
            }
        }
    }

    uint32_t global_lane_offset = 0;  // Stack lanes within each block

    for (const auto& tensor : tensors) {
        for (uint32_t block_id = 0; block_id < tensor.total_blocks; ++block_id) {
            for (uint32_t lane_id = 0; lane_id < tensor.num_lanes; ++lane_id) {
                uint32_t base_offset = block_id * tensor.num_lanes * tensor.row_stride +
                                      lane_id * tensor.row_stride;

                uint16_t sm_id = tensor.host_buffer[base_offset];
                uint32_t write_offset_bytes = tensor.host_buffer[base_offset + 1];
                uint64_t anchor_time = tensor.host_buffer[base_offset + 2]
                    | (static_cast<uint64_t>(tensor.host_buffer[base_offset + 3]) << 32);

                if (write_offset_bytes == 0) continue;

                // Compute event count from byte offset (device writes raw offset to avoid division)
                uint32_t base_offset_bytes = base_offset * 4;
                uint32_t event_width_bytes = tensor.event_width * 4;
                uint32_t row_stride_bytes = tensor.row_stride * 4;
                uint32_t max_offset_bytes = base_offset_bytes + row_stride_bytes;

                // Check for overflow
                if (write_offset_bytes > max_offset_bytes) {
                    Fail("Trace lane write exceeded its allocated capacity");
                }

                constexpr uint32_t HEADER_BYTES = 4 * sizeof(uint32_t);
                uint32_t event_count =
                    (write_offset_bytes - base_offset_bytes - HEADER_BYTES)
                    / event_width_bytes;

                if (event_count == 0) continue;

                uint32_t event_offset = base_offset + 4;

                for (uint32_t event_idx = 0; event_idx < event_count; ++event_idx) {
                    uint32_t start_time = tensor.host_buffer[event_offset];
                    uint32_t end_time = tensor.host_buffer[event_offset + 1];
                    uint32_t duration = end_time - start_time;  // Unsigned wrap handles it correctly

                    if (start_time == 0) continue;  // Skip invalid events

                    parsed_event evt;
                    evt.block_id = block_id;  // Same block ID across tensors
                    evt.lane_id = global_lane_offset + lane_id;  // Offset lane ID by tensor
                    evt.anchor_time = anchor_time;
                    evt.time_offset = anchor_time
                        - static_cast<uint32_t>(
                            static_cast<uint32_t>(anchor_time) - start_time);
                    evt.duration = duration;
                    evt.sm_id = sm_id;
                    evt.param_count = 0;
                    const std::unordered_map<uint32_t, uint16_t>::const_iterator
                        track_override =
                            tensor.lane_track_format_ids.find(lane_id);
                    evt.track_format_id = track_override
                            != tensor.lane_track_format_ids.end()
                        ? track_override->second
                        : tensor.default_track_format_id;

                    // Extract format ID and params
                    if (tensor.lane_format_ids[lane_id] == 0xFFFF) {
                        // Dynamic lane
                        evt.format_id = tensor.host_buffer[event_offset + 2] & 0xFFFF;
                        for (const auto& fmt : formats) {
                            if (fmt.id == evt.format_id) {
                                evt.param_count = fmt.param_count;
                                for (uint8_t p = 0; p < fmt.param_count; ++p) {
                                    evt.params[p] = tensor.host_buffer[event_offset + 3 + p];
                                }
                                break;
                            }
                        }
                    } else {
                        // Static lane
                        evt.format_id = tensor.lane_format_ids[lane_id];
                        if (tensor.event_width >= 4) {
                            for (const auto& fmt : formats) {
                                if (fmt.id == evt.format_id) {
                                    evt.param_count = fmt.param_count;
                                    for (uint8_t p = 0; p < fmt.param_count; ++p) {
                                        evt.params[p] = tensor.host_buffer[event_offset + 2 + p];
                                    }
                                    break;
                                }
                            }
                        }
                    }

                    events.push_back(std::move(evt));
                    event_offset += tensor.event_width;
                }
            }
        }

        global_lane_offset += tensor.num_lanes;  // Advance lane offset for next tensor
    }

    return events;
}

static std::string ReplaceAll(std::string value,
    const std::string& pattern, const std::string& replacement)
{
    size_t offset = 0;

    while ((offset = value.find(pattern, offset)) != std::string::npos)
    {
        value.replace(offset, pattern.size(), replacement);
        offset += replacement.size();
    }

    return value;
}

static const format_descriptor* FindFormat(
    const std::vector<format_descriptor>& formats, uint16_t format_id)
{
    for (const format_descriptor& format : formats)
    {
        if (format.id == format_id)
        {
            return &format;
        }
    }

    return nullptr;
}

static std::string FormatBlockName(const format_descriptor* format,
    uint32_t block_id, dim3 grid_dims, dim3 cluster_dims)
{
    if (!format || !format->label_string)
    {
        return "Block " + std::to_string(block_id);
    }

    uint32_t block_x = block_id % grid_dims.x;
    uint32_t block_y = (block_id / grid_dims.x) % grid_dims.y;
    uint32_t block_z = block_id / (grid_dims.x * grid_dims.y);
    uint32_t cluster_size_x = cluster_dims.x == 0 ? 1 : cluster_dims.x;
    uint32_t cluster_size_y = cluster_dims.y == 0 ? 1 : cluster_dims.y;
    uint32_t cluster_size_z = cluster_dims.z == 0 ? 1 : cluster_dims.z;
    uint32_t cluster_x = block_x / cluster_size_x;
    uint32_t cluster_y = block_y / cluster_size_y;
    uint32_t cluster_z = block_z / cluster_size_z;
    uint32_t cluster_count_x =
        (grid_dims.x + cluster_size_x - 1) / cluster_size_x;
    uint32_t cluster_count_y =
        (grid_dims.y + cluster_size_y - 1) / cluster_size_y;
    uint32_t cluster_id = cluster_x
        + cluster_count_x * (cluster_y + cluster_count_y * cluster_z);
    std::string name = format->label_string;
    name = ReplaceAll(std::move(name), "{blockX}", std::to_string(block_x));
    name = ReplaceAll(std::move(name), "{blockY}", std::to_string(block_y));
    name = ReplaceAll(std::move(name), "{blockZ}", std::to_string(block_z));
    name = ReplaceAll(
        std::move(name), "{blockLinear}", std::to_string(block_id));
    name = ReplaceAll(
        std::move(name), "{clusterX}", std::to_string(cluster_x));
    name = ReplaceAll(
        std::move(name), "{clusterY}", std::to_string(cluster_y));
    name = ReplaceAll(
        std::move(name), "{clusterZ}", std::to_string(cluster_z));
    return ReplaceAll(
        std::move(name), "{clusterLinear}", std::to_string(cluster_id));
}

static std::string FormatTrackName(const format_descriptor* format,
    uint32_t lane_id)
{
    if (!format || !format->label_string)
    {
        return "Lane " + std::to_string(lane_id);
    }

    return ReplaceAll(format->label_string,
        "{lane}", std::to_string(lane_id));
}

bool trace_writer::AppendToSession(TraceSession& session,
    TrackId parent_track, ClockId gpu_clock, ClockId anchor_clock,
    uint64_t reference_anchor_ns, EventId parent_event,
    uint64_t uncertainty_ns,
    const std::vector<event_parent_interval>* parent_intervals,
    uint32_t displayed_block_count,
    bool parents_indexed_by_block)
{
    std::vector<parsed_event> events = parse_all_events();

    if (events.empty())
    {
#ifndef NANOTRACE_NO_LOG
        fprintf(stderr, "Warning: No trace events found\n");
#endif
        return false;
    }

    uint64_t raw_anchor = 0;
    for (const parsed_event& event : events)
    {
        raw_anchor = std::max(raw_anchor, event.anchor_time);
    }

    std::vector<EventId> resolved_parent_events(events.size(), parent_event);
    std::vector<uint64_t> resolved_time_offsets;
    resolved_time_offsets.reserve(events.size());
    for (const parsed_event& event : events)
    {
        resolved_time_offsets.push_back(event.time_offset);
    }

    if (parent_intervals)
    {
        for (size_t event_index = 0; event_index < events.size(); ++event_index)
        {
            const parsed_event& event = events[event_index];
            uint64_t event_start_ns;
            uint64_t event_end_ns;
            const event_parent_interval* parent = nullptr;

            if (parents_indexed_by_block)
            {
                if (event.block_id >= parent_intervals->size()
                    || event.anchor_time < event.time_offset + event.duration)
                {
#ifndef NANOTRACE_NO_LOG
                    fprintf(stderr, "Nanotrace block has no indexed parent\n");
#endif
                    return false;
                }

                parent = &(*parent_intervals)[event.block_id];
                uint64_t start_to_anchor =
                    event.anchor_time - event.time_offset;
                uint64_t end_to_anchor = event.anchor_time
                    - (event.time_offset + event.duration);
                if (parent->end_ns < start_to_anchor)
                {
#ifndef NANOTRACE_NO_LOG
                    fprintf(stderr, "Nanotrace event predates its parent\n");
#endif
                    return false;
                }

                event_start_ns = parent->end_ns - start_to_anchor;
                event_end_ns = parent->end_ns - end_to_anchor;
                resolved_time_offsets[event_index] = event_start_ns
                    <= reference_anchor_ns
                    ? raw_anchor - (reference_anchor_ns - event_start_ns)
                    : raw_anchor + (event_start_ns - reference_anchor_ns);
            }
            else
            {
                event_start_ns = event.time_offset <= raw_anchor
                    ? reference_anchor_ns - (raw_anchor - event.time_offset)
                    : reference_anchor_ns + (event.time_offset - raw_anchor);
                event_end_ns = event_start_ns + event.duration;
                std::vector<event_parent_interval>::const_iterator position =
                    std::upper_bound(parent_intervals->begin(),
                        parent_intervals->end(), event_start_ns,
                        [](uint64_t timestamp,
                            const event_parent_interval& interval)
                        {
                            return timestamp < interval.start_ns;
                        });

                if (position == parent_intervals->begin())
                {
#ifndef NANOTRACE_NO_LOG
                    fprintf(stderr, "Nanotrace event has no enclosing parent\n");
#endif
                    return false;
                }

                parent = &*--position;
            }

            if (event_start_ns + uncertainty_ns < parent->start_ns
                || event_end_ns > parent->end_ns + uncertainty_ns)
            {
#ifndef NANOTRACE_NO_LOG
                fprintf(stderr,
                    "Nanotrace event block %u lane %u [%llu, %llu] falls "
                    "outside parent [%llu, %llu], anchor %llu -> %llu\n",
                    event.block_id, event.lane_id,
                    static_cast<unsigned long long>(event_start_ns),
                    static_cast<unsigned long long>(event_end_ns),
                    static_cast<unsigned long long>(parent->start_ns),
                    static_cast<unsigned long long>(parent->end_ns),
                    static_cast<unsigned long long>(raw_anchor),
                    static_cast<unsigned long long>(reference_anchor_ns));
#endif
                return false;
            }

            resolved_parent_events[event_index] = parent->event_id;
        }
    }

    session.AddClockSnapshot(gpu_clock, anchor_clock,
        raw_anchor, reference_anchor_ns, uncertainty_ns);

    TrackId kernel_track = session.AddTrack(kernel_name.c_str(),
        TrackKind::Kernel, gpu_clock, parent_track, 0);
    std::unordered_map<uint16_t, TrackId> sm_tracks;
    std::map<std::pair<uint16_t, uint32_t>, TrackId> block_tracks;
    std::map<std::tuple<uint16_t, uint32_t, uint32_t>, TrackId> lane_tracks;
    const format_descriptor* block_format = FindFormat(
        formats, default_block_format_id);
    dim3 grid_dims = tensors.front().grid_dims;
    dim3 cluster_dims = tensors.front().cluster_dims;

    std::unordered_map<uint16_t, bool> registered_event_formats;
    for (const parsed_event& event : events)
    {
        if (!registered_event_formats.emplace(event.format_id, true).second)
        {
            continue;
        }

        const format_descriptor* format = FindFormat(formats, event.format_id);
        if (format)
        {
            session.RegisterEventFormat(format->label_string,
                format->tooltip_string, format->param_count);
        }
    }

    for (size_t event_index = 0; event_index < events.size(); ++event_index)
    {
        const parsed_event& event = events[event_index];
        uint32_t displayed_block_id = displayed_block_count == 0
            ? event.block_id : event.block_id % displayed_block_count;
        TrackId sm_track;
        const std::unordered_map<uint16_t, TrackId>::const_iterator sm =
            sm_tracks.find(event.sm_id);
        if (sm == sm_tracks.end())
        {
            std::string name = "SM " + std::to_string(event.sm_id);
            sm_track = session.AddTrack(name.c_str(),
                TrackKind::StreamingMultiprocessor, gpu_clock,
                kernel_track, event.sm_id, event.sm_id);
            sm_tracks.emplace(event.sm_id, sm_track);
        }
        else
        {
            sm_track = sm->second;
        }

        std::pair<uint16_t, uint32_t> block_key{
            event.sm_id, displayed_block_id };
        TrackId block_track;
        const std::map<std::pair<uint16_t, uint32_t>, TrackId>::const_iterator
            block = block_tracks.find(block_key);
        if (block == block_tracks.end())
        {
            std::string name = FormatBlockName(
                block_format, displayed_block_id, grid_dims, cluster_dims);
            block_track = session.AddTrack(name.c_str(),
                TrackKind::ThreadBlock, gpu_clock, sm_track,
                static_cast<int32_t>(displayed_block_id), displayed_block_id);
            block_tracks.emplace(block_key, block_track);
        }
        else
        {
            block_track = block->second;
        }

        std::tuple<uint16_t, uint32_t, uint32_t> lane_key{
            event.sm_id, displayed_block_id, event.lane_id };
        TrackId lane_track;
        const std::map<std::tuple<uint16_t, uint32_t, uint32_t>,
            TrackId>::const_iterator
            lane = lane_tracks.find(lane_key);
        if (lane == lane_tracks.end())
        {
            std::string name = FormatTrackName(
                FindFormat(formats, event.track_format_id), event.lane_id);
            lane_track = session.AddTrack(name.c_str(), TrackKind::Warp,
                gpu_clock, block_track, static_cast<int32_t>(event.lane_id),
                event.lane_id);
            lane_tracks.emplace(lane_key, lane_track);
        }
        else
        {
            lane_track = lane->second;
        }

        const format_descriptor* format = FindFormat(
            formats, event.format_id);
        const char* event_name = format && format->label_string
            ? format->label_string : "GPU event";
        uint64_t event_duration = event.duration == 0 ? 32 : event.duration;
        uint64_t event_start = resolved_time_offsets[event_index];
        if (!parent_intervals && parent_event != INVALID_EVENT_ID
            && event_start + event_duration > reference_anchor_ns)
        {
            event_start = reference_anchor_ns >= event_duration
                ? reference_anchor_ns - event_duration : 0;
        }
        EventId event_id = session.AddSlice(lane_track, event_name,
            event_start, event_duration,
            0, resolved_parent_events[event_index], 0);
        for (uint8_t i = 0; i < event.param_count; ++i)
        {
            std::string fallback_name = "parameter_" + std::to_string(i);
            const char* argument_name = format && format->param_names
                ? format->param_names[i] : fallback_name.c_str();
            session.AddUnsignedArgument(
                event_id, argument_name, event.params[i]);
        }
    }

    return true;
}

void trace_writer::write(const char* filename, bool compress)
{
    static_cast<void>(compress);
    TraceSession session{ kernel_name.c_str() };
    ClockId gpu_clock = session.AddClock(
        "GPU global timer", ClockKind::GpuGlobalTimer);

    if (!AppendToSession(session, INVALID_TRACK_ID, gpu_clock,
        session.ReferenceClock(), 0))
    {
        return;
    }

    if (!session.Write(filename))
    {
#ifndef NANOTRACE_NO_LOG
        fprintf(stderr, "Nanotrace write failed: %s\n",
            session.LastError().c_str());
#endif
    }
}

} // namespace nanotrace

#endif
