#include "nanotrace/nanotrace_host.h"
#include <fstream>
#include <cstring>
#include <cstdio>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <stdexcept>

#ifdef NANOTRACE_WITH_MINIZ
#include "miniz.h"
#endif

namespace nanotrace {

// Hash functions for unordered containers with composite keys
struct pair_hash {
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

struct tuple3_hash {
    template <typename T1, typename T2, typename T3>
    std::size_t operator()(const std::tuple<T1, T2, T3>& t) const {
        auto h1 = std::hash<T1>{}(std::get<0>(t));
        auto h2 = std::hash<T2>{}(std::get<1>(t));
        auto h3 = std::hash<T3>{}(std::get<2>(t));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

trace_writer::trace_writer(const char* kernel_name)
    : kernel_name(kernel_name)
    , default_block_format_id(0)  // Must call set_block_type<>() before write()
    , total_lanes_so_far(0)
{}

trace_writer::~trace_writer() = default;

void trace_writer::write_uint8(std::vector<uint8_t>& buf, uint8_t val) {
    buf.push_back(val);
}

void trace_writer::write_uint16(std::vector<uint8_t>& buf, uint16_t val) {
    buf.push_back(val & 0xFF);
    buf.push_back((val >> 8) & 0xFF);
}

void trace_writer::write_uint32(std::vector<uint8_t>& buf, uint32_t val) {
    buf.push_back(val & 0xFF);
    buf.push_back((val >> 8) & 0xFF);
    buf.push_back((val >> 16) & 0xFF);
    buf.push_back((val >> 24) & 0xFF);
}

void trace_writer::write_uint64(std::vector<uint8_t>& buf, uint64_t val) {
    write_uint32(buf, val & 0xFFFFFFFF);
    write_uint32(buf, (val >> 32) & 0xFFFFFFFF);
}

void trace_writer::write_string(std::vector<uint8_t>& buf, const std::string& str) {
    write_uint16(buf, str.length());
    buf.insert(buf.end(), str.begin(), str.end());
}

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
                throw std::runtime_error("All tensors must have the same grid dimensions for lane stacking");
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

                // Compute event count from byte offset (device writes raw offset to avoid division)
                uint32_t base_offset_bytes = base_offset * 4;
                uint32_t event_width_bytes = tensor.event_width * 4;
                uint32_t row_stride_bytes = tensor.row_stride * 4;
                uint32_t max_offset_bytes = base_offset_bytes + row_stride_bytes;

                // Check for overflow
                if (write_offset_bytes > max_offset_bytes) {
                    throw std::runtime_error(
                        "Overflow detected in block " + std::to_string(block_id) +
                        ", lane " + std::to_string(lane_id) + " (SM " + std::to_string(sm_id) + "): " +
                        "write_offset=" + std::to_string(write_offset_bytes) + " bytes exceeds " +
                        "allocated capacity=" + std::to_string(max_offset_bytes) + " bytes. " +
                        "Allocated " + std::to_string((row_stride_bytes - event_width_bytes) / event_width_bytes) + " events, " +
                        "attempted " + std::to_string((write_offset_bytes - base_offset_bytes - event_width_bytes) / event_width_bytes) + " events.");
                }

                uint32_t event_count = (write_offset_bytes - base_offset_bytes - event_width_bytes) / event_width_bytes;

                if (event_count == 0) continue;

                uint32_t event_offset = base_offset + tensor.event_width;

                for (uint32_t event_idx = 0; event_idx < event_count; ++event_idx) {
                    uint32_t start_time = tensor.host_buffer[event_offset];
                    uint32_t end_time = tensor.host_buffer[event_offset + 1];
                    uint32_t duration = end_time - start_time;  // Unsigned wrap handles it correctly

                    if (start_time == 0) continue;  // Skip invalid events

                    parsed_event evt;
                    evt.block_id = block_id;  // Same block ID across tensors
                    evt.cluster_id = 0;  // TODO: Compute from cluster dims when available
                    evt.lane_id = global_lane_offset + lane_id;  // Offset lane ID by tensor
                    evt.time_offset = start_time;  // Will fix wraparound later
                    evt.duration = duration;
                    evt.sm_id = sm_id;
                    evt.param_count = 0;

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

// Detect and fix timer wraparound (32-bit timer wraps every ~4.3 seconds)
void trace_writer::fix_timer_wraparound(std::vector<parsed_event>& events) {
    if (events.empty()) return;

    // Sort by raw time to detect wraparound
    std::sort(events.begin(), events.end(), [](const parsed_event& a, const parsed_event& b) {
        return a.time_offset < b.time_offset;
    });

    // After sorting, if there was a wraparound:
    // - Post-wrap events (small values like 100, 200) come first
    // - Pre-wrap events (large values like 0xFFFFFFF0) come later
    // Detect large FORWARD jump from small to large (crossing from post-wrap to pre-wrap)
    const uint64_t WRAP_THRESHOLD = 0x80000000ULL;  // 2^31

    size_t wrap_point = 0;
    bool found_wrap = false;

    for (size_t i = 1; i < events.size(); ++i) {
        uint32_t prev_time = static_cast<uint32_t>(events[i - 1].time_offset);
        uint32_t curr_time = static_cast<uint32_t>(events[i].time_offset);

        if (curr_time > prev_time) {
            uint32_t jump = curr_time - prev_time;

            // Detect large forward jump (post-wrap to pre-wrap transition)
            if (jump > WRAP_THRESHOLD) {
                wrap_point = i;
                found_wrap = true;
                break;
            }
        }
    }

    if (found_wrap) {
        // Events before wrap_point are post-wrap (need 2^32 added)
        // Events from wrap_point onward are pre-wrap (no adjustment)
        for (size_t i = 0; i < wrap_point; ++i) {
            events[i].time_offset = static_cast<uint32_t>(events[i].time_offset) + 0x100000000ULL;
        }
    }
}

// Find the earliest timestamp (kernel start time)
uint64_t trace_writer::find_kernel_start_time(const std::vector<parsed_event>& events) {
    uint64_t min_time = UINT64_MAX;
    for (const auto& evt : events) {
        if (evt.time_offset < min_time) {
            min_time = evt.time_offset;
        }
    }
    return min_time;
}

// Convert absolute timestamps to offsets from kernel start
void trace_writer::convert_to_offsets(std::vector<parsed_event>& events, uint64_t kernel_start_time) {
    for (auto& evt : events) {
        uint64_t absolute_time = evt.time_offset;
        uint64_t offset = absolute_time - kernel_start_time;
        evt.time_offset = offset;

        // Check for overflow (kernel ran longer than 4.3 seconds)
        if (offset > UINT32_MAX) {
            throw std::runtime_error("Timestamp overflow detected (kernel duration > 4.3s)");
        }
    }
}

void trace_writer::write(const char* filename, bool compress) {
    // Step 1: Parse all events from all tensors
    std::vector<parsed_event> events = parse_all_events();

    if (events.empty()) {
#ifndef NANOTRACE_NO_LOG
        fprintf(stderr, "Warning: No trace events found\n");
#endif
        return;
    }

    // Step 2: Fix timer wraparound
    fix_timer_wraparound(events);

    // Step 3: Find kernel start time
    uint64_t kernel_start_time = find_kernel_start_time(events);

    // Step 4: Convert to offsets from kernel start
    convert_to_offsets(events, kernel_start_time);

    // Step 5: Build format ID mapping (original TraceType::id -> file index)
    std::unordered_map<uint16_t, uint16_t> format_id_map;
    for (size_t i = 0; i < formats.size(); ++i) {
        format_id_map[formats[i].id] = static_cast<uint16_t>(i);
    }

    // Translate default block and track format IDs to file indices
    auto block_it = format_id_map.find(default_block_format_id);
    if (block_it != format_id_map.end()) {
        default_block_format_id = block_it->second;
    } else {
        throw std::runtime_error("Default block type not registered");
    }

    // Translate per-tensor track format IDs to file indices
    for (auto& tensor : tensors) {
        // Translate default track format ID
        if (tensor.default_track_format_id != 0) {
            auto track_it = format_id_map.find(tensor.default_track_format_id);
            if (track_it != format_id_map.end()) {
                tensor.default_track_format_id = track_it->second;
            } else {
                throw std::runtime_error("Tensor track type not registered");
            }
        }

        // Translate per-lane overrides
        for (auto& [lane, format_id] : tensor.lane_track_format_ids) {
            auto it = format_id_map.find(format_id);
            if (it != format_id_map.end()) {
                format_id = it->second;
            } else {
                throw std::runtime_error("Lane track type not registered");
            }
        }
    }

    // Step 6: Translate event format IDs to file indices
    for (auto& evt : events) {
        auto it = format_id_map.find(evt.format_id);
        if (it != format_id_map.end()) {
            evt.format_id = it->second;
        } else {
            throw std::runtime_error("Event references unregistered format ID");
        }
    }

    // Step 7: Group events by (block_id, lane_id) for writing
    // Use map (not unordered_map) to ensure tracks are written in sorted order
    std::map<std::pair<uint32_t, uint32_t>, std::vector<parsed_event*>> tracks;
    for (auto& evt : events) {
        tracks[{evt.block_id, evt.lane_id}].push_back(&evt);
    }

    // Step 8: Collect unique block descriptors (block_id, cluster_id, sm_id tuples)
    struct block_desc { uint32_t block_id; uint32_t cluster_id; uint16_t sm_id; };
    std::vector<block_desc> block_descriptors;
    std::unordered_set<std::tuple<uint32_t, uint32_t, uint16_t>, tuple3_hash> seen_blocks;

    for (const auto& evt : events) {
        auto key = std::make_tuple(evt.block_id, evt.cluster_id, evt.sm_id);
        if (seen_blocks.insert(key).second) {
            block_descriptors.push_back({evt.block_id, evt.cluster_id, evt.sm_id});
        }
    }

    // Sort block descriptors by grid ID (required by visualizer parser)
    std::sort(block_descriptors.begin(), block_descriptors.end(),
        [](const block_desc& a, const block_desc& b) {
            return a.block_id < b.block_id;
        });

    // Build block descriptor ID map after sorting
    std::unordered_map<std::tuple<uint32_t, uint32_t, uint16_t>, uint32_t, tuple3_hash> block_id_map;
    for (size_t i = 0; i < block_descriptors.size(); ++i) {
        const auto& desc = block_descriptors[i];
        block_id_map[std::make_tuple(desc.block_id, desc.cluster_id, desc.sm_id)] = static_cast<uint32_t>(i);
    }

    // Step 7: Write binary file
    std::vector<uint8_t> header;
    std::vector<uint8_t> payload;

    // Write magic number and version (uncompressed)
    const char* magic = "nanotrace";
    for (int i = 0; i < 10; ++i) {
        write_uint8(header, magic[i]);
    }
    write_uint8(header, 1);  // Format version

    // Check if compression is actually available
    bool can_compress = false;
#ifdef NANOTRACE_WITH_MINIZ
    can_compress = compress;
#endif
    write_uint8(header, can_compress ? 1 : 0);

    // Write kernel name
    write_string(payload, kernel_name);

    // Write grid dimensions (use first tensor's grid, all should be the same)
    if (!tensors.empty()) {
        write_uint32(payload, tensors[0].grid_dims.x);
        write_uint32(payload, tensors[0].grid_dims.y);
        write_uint32(payload, tensors[0].grid_dims.z);
    } else {
        write_uint32(payload, 0);
        write_uint32(payload, 0);
        write_uint32(payload, 0);
    }

    // Write cluster dimensions (use first tensor's cluster dims, all should be the same)
    if (!tensors.empty()) {
        write_uint32(payload, tensors[0].cluster_dims.x);
        write_uint32(payload, tensors[0].cluster_dims.y);
        write_uint32(payload, tensors[0].cluster_dims.z);
    } else {
        write_uint32(payload, 0);
        write_uint32(payload, 0);
        write_uint32(payload, 0);
    }

    // Write counts
    write_uint32(payload, formats.size());
    write_uint32(payload, block_descriptors.size());
    write_uint32(payload, tracks.size());
    write_uint64(payload, events.size());

    // Write format descriptors with both label and tooltip strings
    for (const auto& fmt : formats) {
        write_string(payload, fmt.label_string);
        write_string(payload, fmt.tooltip_string);
        write_uint8(payload, fmt.param_count);
    }

    // Write block descriptors
    // Format: [block_id (uint32), cluster_id (uint32), sm_id (uint16), format_id (uint16)]
    for (const auto& desc : block_descriptors) {
        write_uint32(payload, desc.block_id);
        write_uint32(payload, desc.cluster_id);
        write_uint16(payload, desc.sm_id);
        write_uint16(payload, default_block_format_id);
    }

    // Write event tracks
    for (const auto& [key, event_ptrs] : tracks) {
        auto [block_id, lane_id] = key;

        if (event_ptrs.empty()) continue;

        // Get sm_id and cluster_id from first event
        uint16_t sm_id = event_ptrs[0]->sm_id;
        uint32_t cluster_id = event_ptrs[0]->cluster_id;

        // Determine which tensor this lane belongs to and get track format ID
        uint16_t track_format_id = 0;
        uint32_t lane_offset = 0;
        for (const auto& tensor : tensors) {
            if (lane_id >= lane_offset && lane_id < lane_offset + tensor.num_lanes) {
                // This lane belongs to this tensor
                uint32_t tensor_relative_lane = lane_id - lane_offset;

                // Check for per-lane override first
                auto track_it = tensor.lane_track_format_ids.find(tensor_relative_lane);
                if (track_it != tensor.lane_track_format_ids.end()) {
                    track_format_id = track_it->second;
                } else {
                    track_format_id = tensor.default_track_format_id;
                }
                break;
            }
            lane_offset += tensor.num_lanes;
        }

        if (track_format_id == 0) {
            throw std::runtime_error("Track format ID not set for lane " + std::to_string(lane_id));
        }

        // Write track header
        write_uint32(payload, block_id_map[std::make_tuple(block_id, cluster_id, sm_id)]);  // Block descriptor ID
        write_uint16(payload, track_format_id);  // Track format ID (from tensor)
        write_uint32(payload, lane_id);  // Lane ID (for {lane} placeholder expansion)
        // No track parameters (track format has 0 params, lane_id is metadata not a param)
        write_uint32(payload, event_ptrs.size());

        // Write events
        for (const auto* evt : event_ptrs) {
            write_uint32(payload, static_cast<uint32_t>(evt->time_offset));
            // Clamp duration to minimum 32ns (global timer resolution)
            uint32_t duration = (evt->duration == 0) ? 32 : evt->duration;
            write_uint32(payload, duration);
            write_uint16(payload, evt->format_id);

            for (uint8_t p = 0; p < evt->param_count; ++p) {
                write_uint32(payload, evt->params[p]);
            }
        }
    }

    // Compress payload if requested and available
    std::vector<uint8_t> final_payload;
    if (can_compress) {
#ifdef NANOTRACE_WITH_MINIZ
        mz_ulong compressed_size = mz_compressBound(payload.size());
        final_payload.resize(compressed_size);

        int result = mz_compress(final_payload.data(), &compressed_size,
                                 payload.data(), payload.size());

        if (result == MZ_OK) {
            final_payload.resize(compressed_size);
        } else {
            // Compression failed, fall back to uncompressed
            final_payload = std::move(payload);
            header[11] = 0;
        }
#endif
    } else {
        final_payload = std::move(payload);
    }

    // Calculate and log statistics per tensor
    uint32_t min_duration = UINT32_MAX;
    uint32_t max_duration = 0;
    uint64_t min_time = UINT64_MAX;
    uint64_t max_time = 0;

    for (const auto& evt : events) {
        uint32_t duration = (evt.duration == 0) ? 32 : evt.duration;
        if (duration < min_duration) min_duration = duration;
        if (duration > max_duration) max_duration = duration;

        uint64_t evt_end = evt.time_offset + evt.duration;
        if (evt.time_offset < min_time) min_time = evt.time_offset;
        if (evt_end > max_time) max_time = evt_end;
    }

    uint64_t total_duration_ns = (events.empty()) ? 0 : (max_time - min_time);

#ifndef NANOTRACE_NO_LOG
    // Log per-tensor statistics
    printf("Nanotrace: %zu tensors, %zu blocks, %zu total events\n",
            tensors.size(), block_descriptors.size(), events.size());

    uint32_t lane_offset = 0;
    for (size_t tensor_idx = 0; tensor_idx < tensors.size(); ++tensor_idx) {
        const auto& tensor = tensors[tensor_idx];

        // Calculate max events per lane for this tensor
        uint32_t max_events_this_tensor = 0;
        for (uint32_t lane_id = lane_offset; lane_id < lane_offset + tensor.num_lanes; ++lane_id) {
            for (const auto& [key, event_ptrs] : tracks) {
                if (key.second == lane_id && event_ptrs.size() > max_events_this_tensor) {
                    max_events_this_tensor = static_cast<uint32_t>(event_ptrs.size());
                }
            }
        }

        printf("  Tensor %zu: %u lanes, max %u events/lane\n",
                tensor_idx, tensor.num_lanes, max_events_this_tensor);

        lane_offset += tensor.num_lanes;
    }

    if (!events.empty()) {
        printf("  Duration: %.6f us total, events %u-%u ns\n",
                total_duration_ns / 1e3, min_duration, max_duration);
    }
    printf("  Output: %s (%zu bytes uncompressed, %zu compressed = %.1f%%)\n",
            filename, payload.size(), final_payload.size(),
            (payload.size() > 0) ? (100.0 * final_payload.size() / payload.size()) : 0.0);
#endif

    // Write to file
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(header.data()), header.size());
    file.write(reinterpret_cast<const char*>(final_payload.data()), final_payload.size());
    file.close();
}

} // namespace nanotrace
