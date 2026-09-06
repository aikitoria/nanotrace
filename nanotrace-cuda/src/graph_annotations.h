#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "hes.h"
#include "nanotrace/gpu_process_trace.h"

namespace nanotrace
{
    // Pure host-side annotation over hardware events. No marker kernel, CUDA
    // event, or dependency is added to the graph being measured.
    class GraphAnnotations
    {
        struct Node
        {
            uint64_t graph = 0;
            std::string name;
            std::vector<GpuGraphRange> ranges;
            bool iteration_end = false;
        };
        struct Graph
        {
            std::vector<std::string> launch_tracks;
            uint64_t anchor = 0;
            bool annotated = false;
        };
        std::unordered_map<uint64_t, Node> _nodes;
        std::map<uint64_t, Graph> _graphs;
        std::unordered_map<uint64_t, uint64_t> _clones;

        uint64_t Original(uint64_t node) const
        {
            auto clone = _clones.find(node);
            return clone == _clones.end() ? node : clone->second;
        }
    public:
        void RegisterNode(uint64_t graph, uint64_t node, const char* name,
            const std::vector<GpuGraphRange>& ranges, bool launch_anchor)
        {
            Graph& info = _graphs[graph];
            if (launch_anchor) info.anchor = node;
            info.annotated |= !ranges.empty();
            _nodes[node] = { graph, name ? name : "", ranges, false };
        }
        void CloneNode(uint64_t original, uint64_t clone)
        {
            _clones[clone] = Original(original);
        }
        void MarkIterationEnd(uint64_t node)
        {
            auto it = _nodes.find(node);
            if (it != _nodes.end()) it->second.iteration_end = true;
        }
        void RecordLaunch(uint64_t graph, const char* dynamic_track)
        {
            _graphs[graph].launch_tracks.emplace_back(dynamic_track ? dynamic_track : "");
        }

        template<typename DeviceTrack>
        bool Append(TraceSession& session, std::vector<HesKernelEvent>& kernels,
            DeviceTrack&& device_track, std::string& error)
        {
            std::map<uint64_t, std::vector<HesKernelEvent*>> by_graph;
            for (HesKernelEvent& kernel : kernels)
            {
                auto node = _nodes.find(Original(kernel.graph_node_id));
                if (node != _nodes.end() && _graphs.at(node->second.graph).annotated)
                    by_graph[node->second.graph].push_back(&kernel);
            }
            for (const auto& [id, graph] : _graphs)
                if (graph.annotated && !graph.launch_tracks.empty() && !by_graph.contains(id))
                {
                    error = "No hardware graph nodes matched semantic graph " + std::to_string(id);
                    return false;
                }
            std::map<std::pair<uint32_t, std::string>, TrackId> tracks;
            for (auto& [graph_id, events] : by_graph)
            {
                std::sort(events.begin(), events.end(), [](const auto* a, const auto* b)
                    { return std::tie(a->start_ns, a->end_ns) < std::tie(b->start_ns, b->end_ns); });
                const Graph& graph = _graphs.at(graph_id);
                std::vector<uint64_t> starts;
                for (const auto* event : events)
                    if (Original(event->graph_node_id) == graph.anchor) starts.push_back(event->start_ns);
                if (starts.size() != graph.launch_tracks.size())
                {
                    error = "Semantic graph " + std::to_string(graph_id) + " has " + std::to_string(starts.size())
                        + " hardware launch anchors for " + std::to_string(graph.launch_tracks.size()) + " recorded launches";
                    return false;
                }
                for (size_t launch = 0; launch < starts.size(); ++launch)
                {
                    const uint64_t end = launch + 1 < starts.size() ? starts[launch + 1] : std::numeric_limits<uint64_t>::max();
                    auto first = std::lower_bound(events.begin(), events.end(), starts[launch],
                        [](const auto* event, uint64_t time) { return event->start_ns < time; });
                    auto last = std::lower_bound(first, events.end(), end,
                        [](const auto* event, uint64_t time) { return event->start_ns < time; });
                    std::vector<uint64_t> iteration_ends;
                    for (auto it = first; it != last; ++it)
                        if (_nodes.at(Original((*it)->graph_node_id)).iteration_end) iteration_ends.push_back((*it)->end_ns);
                    std::sort(iteration_ends.begin(), iteration_ends.end());
                    struct Range
                    {
                        uint64_t start = std::numeric_limits<uint64_t>::max();
                        uint64_t end = 0;
                        uint64_t count = 0;
                        uint32_t color = 0;
                        uint32_t device = 0;
                        ClockId clock = 0;
                    };
                    using Key = std::tuple<size_t, uint64_t, std::string, std::string>;
                    std::map<Key, Range> ranges;
                    for (auto it = first; it != last; ++it)
                    {
                        HesKernelEvent& event = **it;
                        const Node& node = _nodes.at(Original(event.graph_node_id));
                        size_t iteration = static_cast<size_t>(std::lower_bound(iteration_ends.begin(), iteration_ends.end(), event.start_ns) - iteration_ends.begin());
                        for (const GpuGraphRange& definition : node.ranges)
                        {
                            const std::string& track = definition.dynamic_track ? graph.launch_tracks[launch] : definition.track;
                            std::string label = track.empty() ? definition.name : track + " " + definition.name;
                            Range& range = ranges[{ iteration, definition.instance, track, label }];
                            range.start = std::min(range.start, event.start_ns);
                            range.end = std::max(range.end, event.end_ns);
                            ++range.count;
                            range.color = definition.color;
                            range.device = event.device_id;
                            range.clock = event.clock_id;
                        }
                    }
                    for (const auto& [key, range] : ranges)
                    {
                        const auto& [iteration, instance, track_name, label] = key;
                        auto [track, inserted] = tracks.try_emplace({ range.device, track_name }, INVALID_TRACK_ID);
                        if (inserted) track->second = session.AddTrack(track_name.c_str(), TrackKind::GpuAnnotation,
                            range.clock, device_track(range.device), -100);
                        EventId event = session.AddSlice(track->second, label.c_str(), range.start, range.end - range.start,
                            0, INVALID_EVENT_ID, range.color);
                        session.AddUnsignedArgument(event, "graph_instance", graph_id);
                        session.AddUnsignedArgument(event, "launch", launch);
                        session.AddUnsignedArgument(event, "iteration", iteration);
                        session.AddUnsignedArgument(event, "range_instance", instance);
                        session.AddUnsignedArgument(event, "kernel_count", range.count);
                    }
                }
            }
            return true;
        }
    };
}
