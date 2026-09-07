#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace nanotrace
{
    class TraceSession;
    class trace_writer;

    struct GpuKernelTraceOptions
    {
        const char* kernel_name_substring = nullptr;
        std::optional<uint32_t> device_id;
        std::optional<uint32_t> context_id;
        size_t expected_invocation_count = 1;
        uint32_t blocks_per_invocation = 1;
    };

    // A named envelope of existing hardware-timed graph nodes. A dynamic
    // track takes its name from RecordGraphLaunch, e.g. a surviving request.
    struct GpuGraphRange
    {
        std::string name;
        std::string track;
        bool dynamic_track = false;
        uint32_t color = 0;
        uint64_t instance = 0; // Distinguishes disjoint scopes with the same label.
        // Select one occurrence of a reused node in a fixed-count device loop.
        // Occurrences count independently per node and reset at each launch.
        uint32_t repetition = 0;
        uint32_t repeat_count = 1;
    };

    class GpuProcessTrace
    {
    public:
        explicit GpuProcessTrace(TraceSession& session,
            const char* kernel_name_prefix_to_strip = nullptr);
        ~GpuProcessTrace();

        GpuProcessTrace(const GpuProcessTrace&) = delete;
        GpuProcessTrace& operator=(const GpuProcessTrace&) = delete;

        explicit operator bool() const;
        bool Begin();
        bool Finish();
        bool Finish(trace_writer& kernel_trace);
        bool AddKernelTrace(trace_writer& kernel_trace);
        bool AddKernelTrace(trace_writer& kernel_trace,
            const GpuKernelTraceOptions& options);
        void RegisterGraphNode(uint64_t graph, uint64_t node, const char* name,
            const std::vector<GpuGraphRange>& ranges, bool launch_anchor);
        void MarkGraphIterationEnd(uint64_t node);
        void RecordGraphLaunch(uint64_t graph, const char* dynamic_track);
        const std::string& LastError() const;

    private:
        class Implementation;
        std::unique_ptr<Implementation> _implementation;
    };
}
