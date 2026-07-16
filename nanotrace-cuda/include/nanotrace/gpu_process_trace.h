#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

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
        const std::string& LastError() const;

    private:
        class Implementation;
        std::unique_ptr<Implementation> _implementation;
    };
}
