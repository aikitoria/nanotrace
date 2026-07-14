#pragma once

#include <memory>
#include <string>

namespace nanotrace
{
    class TraceSession;
    class trace_writer;

    class GpuTrace
    {
    public:
        explicit GpuTrace(const char* session_name);
        ~GpuTrace();

        GpuTrace(const GpuTrace&) = delete;
        GpuTrace& operator=(const GpuTrace&) = delete;

        explicit operator bool() const;
        bool Begin();
        bool Write(const char* filename, trace_writer& kernel_trace);
        TraceSession& Session();
        const std::string& LastError() const;

    private:
        class Implementation;
        std::unique_ptr<Implementation> _implementation;
    };
}
