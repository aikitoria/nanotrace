#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "nanotrace/session.h"

namespace nanotrace
{
    struct HesKernelEvent
    {
        EventId event_id = INVALID_EVENT_ID;
        TrackId track_id = INVALID_TRACK_ID;
        ClockId clock_id = INVALID_CLOCK_ID;
        uint32_t device_id = 0;
        uint32_t context_id = 0;
        uint32_t stream_id = 0;
        uint32_t graph_id = 0;
        uint64_t graph_node_id = 0;
        uint64_t correlation_id = 0;
        uint64_t start_ns = 0;
        uint64_t end_ns = 0;
        const char* name = nullptr;
        const char* match_name = nullptr;
    };

    class HesTracer
    {
    public:
        explicit HesTracer(TraceSession& session,
            TrackId parent_track = INVALID_TRACK_ID,
            const char* kernel_name_prefix_to_strip = nullptr);
        ~HesTracer();

        HesTracer(const HesTracer&) = delete;
        HesTracer& operator=(const HesTracer&) = delete;

        bool Initialize();
        bool BeginCapture();
        bool ResetCaptureStart();
        bool CaptureClockSnapshot();
        bool Stop();
        bool IsInitialized() const;
        const std::string& LastError() const;
        const std::vector<HesKernelEvent>& KernelEvents() const;

    private:
        class Implementation;
        std::unique_ptr<Implementation> _implementation;
    };
}
