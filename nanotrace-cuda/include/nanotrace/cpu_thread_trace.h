#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "session.h"

namespace nanotrace
{
    struct BufferedCpuEvent
    {
        const char* name = nullptr;
        uint64_t timestamp = 0;
        uint64_t duration = 0;
        uint64_t correlation_id = 0;
        uint32_t color = 0;
        EventKind kind = EventKind::Slice;
    };

    struct CpuEventToken
    {
        uint64_t timestamp = 0;
    };

    class CpuThreadTrace
    {
    public:
        CpuThreadTrace(TraceSession& session, const char* thread_name,
            size_t event_capacity = 4096,
            TrackId parent_id = INVALID_TRACK_ID,
            int32_t sort_order = 0);
        ~CpuThreadTrace();

        CpuThreadTrace(const CpuThreadTrace&) = delete;
        CpuThreadTrace& operator=(const CpuThreadTrace&) = delete;

        CpuEventToken Begin() const;
        void End(CpuEventToken token, const char* name,
            uint64_t correlation_id = 0, uint32_t color = 0);
        void Bookmark(const char* name, uint64_t correlation_id = 0,
            uint32_t color = 0);
        bool Flush();

        TrackId Track() const { return _track_id; }
        size_t DroppedEventCount() const { return _dropped_event_count; }

    private:
        void AddBufferedEvent(const char* name, EventKind kind,
            uint64_t timestamp, uint64_t duration,
            uint64_t correlation_id, uint32_t color);

        TraceSession* _session;
        TrackId _track_id;
        std::vector<BufferedCpuEvent> _events;
        size_t _event_capacity = 0;
        size_t _event_count = 0;
        size_t _dropped_event_count = 0;
    };

    class CpuScope
    {
    public:
        CpuScope(CpuThreadTrace& trace, const char* name,
            uint64_t correlation_id = 0, uint32_t color = 0)
            : _trace{ &trace }
            , _name{ name }
            , _correlation_id{ correlation_id }
            , _color{ color }
            , _token{ trace.Begin() }
        {
        }

        ~CpuScope()
        {
            _trace->End(_token, _name, _correlation_id, _color);
        }

        CpuScope(const CpuScope&) = delete;
        CpuScope& operator=(const CpuScope&) = delete;

    private:
        CpuThreadTrace* _trace;
        const char* _name;
        uint64_t _correlation_id;
        uint32_t _color;
        CpuEventToken _token;
    };
}
