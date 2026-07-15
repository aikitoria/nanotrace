#include <memory>
#include <utility>
#include <vector>

#include "nanotrace/cpu_thread_trace.h"

namespace nanotrace
{
    namespace
    {
        bool ReadCpuEvent(const void* source, size_t event_index,
            BufferedEventWithArguments* output, BufferedEventArgument*, size_t)
        {
            const BufferedCpuEvent* events =
                static_cast<const BufferedCpuEvent*>(source);
            const BufferedCpuEvent& event = events[event_index];
            output->event = BufferedTraceEvent{
                event.name,
                INVALID_TRACK_ID,
                event.kind,
                event.timestamp,
                event.duration,
                event.correlation_id,
                INVALID_EVENT_ID,
                event.color,
            };
            output->arguments = nullptr;
            output->argument_count = 0;
            return true;
        }
    }

    CpuThreadTrace::CpuThreadTrace(TraceSession& session,
        const char* thread_name, size_t event_capacity, TrackId parent_id,
        int32_t sort_order)
        : _session{ &session }
        , _track_id{ session.AddTrack(thread_name, TrackKind::CpuThread,
            session.ReferenceClock(), parent_id, sort_order) }
        , _events(event_capacity)
        , _event_capacity{ event_capacity }
    {
    }

    CpuThreadTrace::~CpuThreadTrace()
    {
        Flush();
    }

    CpuEventToken CpuThreadTrace::Begin() const
    {
        return CpuEventToken{ TraceSession::MonotonicRawNowNs() };
    }

    void CpuThreadTrace::AddBufferedEvent(const char* name, EventKind kind,
        uint64_t timestamp, uint64_t duration, uint64_t correlation_id,
        uint32_t color)
    {
        if (_event_count >= _events.size())
        {
            _dropped_event_count++;
            return;
        }

        _events[_event_count++] = BufferedCpuEvent{
            name, timestamp, duration, correlation_id, color, kind,
        };
    }

    void CpuThreadTrace::End(CpuEventToken token, const char* name,
        uint64_t correlation_id, uint32_t color)
    {
        uint64_t end = TraceSession::MonotonicRawNowNs();
        AddBufferedEvent(name, EventKind::Slice, token.timestamp,
            end - token.timestamp, correlation_id, color);
    }

    void CpuThreadTrace::Bookmark(const char* name,
        uint64_t correlation_id, uint32_t color)
    {
        AddBufferedEvent(name, EventKind::Bookmark,
            TraceSession::MonotonicRawNowNs(), 0, correlation_id, color);
    }

    bool CpuThreadTrace::Flush()
    {
        if (!_session)
        {
            return false;
        }

        if (_event_count == 0)
        {
            return true;
        }

        std::shared_ptr<std::vector<BufferedCpuEvent>> events =
            std::make_shared<std::vector<BufferedCpuEvent>>(
                std::move(_events));
        std::shared_ptr<const void> owner = events;
        if (_session->AddBufferedEventSource(std::move(owner), events->data(),
            _event_count, ReadCpuEvent, _track_id) == INVALID_EVENT_ID)
        {
            _events = std::move(*events);
            return false;
        }

        _events = std::vector<BufferedCpuEvent>(_event_capacity);
        _event_count = 0;
        return true;
    }
}
