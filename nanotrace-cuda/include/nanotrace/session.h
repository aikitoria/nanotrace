#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace nanotrace
{
    using ClockId = uint32_t;
    using StringId = uint32_t;
    using TrackId = uint64_t;
    using EventId = uint64_t;

    inline constexpr ClockId INVALID_CLOCK_ID = 0;
    inline constexpr TrackId INVALID_TRACK_ID = 0;
    inline constexpr EventId INVALID_EVENT_ID = 0;

    enum class ClockKind : uint8_t
    {
        MonotonicRaw,
        CpuTsc,
        Cupti,
        GpuGlobalTimer,
    };

    enum class TrackKind : uint8_t
    {
        Process,
        CpuThread,
        GpuDevice,
        GpuStream,
        Kernel,
        StreamingMultiprocessor,
        ThreadBlock,
        Warp,
        Generic,
    };

    enum class EventKind : uint8_t
    {
        Slice,
        Bookmark,
        Counter,
        Flow,
    };

    enum class ArgumentKind : uint8_t
    {
        Unsigned,
        Signed,
        Floating,
        String,
        Boolean,
    };

    struct ClockDescriptor
    {
        ClockId id = INVALID_CLOCK_ID;
        ClockKind kind = ClockKind::MonotonicRaw;
        StringId name = 0;
        uint32_t device_id = 0;
        uint64_t frequency_hz = 1'000'000'000;
    };

    struct ClockSnapshot
    {
        ClockId source_clock = INVALID_CLOCK_ID;
        ClockId reference_clock = INVALID_CLOCK_ID;
        uint64_t source_timestamp = 0;
        uint64_t reference_timestamp = 0;
        uint64_t uncertainty_ns = 0;
    };

    struct TrackDescriptor
    {
        TrackId id = INVALID_TRACK_ID;
        TrackId parent_id = INVALID_TRACK_ID;
        ClockId clock_id = INVALID_CLOCK_ID;
        StringId name = 0;
        TrackKind kind = TrackKind::Generic;
        int32_t sort_order = 0;
        uint64_t source_id = 0;
    };

    struct EventArgument
    {
        StringId name = 0;
        ArgumentKind kind = ArgumentKind::Unsigned;
        uint64_t value = 0;
    };

    struct EventFormatDescriptor
    {
        StringId label = 0;
        StringId tooltip = 0;
        uint8_t parameter_count = 0;
    };

    struct TraceEvent
    {
        EventId id = INVALID_EVENT_ID;
        EventId parent_id = INVALID_EVENT_ID;
        TrackId track_id = INVALID_TRACK_ID;
        StringId name = 0;
        EventKind kind = EventKind::Slice;
        uint64_t timestamp = 0;
        uint64_t duration = 0;
        uint64_t correlation_id = 0;
        uint32_t first_argument = 0;
        uint16_t argument_count = 0;
        uint32_t color = 0;
    };

    struct BufferedTraceEvent
    {
        const char* name = nullptr;
        TrackId track_id = INVALID_TRACK_ID;
        EventKind kind = EventKind::Slice;
        uint64_t timestamp = 0;
        uint64_t duration = 0;
        uint64_t correlation_id = 0;
        EventId parent_id = INVALID_EVENT_ID;
        uint32_t color = 0;
    };

    struct BufferedEventArgument
    {
        const char* name = nullptr;
        ArgumentKind kind = ArgumentKind::Unsigned;
        uint64_t value = 0;
        const char* string_value = nullptr;
    };

    struct BufferedEventWithArguments
    {
        BufferedTraceEvent event;
        const BufferedEventArgument* arguments = nullptr;
        size_t argument_count = 0;
    };

    inline constexpr size_t BUFFERED_EVENT_ARGUMENT_CAPACITY = 8;
    using BufferedEventReader = bool (*)(const void* source,
        size_t event_index, BufferedEventWithArguments* event,
        BufferedEventArgument* argument_storage, size_t argument_capacity);

    class TraceSession
    {
    public:
        explicit TraceSession(const char* name);

        TraceSession(const TraceSession&) = delete;
        TraceSession& operator=(const TraceSession&) = delete;

        ClockId ReferenceClock() const { return _reference_clock; }
        ClockId AddClock(const char* name, ClockKind kind,
            uint32_t device_id = 0, uint64_t frequency_hz = 1'000'000'000);
        void AddClockSnapshot(ClockId source_clock, ClockId reference_clock,
            uint64_t source_timestamp, uint64_t reference_timestamp,
            uint64_t uncertainty_ns = 0);

        TrackId AddTrack(const char* name, TrackKind kind, ClockId clock_id,
            TrackId parent_id = INVALID_TRACK_ID, int32_t sort_order = 0,
            uint64_t source_id = 0);
        bool SetTrackParent(TrackId track_id, TrackId parent_id);
        EventId AddSlice(TrackId track_id, const char* name,
            uint64_t timestamp, uint64_t duration,
            uint64_t correlation_id = 0, EventId parent_id = INVALID_EVENT_ID,
            uint32_t color = 0);
        EventId AddBookmark(TrackId track_id, const char* name,
            uint64_t timestamp, uint64_t correlation_id = 0,
            EventId parent_id = INVALID_EVENT_ID, uint32_t color = 0);
        void RegisterEventFormat(const char* label, const char* tooltip,
            uint8_t parameter_count);
        void AddUnsignedArgument(EventId event_id, const char* name, uint64_t value);
        void AddSignedArgument(EventId event_id, const char* name, int64_t value);
        void AddStringArgument(EventId event_id, const char* name, const char* value);
        EventId AddBufferedEventSource(std::shared_ptr<const void> owner,
            const void* source, size_t event_count, BufferedEventReader reader,
            TrackId fixed_track_id = INVALID_TRACK_ID);

        bool Write(const char* filename);
        const std::string& LastError() const { return _last_error; }

        static uint64_t MonotonicRawNowNs();

    private:
        StringId Intern(const char* value);
        EventId AddEvent(TrackId track_id, const char* name, EventKind kind,
            uint64_t timestamp, uint64_t duration, uint64_t correlation_id,
            EventId parent_id, uint32_t color);
        TraceEvent* FindEvent(EventId event_id);
        void SetError(const char* message);

        struct BufferedEventSourceDescriptor
        {
            std::shared_ptr<const void> owner;
            const void* source;
            size_t event_count;
            size_t argument_count;
            BufferedEventReader reader;
            TrackId fixed_track_id;
            EventId first_event_id;
            std::vector<StringId> event_name_ids;
            std::vector<StringId> argument_name_ids;
            std::vector<ArgumentKind> argument_kinds;
            std::vector<uint64_t> argument_values;
        };

        std::mutex _mutex;
        std::string _name;
        std::string _last_error;
        std::vector<std::string> _strings;
        std::unordered_map<std::string, StringId> _string_ids;
        std::vector<ClockDescriptor> _clocks;
        std::vector<ClockSnapshot> _clock_snapshots;
        std::vector<TrackDescriptor> _tracks;
        std::vector<TraceEvent> _events;
        std::vector<EventArgument> _arguments;
        std::vector<EventFormatDescriptor> _event_formats;
        std::unordered_map<StringId, size_t> _event_format_indices;
        std::vector<BufferedEventSourceDescriptor> _event_sources;
        ClockId _reference_clock = INVALID_CLOCK_ID;
        TrackId _next_track_id = 1;
        EventId _next_event_id = 1;
    };

}
