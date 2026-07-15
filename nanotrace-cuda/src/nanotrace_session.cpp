#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <time.h>
#include <unordered_map>

#include "nanotrace/nanotrace_session.h"

#if defined(NANOTRACE_WITH_MINIZ)
#include "miniz.h"
#endif

namespace nanotrace
{
    namespace
    {
        enum class ChunkType : uint32_t
        {
            Session = 1,
            Strings = 2,
            Clocks = 3,
            ClockSnapshots = 4,
            Tracks = 5,
            Events = 6,
            Arguments = 7,
            EventFormats = 8,
        };

        class BinaryWriter
        {
        public:
            void Reserve(size_t size) { _data.reserve(size); }

            void U8(uint8_t value) { _data.push_back(value); }

            void U16(uint16_t value)
            {
                U8(static_cast<uint8_t>(value));
                U8(static_cast<uint8_t>(value >> 8));
            }

            void U32(uint32_t value)
            {
                U16(static_cast<uint16_t>(value));
                U16(static_cast<uint16_t>(value >> 16));
            }

            void I32(int32_t value) { U32(std::bit_cast<uint32_t>(value)); }

            void U64(uint64_t value)
            {
                U32(static_cast<uint32_t>(value));
                U32(static_cast<uint32_t>(value >> 32));
            }

            void VarU64(uint64_t value)
            {
                while (value >= 0x80)
                {
                    U8(static_cast<uint8_t>(value) | 0x80);
                    value >>= 7;
                }

                U8(static_cast<uint8_t>(value));
            }

            void Bytes(const void* source, size_t size)
            {
                const uint8_t* begin = static_cast<const uint8_t*>(source);
                _data.insert(_data.end(), begin, begin + size);
            }

            void String(const std::string& value)
            {
                U32(static_cast<uint32_t>(value.size()));
                Bytes(value.data(), value.size());
            }

            void Chunk(ChunkType type, uint64_t count, const BinaryWriter& payload)
            {
                U32(static_cast<uint32_t>(type));
                U32(0);
                U64(payload._data.size());
                U64(count);
                Bytes(payload._data.data(), payload._data.size());
            }

            const std::vector<uint8_t>& Data() const { return _data; }

        private:
            std::vector<uint8_t> _data;
        };

        constexpr uint8_t EVENT_KIND_MASK = 0x03;
        constexpr uint8_t EVENT_HAS_PARENT = 0x04;
        constexpr uint8_t EVENT_HAS_CORRELATION = 0x08;
        constexpr uint8_t EVENT_HAS_ARGUMENTS = 0x10;
        constexpr uint8_t EVENT_HAS_COLOR = 0x20;

        uint64_t ZigZagEncode(int64_t value)
        {
            uint64_t bits = std::bit_cast<uint64_t>(value);
            return (bits << 1) ^ static_cast<uint64_t>(value >> 63);
        }

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

    TraceSession::TraceSession(const char* name)
        : _name{ name ? name : "Nanotrace session" }
    {
        _strings.emplace_back();
        _string_ids.emplace(std::string{}, 0);
        _reference_clock = AddClock(
            "CLOCK_MONOTONIC_RAW", ClockKind::MonotonicRaw);
    }

    StringId TraceSession::Intern(const char* value)
    {
        std::string string_value = value ? value : "";
        const std::unordered_map<std::string, StringId>::const_iterator existing =
            _string_ids.find(string_value);

        if (existing != _string_ids.end())
        {
            return existing->second;
        }

        if (_strings.size() >= std::numeric_limits<StringId>::max())
        {
            SetError("String table exhausted");
            return 0;
        }

        StringId id = static_cast<StringId>(_strings.size());
        _strings.push_back(string_value);
        _string_ids.emplace(std::move(string_value), id);
        return id;
    }

    ClockId TraceSession::AddClock(const char* name, ClockKind kind,
        uint32_t device_id, uint64_t frequency_hz)
    {
        std::lock_guard<std::mutex> lock{ _mutex };
        ClockId id = static_cast<ClockId>(_clocks.size() + 1);
        _clocks.push_back(ClockDescriptor{
            id, kind, Intern(name), device_id, frequency_hz,
        });
        return id;
    }

    void TraceSession::AddClockSnapshot(ClockId source_clock,
        ClockId reference_clock, uint64_t source_timestamp,
        uint64_t reference_timestamp, uint64_t uncertainty_ns)
    {
        std::lock_guard<std::mutex> lock{ _mutex };
        _clock_snapshots.push_back(ClockSnapshot{
            source_clock, reference_clock, source_timestamp,
            reference_timestamp, uncertainty_ns,
        });
    }

    TrackId TraceSession::AddTrack(const char* name, TrackKind kind,
        ClockId clock_id, TrackId parent_id, int32_t sort_order,
        uint64_t source_id)
    {
        std::lock_guard<std::mutex> lock{ _mutex };
        TrackId id = _next_track_id++;
        _tracks.push_back(TrackDescriptor{
            id, parent_id, clock_id, Intern(name), kind, sort_order, source_id,
        });
        return id;
    }

    bool TraceSession::SetTrackParent(TrackId track_id, TrackId parent_id)
    {
        std::lock_guard<std::mutex> lock{ _mutex };

        if (track_id == INVALID_TRACK_ID || track_id > _tracks.size()
            || parent_id == INVALID_TRACK_ID || parent_id > _tracks.size()
            || track_id == parent_id)
        {
            SetError("Track parent relationship is invalid");
            return false;
        }

        TrackId ancestor_id = parent_id;

        while (ancestor_id != INVALID_TRACK_ID)
        {
            if (ancestor_id == track_id)
            {
                SetError("Track parent relationship contains a cycle");
                return false;
            }

            ancestor_id = _tracks[ancestor_id - 1].parent_id;
        }

        _tracks[track_id - 1].parent_id = parent_id;
        return true;
    }

    EventId TraceSession::AddEvent(TrackId track_id, const char* name,
        EventKind kind, uint64_t timestamp, uint64_t duration,
        uint64_t correlation_id, EventId parent_id, uint32_t color)
    {
        std::lock_guard<std::mutex> lock{ _mutex };
        EventId id = _next_event_id++;
        _events.push_back(TraceEvent{
            id, parent_id, track_id, Intern(name), kind, timestamp, duration,
            correlation_id, static_cast<uint32_t>(_arguments.size()), 0, color,
        });
        return id;
    }

    EventId TraceSession::AddSlice(TrackId track_id, const char* name,
        uint64_t timestamp, uint64_t duration, uint64_t correlation_id,
        EventId parent_id, uint32_t color)
    {
        return AddEvent(track_id, name, EventKind::Slice, timestamp, duration,
            correlation_id, parent_id, color);
    }

    EventId TraceSession::AddBookmark(TrackId track_id, const char* name,
        uint64_t timestamp, uint64_t correlation_id, EventId parent_id,
        uint32_t color)
    {
        return AddEvent(track_id, name, EventKind::Bookmark, timestamp, 0,
            correlation_id, parent_id, color);
    }

    void TraceSession::RegisterEventFormat(const char* label,
        const char* tooltip, uint8_t parameter_count)
    {
        std::lock_guard<std::mutex> lock{ _mutex };
        StringId label_id = Intern(label);
        StringId tooltip_id = Intern(tooltip ? tooltip : label);
        const std::unordered_map<StringId, size_t>::const_iterator existing =
            _event_format_indices.find(label_id);

        if (existing != _event_format_indices.end())
        {
            const EventFormatDescriptor& format =
                _event_formats[existing->second];
            if (format.tooltip != tooltip_id
                || format.parameter_count != parameter_count)
            {
                SetError("Event label has conflicting format descriptors");
            }
            return;
        }

        _event_format_indices.emplace(label_id, _event_formats.size());
        _event_formats.push_back(EventFormatDescriptor{
            label_id, tooltip_id, parameter_count });
    }

    TraceEvent* TraceSession::FindEvent(EventId event_id)
    {
        if (event_id == 0 || event_id >= _next_event_id)
        {
            return nullptr;
        }
        if (!_events.empty() && _events.back().id == event_id)
        {
            return &_events.back();
        }

        const std::vector<TraceEvent>::iterator event = std::lower_bound(
            _events.begin(), _events.end(), event_id,
            [](const TraceEvent& candidate, EventId id)
            {
                return candidate.id < id;
            });
        return event != _events.end() && event->id == event_id
            ? &*event : nullptr;
    }

    void TraceSession::AddUnsignedArgument(EventId event_id,
        const char* name, uint64_t value)
    {
        std::lock_guard<std::mutex> lock{ _mutex };
        TraceEvent* event = FindEvent(event_id);

        if (!event)
        {
            SetError("Argument references an unknown event");
            return;
        }

        _arguments.push_back(EventArgument{
            Intern(name), ArgumentKind::Unsigned, value,
        });
        event->argument_count++;
    }

    void TraceSession::AddSignedArgument(EventId event_id,
        const char* name, int64_t value)
    {
        std::lock_guard<std::mutex> lock{ _mutex };
        TraceEvent* event = FindEvent(event_id);

        if (!event)
        {
            SetError("Argument references an unknown event");
            return;
        }

        _arguments.push_back(EventArgument{
            Intern(name), ArgumentKind::Signed, std::bit_cast<uint64_t>(value),
        });
        event->argument_count++;
    }

    void TraceSession::AddStringArgument(EventId event_id,
        const char* name, const char* value)
    {
        std::lock_guard<std::mutex> lock{ _mutex };
        TraceEvent* event = FindEvent(event_id);

        if (!event)
        {
            SetError("Argument references an unknown event");
            return;
        }

        _arguments.push_back(EventArgument{
            Intern(name), ArgumentKind::String, Intern(value),
        });
        event->argument_count++;
    }

    EventId TraceSession::AddBufferedEventSource(const void* source,
        size_t event_count, BufferedEventReader reader,
        TrackId fixed_track_id)
    {
        std::lock_guard<std::mutex> lock{ _mutex };

        if ((!source && event_count != 0) || !reader)
        {
            SetError("Buffered event source is invalid");
            return INVALID_EVENT_ID;
        }

        if (fixed_track_id != INVALID_TRACK_ID
            && fixed_track_id > _tracks.size())
        {
            SetError("Buffered event source references an unknown track");
            return INVALID_EVENT_ID;
        }

        if (event_count > std::numeric_limits<EventId>::max()
            - _next_event_id)
        {
            SetError("Event ID space exhausted");
            return INVALID_EVENT_ID;
        }

        EventId first_event_id = _next_event_id;
        _event_sources.push_back(BufferedEventSourceDescriptor{
            source, event_count, reader, fixed_track_id, first_event_id,
        });
        _next_event_id += event_count;
        return first_event_id;
    }

    void TraceSession::SetError(const char* message)
    {
        if (_last_error.empty())
        {
            _last_error = message ? message : "Unknown nanotrace error";
        }
    }

    bool TraceSession::Write(const char* filename)
    {
        std::lock_guard<std::mutex> lock{ _mutex };

        if (!_last_error.empty())
        {
            return false;
        }

        if (!filename || filename[0] == '\0')
        {
            SetError("Trace filename is empty");
            return false;
        }

        uint64_t total_event_count = _events.size();
        uint64_t total_argument_count = _arguments.size();
        std::array<BufferedEventArgument,
            BUFFERED_EVENT_ARGUMENT_CAPACITY> argument_storage;
        std::unordered_map<const char*, StringId> source_string_ids;
        source_string_ids.reserve(256);

        const auto intern_source_string = [&](const char* value)
        {
            const char* valid_value = value ? value : "";
            const std::unordered_map<const char*, StringId>::const_iterator
                existing = source_string_ids.find(valid_value);

            if (existing != source_string_ids.end())
            {
                return existing->second;
            }

            StringId id = Intern(valid_value);
            source_string_ids.emplace(valid_value, id);
            return id;
        };

        for (const BufferedEventSourceDescriptor& source : _event_sources)
        {
            total_event_count += source.event_count;

            for (size_t i = 0; i < source.event_count; ++i)
            {
                BufferedEventWithArguments event;

                if (!source.reader(source.source, i, &event,
                    argument_storage.data(), argument_storage.size()))
                {
                    SetError("Failed to read buffered event source");
                    return false;
                }

                if (source.fixed_track_id != INVALID_TRACK_ID)
                {
                    event.event.track_id = source.fixed_track_id;
                }

                if (event.event.track_id == INVALID_TRACK_ID
                    || event.event.track_id > _tracks.size())
                {
                    SetError("Buffered event references an unknown track");
                    return false;
                }

                if (event.argument_count > argument_storage.size()
                    || (!event.arguments && event.argument_count != 0))
                {
                    SetError("Buffered event arguments are invalid");
                    return false;
                }

                intern_source_string(event.event.name);
                total_argument_count += event.argument_count;

                for (size_t argument_index = 0;
                    argument_index < event.argument_count; ++argument_index)
                {
                    const BufferedEventArgument& argument =
                        event.arguments[argument_index];
                    intern_source_string(argument.name);

                    if (argument.kind == ArgumentKind::String)
                    {
                        intern_source_string(argument.string_value);
                    }
                }
            }
        }

        BinaryWriter body;
        body.Reserve(_strings.size() * 32
            + _clocks.size() * 24
            + _clock_snapshots.size() * 40
            + _tracks.size() * 48
            + total_event_count * 12
            + total_argument_count * 8);

        BinaryWriter session;
        session.String(_name);
        session.U32(_reference_clock);
        session.U32(0);
        body.Chunk(ChunkType::Session, 1, session);

        BinaryWriter strings;
        for (const std::string& value : _strings)
        {
            strings.String(value);
        }
        body.Chunk(ChunkType::Strings, _strings.size(), strings);

        BinaryWriter event_formats;
        for (const EventFormatDescriptor& format : _event_formats)
        {
            event_formats.VarU64(format.label);
            event_formats.VarU64(format.tooltip);
            event_formats.U8(format.parameter_count);
        }
        body.Chunk(ChunkType::EventFormats,
            _event_formats.size(), event_formats);

        BinaryWriter clocks;
        for (const ClockDescriptor& clock : _clocks)
        {
            clocks.U32(clock.id);
            clocks.U8(static_cast<uint8_t>(clock.kind));
            clocks.U8(0);
            clocks.U16(0);
            clocks.U32(clock.name);
            clocks.U32(clock.device_id);
            clocks.U64(clock.frequency_hz);
        }
        body.Chunk(ChunkType::Clocks, _clocks.size(), clocks);

        BinaryWriter snapshots;
        for (const ClockSnapshot& snapshot : _clock_snapshots)
        {
            snapshots.U32(snapshot.source_clock);
            snapshots.U32(snapshot.reference_clock);
            snapshots.U64(snapshot.source_timestamp);
            snapshots.U64(snapshot.reference_timestamp);
            snapshots.U64(snapshot.uncertainty_ns);
        }
        body.Chunk(ChunkType::ClockSnapshots,
            _clock_snapshots.size(), snapshots);

        BinaryWriter tracks;
        for (const TrackDescriptor& track : _tracks)
        {
            tracks.U64(track.id);
            tracks.U64(track.parent_id);
            tracks.U32(track.clock_id);
            tracks.U32(track.name);
            tracks.U8(static_cast<uint8_t>(track.kind));
            tracks.U8(0);
            tracks.U16(0);
            tracks.I32(track.sort_order);
            tracks.U32(0);
            tracks.U64(track.source_id);
        }
        body.Chunk(ChunkType::Tracks, _tracks.size(), tracks);

        BinaryWriter arguments;
        arguments.Reserve(total_argument_count * 8);
        for (const EventArgument& argument : _arguments)
        {
            arguments.VarU64(argument.name);
            arguments.U8(static_cast<uint8_t>(argument.kind));

            if (argument.kind == ArgumentKind::Signed)
            {
                arguments.VarU64(ZigZagEncode(
                    std::bit_cast<int64_t>(argument.value)));
            }
            else if (argument.kind == ArgumentKind::Floating)
            {
                arguments.U64(argument.value);
            }
            else
            {
                arguments.VarU64(argument.value);
            }
        }

        BinaryWriter events;
        events.Reserve(total_event_count * 12);
        events.VarU64(total_event_count);
        std::vector<uint64_t> previous_timestamps(_tracks.size() + 1);
        std::vector<uint8_t> track_has_timestamp(_tracks.size() + 1);

        const auto encode_event = [&](TrackId track_id, uint64_t timestamp,
            uint64_t duration, StringId name, EventKind kind,
            uint64_t correlation_id, EventId parent_id, uint32_t color,
            uint64_t first_argument, size_t argument_count)
        {
            if (track_id == INVALID_TRACK_ID || track_id > _tracks.size())
            {
                SetError("Event references an unknown track");
                return false;
            }

            uint64_t previous_timestamp = track_has_timestamp[track_id]
                ? previous_timestamps[track_id] : 0;
            int64_t timestamp_delta = 0;

            if (timestamp >= previous_timestamp)
            {
                uint64_t delta = timestamp - previous_timestamp;

                if (delta > static_cast<uint64_t>(
                    std::numeric_limits<int64_t>::max()))
                {
                    SetError("Positive event timestamp delta exceeds int64");
                    return false;
                }

                timestamp_delta = static_cast<int64_t>(delta);
            }
            else
            {
                uint64_t delta = previous_timestamp - timestamp;

                if (delta > static_cast<uint64_t>(
                    std::numeric_limits<int64_t>::max()))
                {
                    SetError("Negative event timestamp delta exceeds int64");
                    return false;
                }

                timestamp_delta = -static_cast<int64_t>(delta);
            }

            uint8_t flags = static_cast<uint8_t>(kind) & EVENT_KIND_MASK;

            if (parent_id != INVALID_EVENT_ID)
            {
                if (parent_id >= _next_event_id)
                {
                    SetError("Event references an unknown parent event");
                    return false;
                }

                flags |= EVENT_HAS_PARENT;
            }

            if (correlation_id != 0)
            {
                flags |= EVENT_HAS_CORRELATION;
            }

            if (argument_count != 0)
            {
                flags |= EVENT_HAS_ARGUMENTS;
            }

            if (color != 0)
            {
                flags |= EVENT_HAS_COLOR;
            }

            events.VarU64(track_id);
            events.VarU64(ZigZagEncode(timestamp_delta));
            events.VarU64(duration);
            events.VarU64(name);
            events.U8(flags);

            if ((flags & EVENT_HAS_PARENT) != 0)
            {
                events.VarU64(parent_id);
            }

            if ((flags & EVENT_HAS_CORRELATION) != 0)
            {
                events.VarU64(correlation_id);
            }

            if ((flags & EVENT_HAS_ARGUMENTS) != 0)
            {
                events.VarU64(first_argument);
                events.VarU64(argument_count);
            }

            if ((flags & EVENT_HAS_COLOR) != 0)
            {
                events.VarU64(color);
            }

            previous_timestamps[track_id] = timestamp;
            track_has_timestamp[track_id] = 1;
            return true;
        };

        size_t event_index = 0;
        size_t source_index = 0;
        EventId next_event_id = 1;
        uint64_t next_argument_index = _arguments.size();

        while (next_event_id < _next_event_id)
        {
            while (source_index < _event_sources.size()
                && _event_sources[source_index].event_count == 0)
            {
                source_index++;
            }

            if (event_index < _events.size()
                && _events[event_index].id == next_event_id)
            {
                const TraceEvent& event = _events[event_index++];
                uint64_t argument_end =
                    static_cast<uint64_t>(event.first_argument)
                    + event.argument_count;

                if (argument_end > _arguments.size()
                    || !encode_event(event.track_id, event.timestamp,
                        event.duration, event.name, event.kind,
                        event.correlation_id, event.parent_id, event.color,
                        event.first_argument, event.argument_count))
                {
                    if (_last_error.empty())
                    {
                        SetError("Event arguments exceed the argument table");
                    }

                    return false;
                }

                next_event_id++;
                continue;
            }

            if (source_index >= _event_sources.size()
                || _event_sources[source_index].first_event_id != next_event_id)
            {
                SetError("Event sources do not cover the event ID sequence");
                return false;
            }

            const BufferedEventSourceDescriptor& source =
                _event_sources[source_index++];

            for (size_t i = 0; i < source.event_count; ++i)
            {
                BufferedEventWithArguments event;

                if (!source.reader(source.source, i, &event,
                    argument_storage.data(), argument_storage.size()))
                {
                    SetError("Failed to read buffered event source");
                    return false;
                }

                if (source.fixed_track_id != INVALID_TRACK_ID)
                {
                    event.event.track_id = source.fixed_track_id;
                }

                uint64_t first_argument = next_argument_index;

                for (size_t argument_index = 0;
                    argument_index < event.argument_count; ++argument_index)
                {
                    const BufferedEventArgument& argument =
                        event.arguments[argument_index];
                    uint64_t value = argument.kind == ArgumentKind::String
                        ? intern_source_string(argument.string_value)
                        : argument.value;
                    StringId name = intern_source_string(argument.name);
                    arguments.VarU64(name);
                    arguments.U8(static_cast<uint8_t>(argument.kind));

                    if (argument.kind == ArgumentKind::Signed)
                    {
                        arguments.VarU64(ZigZagEncode(
                            std::bit_cast<int64_t>(value)));
                    }
                    else if (argument.kind == ArgumentKind::Floating)
                    {
                        arguments.U64(value);
                    }
                    else
                    {
                        arguments.VarU64(value);
                    }

                    next_argument_index++;
                }

                const BufferedTraceEvent& buffered = event.event;

                if (!encode_event(buffered.track_id, buffered.timestamp,
                    buffered.duration, intern_source_string(buffered.name),
                    buffered.kind,
                    buffered.correlation_id, buffered.parent_id,
                    buffered.color, first_argument, event.argument_count))
                {
                    return false;
                }

                next_event_id++;
            }
        }

        body.Chunk(ChunkType::Events, total_event_count, events);
        body.Chunk(ChunkType::Arguments, total_argument_count, arguments);

        uint8_t file_flags = 0;
        const std::vector<uint8_t>* stored_body = &body.Data();

#if defined(NANOTRACE_WITH_MINIZ)
        if (body.Data().size() > std::numeric_limits<mz_ulong>::max())
        {
            SetError("Trace body exceeds miniz input size");
            return false;
        }

        mz_ulong source_size = static_cast<mz_ulong>(body.Data().size());
        mz_ulong compressed_size = mz_compressBound(source_size);
        std::vector<uint8_t> compressed_body(compressed_size);
        int compression_result = mz_compress2(compressed_body.data(),
            &compressed_size, body.Data().data(), source_size, MZ_BEST_SPEED);

        if (compression_result != MZ_OK)
        {
            SetError("Failed to compress trace body");
            return false;
        }

        compressed_body.resize(compressed_size);
        stored_body = &compressed_body;
        file_flags |= 1;
#endif

        BinaryWriter file;
        file.Reserve(32 + stored_body->size());
        const char magic[8]{ 'N', 'T', 'R', 'A', 'C', 'E', '4', '\0' };
        file.Bytes(magic, sizeof(magic));
        file.U16(4);
        file.U16(0);
        file.U8(1);
        file.U8(file_flags);
        file.U16(0);
        file.U64(body.Data().size());
        file.U64(stored_body->size());
        file.Bytes(stored_body->data(), stored_body->size());

        std::ofstream output{ filename, std::ios::binary | std::ios::trunc };

        if (!output)
        {
            SetError("Failed to open trace output file");
            return false;
        }

        const std::vector<uint8_t>& data = file.Data();
        output.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size()));
        output.close();

        if (!output)
        {
            SetError("Failed to write trace output file");
            return false;
        }

        return true;
    }

    uint64_t TraceSession::MonotonicRawNowNs()
    {
        timespec timestamp{};
        clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp);
        return static_cast<uint64_t>(timestamp.tv_sec) * 1'000'000'000ULL
            + static_cast<uint64_t>(timestamp.tv_nsec);
    }

    CpuThreadContext::CpuThreadContext(TraceSession& session,
        const char* thread_name, size_t event_capacity, TrackId parent_id,
        int32_t sort_order)
        : _session{ &session }
        , _track_id{ session.AddTrack(thread_name, TrackKind::CpuThread,
            session.ReferenceClock(), parent_id, sort_order) }
        , _events(event_capacity)
    {
    }

    CpuThreadContext::~CpuThreadContext()
    {
        Flush();
    }

    CpuEventToken CpuThreadContext::Begin() const
    {
        return CpuEventToken{ TraceSession::MonotonicRawNowNs() };
    }

    void CpuThreadContext::AddBufferedEvent(const char* name, EventKind kind,
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

    void CpuThreadContext::End(CpuEventToken token, const char* name,
        uint64_t correlation_id, uint32_t color)
    {
        uint64_t end = TraceSession::MonotonicRawNowNs();
        AddBufferedEvent(name, EventKind::Slice, token.timestamp,
            end - token.timestamp, correlation_id, color);
    }

    void CpuThreadContext::Bookmark(const char* name,
        uint64_t correlation_id, uint32_t color)
    {
        AddBufferedEvent(name, EventKind::Bookmark,
            TraceSession::MonotonicRawNowNs(), 0, correlation_id, color);
    }

    void CpuThreadContext::Flush()
    {
        if (!_session || _events_registered)
        {
            return;
        }

        if (_event_count != 0 && _session->AddBufferedEventSource(
            _events.data(), _event_count, ReadCpuEvent, _track_id)
            == INVALID_EVENT_ID)
        {
            return;
        }

        _events_registered = true;
    }
}
