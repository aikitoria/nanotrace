#include <algorithm>
#include <atomic>
#include <cxxabi.h>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <cupti.h>

#include "hes.h"

namespace nanotrace
{
    namespace
    {
        constexpr size_t ACTIVITY_BUFFER_SIZE = 64ULL * 1024 * 1024;
        constexpr size_t ACTIVITY_BUFFER_ALIGNMENT = 64;

        std::string DemangleKernelName(const char* name)
        {
            if (!name)
            {
                return "CUDA kernel";
            }

            int status = 0;
            char* demangled = abi::__cxa_demangle(name, nullptr, nullptr,
                &status);
            if (status != 0 || !demangled)
            {
                std::free(demangled);
                return name;
            }

            std::string result{ demangled };
            std::free(demangled);
            return result;
        }

        std::string KernelDisplayName(const std::string& signature,
            std::string_view prefix_to_strip)
        {
            uint32_t depth = 0;
            std::string name = signature;

            for (size_t i = signature.size(); i > 0; --i)
            {
                char character = signature[i - 1];
                if (character == ')')
                {
                    depth++;
                }
                else if (character == '(' && depth != 0)
                {
                    depth--;
                    if (depth == 0)
                    {
                        name = signature.substr(0, i - 1);
                        break;
                    }
                }
            }

            constexpr std::string_view VOID_PREFIX{ "void " };
            if (name.starts_with(VOID_PREFIX))
            {
                name.erase(0, VOID_PREFIX.size());
            }

            if (!prefix_to_strip.empty()
                && name.starts_with(prefix_to_strip))
            {
                name.erase(0, prefix_to_strip.size());
            }
            return name;
        }

        struct StreamKey
        {
            uint32_t device_id;
            uint32_t context_id;
            uint32_t stream_id;

            bool operator==(const StreamKey& other) const
            {
                return device_id == other.device_id
                    && context_id == other.context_id
                    && stream_id == other.stream_id;
            }
        };

        struct StreamKeyHash
        {
            size_t operator()(const StreamKey& key) const
            {
                size_t hash = std::hash<uint32_t>{}(key.device_id);
                hash ^= std::hash<uint32_t>{}(key.context_id)
                    + 0x9e3779b9U + (hash << 6) + (hash >> 2);
                hash ^= std::hash<uint32_t>{}(key.stream_id)
                    + 0x9e3779b9U + (hash << 6) + (hash >> 2);
                return hash;
            }
        };

        struct TransparentStringHash
        {
            using is_transparent = void;

            size_t operator()(std::string_view value) const
            {
                return std::hash<std::string_view>{}(value);
            }
        };

        struct TransparentStringEqual
        {
            using is_transparent = void;

            bool operator()(std::string_view first,
                std::string_view second) const
            {
                return first == second;
            }
        };

        struct KernelNames
        {
            std::string name;
            std::string match_name;
        };

        struct GraphExecutionKey
        {
            uint32_t device_id;
            uint32_t context_id;
            uint32_t graph_id;
            uint64_t correlation_id;

            bool operator==(const GraphExecutionKey& other) const
            {
                return device_id == other.device_id
                    && context_id == other.context_id
                    && graph_id == other.graph_id
                    && correlation_id == other.correlation_id;
            }
        };

        struct GraphExecutionKeyHash
        {
            size_t operator()(const GraphExecutionKey& key) const
            {
                size_t hash = std::hash<uint32_t>{}(key.device_id);
                hash ^= std::hash<uint32_t>{}(key.context_id)
                    + 0x9e3779b9U + (hash << 6) + (hash >> 2);
                hash ^= std::hash<uint32_t>{}(key.graph_id)
                    + 0x9e3779b9U + (hash << 6) + (hash >> 2);
                hash ^= std::hash<uint64_t>{}(key.correlation_id)
                    + 0x9e3779b9U + (hash << 6) + (hash >> 2);
                return hash;
            }
        };

        struct GraphExecutionEvent
        {
            EventId event_id = INVALID_EVENT_ID;
            TrackId track_id = INVALID_TRACK_ID;
            ClockId clock_id = INVALID_CLOCK_ID;
            uint32_t device_id = 0;
            uint32_t context_id = 0;
            uint32_t graph_id = 0;
            uint64_t correlation_id = 0;
            uint64_t start_ns = 0;
            uint64_t end_ns = 0;
            uint32_t activity_count = 0;
        };

        using KernelNameMap = std::unordered_map<std::string, KernelNames,
            TransparentStringHash, TransparentStringEqual>;
        using DeviceTrackMap = std::unordered_map<uint32_t, TrackId>;
        using StreamTrackMap = std::unordered_map<StreamKey, TrackId,
            StreamKeyHash>;
        using GraphExecutionMap = std::unordered_map<GraphExecutionKey,
            size_t, GraphExecutionKeyHash>;

        bool ReadHesEvent(const void* source, size_t event_index,
            BufferedEventWithArguments* output,
            BufferedEventArgument* argument_storage, size_t argument_capacity)
        {
            if (argument_capacity < 3)
            {
                return false;
            }

            const HesKernelEvent* kernels =
                static_cast<const HesKernelEvent*>(source);
            const HesKernelEvent& kernel = kernels[event_index];
            argument_storage[0] = BufferedEventArgument{
                "graph_id", ArgumentKind::Unsigned, kernel.graph_id, nullptr };
            argument_storage[1] = BufferedEventArgument{
                "graph_node_id", ArgumentKind::Unsigned,
                kernel.graph_node_id, nullptr };
            argument_storage[2] = BufferedEventArgument{
                "context_id", ArgumentKind::Unsigned,
                kernel.context_id, nullptr };
            output->event = BufferedTraceEvent{
                kernel.name,
                kernel.track_id,
                EventKind::Slice,
                kernel.start_ns,
                kernel.end_ns - kernel.start_ns,
                kernel.correlation_id,
                INVALID_EVENT_ID,
                0,
            };
            output->arguments = argument_storage;
            output->argument_count = 3;
            return true;
        }

        bool ReadGraphExecutionEvent(const void* source, size_t event_index,
            BufferedEventWithArguments* output,
            BufferedEventArgument* argument_storage, size_t argument_capacity)
        {
            if (argument_capacity < 3)
            {
                return false;
            }

            const GraphExecutionEvent* executions =
                static_cast<const GraphExecutionEvent*>(source);
            const GraphExecutionEvent& execution = executions[event_index];
            argument_storage[0] = BufferedEventArgument{
                "graph_id", ArgumentKind::Unsigned,
                execution.graph_id, nullptr };
            argument_storage[1] = BufferedEventArgument{
                "context_id", ArgumentKind::Unsigned,
                execution.context_id, nullptr };
            argument_storage[2] = BufferedEventArgument{
                "activity_count", ArgumentKind::Unsigned,
                execution.activity_count, nullptr };
            output->event = BufferedTraceEvent{
                "CUDA Graph",
                execution.track_id,
                EventKind::Slice,
                execution.start_ns,
                execution.end_ns - execution.start_ns,
                execution.correlation_id,
                INVALID_EVENT_ID,
                0,
            };
            output->arguments = argument_storage;
            output->argument_count = 3;
            return true;
        }

    }

    class HesTracer::Implementation
    {
    public:
        Implementation(TraceSession& session, TrackId parent_track,
            const char* kernel_name_prefix_to_strip)
            : _session{ &session }
            , _parent_track{ parent_track }
            , _cupti_clock{ session.AddClock(
                "CUPTI hardware clock", ClockKind::Cupti) }
            , _kernel_name_prefix_to_strip{
                kernel_name_prefix_to_strip
                    ? kernel_name_prefix_to_strip : "" }
        {
        }

        ~Implementation()
        {
            Stop();
        }

        bool Initialize()
        {
            std::lock_guard<std::mutex> lock{ _state_mutex };

            if (_initialized)
            {
                SetError("HES tracer is already initialized");
                return false;
            }

            if (_active.load(std::memory_order_acquire) != nullptr)
            {
                SetError("Only one nanotrace HES tracer can be active");
                return false;
            }

            uint32_t cupti_version = 0;

            if (!Check(cuptiGetVersion(&cupti_version),
                "query CUPTI version"))
            {
                return false;
            }

            if (cupti_version < 130300)
            {
                SetError("CUPTI 13.3 or newer is required for HES tracing");
                return false;
            }

            CUpti_SubscriberParams parameters{};
            parameters.structSize = CUpti_SubscriberParams_STRUCT_SIZE;
            parameters.allowMultipleSubscribers = 0;

            if (!Check(cuptiSubscribe_v2(&_subscriber, SubscriberCallback, this,
                &parameters), "subscribe"))
            {
                return false;
            }

            _active.store(this, std::memory_order_release);

            if (!Check(cuptiActivityRegisterCallbacks_v2(_subscriber,
                RequestBuffer, CompleteBuffer), "register activity buffers"))
            {
                _active.store(nullptr, std::memory_order_release);
                cuptiUnsubscribe(_subscriber);
                _subscriber = nullptr;
                return false;
            }

            if (!Check(cuptiEnableCallback(1, _subscriber,
                CUPTI_CB_DOMAIN_STATE, CUPTI_CBID_STATE_FATAL_ERROR),
                "enable fatal error callback"))
            {
                _active.store(nullptr, std::memory_order_release);
                cuptiUnsubscribe(_subscriber);
                _subscriber = nullptr;
                return false;
            }

            if (!Check(cuptiActivityEnable_v2(_subscriber,
                CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, nullptr),
                "enable concurrent kernel activity"))
            {
                _active.store(nullptr, std::memory_order_release);
                cuptiUnsubscribe(_subscriber);
                _subscriber = nullptr;
                return false;
            }

            uint8_t enable_hes = 1;
            size_t enable_hes_size = sizeof(enable_hes);

            if (!Check(cuptiActivitySetAttribute_v2(_subscriber,
                CUPTI_ACTIVITY_ATTR_ENABLE_HES, &enable_hes_size,
                &enable_hes), "request HES hardware tracing"))
            {
                _active.store(nullptr, std::memory_order_release);
                cuptiUnsubscribe(_subscriber);
                _subscriber = nullptr;
                return false;
            }

            CUresult driver_result = cuInit(0);

            if (driver_result != CUDA_SUCCESS)
            {
                const char* driver_error = nullptr;
                cuGetErrorString(driver_result, &driver_error);
                _last_error = "initialize CUDA driver: ";
                _last_error += driver_error ? driver_error
                    : "unknown CUDA driver error";
                _active.store(nullptr, std::memory_order_release);
                cuptiUnsubscribe(_subscriber);
                _subscriber = nullptr;
                return false;
            }

            uint8_t hes_enabled = 0;
            size_t hes_enabled_size = sizeof(hes_enabled);

            if (!Check(cuptiActivityGetAttribute_v2(nullptr,
                CUPTI_ACTIVITY_ATTR_ENABLE_HES, &hes_enabled_size,
                &hes_enabled), "query HES state"))
            {
                _active.store(nullptr, std::memory_order_release);
                cuptiUnsubscribe(_subscriber);
                _subscriber = nullptr;
                return false;
            }

            if (hes_enabled == 0)
            {
                SetError("HES did not activate during CUDA initialization");
                _active.store(nullptr, std::memory_order_release);
                cuptiUnsubscribe(_subscriber);
                _subscriber = nullptr;
                return false;
            }

            _hes_confirmed = true;

            if (!Check(cuptiActivityDisable_v2(_subscriber,
                CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, nullptr),
                "disable concurrent kernel activity until capture"))
            {
                _active.store(nullptr, std::memory_order_release);
                cuptiUnsubscribe(_subscriber);
                _subscriber = nullptr;
                return false;
            }

            _initialized = true;
            return true;
        }

        bool CaptureClockSnapshot()
        {
            std::lock_guard<std::mutex> lock{ _state_mutex };

            if (!_initialized)
            {
                SetError("HES tracer is not initialized");
                return false;
            }

            return CaptureClockSnapshotUnlocked();
        }

        bool ResetCaptureStart()
        {
            std::lock_guard<std::mutex> lock{ _state_mutex };

            if (!_capture_started)
            {
                SetError("HES capture is not active");
                return false;
            }

            return Check(cuptiGetTimestamp_v2(
                _subscriber, &_capture_start_ns),
                "reset capture start timestamp");
        }

        bool BeginCapture()
        {
            std::lock_guard<std::mutex> lock{ _state_mutex };

            if (!_initialized)
            {
                SetError("HES tracer is not initialized");
                return false;
            }

            if (_capture_started)
            {
                SetError("HES capture is already active");
                return false;
            }

            if (!EnableCaptureActivities())
            {
                return false;
            }

            if (!Check(cuptiGetTimestamp_v2(
                _subscriber, &_capture_start_ns),
                "capture begin timestamp"))
            {
                DisableCaptureActivities();
                return false;
            }

            _capture_started = true;
            return true;
        }

        bool Stop()
        {
            std::lock_guard<std::mutex> lock{ _state_mutex };

            if (!_initialized)
            {
                return _last_error.empty();
            }

            bool success = CaptureClockSnapshotUnlocked();
            success = Check(cuptiGetLastError(),
                "query pending CUPTI error") && success;
            success = Check(cuptiActivityFlushAll(
                CUPTI_ACTIVITY_FLAG_FLUSH_FORCED), "flush activities")
                && success;
            success = Check(cuptiUnsubscribe(_subscriber), "unsubscribe")
                && success;
            _subscriber = nullptr;
            _active.store(nullptr, std::memory_order_release);
            _initialized = false;

            return AppendEventsToSession() && success;
        }

        bool IsInitialized() const { return _initialized; }
        const std::string& LastError() const { return _last_error; }
        const std::vector<HesKernelEvent>& KernelEvents() const
        {
            return _completed_kernel_events
                ? *_completed_kernel_events : _kernel_events;
        }

    private:
        bool EnableCaptureActivities()
        {
            return Check(cuptiActivityEnable_v2(_subscriber,
                CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, nullptr),
                "enable concurrent kernel activity");
        }

        void DisableCaptureActivities()
        {
            cuptiActivityDisable_v2(_subscriber,
                CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, nullptr);
        }

        static void CUPTIAPI SubscriberCallback(void* user_data,
            CUpti_CallbackDomain domain, CUpti_CallbackId,
            const void* callback_data)
        {
            Implementation* implementation =
                static_cast<Implementation*>(user_data);

            if (!implementation)
            {
                return;
            }

            if (domain == CUPTI_CB_DOMAIN_STATE && callback_data)
            {
                const CUpti_StateData* state =
                    static_cast<const CUpti_StateData*>(callback_data);
                implementation->SetError("CUPTI state notification",
                    state->notification.result);
            }
        }

        static void CUPTIAPI RequestBuffer(uint8_t** buffer,
            size_t* size, size_t* max_records,
            CUpti_BufferCallbackRequestInfo*)
        {
            *buffer = static_cast<uint8_t*>(std::aligned_alloc(
                ACTIVITY_BUFFER_ALIGNMENT, ACTIVITY_BUFFER_SIZE));
            *size = *buffer ? ACTIVITY_BUFFER_SIZE : 0;
            *max_records = 0;
        }

        static void CUPTIAPI CompleteBuffer(uint8_t* buffer, size_t,
            size_t valid_size, CUpti_BufferCallbackCompleteInfo*)
        {
            Implementation* active =
                _active.load(std::memory_order_acquire);

            if (active && valid_size != 0)
            {
                active->ParseBuffer(buffer, valid_size);
            }

            std::free(buffer);
        }

        void ParseBuffer(uint8_t* buffer, size_t valid_size)
        {
            std::lock_guard<std::mutex> lock{ _event_mutex };
            CUpti_Activity* activity = nullptr;

            while (true)
            {
                CUptiResult result = cuptiActivityGetNextRecord_v2(
                    _subscriber, buffer, valid_size, &activity);

                if (result == CUPTI_ERROR_MAX_LIMIT_REACHED)
                {
                    break;
                }

                if (result != CUPTI_SUCCESS)
                {
                    SetError("Failed to parse a CUPTI activity buffer", result);
                    break;
                }

                if (activity->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL
                    || activity->kind == CUPTI_ACTIVITY_KIND_KERNEL)
                {
                    const CUpti_ActivityKernel12* kernel =
                        reinterpret_cast<const CUpti_ActivityKernel12*>(activity);
                    const char* raw_name = kernel->name
                        ? kernel->name : "CUDA kernel";
                    const KernelNameMap::iterator existing =
                        _kernel_names.find(std::string_view{ raw_name });
                    const KernelNames* names = nullptr;

                    if (existing != _kernel_names.end())
                    {
                        names = &existing->second;
                    }
                    else
                    {
                        std::string signature = DemangleKernelName(raw_name);
                        std::string match_name = KernelDisplayName(
                            signature, {});
                        std::string name = KernelDisplayName(signature,
                            _kernel_name_prefix_to_strip);
                        const std::pair<KernelNameMap::iterator, bool> inserted =
                            _kernel_names.emplace(raw_name, KernelNames{
                                std::move(name), std::move(match_name) });
                        names = &inserted.first->second;
                    }

                    _raw_kernel_events.push_back(HesKernelEvent{
                        INVALID_EVENT_ID,
                        INVALID_TRACK_ID,
                        _cupti_clock,
                        kernel->deviceId,
                        kernel->contextId,
                        kernel->streamId,
                        kernel->graphId,
                        kernel->graphNodeId,
                        kernel->correlationId,
                        kernel->start,
                        kernel->end,
                        names->name.c_str(),
                        names->match_name.c_str(),
                    });
                    continue;
                }

            }
        }

        bool AppendEventsToSession()
        {
            std::lock_guard<std::mutex> lock{ _event_mutex };
            DeviceTrackMap device_tracks;
            StreamTrackMap stream_tracks;
            GraphExecutionMap graph_execution_indices;
            _kernel_events.reserve(
                _kernel_events.size() + _raw_kernel_events.size());

            const auto get_device_track = [this, &device_tracks](
                uint32_t device_id) -> TrackId
            {
                const DeviceTrackMap::const_iterator existing =
                    device_tracks.find(device_id);
                if (existing != device_tracks.end())
                {
                    return existing->second;
                }

                std::string name = "GPU " + std::to_string(device_id);
                TrackId track = _session->AddTrack(name.c_str(),
                    TrackKind::GpuDevice, _cupti_clock, _parent_track,
                    static_cast<int32_t>(device_id), device_id);
                device_tracks.emplace(device_id, track);
                return track;
            };

            const auto get_stream_track = [this, &stream_tracks,
                &get_device_track](uint32_t device_id, uint32_t context_id,
                uint32_t stream_id, bool graph_track) -> TrackId
            {
                StreamKey key{ device_id, context_id, stream_id };
                const StreamTrackMap::const_iterator existing =
                    stream_tracks.find(key);
                if (existing != stream_tracks.end())
                {
                    return existing->second;
                }

                std::string name = graph_track
                    ? "Graph executions"
                    : "Stream " + std::to_string(stream_id);
                TrackId track = _session->AddTrack(name.c_str(),
                    TrackKind::GpuStream, _cupti_clock,
                    get_device_track(device_id),
                    graph_track ? -1 : static_cast<int32_t>(stream_id),
                    stream_id);
                stream_tracks.emplace(key, track);
                return track;
            };

            const auto record_graph_execution = [this,
                &graph_execution_indices](uint32_t device_id,
                uint32_t context_id, uint32_t graph_id,
                uint64_t correlation_id, uint64_t start_ns,
                uint64_t end_ns)
            {
                if (graph_id == 0 || correlation_id == 0)
                {
                    return;
                }

                GraphExecutionKey key{
                    device_id, context_id, graph_id, correlation_id };
                const GraphExecutionMap::const_iterator existing =
                    graph_execution_indices.find(key);
                if (existing == graph_execution_indices.end())
                {
                    size_t index = _graph_events.size();
                    _graph_events.push_back(GraphExecutionEvent{
                        INVALID_EVENT_ID,
                        INVALID_TRACK_ID,
                        _cupti_clock,
                        device_id,
                        context_id,
                        graph_id,
                        correlation_id,
                        start_ns,
                        end_ns,
                        1,
                    });
                    graph_execution_indices.emplace(key, index);
                    return;
                }

                GraphExecutionEvent& execution =
                    _graph_events[existing->second];
                execution.start_ns = std::min(execution.start_ns, start_ns);
                execution.end_ns = std::max(execution.end_ns, end_ns);
                execution.activity_count++;
            };

            for (HesKernelEvent& kernel : _raw_kernel_events)
            {
                if (kernel.start_ns == 0 || kernel.end_ns < kernel.start_ns
                    || (_capture_started
                        && kernel.start_ns < _capture_start_ns))
                {
                    continue;
                }

                kernel.track_id = get_stream_track(kernel.device_id,
                    kernel.context_id, kernel.stream_id, false);
                record_graph_execution(kernel.device_id, kernel.context_id,
                    kernel.graph_id, kernel.correlation_id,
                    kernel.start_ns, kernel.end_ns);
                _kernel_events.push_back(std::move(kernel));
            }

            _raw_kernel_events.clear();

            constexpr uint32_t GRAPH_TRACK_STREAM_ID =
                std::numeric_limits<uint32_t>::max();
            for (GraphExecutionEvent& execution : _graph_events)
            {
                execution.track_id = get_stream_track(execution.device_id,
                    execution.context_id, GRAPH_TRACK_STREAM_ID, true);
            }

            if (!_graph_events.empty())
            {
                std::shared_ptr<std::vector<GraphExecutionEvent>> events =
                    std::make_shared<std::vector<GraphExecutionEvent>>(
                        std::move(_graph_events));
                std::shared_ptr<const void> owner = events;
                EventId first_graph_event_id =
                    _session->AddBufferedEventSource(std::move(owner),
                        events->data(), events->size(),
                        ReadGraphExecutionEvent);
                if (first_graph_event_id == INVALID_EVENT_ID)
                {
                    SetError(_session->LastError().c_str());
                    return false;
                }

                for (size_t i = 0; i < events->size(); ++i)
                {
                    (*events)[i].event_id = first_graph_event_id + i;
                }
            }

            if (!_kernel_events.empty())
            {
                std::shared_ptr<std::vector<HesKernelEvent>> events =
                    std::make_shared<std::vector<HesKernelEvent>>(
                        std::move(_kernel_events));
                std::shared_ptr<const void> owner = events;
                EventId first_kernel_event_id =
                    _session->AddBufferedEventSource(std::move(owner),
                        events->data(), events->size(), ReadHesEvent);
                if (first_kernel_event_id == INVALID_EVENT_ID)
                {
                    SetError(_session->LastError().c_str());
                    return false;
                }

                for (size_t i = 0; i < events->size(); ++i)
                {
                    (*events)[i].event_id = first_kernel_event_id + i;
                }
                _completed_kernel_events = std::move(events);
            }

            return true;
        }

        bool CaptureClockSnapshotUnlocked()
        {
            if (!_hes_confirmed)
            {
                uint8_t hes_enabled = 0;
                size_t hes_enabled_size = sizeof(hes_enabled);

                if (!Check(cuptiActivityGetAttribute_v2(nullptr,
                    CUPTI_ACTIVITY_ATTR_ENABLE_HES, &hes_enabled_size,
                    &hes_enabled), "query HES state"))
                {
                    return false;
                }

                if (hes_enabled == 0)
                {
                    SetError("HES did not activate during CUDA initialization");
                    return false;
                }

                _hes_confirmed = true;
            }

            uint64_t reference_before = TraceSession::MonotonicRawNowNs();
            uint64_t cupti_timestamp = 0;

            if (!Check(cuptiGetTimestamp_v2(
                _subscriber, &cupti_timestamp), "get CUPTI timestamp"))
            {
                return false;
            }

            uint64_t reference_after = TraceSession::MonotonicRawNowNs();
            uint64_t elapsed = reference_after - reference_before;
            _session->AddClockSnapshot(_cupti_clock,
                _session->ReferenceClock(), cupti_timestamp,
                reference_before + elapsed / 2, (elapsed + 1) / 2);
            return true;
        }

        bool Check(CUptiResult result, const char* operation)
        {
            if (result == CUPTI_SUCCESS)
            {
                return true;
            }

            SetError(operation, result);
            return false;
        }

        void SetError(const char* operation, CUptiResult result)
        {
            const char* result_string = nullptr;
            cuptiGetResultString(result, &result_string);
            _last_error = operation ? operation : "CUPTI failure";
            _last_error += ": ";
            _last_error += result_string ? result_string : "unknown CUPTI error";
        }

        void SetError(const char* message)
        {
            _last_error = message ? message : "Unknown HES tracer error";
        }

        TraceSession* _session;
        TrackId _parent_track;
        ClockId _cupti_clock;
        CUpti_SubscriberHandle _subscriber = nullptr;
        bool _hes_confirmed = false;
        bool _initialized = false;
        bool _capture_started = false;
        uint64_t _capture_start_ns = 0;
        std::string _last_error;
        std::mutex _state_mutex;
        std::mutex _event_mutex;
        std::vector<HesKernelEvent> _raw_kernel_events;
        std::vector<HesKernelEvent> _kernel_events;
        std::shared_ptr<std::vector<HesKernelEvent>>
            _completed_kernel_events;
        std::vector<GraphExecutionEvent> _graph_events;
        KernelNameMap _kernel_names;
        std::string _kernel_name_prefix_to_strip;
        static inline std::atomic<Implementation*> _active{ nullptr };
    };

    HesTracer::HesTracer(TraceSession& session, TrackId parent_track,
        const char* kernel_name_prefix_to_strip)
        : _implementation{
            std::make_unique<Implementation>(session, parent_track,
                kernel_name_prefix_to_strip) }
    {
    }

    HesTracer::~HesTracer() = default;

    bool HesTracer::Initialize() { return _implementation->Initialize(); }
    bool HesTracer::BeginCapture()
    {
        return _implementation->BeginCapture();
    }
    bool HesTracer::ResetCaptureStart()
    {
        return _implementation->ResetCaptureStart();
    }
    bool HesTracer::CaptureClockSnapshot()
    {
        return _implementation->CaptureClockSnapshot();
    }
    bool HesTracer::Stop() { return _implementation->Stop(); }
    bool HesTracer::IsInitialized() const
    {
        return _implementation->IsInitialized();
    }
    const std::string& HesTracer::LastError() const
    {
        return _implementation->LastError();
    }
    const std::vector<HesKernelEvent>& HesTracer::KernelEvents() const
    {
        return _implementation->KernelEvents();
    }
}
