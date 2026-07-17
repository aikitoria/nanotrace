#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "nanotrace/gpu_process_trace.h"
#include "hes.h"
#include "nanotrace/trace_writer.h"
#include "nanotrace/session.h"

namespace nanotrace
{
#if !defined(NANOTRACE_DISABLED)
    namespace
    {
        __global__ void PrimeHardwareEventCapture()
        {
        }
    }

    class GpuProcessTrace::Implementation
    {
    public:
        explicit Implementation(TraceSession& session,
            const char* kernel_name_prefix_to_strip)
            : _session{ &session }
            , _hes{ session, INVALID_TRACK_ID,
                kernel_name_prefix_to_strip }
        {
            if (!_hes.Initialize())
            {
                _last_error = _hes.LastError();
            }
        }

        bool IsReady() const
        {
            return _last_error.empty() && !_finished;
        }

        bool Begin()
        {
            if (!IsReady())
            {
                return false;
            }
            if (_capturing)
            {
                _last_error = "GPU trace capture has already begun";
                return false;
            }
            if (!_hes.BeginCapture())
            {
                _last_error = _hes.LastError();
                return false;
            }

            PrimeHardwareEventCapture<<<1, 1>>>();
            cudaError_t result = cudaGetLastError();
            if (result == cudaSuccess)
            {
                result = cudaDeviceSynchronize();
            }

            if (result != cudaSuccess)
            {
                _last_error = "Failed to prime GPU hardware tracing: ";
                _last_error += cudaGetErrorString(result);
                _hes.Stop();
                return false;
            }
            if (!_hes.ResetCaptureStart() || !_hes.CaptureClockSnapshot())
            {
                _last_error = _hes.LastError();
                _hes.Stop();
                return false;
            }

            _capturing = true;
            return true;
        }

        bool Finish()
        {
            if (!_capturing)
            {
                _last_error = "GPU trace capture has not begun";
                return false;
            }
            if (!_hes.Stop())
            {
                _last_error = _hes.LastError();
                return false;
            }
            _capturing = false;
            _finished = true;
            return true;
        }

        bool AddKernelTrace(trace_writer& kernel_trace)
        {
            return AddKernelTrace(kernel_trace, GpuKernelTraceOptions{});
        }

        bool AddKernelTrace(trace_writer& kernel_trace,
            const GpuKernelTraceOptions& options)
        {
            if (!_finished)
            {
                _last_error = "GPU trace capture has not finished";
                return false;
            }

            if (options.expected_invocation_count == 0
                || (options.expected_invocation_count > 1
                    && options.blocks_per_invocation == 0))
            {
                _last_error = "GPU kernel trace options are invalid";
                return false;
            }

            const char* kernel_name = options.kernel_name_substring
                ? options.kernel_name_substring : kernel_trace.KernelName();
            std::vector<const HesKernelEvent*> matches;
            const std::vector<HesKernelEvent>& kernels = _hes.KernelEvents();
            for (const HesKernelEvent& kernel : kernels)
            {
                const char* match_name = kernel.match_name
                    ? kernel.match_name : kernel.name;
                if (match_name && std::strstr(match_name, kernel_name)
                    && (!options.device_id
                        || kernel.device_id == *options.device_id)
                    && (!options.context_id
                        || kernel.context_id == *options.context_id))
                {
                    matches.push_back(&kernel);
                }
            }

            if (!matches.empty())
            {
                const HesKernelEvent& first_kernel = *matches.front();
                for (const HesKernelEvent* kernel : matches)
                {
                    if (kernel->device_id != first_kernel.device_id
                        || kernel->context_id != first_kernel.context_id)
                    {
                        _last_error = "GPU kernel trace matched more than one "
                            "CUDA context; select a device and context "
                            "explicitly";
                        return false;
                    }
                }
            }

            if (matches.size() != options.expected_invocation_count)
            {
                _last_error = "GPU hardware trace captured ";
                _last_error += std::to_string(matches.size());
                _last_error += " invocations, expected ";
                _last_error += std::to_string(
                    options.expected_invocation_count);
                _last_error += " for kernel: ";
                _last_error += kernel_name;
                return false;
            }

            std::sort(matches.begin(), matches.end(),
                [](const HesKernelEvent* first,
                    const HesKernelEvent* second)
                {
                    return first->start_ns < second->start_ns;
                });

            const HesKernelEvent& last_kernel = *matches.back();
            ClockId gpu_clock = _session->AddClock(
                "GPU global timer", ClockKind::GpuGlobalTimer,
                last_kernel.device_id);

            bool appended = false;
            if (matches.size() == 1)
            {
                appended = kernel_trace.AppendToSession(*_session,
                    last_kernel.track_id, gpu_clock, last_kernel.clock_id,
                    last_kernel.end_ns, last_kernel.event_id);
            }
            else
            {
                std::vector<event_parent_interval> parent_intervals;
                parent_intervals.reserve(matches.size());
                for (const HesKernelEvent* kernel : matches)
                {
                    parent_intervals.push_back(event_parent_interval{
                        kernel->start_ns, kernel->end_ns, kernel->event_id });
                }

                appended = kernel_trace.AppendToSession(*_session,
                    last_kernel.track_id, gpu_clock, last_kernel.clock_id,
                    last_kernel.end_ns, INVALID_EVENT_ID, 0,
                    &parent_intervals, options.blocks_per_invocation);
            }

            if (!appended)
            {
                _last_error = "Kernel trace contains no intra-kernel events";
                return false;
            }
            return true;
        }

        bool Finish(trace_writer& kernel_trace)
        {
            return Finish() && AddKernelTrace(kernel_trace);
        }

        const std::string& LastError() const
        {
            return _last_error;
        }

    private:
        TraceSession* _session;
        HesTracer _hes;
        std::string _last_error;
        bool _capturing = false;
        bool _finished = false;
    };
#else
    class GpuProcessTrace::Implementation
    {
    public:
        explicit Implementation(TraceSession&, const char*)
        {
        }

        bool IsReady() const { return true; }
        bool Begin() { return true; }
        bool Finish() { return true; }
        bool Finish(trace_writer&) { return true; }
        bool AddKernelTrace(trace_writer&) { return true; }
        bool AddKernelTrace(trace_writer&,
            const GpuKernelTraceOptions&) { return true; }
        const std::string& LastError() const { return _last_error; }

    private:
        std::string _last_error;
    };
#endif

    GpuProcessTrace::GpuProcessTrace(TraceSession& session,
        const char* kernel_name_prefix_to_strip)
        : _implementation{ std::make_unique<Implementation>(session,
            kernel_name_prefix_to_strip) }
    {
    }

    GpuProcessTrace::~GpuProcessTrace() = default;

    GpuProcessTrace::operator bool() const
    {
        return _implementation->IsReady();
    }

    bool GpuProcessTrace::Begin()
    {
        return _implementation->Begin();
    }

    bool GpuProcessTrace::Finish()
    {
        return _implementation->Finish();
    }

    bool GpuProcessTrace::Finish(trace_writer& kernel_trace)
    {
        return _implementation->Finish(kernel_trace);
    }

    bool GpuProcessTrace::AddKernelTrace(trace_writer& kernel_trace)
    {
        return _implementation->AddKernelTrace(kernel_trace);
    }

    bool GpuProcessTrace::AddKernelTrace(trace_writer& kernel_trace,
        const GpuKernelTraceOptions& options)
    {
        return _implementation->AddKernelTrace(kernel_trace, options);
    }

    const std::string& GpuProcessTrace::LastError() const
    {
        return _implementation->LastError();
    }
}
