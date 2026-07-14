#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "nanotrace/nanotrace_gpu.h"
#include "nanotrace/nanotrace_hes.h"
#include "nanotrace/nanotrace_host.h"
#include "nanotrace/nanotrace_session.h"

namespace nanotrace
{
#if !defined(NANOTRACE_DISABLED)
    namespace
    {
        __global__ void PrimeHardwareEventCapture()
        {
        }
    }

    class GpuTrace::Implementation
    {
    public:
        explicit Implementation(const char* session_name)
            : _session{ session_name }
            , _hes{ _session }
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

        bool Write(const char* filename, trace_writer& kernel_trace)
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

            const std::vector<HesKernelEvent>& kernels = _hes.KernelEvents();
            const HesKernelEvent* matched_kernel = nullptr;
            for (std::vector<HesKernelEvent>::const_reverse_iterator it =
                    kernels.rbegin(); it != kernels.rend(); ++it)
            {
                if (it->name
                    && std::strstr(it->name, kernel_trace.KernelName()))
                {
                    matched_kernel = &*it;
                    break;
                }
            }
            if (!matched_kernel)
            {
                _last_error = "GPU hardware trace did not contain kernel: ";
                _last_error += kernel_trace.KernelName();
                return false;
            }

            ClockId gpu_clock = _session.AddClock(
                "GPU global timer", ClockKind::GpuGlobalTimer,
                matched_kernel->device_id);
            if (!kernel_trace.AppendToSession(_session,
                matched_kernel->track_id, gpu_clock, matched_kernel->clock_id,
                matched_kernel->end_ns, matched_kernel->event_id))
            {
                _last_error = "Kernel trace contains no intra-kernel events";
                return false;
            }
            if (!_session.Write(filename))
            {
                _last_error = _session.LastError();
                return false;
            }

            _finished = true;
            return true;
        }

        const std::string& LastError() const
        {
            return _last_error;
        }

        TraceSession& Session()
        {
            return _session;
        }

    private:
        TraceSession _session;
        HesTracer _hes;
        std::string _last_error;
        bool _capturing = false;
        bool _finished = false;
    };
#else
    class GpuTrace::Implementation
    {
    public:
        explicit Implementation(const char* session_name)
            : _session{ session_name }
        {
        }

        bool IsReady() const { return true; }
        bool Begin() { return true; }
        bool Write(const char*, trace_writer&) { return true; }
        TraceSession& Session() { return _session; }
        const std::string& LastError() const { return _last_error; }

    private:
        TraceSession _session;
        std::string _last_error;
    };
#endif

    GpuTrace::GpuTrace(const char* session_name)
        : _implementation{ std::make_unique<Implementation>(session_name) }
    {
    }

    GpuTrace::~GpuTrace() = default;

    GpuTrace::operator bool() const
    {
        return _implementation->IsReady();
    }

    bool GpuTrace::Begin()
    {
        return _implementation->Begin();
    }

    bool GpuTrace::Write(const char* filename, trace_writer& kernel_trace)
    {
        return _implementation->Write(filename, kernel_trace);
    }

    TraceSession& GpuTrace::Session()
    {
        return _implementation->Session();
    }

    const std::string& GpuTrace::LastError() const
    {
        return _implementation->LastError();
    }
}
