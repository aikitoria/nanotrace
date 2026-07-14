#include <cstdio>

#include <nanotrace/nanotrace_session.h>

int main(int argc, char** argv)
{
    const char* output_path = argc > 1
        ? argv[1] : "cpu_hierarchy.nanotrace";

    nanotrace::TraceSession session{ "CPU hierarchy trace" };
    nanotrace::ClockId clock = session.ReferenceClock();
    nanotrace::TrackId main_thread = session.AddTrack(
        "Main thread", nanotrace::TrackKind::CpuThread, clock);
    nanotrace::TrackId worker_0 = session.AddTrack(
        "Worker 0", nanotrace::TrackKind::CpuThread, clock, main_thread, 0);
    nanotrace::TrackId worker_1 = session.AddTrack(
        "Worker 1", nanotrace::TrackKind::CpuThread, clock, main_thread, 1);
    nanotrace::TrackId io_thread = session.AddTrack(
        "I/O thread", nanotrace::TrackKind::CpuThread, clock, main_thread, 2);

    uint64_t start = nanotrace::TraceSession::MonotonicRawNowNs();
    session.AddSlice(main_thread, "Execute request", start, 240'000);
    session.AddSlice(main_thread, "Dispatch workers", start + 8'000, 22'000);
    session.AddSlice(main_thread, "Wait for workers", start + 34'000, 174'000);
    session.AddSlice(main_thread, "Publish result", start + 214'000, 18'000);

    session.AddSlice(worker_0, "Prepare shard", start + 24'000, 34'000);
    session.AddSlice(worker_0, "Compute shard", start + 62'000, 118'000);
    session.AddSlice(worker_0, "Commit shard", start + 184'000, 16'000);

    session.AddSlice(worker_1, "Prepare shard", start + 30'000, 28'000);
    session.AddSlice(worker_1, "Compute shard", start + 62'000, 126'000);
    session.AddSlice(worker_1, "Commit shard", start + 192'000, 12'000);

    session.AddSlice(io_thread, "Read input", start + 12'000, 44'000);
    session.AddSlice(io_thread, "Write output", start + 202'000, 26'000);

    if (!session.Write(output_path))
    {
        std::fprintf(stderr, "Trace write failed: %s\n",
            session.LastError().c_str());
        return 1;
    }

    std::printf("Wrote %s\n", output_path);
    return 0;
}
