# nanotrace

Nanotrace records CPU scopes, CUDA kernel timings, and events inside kernels
on a shared timeline.

**[Open the viewer](https://aikitoria.github.io/nanotrace/)** to try a sample
or drop in a `.nanotrace` file. Traces are processed locally in your browser.
The viewer requires WebGPU.

![GPU execution and CPU worker tracks in the Nanotrace viewer](docs/img0.png)

Zoom into individual operations, expand instrumented kernels to inspect
blocks and lanes, or select a time range to compare event counts and durations.

![Trace overview with selection statistics](docs/img1.png)

## Record a trace

The CUDA examples require a Blackwell GPU, CUDA and CUPTI 13.3+, a C++20
compiler, CMake, Ninja, and libiberty development headers and library. They
build for `sm_120a`. See the [build guide](nanotrace-cuda/README.md#requirements)
for details.

From the repository root:

```bash
cmake -S nanotrace-cuda -B build -GNinja -DBUILD_EXAMPLES=ON
cmake --build build
CUDA_VISIBLE_DEVICES=0 ./build/examples/unified_trace
```

Open `unified_trace.nanotrace` in the viewer. It contains CPU scopes, hardware
timings for three CUDA kernels, and events recorded inside a kernel.

## Instrument your application

CPU recording uses per-thread buffers. CUDA kernel timing uses CUPTI HES;
events inside kernels use explicit instrumentation.

Create a `nanotrace::TraceSession` and attach `nanotrace::GpuProcessTrace`
**before initializing CUDA**. One collector captures all devices in the process.
Add scopes around the work you want to measure, then write a `.nanotrace` file.

See the [CPU and CUDA API guide](nanotrace-cuda/README.md) and the
[complete example](nanotrace-cuda/examples/unified_trace.cu) for setup and usage.
The [file format](docs/nanotrace.md) is documented separately.

## Run the viewer locally

Requires Node.js 26 and npm 11.

```bash
cd visualizer
npm ci
npm run dev
```

Open the URL printed by Vite. The included samples cover CUDA graphs, CPU
worker tracks, TMA transfers, and GLM-5.3 inference on eight GPUs.

The [viewer guide](visualizer/README.md) covers production builds, trace
validation, and installing the viewer as an app.

## License

[MIT](LICENSE). The bundled IBM Plex Sans font uses the
[SIL Open Font License](visualizer/src/assets/fonts/OFL.txt).
