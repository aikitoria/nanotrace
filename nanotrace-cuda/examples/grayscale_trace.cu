#include <nanotrace/nanotrace.cuh>
#include <nanotrace/nanotrace_host.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Define trace types
NANOTRACE_DEFINE_TRACE_TYPE(TraceWarp, "Warp", "Warp execution", 0, nanotrace::lane_type::STATIC);

// Define block type
NANOTRACE_DEFINE_BLOCK_TYPE(GrayscaleBlock, "Block {blockLinear}", "Block {blockLinear}");

// Define track type
NANOTRACE_DEFINE_TRACK_TYPE(WarpTrack, "Warp {lane}", "Warp {lane}", 0);

// L2 cache flush kernel
__global__ void flush_l2_kernel(float* buffer, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    float sum = 0.0f;
    for (size_t i = idx; i < size; i += stride) {
        sum += buffer[i];
    }

    // Write back to prevent optimization
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        buffer[0] = sum;
    }
}

// Grayscale kernel with nanotrace instrumentation
// Only first warp (warp 0) traces (1 lane total)
__global__ void rgb_to_grayscale_kernel(
    const float* __restrict__ rgb,
    float* __restrict__ gray,
    const int total_pixels,
    nanotrace::static_tensor_handle<1, 2> trace_handle)
{
    const float w_r = 0.2989f;
    const float w_g = 0.5870f;
    const float w_b = 0.1140f;

    // Only thread 0 does tracing (warp 0, lane 0)
    bool should_trace = (threadIdx.x == 0);

    // Start timing (only thread 0)
    auto lane = nanotrace::begin_lane(trace_handle, blockIdx.x, 0, should_trace);
    auto s = nanotrace::start();

    // Each thread processes 1 iteration of 4 pixels = 4 pixels total
    constexpr int iterations = 1;
    constexpr int pixels_per_iter = 4;

    int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * pixels_per_iter * iterations;

    #pragma unroll
    for (int iter = 0; iter < iterations; iter++) {
        int idx = base_idx + iter * pixels_per_iter;

        if (idx + 3 < total_pixels) {
            int rgb_base = idx * 3;

            float4 vec0 = *reinterpret_cast<const float4*>(&rgb[rgb_base]);
            float4 vec1 = *reinterpret_cast<const float4*>(&rgb[rgb_base + 4]);
            float4 vec2 = *reinterpret_cast<const float4*>(&rgb[rgb_base + 8]);

            float r0 = vec0.x, g0 = vec0.y, b0 = vec0.z;
            float r1 = vec0.w, g1 = vec1.x, b1 = vec1.y;
            float r2 = vec1.z, g2 = vec1.w, b2 = vec2.x;
            float r3 = vec2.y, g3 = vec2.z, b3 = vec2.w;

            float y0 = fmaf(w_r, r0, fmaf(w_g, g0, w_b * b0));
            float y1 = fmaf(w_r, r1, fmaf(w_g, g1, w_b * b1));
            float y2 = fmaf(w_r, r2, fmaf(w_g, g2, w_b * b2));
            float y3 = fmaf(w_r, r3, fmaf(w_g, g3, w_b * b3));

            float4 result = {y0, y1, y2, y3};
            *reinterpret_cast<float4*>(&gray[idx]) = result;
        }
    }

    // End timing (only thread 0)
    nanotrace::end(s, trace_handle, lane, TraceWarp{});
    nanotrace::finish_lane(trace_handle, lane);
}

int main() {
    printf("nanotrace-cuda grayscale example\n");

    // Image configuration: 16384x16384 = 268,435,456 pixels
    const int width = 16384;
    const int height = 16384;
    const int total_pixels = width * height;

    printf("Image size: %dx%d = %d pixels\n", width, height, total_pixels);

    // Allocate RGB input (3 channels) and grayscale output (1 channel)
    float* d_rgb;
    float* d_gray;

    size_t rgb_size = total_pixels * 3 * sizeof(float);
    size_t gray_size = total_pixels * sizeof(float);

    cudaMalloc(&d_rgb, rgb_size);
    cudaMalloc(&d_gray, gray_size);

    // Initialize RGB with random data
    float* h_rgb = new float[total_pixels * 3];
    for (int i = 0; i < total_pixels * 3; i++) {
        h_rgb[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    cudaMemcpy(d_rgb, h_rgb, rgb_size, cudaMemcpyHostToDevice);

    // Kernel configuration
    const int threads = 160;  // 5 warps
    const int pixels_per_thread = 4;
    const int blocks = (total_pixels + threads * pixels_per_thread - 1) / (threads * pixels_per_thread);

    dim3 grid(blocks, 1, 1);
    dim3 block(threads, 1, 1);

    printf("Launching kernel with grid (%u, %u, %u) and block (%u, %u, %u)\n",
           grid.x, grid.y, grid.z, block.x, block.y, block.z);
    printf("Total blocks: %d, threads per block: %d, warps per block: %d\n",
           blocks, threads, threads / 32);

    // Create trace tensor (1 lane = only warp 0 per block)
    using TraceConfig = nanotrace::static_trace_builder<1, TraceWarp>;

    TraceConfig trace_tensor(1, grid);  // 1 event per lane
    trace_tensor.set_track_type<WarpTrack>();

    // Allocate L2 flush buffer (128 MB should be enough to flush L2)
    float* d_flush_buffer;
    size_t flush_size = 128 * 1024 * 1024 / sizeof(float);  // 128 MB in floats
    cudaMalloc(&d_flush_buffer, flush_size * sizeof(float));
    cudaMemset(d_flush_buffer, 0, flush_size * sizeof(float));

    // Flush L2 before warmup
    flush_l2_kernel<<<256, 256>>>(d_flush_buffer, flush_size);
    cudaDeviceSynchronize();

    // Warmup: run 10 iterations to warm up GPU
    printf("Warming up GPU (10 iterations)...\n");
    for (int i = 0; i < 10; i++) {
        rgb_to_grayscale_kernel<<<grid, block>>>(
            d_rgb,
            d_gray,
            total_pixels,
            trace_tensor.get_handle());
    }
    cudaDeviceSynchronize();

    // Reset trace tensor before traced run
    trace_tensor.reset();

    // Flush L2 before traced run
    flush_l2_kernel<<<256, 256>>>(d_flush_buffer, flush_size);
    cudaDeviceSynchronize();

    // Traced run: single iteration with fresh trace data
    printf("Running traced iteration...\n");
    rgb_to_grayscale_kernel<<<grid, block>>>(
        d_rgb,
        d_gray,
        total_pixels,
        trace_tensor.get_handle());

    // Wait for completion
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        cudaFree(d_rgb);
        cudaFree(d_gray);
        delete[] h_rgb;
        return 1;
    }

    printf("Kernel completed successfully\n");

    // Verify output (simple sanity check)
    float* h_gray = new float[total_pixels];
    cudaMemcpy(h_gray, d_gray, gray_size, cudaMemcpyDeviceToHost);

    // Check a few pixels
    printf("Sample grayscale values: %.3f, %.3f, %.3f\n",
           h_gray[0], h_gray[100], h_gray[1000]);

    // Write trace file
    printf("Writing trace file...\n");
    nanotrace::trace_writer writer("rgb_to_grayscale");
    writer.set_block_type<GrayscaleBlock>();
    writer.add_tensor(trace_tensor);
    writer.write("grayscale_trace.nanotrace");

    printf("Trace written to grayscale_trace.nanotrace\n");

    // Cleanup
    cudaFree(d_rgb);
    cudaFree(d_gray);
    cudaFree(d_flush_buffer);
    delete[] h_rgb;
    delete[] h_gray;

    return 0;
}
