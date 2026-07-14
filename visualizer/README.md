# Nanotrace visualizer

WebGPU viewer for nanotrace v4 CPU, CUPTI HES, and CUDA intra-kernel tracks.

Kernel details expand progressively through SM, block, and lane rows. Click
zone or row disclosure controls to expand one level, or use the top-bar control
to expand or collapse the complete hierarchy. Multi-stream traces are grouped
under collapsible GPU rows. Tooltips include serialized event arguments, and
all displayed timestamps use adaptive units.

Requires Node.js 26 and npm 11.

## Development

~~~bash
npm ci
npm run dev
~~~

## Production build

~~~bash
npm run build
~~~

The built files are written to `dist/`.

## Validation

~~~bash
npm run validate -- /path/to/trace.nanotrace
~~~

The validator uses the production parser and checks correlated event bounds,
event parents, track references, non-overlapping sublanes, hover lookup, and
collapsed and expanded projections. It contains no sample-specific names or
expected event counts.

## Bundled samples

- `unified_trace.nanotrace`: three CUDA kernels with CPU, HES, and explicit
  intra-kernel events;
- `multistream_graph.nanotrace`: a CUDA graph with two execution streams;
- `cpu_hierarchy.nanotrace`: application-defined CPU parent and worker tracks;
- `tma_bandwidth_static_sm120a.nanotrace`: statically scheduled TMA transfers;
- `tma_bandwidth_atomic_sm120a.nanotrace`: dynamically scheduled TMA transfers.

The files are produced by the C++ examples in `nanotrace-cuda/examples`, not by
a separate browser-format generator. The GPU samples are generated from the
repository's `sm_120a` build.
