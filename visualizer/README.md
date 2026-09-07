# Nanotrace visualizer

WebGPU viewer for nanotrace v4 CPU, CUPTI HES, and CUDA intra-kernel tracks.

Click a track label to expand or collapse its children. Instrumented kernels
also have zone disclosure controls for their SM, block, and lane details. The
top-bar control expands or collapses the complete hierarchy. Multi-stream
traces are grouped under GPU rows. Tooltips include serialized event arguments,
and displayed timestamps use adaptive units.

Region annotations span existing track groups without adding rows. Their
colored labels sit between groups and use the same font and minimum draw sizes
as zone labels. Horizontal borders mark their extent; a background tint appears
on hover. Region tooltips show duration, and right-click copying retains exact
start, end, and duration values. Drawn labels omit `mina::kernels::` and
`mina::components::`; trace metadata and copied names retain the full names.
Groups follow the track metadata.

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

The built files are written to `dist/`. GitHub CI supplies `NANOTRACE_GIT_HASH`
so the viewer identifies the deployed commit. Local builds show `dev` unless
that variable is set.

## Install as an app

After the GitHub Pages deployment, open
<https://aikitoria.github.io/nanotrace/> and use the browser's **Install app**
action (or **Add to Home Screen** / **Add to Dock**). The manifest launches
Nanotrace in a standalone window and keeps its identity, start URL and service
worker scope under `/nanotrace/`.

The app requires a WebGPU-capable browser. Like the online viewer, the installed
app needs a network connection to launch; the service worker does not cache
application assets or trace files. Updates do not reload an active analysis.
The Vite dev server does not register the worker. Installation requires HTTPS
or localhost; use the deployed HTTPS site to install from another machine.

`npm run validate:pwa` checks the built Pages paths, manifest, icons and service
worker before CI uploads the deployment artifact. Run it after `npm run build`.

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
- `mina_cuda_graph.nanotrace`: a Mina GLM-5.3 server capture from 2026-09-07,
  running on eight GPUs with a 200-token prompt and 100 generated tokens;
- `multistream_graph.nanotrace`: a CUDA graph with two execution streams;
- `cpu_hierarchy.nanotrace`: application-defined CPU parent and worker tracks;
- `tma_bandwidth_static_sm120a.nanotrace`: statically scheduled TMA transfers;
- `tma_bandwidth_atomic_sm120a.nanotrace`: dynamically scheduled TMA transfers.

The Mina capture comes from an external application using the Nanotrace C++
writer. It includes CPU scopes, hardware kernel events, and region annotations,
with no intra-kernel events. The other samples come from
`nanotrace-cuda/examples`; their CUDA producers target `sm_120a`.
