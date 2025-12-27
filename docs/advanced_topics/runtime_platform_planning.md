# Runtime performance, pipelines, and platform planning

This guide collects practical guidance for running WeatherFlow models inside
interactive experiences such as training dashboards, flight/ferry mission
replays, or real-time situational tools. It establishes performance budgets,
profiling hooks, data pipelines, and platform targets so you can ship a smooth
experience across PC, VR, and console surfaces.

## Frame-time budgets and profiling hooks

### Target budgets

| Mode | Target fps | Frame budget | Main-thread budget | GPU budget |
| --- | --- | --- | --- | --- |
| Desktop visualization | 60 fps | 16.6 ms | 6–8 ms | 8–10 ms |
| High-end desktop | 90 fps | 11.1 ms | 4–6 ms | 5–7 ms |
| Esports/fast camera fly-throughs | 120 fps | 8.3 ms | 3–4 ms | 4–5 ms |
| VR (comfort) | 90 fps | 11.1 ms | 3–4 ms | 6–7 ms |
| VR (premium) | 120 fps | 8.3 ms | 3–4 ms | 4–5 ms |
| Console (quality) | 60 fps | 16.6 ms | 6–8 ms | 8–10 ms |
| Console (performance) | 40–60 fps | 16.6–25 ms | 8–10 ms | 10–12 ms |

Use the budgets to partition work between the simulation step (model forward +
solver integration), rendering (slicing volume fields into cross-sections or
tiles), and streaming. Maintain a small 1–2 ms headroom per frame for IO and
buffer swaps.

### Profiling hooks for CPU/GPU

- **PyTorch profiler** – Wrap inference and solver calls with `torch.profiler`
  to capture CPU, CUDA, and memory events. Use NVTX to mark stages such as
  `"data_prefetch"`, `"flow_inference"`, `"ode_step"`, and `"postprocess"` so
  Nsight Systems/Graphics produces readable timelines.
- **CUDA events** – Place paired `torch.cuda.Event` markers around the
  upscaling/visualization pass to monitor GPU-bound work independently from
  model execution.
- **Custom timers** – Instrument the FastAPI server (`weatherflow/server/app.py`)
  endpoints with lightweight timers that emit Prometheus-style metrics
  (`inference_ms`, `frame_ms`, `cpu_mem_mb`, `gpu_mem_mb`, `dropped_frames`).
- **Frame budget watchdog** – Add a watchdog thread that reports when the
  render loop exceeds the target budget for more than N consecutive frames and
  triggers a feature backoff (see below).
- **Saved traces** – Persist profiler traces alongside run metadata so replay
  sessions can be compared to live captures when diagnosing regressions.

## Upscaling and reconstruction (FSR/DLSS/TSR)

1. Render weather fields (heatmaps, streamlines, ray-marched volumes) at an
   internal resolution that fits the GPU budget, then feed the frame into an
   upscaler module.
2. Support pluggable upscalers: FidelityFX Super Resolution (FSR) for an open
   option, DLSS on NVIDIA for neural super-resolution, and Temporal Super
   Resolution (TSR) for vendors without dedicated hardware. Keep a bilinear
   fallback for headless or CI runs.
3. Share per-frame motion vectors and depth from the weather field rasterizer
   to the TSR/DLSS pass to stabilise thin features like jet streaks and frontal
   boundaries.
4. Expose quality presets: `performance` (50–67% internal res), `balanced`
   (67–75%), and `quality` (75–83%). Bind them to the platform backoff matrix
   so the watchdog can step down automatically.

## Asset and content pipeline with hot reload

Goal: deterministic, reproducible content with rapid iteration for procedural
terrain/sea state, weather presets, and mission scripting.

- **Source data staging** – Store raw ERA5/CMIP or custom ocean/terrain rasters
  in a `data/raw/` bucket. Normalise to common grids (e.g., 0.25°) and units via
  preprocessing notebooks or `weatherflow.data` pipelines.
- **Procedural synthesis** – Generate perturbations (e.g., mesoscale eddies,
  wave spectra, small-planet terrain) as parameterised scripts. Emit `xarray`
  datasets or NetCDF tiles tagged with the parameter seed used for synthesis.
- **Preset manifests** – Describe weather presets and mission scripting in
  declarative YAML/JSON (scenario name, start time, forcing, camera path,
  objectives). Keep manifests in `content/presets/` and version them alongside
  model checkpoints.
- **Build steps** – Use a content build script that: (1) validates manifests,
  (2) bakes procedural outputs into streamable chunks (Zarr/NetCDF), (3)
  computes checksums for cache-busting, and (4) emits a `build_index.json`
  consumed by the renderer/server.
- **Hot reload** – Watch `content/` for manifest or asset changes using
  `watchdog`; on change, invalidate the in-memory cache, reload manifests, and
  notify connected clients (WebSocket event) to swap in updated presets without
  restarting the server.
- **CI guardrails** – Add schema checks for manifests and deterministic seed
  assertions so procedurally generated assets remain reproducible in tests.

## State management, determinism, and capture

- **Save/load** – Persist replayable state objects containing: model version
  hash, solver parameters, RNG seeds, current simulation time, camera/view
  state, active preset/mission, and any user annotations. Store as JSON +
  binary tensors (e.g., PyTorch `state_dict`) so they can be restored without
  re-running preprocessing.
- **Deterministic seeds** – Propagate a root seed through PyTorch, NumPy, and
  any procedural generators. Include the seed in the preset manifest and the
  saved state metadata for reproducibility across platforms.
- **Replay sharing** – Compress replay state and metadata into shareable files
  (zip/JSON) and surface a checksum to detect tampering. Allow headless
  playback that re-runs the solver with the recorded seed and camera path to
  regenerate identical frames.
- **Photo/video capture** – Provide utilities to export: (1) cross-section PNGs
  or GeoTIFFs for horizontal/vertical slices, (2) fly-through MP4/WebM rendered
  from recorded camera splines, and (3) short GIFs for quick sharing. Tie
  captures to the same metadata as replays (seed, timestep, preset).

## Platform targets and feature backoffs

| Platform | Target resolution | Target fps | Default upscaler | Backoffs |
| --- | --- | --- | --- | --- |
| PC (mid) | 1440p | 60 | FSR Balanced | Drop to 1080p internal, lower particle counts, decimate solver substeps by 2x |
| PC (high) | 4K | 90 | DLSS Quality | Switch to Balanced/Performance, reduce ray samples for volume render, cap wave detail |
| VR (PCVR) | 1832×1920 per-eye | 90 | DLSS/FSR Quality | Fixed foveated rendering, halve post-process filters, reduce solver substeps in peripheral tiles |
| Console (quality) | 4K | 60 | TSR Quality | Switch to TSR Balanced, trim cross-section overlays, disable expensive vorticity overlays |
| Console (performance) | 1440p | 40–60 | TSR Balanced | Step to 1080p internal, disable secondary particles, decimate simulation output frames |

Link the backoff matrix to the watchdog so the system can automatically step
down when frame times exceed the budget and step back up when headroom is
available. Keep per-platform defaults in configuration so QA can lock targets
for certification or benchmarking.
