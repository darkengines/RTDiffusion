# RTDiffusion

RTDiffusion is an experimental local realtime diffusion and image-to-video workbench. The frontend is a Vite/Lit/Fabric.js canvas editor. The backend is a FastAPI service that exposes realtime inpaint streams, layer generation jobs, model discovery, GPU discovery, and motion/video jobs.

This repository is currently a research prototype, not a finished realtime product. The UI exposes several renderer tabs, but the important status is:

- SDXL/Z-Image style image streaming works as a Diffusers-based realtime baseline.
- Docker CUDA now exposes Torch CUDA, xFormers, and TensorRT runtime detection.
- StreamDiffusion is not working yet as the upstream paper/GitHub implementation promises. The app has an internal StreamDiffusion-style path, but it is not a proven high-FPS Stream Batch worker and should be treated as incomplete.
- Causal-Forcing is not working at the moment. The frontend/backend task shell exists, but no configured runtime currently generates usable clips.
- TensorRT is installed/detected in Docker and the Windows venv, but no U-Net/VAE/transformer TensorRT engine export/cache/inference path is implemented yet.

## Current Status

### Working Enough To Use

- Canvas editor with image import, painting, masking, layers, and project state.
- Binary WebSocket image stream for realtime SDXL/Z-Image-like inpaint frames.
- Model, LoRA, and video-model asset discovery from local ComfyUI folders.
- Multi-GPU selection in the frontend, backed by `/system/gpus`.
- Docker CUDA backend with CUDA visible on both GPUs, xFormers CUDA kernels verified, and TensorRT import verified.
- StreamDiffusion preset controls for Diffusers baseline, LCM LoRA SDXL, SDXL Lightning 4-step LoRA, and SDXL Turbo.
- Live drawing snapshots now include the active Fabric upper canvas stroke instead of waiting for mouseup.
- StreamDiffusion frames now draw immediately without the previous frame-blending crossfade.
- Layer variation task API with progress WebSocket.
- Motion task API, frame cache, MP4 playback cache, and progress WebSocket.
- Native WAN/FastVideo adapter shell and GGUF external-command routing shell.

### Not Working Yet

- True upstream StreamDiffusion behavior: persistent IO queues, Stream Batch, real latent R-CFG, high-FPS direct component execution, and TensorRT-backed fixed-shape engines are not implemented end to end.
- Causal-Forcing: the tab and task API exist, but the runtime is unconfigured/nonfunctional right now.
- Krea Realtime Video: exposed as a research target, but not a usable realtime frontend path here.
- TensorRT acceleration of actual diffusion inference: runtime packages are installed, but RTDiffusion does not yet export/build/load TensorRT engines for the model components.
- Dasiwa/WAN GGUF native inference: Diffusers and TensorRT do not load these directly. They require an external Wan-Video/Dasiwa runner command.

## Run

### Frontend

```powershell
Set-Location h:\RTDiffusion\frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

### Windows Backend

```powershell
Set-Location h:\RTDiffusion
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .\backend[gpu]
uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Optional TensorRT runtime detection on Windows:

```powershell
python -m pip install -e .\backend[tensorrt] --extra-index-url https://pypi.nvidia.com
```

Windows xFormers is currently not viable with the tested `Python 3.12 + torch 2.5.1+cu121` environment. The available wheel installs but its CUDA/C++ extensions are built for a different Torch/Python ABI.

### Docker CUDA Backend

Docker is the preferred accelerator environment for this project right now.

```powershell
Set-Location h:\RTDiffusion
docker compose -f compose.backend.cuda.yml build backend
docker compose -f compose.backend.cuda.yml up -d --no-build backend
```

The compose backend maps:

- `D:/comfyui/comfy/models` to `/models`
- `./outputs` to `/app/outputs`
- `./outputs/private-checkpoints` to `/private-checkpoints` by default
- `./outputs/private-loras` to `/private-loras` by default

Override private mounts when needed:

```powershell
$env:RTD_PRIVATE_CHECKPOINTS_HOST = "t:/models/checkpoints"
$env:RTD_PRIVATE_LORAS_HOST = "t:/models/loras"
```

Docker must use `.env.docker` paths such as `/models/diffusers/sdxl-turbo`, not Windows `D:\...` paths. In Docker, the backend loader reads `.env.docker` and avoids `.env.local` so host Windows paths do not leak into the Linux container.

Useful checks:

```powershell
docker compose -f compose.backend.cuda.yml ps
docker logs --tail 80 rtdiffusion-backend-cuda
Invoke-RestMethod http://127.0.0.1:8000/system/gpus
Invoke-RestMethod http://127.0.0.1:8000/renderers/capabilities
```

## Frontend Features And Expected Behavior

### Scene Canvas

Purpose: provide the source image, masks, paint edits, and layer composition used by realtime renderers and video tasks.

Expected behavior:

- Import or load a source image into the stage.
- Paint regular strokes or mask strokes directly on the canvas.
- While streaming, drawing should update the next submitted frame live during mouse movement, not only after releasing the mouse.
- Canvas size should stay stable during generation; renderer output should not resize the editor.

### SDXL Tab

Purpose: baseline realtime inpaint/image-to-image renderer using Diffusers.

Expected behavior:

- Start a binary WebSocket stream to `/ws/inpaint`.
- Send the current canvas, mask, prompt, selected model, LoRAs, sampler settings, and selected GPU.
- Display returned frames in the output panel.
- Show browser display FPS separately from backend generation FPS/latency.
- Keep only one frame in flight so the UI does not flood the backend.

Current caveat: this is still a stock Diffusers-style path. It can be useful, but it is not paper-level realtime diffusion.

### Z-Image Tab

Purpose: route compatible selected models through the same realtime image stream contract.

Expected behavior:

- Behave like the SDXL realtime tab from the user's perspective.
- Use the model picker and GPU selector.
- Report capability truthfully via `/renderers/capabilities`.

Current caveat: compatibility depends on the selected model and the backend Diffusers support.

### StreamDiffusion Tab

Purpose: intended high-FPS input-driven realtime renderer with StreamDiffusion-style controls.

Expected behavior target:

- Keep a persistent worker/session in memory.
- Precompute prompt embeddings, timesteps, noise/scaling coefficients, and reusable state.
- Use Stream Batch or equivalent direct component execution instead of calling a full Diffusers pipeline per UI frame.
- Use LCM, Lightning, Turbo, or Hyper-SD-compatible models.
- Support image-driven transitions and perceptible smooth changes while the user paints.
- Use xFormers/SDPA and eventually TensorRT engines when available.

Current reality:

- The tab has presets for Diffusers, LCM LoRA SDXL, SDXL Lightning 4-step LoRA, and SDXL Turbo.
- The backend reports `internal-streamdiffusion` when `RTD_STREAM_NATIVE=1`.
- There is an internal persistent denoising-session attempt with warmup and optional Tiny VAE.
- It is not confirmed to achieve true StreamDiffusion behavior or high FPS.
- Treat this tab as incomplete until the direct worker is rewritten and measured.

### Causal-Forcing Tab

Purpose: generate autoregressive image-to-video clips from the current canvas.

Expected behavior target:

- Create a task with `/motion/tasks`.
- Generate an arrival/keyframe image if needed.
- Run a configured Causal-Forcing runtime.
- Stream progress over `/ws/motion/{task_id}`.
- Show preview frames and final MP4 in the frontend.

Current reality:

- Not working at the moment.
- The shell exists, but the required external repo/checkpoint/runtime is not configured into a usable path.
- The frontend should show it as unavailable or error clearly, not pretend it is native realtime.

### FastVideo / WAN Tab

Purpose: generate motion/video clips from WAN/FastVideo-compatible local assets.

Expected behavior:

- Let the user select a WAN/FastVideo model or component path.
- Submit a motion task with current canvas, optional arrival prompt/image, prompt, duration, FPS, and GPU.
- Stream task progress and preview frames.
- Cache PNG frames and `playback.mp4` under `outputs/motion-cache`.

Current caveats:

- Native WAN/FastVideo is heavy and may need GPU memory handoff from realtime image rendering.
- GGUF/NVFP4 files need an external Wan-Video/Dasiwa runner command; they are not native Diffusers/TensorRT inputs here.

### Layer Generator

Purpose: generate standalone layer variations with transparent-background support.

Expected behavior:

- Submit `/layer/tasks` with prompt/model/sampling settings.
- Pause/unload realtime sampling if it uses the same GPU.
- Stream progress through `/ws/tasks/{task_id}`.
- Return selectable generated layer variations.
- Resume realtime sampling after the priority layer task when possible.

### Asset Picker

Purpose: expose local models and LoRAs without hardcoding every file.

Expected behavior:

- Scan configured model, LoRA, and video-model folders.
- Hide unsupported native assets from incompatible pickers.
- Treat Diffusers model folders as a single selectable asset when they contain `model_index.json`.
- Avoid listing internal component files inside Diffusers model directories as ordinary checkpoints.

### Renderer Status Panel

Purpose: stop fake labels.

Expected behavior:

- Call `/renderers/capabilities`.
- Show whether each tab is native, external, fallback, configured, streamable, realtime, and which accelerators are detected.
- Make unavailable renderers obvious before the user spends minutes waiting.

## Backend Summary

### Main Modules

- `backend/app/main.py`: FastAPI app, HTTP endpoints, WebSockets, task orchestration, per-device engine cache, GPU locks, priority layer/video scheduling.
- `backend/app/engine.py`: Diffusers image engine, layer generation, realtime image stream handling, internal StreamDiffusion-style experiment, LoRA loading, scheduler/model normalization.
- `backend/app/renderers.py`: capability reporting for SDXL, Z-Image, StreamDiffusion, Causal-Forcing, and FastVideo.
- `backend/app/assets.py`: model/LoRA/video asset discovery and filtering.
- `backend/app/motion.py`: motion task orchestration, frame cache, MP4 encoding, external command adapters, Causal/Krea/FastVideo routing.
- `backend/app/video_pipeline.py`: native WAN/FastVideo component pipeline attempts.
- `backend/app/config.py`: local env loading. Uses `.env.local` on Windows/local runs and `.env.docker` inside Docker unless `RTD_ENV_FILE` is set.
- `backend/app/image_io.py`: data URL decode/encode and binary WebSocket helpers.
- `backend/app/schemas.py`: Pydantic request/response contracts.

### HTTP Endpoints

- `GET /health`: backend status and current engine mode.
- `GET /system/gpus`: CPU/CUDA device list for frontend selectors.
- `GET /assets`: discovered models, video models, and LoRAs.
- `GET /video/capabilities`: configured/unconfigured status for motion/video models.
- `GET /renderers/capabilities`: renderer truth table plus installed runtime versions.
- `GET /sources`: recursively list image files under a selected source root.
- `POST /sources/pick-root`: desktop folder picker for local runs.
- `GET /sources/image`: serve a selected source image.
- `POST /layer/generate`: synchronous layer variation generation.
- `POST /layer/tasks`: async layer generation task.
- `POST /motion/tasks`: async motion/video generation task.
- `GET /motion/tasks/{task_id}`: poll motion task progress.
- `GET /motion/clips/{clip_id}/video`: serve cached MP4 playback.
- `GET /motion/clips/{clip_id}/frames/{frame_index}`: serve cached PNG motion frame.
- `GET /motion/clips/{clip_id}/previews/{preview_name}`: serve cached preview image.

### WebSockets

- `WS /ws/inpaint`: realtime image stream. The frontend sends `InpaintFrame` JSON. In binary mode, the backend replies with JSON metadata followed by raw image bytes.
- `WS /ws/tasks/{task_id}`: layer task progress stream.
- `WS /ws/motion/{task_id}`: motion/video task progress stream.

### Renderer Implementations

- SDXL: native Diffusers image renderer over binary WebSocket. Works as the baseline.
- Z-Image: Diffusers-compatible image renderer route. Compatibility depends on the selected model.
- StreamDiffusion: internal experimental path, not a full upstream implementation yet.
- Causal-Forcing: external-command/task shell only; not working currently.
- FastVideo/WAN: external command support plus native WAN/FastVideo component attempts; heavy and still experimental.
- GGUF/Dasiwa: external command routing only. Not loaded by Diffusers or TensorRT directly.

### What Has Been Done So Far

- Added binary WebSocket image transport to reduce base64 JSON frame overhead.
- Split frontend display FPS from backend generation FPS/latency.
- Added renderer capability API and frontend status UI.
- Added multi-GPU discovery and per-device engine cache/locks.
- Added StreamDiffusion presets and local model/LoRA path handling.
- Added direct file and parent-folder LoRA loading fallback.
- Added Diffusers model-folder normalization for assets with `model_index.json`.
- Added an internal StreamDiffusion-style session experiment with cached state, warmup, optional Tiny VAE, and direct denoise attempts.
- Added live Fabric upper-canvas compositing so in-progress strokes can be included in realtime frames.
- Fixed StreamDiffusion frame delivery: removed hardcoded crossfade, frames now draw immediately.
- Fixed StreamDiffusion stochastic similarity filter: default max skip frames lowered to 0 so no frames are skipped during active painting.
- Fixed StreamDiffusion collapse detection: removed false-positive condition that reset sessions for valid gray/uniform outputs.
- Fixed StreamDiffusion collapse recovery: session warmup and embeddings are preserved on collapse; only the latent buffer is reset.
- Fixed StreamDiffusion fallback step count: no longer forces steps to 4 when the direct path is unavailable.
- Added Docker CUDA backend with xFormers and TensorRT enabled.
- Verified Docker xFormers by running a real CUDA memory-efficient attention kernel.
- Verified Docker TensorRT import and `/renderers/capabilities` accelerator detection.
- Added TensorRT optional dependency group for the Windows venv.
- Added `.env.docker` separation to avoid Windows paths inside the Linux container.
- Added motion cache, frame serving, MP4 serving, and motion task progress plumbing.

## What Failed And Why

### StreamDiffusion

Failure: the app does not yet behave like upstream StreamDiffusion. It remains too slow and does not provide convincing smooth motion/transitions.

Reasons:

- The upstream PyPI package pins old Diffusers and breaks the current environment.
- The existing internal path is a compatibility experiment, not a full IO queue plus Stream Batch worker.
- Full Diffusers pipeline calls are too expensive per frame.
- TensorRT runtime availability alone does not accelerate anything until the model components are exported to engines and invoked through those engines.
- Standard SDXL inpaint checkpoints are not the right default for few-step StreamDiffusion behavior; Turbo/LCM/Lightning-style models are required.

### Causal-Forcing

Failure: the tab is not currently usable.

Reasons:

- It needs an external repo, matching checkpoint, matching Wan2.1 base assets, and a compatible Python environment.
- The current adapter is task/subprocess-oriented, not a persistent video worker.
- Paths and geometry must be exact; mismatches lead to long failures after model load.
- It is not integrated as a native in-process renderer.

## To Do

### Fixes

- Make StreamDiffusion status in the frontend stricter: label it incomplete unless a measured native worker is active.
- Prevent users from starting Causal-Forcing unless a real config exists and a smoke test passes.
- Add a startup health check that validates selected model paths inside Docker before the first generation.
- Add clearer frontend errors for missing external runners, unsupported GGUF/NVFP4 assets, and wrong path roots.
- Add per-renderer cold-start/loading/progress messages so the app does not look frozen during model load.
- Add automatic fallback to local private mount folders when optional drives are unavailable.

### Investigations

- Build a minimal direct-component SDXL Turbo path: VAE encode, U-Net/transformer denoise, VAE/Tiny VAE decode, no full Diffusers pipeline call.
- Implement a real persistent worker with latest-frame input queue and output queue.
- Measure true backend throughput separately from browser display FPS.
- Compare SDPA vs xFormers attention inside Docker for the actual model path.
- Export fixed-shape ONNX/TensorRT engines for SDXL Turbo U-Net and VAE/Tiny VAE.
- Determine whether TensorRT helps the internal StreamDiffusion path enough to justify engine build complexity.
- Re-evaluate Causal-Forcing with a pinned external environment and a one-frame smoke test before wiring UI flows.
- Investigate whether a ComfyUI/Wan-Video runner is the most practical GGUF/Dasiwa integration.

### Features

- Real Stream Batch implementation with staggered latent ring buffer.
- Real latent-space R-CFG, not output-space approximation.
- TensorRT engine cache keyed by model, dtype, resolution, batch, scheduler/timestep plan, and LoRA fusion state.
- Prompt interpolation and image-to-image transition mode between two prompts/keyframes.
- Persistent video workers for Causal/Krea/FastVideo to avoid per-task model reloads.
- Frontend benchmark panel with cold-start time, first-frame time, average generation FPS, display FPS, VRAM, and skip rate.
- Renderer-specific presets that set sane model/LoRA/sampler/step/cfg/strength combinations and explain unavailable requirements through status, not marketing copy.

## Points Of Attention

- Do not call something StreamDiffusion just because frames are streamed. The key requirements are persistent worker state, direct component execution, Stream Batch or equivalent batching, cached embeddings/state, and measured throughput.
- Do not trust `import xformers` on Windows. Verify that xFormers CUDA extensions load and run a memory-efficient attention kernel.
- TensorRT installed is not TensorRT used. Acceleration needs exported engines and an inference path that actually calls them.
- Docker bind mounts from Windows can make model loading slower. This is expected and separate from inference speed.
- `.env.local` contains Windows paths. Docker must load `.env.docker` or explicit container paths.
- Compose `env_file` injects container environment variables, but variable interpolation in `compose.backend.cuda.yml` uses the host shell/.env context. Be explicit when debugging resolved config.
- Diffusers model folders should be selected by folder containing `model_index.json`, not by internal component `.safetensors` files.
- SDXL Turbo, Lightning, LCM, and inpaint checkpoints have different scheduler/CFG/step expectations. One preset cannot safely fit all.
- Standard inpaint checkpoints need `num_inference_steps * strength >= 1`; too-low settings can produce invalid/empty behavior.
- Fabric.js active freehand strokes live on the upper canvas until committed. Realtime exports must composite that upper canvas to include live drawing.
- Browser FPS is not backend generation FPS. Keep them separate.
- Keep one realtime frame in flight unless a real queue/worker is implemented. Flooding the backend increases latency and hides the real bottleneck.
- Video models can force unloading image engines from the same GPU to avoid VRAM conflicts.
- WAN frame counts often have divisibility constraints; tiny frame requests can be rounded down or fail.
- GGUF/NVFP4 assets are not Diffusers/TensorRT-native in this backend. They need a dedicated external runtime.
- Long model-load failures are expensive. Add cheap preflight checks before launching video jobs.

## Quick Reality Check Commands

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/system/gpus
Invoke-RestMethod http://127.0.0.1:8000/renderers/capabilities | ConvertTo-Json -Depth 6
```

Expected Docker accelerator signal after a successful CUDA build:

```text
xformers: true
tensorrt: true
streamdiffusion.runtime: internal-streamdiffusion
```

That signal only means the runtime packages and experimental path are present. It does not mean true StreamDiffusion or TensorRT-accelerated diffusion inference is complete.