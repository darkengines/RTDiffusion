# RTDiffusion

RTDiffusion is a small browser app for real-time prompt-driven inpainting. The frontend is Vite, Lit, TypeScript, and Fabric.js. The backend is FastAPI with a WebSocket endpoint that streams the current canvas, mask, and prompt to a diffusion engine.

## Research Notes

- `arXiv:2403.16627`, SDXS, reports one-step latent diffusion with about 100 FPS at 512px and about 30 FPS at 1024px on a single GPU by distilling the U-Net and decoder.
- StreamDiffusion adds a practical streaming pipeline with batching, IO queues, residual CFG, stochastic similarity filtering, KV-cache precomputation, and optional TensorRT acceleration.
- SDXL Turbo is easy to run through Diffusers with `guidance_scale=0.0` and 1 to 4 inference steps. It is a good baseline for interactive prompting, while true inpainting needs an inpainting-capable checkpoint or a StreamDiffusion-style masked pipeline.
- TAESD / TAESDXL is useful for faster latent preview and lower decode cost.
- Fabric.js is the browser editor choice here because it is TypeScript-friendly, framework-neutral, and gives ergonomic canvas editing primitives without tying the app to React.

## Run

Install the frontend:

```powershell
Set-Location h:\RTDiffusion\frontend
npm install
npm run dev
```

Install the backend in another terminal:

```powershell
Set-Location h:\RTDiffusion
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .\backend
uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Open `http://localhost:5173`, upload an image, paint a mask, edit the prompt, and start the stream.

## GPU Mode

The backend starts in mock mode if PyTorch or Diffusers is not installed. For a real SDXL Turbo path, install PyTorch for your CUDA version, then install the optional backend extras:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .\backend[gpu]
$env:RTD_DEVICE = "cuda"
$env:RTD_GUIDANCE_SCALE = "1.5"
uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Set `RTD_MODEL_PATH` to a local checkpoint or `RTD_MODEL_ID` to a Diffusers model ID. To let the asset picker scan local folders, set `RTD_MODEL_DIRS` and optionally `RTD_LORA_DIRS` to path-list values for your platform.

For 35+ FPS, use 384 or 512 resolution, `steps=1`, `strength` around `0.25` to `0.45`, `guidance_scale=0`, a high-end CUDA GPU, and preferably a StreamDiffusion or TensorRT-backed masked engine. The current backend includes the WebSocket contract and a Diffusers inpainting baseline; the engine is isolated in `backend/app/engine.py` so a StreamDiffusion/TensorRT implementation can replace it without changing the UI.

For standard Stable Diffusion inpaint checkpoints, Diffusers needs `num_inference_steps * strength >= 1`; the backend automatically raises the effective step count when the UI requests a lower value. Turbo models still run with guidance scale `0`, while local non-turbo checkpoints use `RTD_GUIDANCE_SCALE`.

The frontend keeps one diffusion frame in flight at a time. Prompt, mask, and source-image edits are picked up on the next completed frame instead of flooding the backend queue.