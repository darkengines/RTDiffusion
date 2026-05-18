# RTDiffusion pipeline architecture

RTDiffusion rendering modes should integrate model components into the RTDiffusion runtime instead of treating complete third-party applications as the product surface.

Rules for realtime and video modes:

- Prefer direct component loading in-process: tokenizer, text encoder, transformer or U-Net, VAE, scheduler, LoRA adapters, and cached prompt or latent state.
- Use external CLI runners only as temporary adapters for research code that has not yet been componentized.
- Keep model instances warm across frames or chunks whenever VRAM allows it.
- Use Hugging Face model IDs and local Hugging Face cache directories as normal model sources.
- Keep ComfyUI paths as model file locations only; do not depend on a ComfyUI runtime.
- Favor pipelines that support LoRA/pretrained component reuse and can expose their scheduler/latent/text-conditioning internals.

Current direction:

- SDXL and Z-Image remain realtime image diffusion renderers.
- StreamDiffusion direct Stream Batch is experimental and opt-in through `RTD_STREAM_DIRECT=1`; normal SDXL/Z-Image checkpoints should not use it by default.
- WAN/FastVideo is treated as offline image-to-video clip generation and routes through RTDiffusion's native WAN 2.2 component pipeline when no external override is configured.
- Causal-Forcing remains a temporary upstream CLI adapter until its frame-wise model components are integrated as a persistent worker or native component runtime.

WAN/FastVideo offline clip ladder:

- Decoded-frame streaming: native WAN/FastVideo writes frames into the motion cache and publishes frame URLs through motion task websocket progress as soon as each decoded frame lands. This is the current product path.
- Replayable MP4: each finished clip is encoded to `playback.mp4` and shown through a browser `<video>` element with controls and loop playback.
- Warm offline worker: keep the WAN/FastVideo transformer pair, VAE, scheduler, and prompt embeddings resident between independent clip requests to avoid cold-start reload.
- Denoise-step latent preview: optional debugging hook only; disabled by default because it asks Diffusers to pass latent tensors through callbacks and can cost memory/bandwidth.
