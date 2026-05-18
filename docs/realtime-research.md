# Realtime Diffusion Research Notes

These notes track what is needed to move RTDiffusion from a Diffusers realtime baseline toward a StreamDiffusion-style runtime.

## What StreamDiffusion Actually Optimizes

- Stream Batch: accepts a new input frame after each denoising step and batches staggered denoising stages, so throughput is not one full sequential denoise loop per UI frame.
- R-CFG: avoids standard CFG's repeated negative-condition U-Net pass. Self-negative R-CFG costs about the same as no CFG; onetime-negative R-CFG adds only one negative pass.
- SSF: compares the current frame against a reference frame and probabilistically skips VAE/U-Net work when frames are highly similar. This is mainly a GPU activation and power gate, not a raw per-frame compute speedup.
- IO queues: move image resize/normalization and output conversion outside the GPU bottleneck path.
- Pre-computation: cache prompt embeddings, K/V state, noise, and scheduler coefficients when prompt/settings are unchanged.
- Acceleration: Tiny VAE/TAESD, xFormers, LCM/LCM-LoRA or SD-Turbo, and optionally TensorRT engines with fixed shape/batch.

## Why The Current Backend Is Still Far Away

The current backend still calls a stock Diffusers pipeline per websocket frame. Even with CFG disabled and SSF skips, each non-skipped frame still pays Diffusers preprocessing, VAE, scheduler setup, U-Net loop, VAE decode, PIL conversion, and websocket base64 encoding. The paper reports 90+ FPS after replacing that pipeline-level control flow.

## Practical Implementation Ladder

1. Stabilize one-step/few-step models.
   - Prefer SD-Turbo, LCM, or an LCM-LoRA-fused model for realtime.
   - Default accelerated scene settings to `steps=1`, `cfg=0`, and fixed 384/512 canvas sizes.

2. Add a dedicated realtime engine interface.
   - Keep `DiffusionEngine.generate()` as the compatibility fallback.
   - Add a long-lived realtime session object that owns prompt embeddings, scheduler state, and frame queues.

3. Add IO queues.
   - Frontend sends newest frame only.
   - Backend preprocess thread keeps a latest decoded tensor.
   - GPU worker consumes latest tensor and output thread encodes results.

4. Replace Diffusers pipeline calls for realtime.
   - Use pipe components directly: tokenizer/text_encoder, VAE encode/decode, U-Net, scheduler.
   - Cache prompt embeddings until prompt/negative prompt/model/LoRA changes.
   - Cache scheduler timesteps/noise coefficients for current steps/strength/size.

5. Implement real R-CFG in latent space.
   - Self-negative mode: derive virtual negative residual from input latent and current latent.
   - Onetime-negative mode: compute negative condition once at the first denoise step, then reuse the residual.

6. Implement Stream Batch.
   - Maintain a ring buffer of latent frames at staggered denoising steps.
   - Run U-Net over the batch of staggered latents per tick.
   - Emit the oldest completed latent each tick.

7. Optional acceleration.
   - TAESD/Tiny VAE for preview decode.
   - xFormers/SDPA attention always on when available.
   - TensorRT engine generation for fixed size, batch, and model once the direct U-Net path is stable.

## Current RTDiffusion Implementation Status

- Residual CFG checkbox currently disables standard CFG on accelerated frames and blends temporal residual output. This is still an output-space approximation, not the paper's latent-space R-CFG.
- SSF checkbox currently uses a probabilistic skip gate and reports `+ssf-skip` when it avoids sampling. It should move from PIL byte probes to tensor cosine similarity to match StreamDiffusion exactly.
- Reuse previous latent now creates a persistent websocket-session latent queue. Diffusers callback latents are captured and passed into the next frame, with prompt embeddings cached across frames.
- Backend mode reports `+stream(...)` when the persistent realtime compatibility path is active.

This is useful for interactive feel and finally carries latent state across frames, but it is still not the full Stream Batch architecture from the paper. The next meaningful speed jump is replacing stock Diffusers pipeline calls with a direct component path that batches staggered latents through the U-Net/transformer in one call.