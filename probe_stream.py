"""
Probe the StreamDiffusion WebSocket endpoint directly.
Sends real frames and measures: latency, mode, errors.
"""
import asyncio
import base64
import io
import json
import sys
import time

import numpy as np
from PIL import Image

MODEL = "D:\\comfyui\\comfy\\models\\checkpoints\\cyberrealisticXL_v80-inpainting.safetensors"
WS_URL = "ws://localhost:8000/ws/inpaint"
WIDTH, HEIGHT = 512, 512
N_FRAMES = 8


def make_data_url(img: Image.Image, fmt="JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def make_test_image() -> Image.Image:
    arr = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    arr[:, :, 0] = 80
    arr[:, :, 1] = 120
    arr[:, :, 2] = 160
    for y in range(HEIGHT):
        arr[y, :, 0] = int(80 + 100 * y / HEIGHT)
    return Image.fromarray(arr)


def make_empty_mask() -> Image.Image:
    return Image.new("L", (WIDTH, HEIGHT), 0)


def make_test_mask() -> Image.Image:
    arr = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    arr[HEIGHT//4:3*HEIGHT//4, WIDTH//4:3*WIDTH//4] = 255
    return Image.fromarray(arr)


def build_frame(image: Image.Image, mask: Image.Image, prompt: str) -> dict:
    return {
        "image": make_data_url(image),
        "mask": make_data_url(mask, "PNG"),
        "prompt": prompt,
        "negative_prompt": "blurry, low quality",
        "width": WIDTH,
        "height": HEIGHT,
        "steps": 2,
        "cfg": 1.2,
        "strength": 0.75,
        "seed": 42,
        "stream_diffusion": True,
        "stream_direct": True,
        "stream_timestep_indices": [32, 45],
        "stream_frame_buffer_size": 1,
        "stream_cfg_type": "self",
        "stream_similarity_threshold": 0.98,
        "stream_max_skip_frames": 0,
        "stochastic_similarity_filter": False,
        "realtime_accel": True,
    }


async def run():
    try:
        import websockets
    except ImportError:
        print("Installing websockets...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets", "--quiet"])
        import websockets

    image = make_test_image()
    empty_mask = make_empty_mask()
    region_mask = make_test_mask()
    prompt = "a beautiful landscape, cinematic lighting, high quality"

    print(f"Connecting to {WS_URL} ...")
    try:
        async with websockets.connect(WS_URL, max_size=50 * 1024 * 1024) as ws:
            print("Connected.\n")

            # --- Test 1: empty mask (no painted region) ---
            print("=== Test 1: 4 frames, empty mask (no painted region) ===")
            frame_data = build_frame(image, empty_mask, prompt)
            latencies = []
            modes = []
            for i in range(4):
                t0 = time.perf_counter()
                await ws.send(json.dumps(frame_data))
                raw = await ws.recv()
                dt = (time.perf_counter() - t0) * 1000
                result = json.loads(raw)
                latencies.append(dt)
                mode = result.get("mode", "?")
                modes.append(mode)
                fps = result.get("fps", 0)
                err = result.get("error", "")
                has_image = bool(result.get("image", ""))
                print(f"  frame {i}: {dt:.0f}ms  fps={fps:.1f}  mode={mode[:80]}  image={'yes' if has_image else 'NO'}  {'ERROR: '+err[:60] if err else ''}")

            print(f"\n  avg latency: {sum(latencies)/len(latencies):.0f}ms  effective fps: {1000/(sum(latencies)/len(latencies)):.1f}")

            # --- Test 2: painted region mask ---
            print("\n=== Test 2: 4 frames, painted region mask (center 50%) ===")
            frame_data2 = build_frame(image, region_mask, prompt)
            latencies2 = []
            for i in range(4):
                t0 = time.perf_counter()
                await ws.send(json.dumps(frame_data2))
                raw = await ws.recv()
                dt = (time.perf_counter() - t0) * 1000
                result = json.loads(raw)
                latencies2.append(dt)
                mode = result.get("mode", "?")
                fps = result.get("fps", 0)
                err = result.get("error", "")
                has_image = bool(result.get("image", ""))
                print(f"  frame {i}: {dt:.0f}ms  fps={fps:.1f}  mode={mode[:80]}  image={'yes' if has_image else 'NO'}  {'ERROR: '+err[:60] if err else ''}")

            print(f"\n  avg latency: {sum(latencies2)/len(latencies2):.0f}ms  effective fps: {1000/(sum(latencies2)/len(latencies2)):.1f}")

    except Exception as exc:
        print(f"Connection failed: {exc}")
        return

    print("\n--- Summary ---")
    print(f"Empty mask  avg: {sum(latencies)/len(latencies):.0f}ms")
    print(f"Region mask avg: {sum(latencies2)/len(latencies2):.0f}ms")
    print(f"Mode (last): {modes[-1]}")
    if "stream direct disabled" in modes[-1] or "stream native disabled" in modes[-1]:
        print("!! STREAM DIRECT IS DISABLED — check RTD_STREAM_DIRECT or frame.stream_direct")
    if "unavailable" in modes[-1]:
        print("!! MODEL NOT SUPPORTED for direct streaming")


if __name__ == "__main__":
    asyncio.run(run())
