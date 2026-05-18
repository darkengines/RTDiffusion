"""
Probe v2: verify outputs actually change, save frames, check image content.
"""
import asyncio, base64, io, json, time, sys
import numpy as np
from PIL import Image

WS_URL = "ws://localhost:8000/ws/inpaint"
W, H = 512, 512


def img_from_data_url(data_url: str) -> Image.Image:
    _, b64 = data_url.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def mean_abs_diff(a: Image.Image, b: Image.Image) -> float:
    aa = np.array(a, dtype=np.float32)
    bb = np.array(b, dtype=np.float32)
    return float(np.abs(aa - bb).mean())


def make_data_url(img: Image.Image, fmt="JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def make_gradient() -> Image.Image:
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        t = y / H
        arr[y, :, 0] = int(50 + 150 * t)
        arr[y, :, 1] = int(100 - 50 * t)
        arr[y, :, 2] = int(200 - 100 * t)
    return Image.fromarray(arr)


def build_frame(image, mask, prompt, motion_phase=0.0):
    # Simulate the sway motion transform by slightly shifting the image
    import math
    dx = int(math.sin(motion_phase) * W * 0.03)
    dy = int(math.sin(motion_phase * 1.7) * H * 0.02)
    shifted = Image.new("RGB", (W, H), (80, 80, 80))
    shifted.paste(image, (dx, dy))
    return {
        "image": make_data_url(shifted),
        "mask": make_data_url(mask, "PNG"),
        "prompt": prompt,
        "negative_prompt": "blurry, low quality",
        "width": W, "height": H,
        "steps": 2, "cfg": 1.4, "strength": 0.75, "seed": 42,
        "stream_diffusion": True, "stream_direct": True,
        "stream_timestep_indices": [32, 45],
        "stream_frame_buffer_size": 1,
        "stream_cfg_type": "self",
        "stream_similarity_threshold": 0.98,
        "stream_max_skip_frames": 0,
        "stochastic_similarity_filter": False,
        "realtime_accel": True,
    }


async def run():
    import websockets
    from pathlib import Path

    out = Path("outputs/probe2")
    out.mkdir(parents=True, exist_ok=True)

    image = make_gradient()
    empty_mask = Image.new("L", (W, H), 0)

    prompt = "a beautiful fantasy forest, cinematic lighting, painterly style"

    print(f"Connecting to {WS_URL} ...")
    async with websockets.connect(WS_URL, max_size=100 * 1024 * 1024) as ws:
        print("Connected. Sending 10 frames with simulated sway motion...\n")

        outputs = []
        latencies = []
        import math
        for i in range(10):
            phase = i * 0.4  # simulate time passing
            frame = build_frame(image, empty_mask, prompt, motion_phase=phase)
            t0 = time.perf_counter()
            await ws.send(json.dumps(frame))
            raw = await ws.recv()
            dt = (time.perf_counter() - t0) * 1000
            result = json.loads(raw)
            latencies.append(dt)

            mode = result.get("mode", "?")
            err = result.get("error", "")
            img_data = result.get("image", "")

            if err:
                print(f"  frame {i:02d}: ERROR {err[:80]}")
                continue

            if not img_data:
                print(f"  frame {i:02d}: NO IMAGE")
                continue

            out_img = img_from_data_url(img_data)
            outputs.append(out_img)
            out_img.save(out / f"frame_{i:02d}.png")

            diff = mean_abs_diff(outputs[-1], outputs[-2]) if len(outputs) >= 2 else 0.0
            print(f"  frame {i:02d}: {dt:.0f}ms  fps={result.get('fps',0):.1f}  diff_from_prev={diff:.2f}  mode={mode[20:70]}")

        print(f"\nFrames saved to {out.resolve()}")
        if len(outputs) >= 2:
            diffs = [mean_abs_diff(outputs[i], outputs[i-1]) for i in range(1, len(outputs))]
            print(f"Avg frame-to-frame pixel diff: {sum(diffs)/len(diffs):.2f} / 255")
            print(f"Max diff: {max(diffs):.2f}")
            if max(diffs) < 1.0:
                print("!! OUTPUT IS STATIC — frames are identical. Check model / mask / session state.")
            elif max(diffs) < 5.0:
                print("!! Very low variation — output barely changes.")
            else:
                print("OK: Output varies between frames — StreamDiffusion is producing motion.")
        print(f"\nAvg latency: {sum(latencies)/len(latencies):.0f}ms  FPS: {1000/(sum(latencies)/len(latencies)):.1f}")


asyncio.run(run())
