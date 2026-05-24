# RTDiffusion Layer System — Formal Specification

> **Status** — Authoritative. Other agents (and humans) MUST consult this file
> before changing anything in the layer / region / mask / aggregation /
> rendering code paths.
>
> If the implementation disagrees with this spec, the spec wins — file an
> issue and update the spec first.
>
> Last revised: 2026-05-21.

---

## 1. Mental model in one sentence

A **Scene** is a z-ordered stack of **Layers**. Each Layer owns one or more
**Regions**. Each Region is a bundle of **paintable channels** — color pixels
and several spatial masks (one per controllable parameter). At render time
the system walks layers bottom-up and produces a renderer-agnostic
**CompositionPlan** that any backend (SDXL, Z-Image, StreamDiffusion, SANA)
can execute as a single-pass, per-layer-pass, or tiled-pass run.

```
Scene
├── Layer N (top)          z-index, visibility, blend_mode, defaults
│   ├── Region n.k
│   │   ├── channel: color        (RGB pixels painted by the user)
│   │   ├── channel: color_mask   (alpha — where this layer's color exists)
│   │   ├── channel: denoise_mask (gradient — per-pixel denoise weight)
│   │   ├── channel: prompt_mask  (gradient — per-pixel prompt influence)
│   │   ├── channel: cfg_mask     (gradient — per-pixel CFG weight)
│   │   └── parameter overrides (prompt, denoise, cfg, …) + operators
│   └── …
├── Layer N-1
├── …
└── Layer 0 (bottom)
```

---

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Scene** | The whole composition. Carries scene-level prompt, base denoise, base CFG, dims, and the ordered list of layers. |
| **Layer** | Container. Has z-index, visibility, blend_mode (for color compositing), and default values that any of its regions can inherit. |
| **Region** | Spatial sub-area inside a Layer. Holds painted channels and parameter overrides. |
| **Channel** | A paintable layer/region surface. Each channel has its own 2D bitmap. |
| **Mask** | A grayscale (`L`-mode) bitmap whose alpha [0,1] modulates *something*: visibility, denoise weight, CFG weight, prompt influence, etc. |
| **Operator** | How a per-pixel parameter combines with the accumulated state below. `replace`, `add`, `average`, `multiply`, `max` for numerics; `replace`, `concat`, `embed_blend` for prompts. |
| **Schedule** | A `(start, end)` window in `[0, 1]`, fraction of diffusion timesteps over which the region is active. |
| **CompositionPlan** | Renderer-agnostic description of how to render. Outputs one of: `SinglePassPlan`, `LayeredPassPlan`, `TiledPassPlan`. |
| **Renderer** | A backend that executes a CompositionPlan: SDXL diffusers, Z-Image, StreamDiffusion (WebRTC), SANA. |

---

## 3. Paintable channels (per region)

Every region exposes the following channels. Each is independently editable
(brush, eraser, eyedropper, fill, etc.) and stored as a separate bitmap.

### 3.1 `color` — RGB
The pixels the user paints in color. They serve as **input to the diffusion
model** (img2img conditioning). They do NOT directly appear in the output —
the diffusion model rewrites the masked regions.

### 3.2 `color_mask` — alpha (L-channel, 0..255)
Where this region's color exists / is visible. Drives **color compositing**
(see §5). Painted with a brush; default value when omitted = derived from the
`color` channel's alpha.

### 3.3 `denoise_mask` — alpha (L-channel, 0..255)
Per-pixel **weight** on the region's `denoise` scalar. Interpretation: at a
given pixel, `effective_denoise = scalar_denoise × mask_alpha`. Painted as a
grayscale gradient. If unpainted, behaves as full-coverage of the region's
`color_mask`.

### 3.4 `prompt_mask` — alpha (L-channel, 0..255)
Per-pixel **weight** on the region's `prompt`. In layered-pass rendering, the
mask is the per-region inpaint mask. In single-pass rendering, the mask
becomes the weight in the prompt aggregation operator (see §6.4).

### 3.5 `cfg_mask` — absolute CFG map
Controls the absolute CFG value per pixel. The realtime transport uses a raw
float32 mask resource (`application/x-rtd-mask-f32`) so CFG values are not
quantized to 8-bit PNG alpha. Legacy PNG masks are still accepted and are
interpreted as normalized absolute CFG (`0..255` maps to CFG `0..30`). Only
honored by renderers that support per-pixel CFG (SDXL, Z-Image);
StreamDiffusion and SANA fall back to a scalar (see §9).

### 3.6 Mask objects and painting tools
The user does NOT choose a global channel in the header. The **Mask tab** is
the source of truth for mask editing.

Each layer displays these mask rows:
- **Default mask** — inherited from the layer color alpha. It behaves like a
  normal mask for composition, but is read-only: it cannot be painted. It can
  be deleted from the layer and restored with **Add default mask**.
- **Regular masks** — user-created spatial masks. Clicking a mask color swatch
  selects both that mask and its paint color. Brush strokes are scoped to that
  mask; the user should never be able to paint outside the selected mask's
  ownership area.
- **CFG mask** — one special fine-tuning mask per layer. Transparent by
  default. It is layer-wide but clipped to the layer/default mask and overrides
  regular CFG weighting where painted.
- **Denoise mask** — one special fine-tuning mask per layer. Transparent by
  default. It is layer-wide but clipped to the layer/default mask and overrides
  regular denoise weighting where painted.

Brush behaviour:
- Clicking a regular mask color selects `color_mask` editing for that mask.
- Clicking the CFG mask swatch selects `cfg_mask` editing for that layer.
- Clicking the Denoise mask swatch selects `denoise_mask` editing for that
  layer.
- For grayscale fine-tuning masks, brush value is `0..255`: `0` means no
  override, `255` means full scalar override.
- Eraser paints alpha/value 0.

Other parameters that may later need special layer-wide override masks:
ControlNet strength, LoRA weight, IP-Adapter/image-conditioning weight, and
guidance-rescale. They are not implemented yet; add them only when a renderer
can actually consume them.

---

## 4. Mask alpha semantics — **multiplier of the region scalar**

A mask alpha is a **per-pixel weight** that multiplies the region's scalar
value for that parameter:

```
effective_param[x,y] = scalar_param × mask_alpha[x,y] / 255
```

Examples:

```
region.denoise = 0.8
denoise_mask[x, y] = 128   (50 %)
→ effective_denoise[x, y] = 0.40

region.cfg = 7.0
cfg_mask[x, y] = 255       (100 %)
→ effective_cfg[x, y] = 7.0
```

Rationale: this is simple and predictable. The scalar tells **how strong** the
effect is at full mask; the mask tells **where and by how much** to apply it.

---

## 5. Color compositing

The painted `color` channels of each layer composite **bottom-up** using the
layer's `blend_mode`:

```
out_rgb[x, y] = blend(below_rgb, top.color, top.color_mask, top.blend_mode)
```

Supported `blend_mode` values per layer:
- `normal` — standard alpha blending (Photoshop "Normal")
- `multiply` — darken
- `screen` — lighten
- `overlay` — contrast boost

This composite is the **input** fed to the diffusion model. The model rewrites
masked regions; everything outside `denoise_mask` ∪ `color_mask` is preserved.

### 5.1 Empty `color_mask`
A region whose `color_mask` is entirely zero (or whose `color` channel was
never painted) contributes **no color** to the composite — same effect as a
fully transparent layer above. It can still contribute to `denoise_mask` /
`prompt_mask` / `cfg_mask` (e.g., a region that only changes the diffusion
parameters without altering the canvas color).

---

## 6. Parameter aggregation (bottom-up)

For each parameter (denoise, cfg, prompt), the system walks the regions of
the scene **in z-order bottom-up** and accumulates the parameter map.

### 6.1 Numeric operators (`denoise_op`, `cfg_op`, `mask_op`)

Given:
- `acc[x, y]` — current accumulator (scalar 2D map)
- `α[x, y] = mask_alpha[x, y] / 255` — region's mask weight at this pixel
- `v` — region's scalar (denoise / cfg / weight)
- one operator from `replace, add, average, multiply, max`

```
replace   acc = acc·(1−α) + v·α          (smooth alpha-weighted overwrite)
add       acc = clamp(acc + v·α, lo, hi)
average   acc = acc·(1−α) + v·α          (same formula as replace, semantically equivalent here)
multiply  acc = acc · (1 + (v−1)·α)
max       acc = max(acc, v·α)
```

The starting accumulator is the scene-level default (e.g.,
`scene.base_denoise`).

### 6.2 Layer defaults + region overrides

For each parameter the lookup chain is:

1. If the region declares its own value → use it
2. Else if the region's layer declares a default → use the layer default
3. Else → use the scene-level base

This applies to: `prompt`, `negative_prompt`, `denoise`, `cfg`, plus the
operator choice for each.

### 6.3 Mask union

The per-pixel mask of a region for parameter P (denoise, prompt, cfg) is the
**point-wise minimum** of:
- the region's `color_mask` (where the region exists at all), and
- the region's `<param>_mask` (per-pixel weight on that param).

If `<param>_mask` was never painted, it defaults to a uniform 1.0 — meaning
the parameter weight follows `color_mask` directly.

### 6.4 Prompt aggregation

Text prompts can't be averaged like scalars, so each prompt operator has a
distinct semantic:

| Operator | Meaning |
|---|---|
| `replace` | Top region's prompt fully wins where its mask is active |
| `concat` | Build `"(p1:w1), (p2:w2), ..."` from contributions, single encode (DEFAULT) |
| `embed_blend` | Encode each prompt separately, blend the embeddings per-region with their masks. SDXL / Z-Image only — others fall back to `concat` with a warning |

The `concat` weight per region is `max(prompt_mask × color_mask)`.

### 6.5 Schedule

Each region has `schedule_start, schedule_end ∈ [0, 1]` — fraction of
diffusion timesteps during which it's active. Renderers that support
per-region scheduling (SDXL, Z-Image) gate the region accordingly. Others
(StreamDiffusion, SANA) treat the schedule as a binary "active y/n" gate and
emit a warning.

---

## 7. Layer activation rule (the "empty layer" question)

A layer is **inactive** — contributes nothing to color, denoise, prompt, or
CFG — unless **at least one pixel** is painted on **any** of its regions'
channels (`color`, `color_mask`, `denoise_mask`, `prompt_mask`, `cfg_mask`).

Rationale chosen by the user: simple and predictable. Adding a layer with
`prompt = "fire"` but no paint produces zero effect; you must paint where
"fire" should go.

The implementation MUST therefore check, per region:

```python
def is_active(region) -> bool:
    return any(channel_has_painted_pixel(c) for c in region.channels)
```

A layer whose every region is inactive is dropped entirely from the
CompositionPlan (no diffusion pass spawned, no prompt contribution).

---

## 8. Rendering strategies

The renderer picks one of three execution shapes based on the user's choice
on the Schedule tab:

### 8.1 Single-pass (`mode = single`)
One diffusion call with the aggregated per-pixel parameter maps. Cheapest.
- Inputs: composed RGB canvas, aggregated `denoise_map`, aggregated `cfg_map`
  (or scalar fallback), `merged_prompt`.
- Output: one image.
- Quality trade-off: regional prompts get only the `concat` aggregation, no
  hard spatial separation between layer prompts.

### 8.2 Layered-pass (`mode = layered`)
One diffusion call **per region that has a prompt**. Each pass uses that
region's mask as the inpaint mask and that region's prompt as the prompt.
Passes execute bottom-up; each pass's output becomes the next pass's
`composite_base` (so regions stack correctly).

- Inputs (per pass): `composite_base` from previous, region's mask, region's
  prompt, region's denoise/cfg.
- Output: one image, composited at the renderer level.
- Quality trade-off: best regional separation, but N inferences per frame.

### 8.3 Tiled-pass (`mode = tiled`)
Canvas split into overlapping tiles snapped to SDXL-optimal aspect ratios
(1024×1024, 1152×896, 832×1216, etc.). For each tile:
- Aggregate the regions intersecting the tile (same logic as single-pass).
- Run the tile's plan at full SDXL resolution.
- Feather-blend tiles back into the canvas; cache tile signatures to skip
  re-rendering unchanged tiles ("dirty-tile cache").

Best quality at high resolution. Slow on first frame; subsequent frames only
re-render tiles whose plan changed.

---

## 9. Renderer capability matrix

Some renderers can't honour every feature. The composition layer detects this
and emits one explicit warning per dropped feature.

| Feature                | SDXL   | Z-Image | StreamDiffusion | SANA               |
|------------------------|--------|---------|-----------------|--------------------|
| Multi-pass (layered)   | ✅     | ✅      | ✅              | ⚠️ structured fallback |
| Per-region prompt      | ✅     | ✅      | ✅ (per-pass)   | ✅ (LLM structure) |
| Per-region CFG         | ✅     | ✅      | ❌ → cfg_scalar | ❌ → cfg_scalar |
| Per-pixel CFG map      | ✅     | ✅      | ❌              | ❌                 |
| Per-region schedule    | ✅     | ✅      | ❌ → binary     | ❌ → binary        |
| Per-pixel denoise map  | ✅     | ✅      | ✅              | ⚠️ scalar          |
| ControlNet per region  | ✅     | ❌      | ✅              | ❌                 |
| `embed_blend` prompt   | ✅     | ✅      | ❌ → concat     | ❌ → structured    |

SANA gets a special structured-prompt formatter (`Background:\n... / Foreground:\n...`) because its Gemma encoder digests labelled sections better than CSV concat. Spatial English locators (`upper-left`, `centre`, …) are added per region.

---

## 10. Schedule tab UI

The Schedule tab is the **conductor's view**: a timeline of all layers, the
order in which they'll be processed, and a live preview of the render cost.

Required operations:
- **Drag** to adjust `schedule_start` / `schedule_end` (timestep range) on
  each layer/region bar.
- **Drag** vertically to reorder z-index of layers.
- **Live preview** at the bottom: `"3 passes (single) | 5 passes (layered) | 12 tiles (tiled)"` based on the current mode and active regions.
- **Toggle visibility** with an eye icon next to each row.

Read-only sections per layer in their property panel can echo the schedule
window text (e.g., `Active 0.2–0.8`), but the Schedule tab is the **source of
truth**.

---

## 11. Aggregation worked example

Given:

```yaml
scene:
  base_prompt: "ancient temple"
  base_denoise: 0.5
  base_cfg: 4.5

layers:
  - id: bg                              # bottom
    blend_mode: normal
    regions:
      - id: bg-fill
        color_mask: full canvas
        prompt: "stormy sky"
        prompt_op: concat
        denoise: 0.6
        denoise_op: replace

  - id: fg                              # top
    blend_mode: normal
    regions:
      - id: knight
        color_mask: roughly centred figure shape (alpha continuous edges)
        prompt: "armoured knight, holding sword"
        prompt_op: concat
        denoise: 0.9
        denoise_op: replace
        cfg: 8.0
```

Expected behaviour:

- Color: the painted "knight" pixels alpha-blend over the background's
  painted sky.
- Denoise map: bottom region replaces base 0.5 → 0.6 across the whole canvas;
  top region replaces 0.6 → 0.9 in the knight's alpha. Final: 0.6 background,
  0.9 inside knight (with feathered transition where the knight's color_mask
  is partial).
- Prompt (single-pass with concat default):
  `"ancient temple, stormy sky, armoured knight, holding sword"`.
- Prompt (layered-pass): two diffusion calls.
  1. Sky pass: mask = bg-fill mask, prompt = "ancient temple, stormy sky".
  2. Knight pass: mask = knight mask, prompt = "ancient temple, stormy sky, armoured knight, holding sword".
- CFG: bg uses scene base (4.5). Knight uses 8.0. Where renderer supports
  per-pixel CFG: smooth transition through alpha. Otherwise: scalar fallback
  = max contributing CFG (8.0).

---

## 12. Implementation map

This spec is implemented across these modules. They MUST stay in sync with
the spec; if you change a semantic, update both the code AND this document.

| Module | Responsibility |
|---|---|
| `backend/app/composition.py` | Pure: Scene/Layer/Region/Region dataclasses, operators, aggregation, `compose()` entry point. No GPU. |
| `backend/app/tile_executor.py` | Pure: orchestrates a `TiledPassPlan` with feather blending and dirty-tile cache. Uses a renderer callback. |
| `backend/app/sana_prompt.py` | Pure: structured-prompt formatter for the SANA fallback. |
| `backend/app/stream/manager.py` | StreamDiffusion session manager — consumes `LayeredPassPlan` via `inference/stream_session.py`. |
| `backend/app/engine/engine.py` | SDXL / Z-Image — consumes the resolved composition mask + per-region passes. |
| `backend/app/inference/registry.py` | Renderer capability matrix (consumed by the composition layer for graceful fallback). |
| `backend/app/rtc/session.py` | Per-connection orchestration: staging input, RTC dispatch, processes the conducted Schedule. |
| Frontend `scene-panel/` + dedicated **Schedule tab** | UI: layer panel for content/properties; Schedule tab for chronograph. |

---

## 13. Legacy bridge & wire protocol

### 13.1 Legacy `layer_conditions` (still supported)

The historical payload shape:

```json
{
  "layer_conditions": [
    {
      "layer_id": "abc",
      "region_id": "def",
      "image": "data:image/png;base64,...",       // RGBA, alpha = legacy single mask
      "mode": "mask|override|add|multiply|prompt_mix",
      "prompt": "...",
      "denoise": 0.8,
      "cfg": 7.0,
      "schedule_start": 0.0,
      "schedule_end": 1.0,
      "mask_operator": "add",
      "denoise_operator": "replace",
      ...
    }
  ]
}
```

`composition.scene_from_legacy()` is the canonical bridge between this flat
schema and the typed `Scene / Layer / Region`. Legacy `mode` maps to the new
operators as follows:

| Legacy `mode` | New `mask_op` | New `denoise_op` |
|---|---|---|
| `mask` (default) | `add` | `average` |
| `override` / `replace` | `replace` | `replace` |
| `add` | `add` | `add` |
| `multiply` | `multiply` | `multiply` |
| `max` | `max` | `max` |
| `prompt_mix` | (skip, no spatial) | (skip) |

Explicit `mask_operator` / `denoise_operator` fields take precedence over the
legacy `mode`.

### 13.2 Realtime input transport — WebRTC data channels

The realtime path uses HTTP exactly once for WebRTC SDP offer/answer:

```
POST /api/rtc/offer
{ "offer": { "type": "offer", "sdp": "..." } }

→ { "pc_id": "...", "answer": { "type": "answer", "sdp": "..." } }
```

After that handshake, **all realtime inputs are staged through WebRTC data
channels**, not HTTP polling:

| Channel | Ordering | Payload | Meaning |
|---|---:|---|---|
| `settings` | ordered | JSON text `{type:"settings", settings:{...}}` | Scene settings, layer conditions, pipeline graph. |
| `frames` | unordered / no retransmit | JSON text `{type:"frame", image:"data:image/jpeg;base64,..."}` | Latest canvas staging frame; old frames may be dropped. |
| `resources` | ordered | Binary envelope | Masks and other resources; server replies with `blob_ack`. |

Binary resource envelope:

```
uint32_be header_byte_length
utf8_json_header
raw_payload_bytes
```

Example header:

```json
{ "type": "blob", "request_id": "42", "mime": "image/png", "name": "Layer 1/inherited:layer-1/denoise" }
```

Server reply on the same `resources` channel:

```json
{ "type": "blob_ack", "request_id": "42", "id": "blake2b-12-byte-hex" }
```

`name` is optional but strongly recommended. It is used by debug inspection so
uploaded inputs appear with stable, human-readable paths instead of only blob
IDs.

The existing HTTP `/api/rtc/{pc_id}/frame`, `/settings`, and `/blob` endpoints
are legacy fallback/compatibility surfaces only. New frontend code MUST use
data channels for realtime input staging.

### 13.3 Per-channel masks — refs and fallback

Each layer condition may carry up to four channel masks:
`color_mask`, `denoise_mask`, `prompt_mask`, `cfg_mask`. Realtime should send
them as binary blobs over the `resources` data channel and then reference the
returned ID:

```json
{ "region_id": "R1",
  "denoise_mask_ref": "blake2b-12-byte-hex-id" }
```

Inline base64 remains accepted for legacy/non-realtime payloads, but it is not
the realtime path:

```json
{ "region_id": "R1",
  "denoise_mask": "data:image/png;base64,iVBORw0KGgo..." }
```

Server semantics:
- IDs are content-hashed (BLAKE2b-12-byte hex) — re-uploading the same bytes
  returns the same ID; the store never grows from duplicate paints.
- Per-session LRU at 128 entries; blobs die with the session.
- The `*_mask_ref` field wins when both `*_mask_ref` and `*_mask` are present
  in the same layer condition. Inline base64 is the legacy fallback.

Rationale: a painted region's mask only changes when the user paints, not per
video frame. The datachannel blob path lets the frontend stream the changed
resource once and reference it cheaply in subsequent settings messages.

### 13.4 Frontend implementation status

When the frontend doesn't yet send per-channel masks, every parameter of a
given region shares the legacy alpha (`Region.mask`). The composition module
resolves missing channels to that fallback automatically. So:

- **Old frontend + new backend** → identical behaviour to the current product.
- **New frontend with `*_mask_ref`** → each parameter uses its own channel
  mask; binary path, no base64 overhead.
- **New frontend with `*_mask` inline** → same semantics, base64 path.

Current frontend wiring:
- The Mask tab exposes mask rows. Clicking a row's swatch selects the mask and
  the editable channel; there is no global channel selector in the header.
- The default mask is displayed as an inherited, read-only mask. It can be
  removed/restored per layer.
- `color` keeps the legacy colored mask canvas and region-color extraction.
- `denoise` and `cfg` paint into separate grayscale canvases per layer. At
  export, each grayscale channel is clipped by each region's `color_mask` /
  region shape, encoded as PNG, streamed through the WebRTC `resources` data
  channel, and referenced by `*_mask_ref` in `layer_conditions`.
- Debug streams expose input resources under stable names:
  `input/frame/*`, `input/layer/<name>/color`, `input/layer/<name>/alpha`,
  `input/mask/<layer>/<region>/<channel>`, and `input/resource/<sent-name>`.
  `meta:input_resources` lists the currently inspectable inputs as text.
- `region_id` is sent for every exported region, including inherited regions.
- Missing channel masks continue to fall back to the legacy image alpha.

---

## 14. Decisions that are NOT in this spec (yet)

These are deliberately left open and require explicit user direction before
implementation:

1. **Schedule preview animation**: Should the Schedule tab show a scrubber
   that animates per timestep, previewing which regions are active?
2. **ControlNet aggregation when two layers share a CN type**: Take top
   layer's CN? Average? Sum?
3. **Seed scoping**: Scene-level seed, or per-region seed override?

When a decision is needed, **ask** before coding. Do not invent semantics
silently.

---

## 15. Invariants (truths the implementation MUST uphold)

Any change that breaks one of these is a regression.

1. **Z-order is the source of truth.** Bottom-to-top evaluation is the only
   ordering allowed. No code path may iterate layers in a different order.
2. **An unpainted layer contributes nothing.** Verified by `is_active()`.
3. **Mask alpha is a multiplier.** No code path may reinterpret it
   (no thresholding, no inversion, no binary clamping inside the composition
   module).
4. **Renderer capability fallback is explicit.** Each dropped feature emits
   exactly one warning (not zero, not many). The plan still works without
   that feature.
5. **Backpressure is staged, never queued.** The latest user input replaces
   prior staged input; no inference enqueues a backlog of user frames.
6. **Frame staging is read just before dispatch.** The `_process_loop` reads
   `self._canvas_url` immediately before `asyncio.to_thread(...)`, not at
   loop-top.
7. **The Schedule tab is the only UI source of truth for layer order and
   timing.** Other panels may display, never edit, those fields.

---

*End of spec.*
