# RTDiffusion Signal Pipeline Plan

Status: implementation plan for closing the current Web GUI -> transport -> GPU gaps.

The authoritative layer semantics remain in `docs/LAYER_SYSTEM.md`. This file tracks the missing implementation work needed to make those semantics true across SDXL, Z-Image, and StreamDiffusion.

## Target Invariants

1. Regional prompting uses prompt softmaps as first-class inputs. A renderer may execute them as cross-attention masking, layered passes, or a documented fallback, but the capability and degradation must be explicit.
2. The render session receives one complete initial scene, then ordered scene events/patches for property changes. Every patch must be replayable against the previous scene.
3. Images and softmaps are resources referenced by the scene. Resource lifecycle is explicit: added, updated, removed. WebRTC data channels are the preferred resource transport; WebSocket is fallback only.
4. SDXL, Z-Image, and StreamDiffusion consume the same typed composition result before renderer-specific execution.

## Work Plan

### Phase 1: Signal Contract

- Add `scene.signals[]` alongside legacy `layer_conditions`.
- Each signal names `id`, `type`, `resource_ref`, and optional `layer_id` / `region_id` / `channel`.
- Preserve legacy fields until all consumers are migrated.
- Add tests that every input image, layer image, color mask, prompt mask, CFG mask, and denoise mask has exactly one signal entry and a matching resource ref.

### Phase 2: Event Stream

- Replace coarse scene patches with typed events: `scene.set`, `signal.add`, `signal.update`, `signal.remove`, `property.set`, `property.unset`, and `layer.reorder`.
- Keep patch support as compatibility input, but normalize patches into events on the backend.
- Add sequence and snapshot identifiers so the backend can reject mixed-time scenes.

Initial slice completed: `scene_patch` now carries replayable `events[]` for settings, input, signals, and legacy layer-condition replacement. Backend event replay is supported while old `patch` input remains valid. Snapshot identity and removal lifecycle coverage are still pending.

### Phase 3: Resource Lifecycle

- Route image and softmap blobs through the WebRTC `resources` data channel when available.
- Emit control events when media tracks/resources are added or removed.
- Keep WebSocket binary resources as a fallback path with an explicit debug/status flag.
- Add tests for add/update/remove and for stale resource refs not entering inference.

### Phase 4: Shared Composition Consumption

- Make RTC build one typed composition result and pass it to all renderer adapters.
- Remove SDXL/Z-Image divergence where they rebuild regional behavior from legacy `layer_conditions`.
- Keep renderer capability warnings close to execution so dropped features are visible in debug output.

### Phase 5: Regional Attention

- Re-enable SDXL/Z-Image regional attention only for prompt softmaps that are intended for embedding/cross-attention blending.
- Keep mask-mode layered inpaint for bounded regional replacement when cross-attention would leak concepts outside a region.
- Add tests proving arbitrary softmap counts install cross-attention processors and that unsupported renderers produce explicit degradation warnings.

## Immediate Slice

The first implementation slice is Phase 1: add `scene.signals[]` without removing legacy `layer_conditions`. This makes the signal stack observable and gives the backend a stable migration target.