# Scene Stream Protocol

RTDiffusion WebRTC input streaming uses a normalized scene protocol. The browser does not stream a full monolithic frame as the transport unit. Instead, it keeps a server-side scene snapshot current by sending structure and resources independently.

## Channels

- `scene`: ordered JSON data channel carrying the current scene structure.
- `resource:<id>`: one ordered binary data channel per immutable resource id. The channel label identifies the resource reference; each message may also include a human-readable debug name.

Legacy `settings` and `frames` data channels are not used by the frontend RTC input path.

## Scene Message

The `scene` channel carries:

```json
{
  "type": "scene",
  "id": "scene_349c0c7dcb55d6e4f6ab2f9a",
  "seq": 1,
  "scene": {
    "id": "scene_349c0c7dcb55d6e4f6ab2f9a",
    "protocol": "rtd.scene.v1",
    "input": {
      "image_ref": "res_89d6db852c87380f7f7cf7a0",
      "mask_ref": "res_58b0bc4d91e7eaf42f3f4bda"
    },
    "settings": {
      "prompt": "...",
      "width": 832,
      "height": 1216,
      "strength": 1.0
    },
    "layer_conditions": [
      {
        "layer_id": "layerA",
        "region_id": "maskX",
        "image_ref": "res_04cc3d1d3d3d498e33c1a991",
        "denoise_mask_ref_name": "res_fcf5fa074bf5246f3af67c02"
      }
    ]
  }
}
```

`id` is the content reference for the scene structure. If any referenced resource id or scene field changes, the scene id changes. `seq` is still monotonic per peer and older scene messages are ignored.

## Scene Patch Events

After the initial full scene, clients may send `scene_patch` messages. A patch keeps the legacy `patch` object for compatibility and also carries a replayable `events` list:

```json
{
  "type": "scene_patch",
  "seq": 2,
  "events": [
    { "type": "property.set", "path": ["settings", "prompt"], "value": "sunlit room" },
    { "type": "signal.update", "signal": {
      "id": "layer.layerA.regionX.prompt_mask",
      "type": "prompt_mask",
      "channel": "prompt",
      "layer_id": "layerA",
      "region_id": "regionX",
      "resource_ref": "res_abc123"
    }},
    { "type": "layer_conditions.replace", "layer_conditions": [] }
  ],
  "patch": {
    "settings": { "prompt": "sunlit room" },
    "signals": [],
    "layer_conditions": []
  }
}
```

Supported event types:

- `property.set`: set a nested scalar/object value by `path`.
- `property.unset`: remove a nested value by `path`.
- `input.update`: replace the scene `input` block.
- `signal.add` / `signal.update`: upsert a signal by `signal.id`.
- `signal.remove`: remove a signal by `id` or `signal_id`.
- `layer_conditions.replace`: replace the legacy bridge list while it remains supported.

The backend prefers `events` when present and falls back to `patch` for older clients. `seq` remains the replay ordering guard.

## Resource Messages

A resource message is a binary envelope:

```json
{
  "type": "resource",
  "id": "res_fcf5fa074bf5246f3af67c02",
  "name": "layer.layerA.maskX.denoise_mask",
  "mime": "image/png"
}
```

The envelope body is the raw blob bytes. A resource is immutable: its id is derived from its bytes and is the reference used by scenes. If the user paints `layerA/maskX`, the mask bytes produce a new resource id; the next scene references that new id. Unchanged layer images, scene settings, and other masks keep the same ids and are not resent.

## Server Materialization

The server stores immutable resources by id and the latest scene structure by scene id. On every scene or resource update, it materializes a complete renderer input snapshot:

- `input.image_ref` becomes the current canvas/input image.
- `input.mask_ref` becomes the scene mask.
- `layer_conditions[*].image_ref` becomes `layer_conditions[*].image`.
- `*_ref_name` fields become the corresponding inline legacy fields for existing render code.

Renderers read this materialized complete scene. They do not need to know whether a field arrived from a structure update or a resource update.

## Freshness Rule

The current server-side scene snapshot is latest-only. User actions update the relevant resource or scene structure; the next render starts from whatever complete snapshot is current when the renderer begins.
