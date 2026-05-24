/**
 * Per-channel mask blob upload + cache.
 *
 * Mask blob upload + cache. In the WebRTC path, raw PNG bytes travel over the
 * ``resources`` data channel and the server replies with a content-hashed ID.
 * HTTP ``POST /api/rtc/{pc_id}/blob`` remains as a legacy fallback. Layer
 * conditions reference masks by ID (``denoise_mask_ref`` / ``prompt_mask_ref`` /
 * ``cfg_mask_ref`` / ``color_mask_ref``) instead of base64-inlining them in
 * settings JSON.
 *
 * This module owns the *local* cache: given a canvas representing a painted
 * channel mask, it computes a browser-side content hash so we only transmit
 * when the bytes actually changed. The backend still returns its own canonical
 * blob ID. ``serializeChannelRefs`` is what callers use to embed those IDs
 * into the layer conditions payload.
 *
 * Spec: ``docs/LAYER_SYSTEM.md`` §13.2.
 */

import { rtcUploadBlob } from './api'

const MAX_PENDING = 128
const _idByHash = new Map<string, string>()
const _inflight = new Map<string, Promise<string>>()
type BlobUploader = (pcId: string, blob: Blob, name?: string) => Promise<{ id: string }>
let _uploader: BlobUploader = rtcUploadBlob

export type ChannelKey = 'denoise' | 'prompt' | 'cfg' | 'color'
export type ChannelRefMap = Partial<Record<ChannelKey, string>>

const _REF_FIELD: Record<ChannelKey, string> = {
  denoise: 'denoise_mask_ref',
  prompt: 'prompt_mask_ref',
  cfg: 'cfg_mask_ref',
  color: 'color_mask_ref',
}

/** Compute a local cache key for a Blob's bytes. The server uses BLAKE2b for
 * its canonical ID; the browser uses SHA-256 truncated to 12 bytes because it
 * is available through Web Crypto and is only used for client-side dedup. */
async function _hashBlob(blob: Blob): Promise<string> {
  const buf = await _blobArrayBuffer(blob)
  const digest = await crypto.subtle.digest('SHA-256', buf)
  const bytes = new Uint8Array(digest, 0, 12)
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')
}

function _blobArrayBuffer(blob: Blob): Promise<ArrayBuffer> {
  if (typeof blob.arrayBuffer === 'function') return blob.arrayBuffer()
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as ArrayBuffer)
    reader.onerror = () => reject(reader.error ?? new Error('Could not read mask blob'))
    reader.readAsArrayBuffer(blob)
  })
}

/**
 * Upload ``blob`` if its content hash isn't already in the local cache.
 * Returns the server-assigned blob ID. Concurrent calls with identical
 * content de-duplicate via the in-flight map.
 */
export async function uploadMaskBlob(pcId: string, blob: Blob, name?: string): Promise<string> {
  const key = `${pcId}|${await _hashBlob(blob)}`
  const cached = _idByHash.get(key)
  if (cached) return cached
  const existing = _inflight.get(key)
  if (existing) return existing
  const promise = (async () => {
    try {
      const { id } = await _uploader(pcId, blob, name)
      _idByHash.set(key, id)
      if (_idByHash.size > MAX_PENDING) {
        // Drop the oldest entry — naive LRU works fine for this size.
        const oldest = _idByHash.keys().next().value
        if (oldest) _idByHash.delete(oldest)
      }
      return id
    } finally {
      _inflight.delete(key)
    }
  })()
  _inflight.set(key, promise)
  return promise
}

export function setMaskBlobUploader(uploader?: BlobUploader) {
  _uploader = uploader ?? rtcUploadBlob
}

/** Drop every cached ref — call when the RTC session changes. */
export function clearMaskBlobCache() {
  _idByHash.clear()
  _inflight.clear()
  _uploader = rtcUploadBlob
}

/**
 * Convert a ``{channel: Blob}`` map into a ``{<channel>_mask_ref: id}`` map
 * suitable for inlining into a layer_conditions item. Uploads each blob
 * (skipping any that are already cached) in parallel.
 */
export async function serializeChannelRefs(
  pcId: string,
  channels: Partial<Record<ChannelKey, Blob>>,
  names: Partial<Record<ChannelKey, string>> = {},
): Promise<Record<string, string>> {
  const entries: [ChannelKey, Blob][] = (Object.entries(channels) as [ChannelKey, Blob | undefined][])
    .filter((entry): entry is [ChannelKey, Blob] => entry[1] instanceof Blob)
  if (entries.length === 0) return {}
  const ids = await Promise.all(entries.map(([channel, blob]) => uploadMaskBlob(pcId, blob, names[channel])))
  const out: Record<string, string> = {}
  for (let i = 0; i < entries.length; i++) {
    out[_REF_FIELD[entries[i][0]]] = ids[i]
  }
  return out
}
