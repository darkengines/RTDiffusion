import { $stream } from '../stores/stream.store'
import { $scene } from '../stores/scene.store'
import { reportError } from '../stores/ui.store'
import { upsertTask } from './layer.service'
import { openInpaintSocket, openRenderSessionSocket, rtcOffer } from './api'
import { clearMaskBlobCache } from './mask-blob.service'
import type { StreamMessage } from '../types'

// ── Singleton connection state (not reactive — lifecycle objects) ─────────────

let _socket: WebSocket | undefined
let _streamRaf: number | undefined
let _streamTimer: number | undefined
let _sessionSocket: WebSocket | undefined
let _webRtcMediaPeer: RTCPeerConnection | undefined
let _webRtcMediaPeerSessionId = ''
let _webRtcMediaNegotiation: Promise<void> | undefined
let _webRtcLiveInputMediaReady = false
let _webRtcLiveInputMediaRegisteredOnce = false
let _webRtcOutputMediaReady = false
let _webRtcOutputMediaStream: MediaStream | undefined
let _webRtcResourceChannel: RTCDataChannel | undefined
let _webRtcPcId = ''
let _frameLoopRaf: number | undefined
let _webRtcLifecycleSeq = 0
let _stoppingWebRtc = false
let _webRtcLastScene = ''
let _webRtcLastSceneObject: Record<string, unknown> | undefined
let _sendingWebRtcScene = false
let _webRtcSceneDirty = false
let _webRtcSceneSeq = 0
let _webRtcSceneRequestVersion = 0
let _webRtcLayerConditionsDirty = false
let _webRtcResourceRequestVersion = 0
let _webRtcResourceSentVersion = 0
let _webRtcSceneScheduled = false
let _webRtcInputReady = false
let _webRtcSeedRotationTimer: number | undefined
let _webRtcSeedRotationLastAt = 0
let _webRtcSeedRotationSerial = 0
let _webRtcSeedRotationPending = false
const _webRtcSentResourceIds = new Set<string>()
let _webRtcWaitingForResourceBackpressure = false
let _awaitingFrame = false
let _awaitingFrameSentMs = 0      // timestamp when _awaitingFrame was set; 0 = not waiting
let _sendingInpaintFrame = false  // guard against concurrent canvas exports
let _inpaintLastSendMs = 0        // throttle to ~30 fps max
let _inpaintSeq = 0
let _inpaintLatestInputSeq = 0
let _inpaintAwaitingSeq = 0
let _inpaintSendRequested = false
let _inpaintRetryTimer: number | undefined
let _fpsCount = 0
let _fpsWindowStart = 0
let _lastWorkingModel = ''
let _lastWorkingLoras: string[] = []

// Callbacks wired in by the canvas-editor component
export type FrameExporter = () => Promise<string | null>
export type SettingsExporter = (pcId?: string) => string | Promise<string>
export type FullFramePayload = string | Record<string, unknown>
export type FullFrameExporter = (pcId?: string) => Promise<FullFramePayload | null>
export type SceneSettingsExporter = () => Record<string, unknown> | Promise<Record<string, unknown>>
export type LayerConditionsMetadataExporter = () => Record<string, unknown>[] | Promise<Record<string, unknown>[]>

let _exportFullFrame: FullFrameExporter = async () => null
let _exportSceneSettings: SceneSettingsExporter = async () => ({})
let _exportLayerConditionsMetadata: LayerConditionsMetadataExporter = async () => []

type SceneResource = { id: string; debugName: string; blob: Blob; mime: string }
export type DebugSurface =
  | { id: string; name: string; kind: 'canvas'; canvas: HTMLCanvasElement }
  | { id: string; name: string; kind: 'stream'; stream: MediaStream }

const _resourceByDataUrl = new Map<string, SceneResource>()

export function wireFullFrameExporter(fn: FullFrameExporter) { _exportFullFrame = fn }
export function wireSceneSettingsExporter(fn: SceneSettingsExporter) { _exportSceneSettings = fn }
export function wireLayerConditionsMetadataExporter(fn: LayerConditionsMetadataExporter) { _exportLayerConditionsMetadata = fn }

function _fullFramePacket(payload: FullFramePayload): Record<string, unknown> {
  return typeof payload === 'string' ? JSON.parse(payload) as Record<string, unknown> : { ...payload }
}

export function requestInpaintFrameUpdate() {
  _inpaintLatestInputSeq++
  _inpaintSendRequested = true
  _scheduleInpaintFrameSend(0)
}

function _scheduleInpaintFrameSend(delayMs: number) {
  if (_inpaintRetryTimer !== undefined) return
  _inpaintRetryTimer = window.setTimeout(() => {
    _inpaintRetryTimer = undefined
    void _sendInpaintFrame()
  }, Math.max(0, delayMs))
}

export function requestWebRtcSettingsUpdate() {
  if (!_webRtcPcId) return
  if ($scene.get().seedRotationMode !== 'off' && _webRtcSeedRotationLastAt <= 0) {
    _webRtcSeedRotationPending = true
    _webRtcSeedRotationSerial++
  }
  _webRtcSceneRequestVersion++
  _webRtcResourceRequestVersion++
  _webRtcSceneDirty = true
  _scheduleWebRtcSceneSend()
}

export function requestWebRtcLiveInputUpdate() {
  if (!_webRtcPcId) return
  if (!_webRtcLastSceneObject) {
    requestWebRtcSettingsUpdate()
    return
  }
  if ($scene.get().seedRotationMode !== 'off' && _webRtcSeedRotationLastAt <= 0) {
    _webRtcSeedRotationPending = true
    _webRtcSeedRotationSerial++
  }
  _webRtcSceneRequestVersion++
  _webRtcSceneDirty = true
  _scheduleWebRtcSceneSend()
}

export function requestWebRtcLayerConditionsUpdate() {
  if (!_webRtcPcId) return
  if (!_webRtcLastSceneObject) {
    requestWebRtcSettingsUpdate()
    return
  }
  if ($scene.get().seedRotationMode !== 'off' && _webRtcSeedRotationLastAt <= 0) {
    _webRtcSeedRotationPending = true
    _webRtcSeedRotationSerial++
  }
  _webRtcSceneRequestVersion++
  _webRtcLayerConditionsDirty = true
  _webRtcSceneDirty = true
  _scheduleWebRtcSceneSend()
}

export function requestWebRtcSceneSettingsPatch() {
  if (!_webRtcPcId) return
  if (!_webRtcLastSceneObject) {
    requestWebRtcSettingsUpdate()
    return
  }
  void _sendWebRtcSceneSettingsPatch()
}

export function requestRealtimeFrameUpdate(options: { refreshResources?: boolean; refreshLayerConditions?: boolean } = {}) {
  if (_webRtcPcId) {
    if (options.refreshResources) requestWebRtcSettingsUpdate()
    else if (options.refreshLayerConditions) requestWebRtcLayerConditionsUpdate()
    else requestWebRtcLiveInputUpdate()
    return
  }
  requestInpaintFrameUpdate()
}

function _rtcResourceChannelOpen() {
  return _webRtcResourceChannel?.readyState === 'open'
}

async function _negotiateWebRtcMediaPeer(pc: RTCPeerConnection, sessionId: string) {
  const run = async () => {
    const offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    const response = await rtcOffer(offer, sessionId)
    await pc.setRemoteDescription(response.answer)
  }
  const previous = _webRtcMediaNegotiation ?? Promise.resolve()
  const current = previous.catch(() => undefined).then(run)
  _webRtcMediaNegotiation = current
  try {
    await current
  } finally {
    if (_webRtcMediaNegotiation === current) _webRtcMediaNegotiation = undefined
  }
}

async function _ensureWebRtcMediaPlane(sessionId: string) {
  if (!sessionId || typeof RTCPeerConnection === 'undefined') return
  if (_webRtcMediaPeer && _webRtcMediaPeerSessionId === sessionId && _webRtcMediaPeer.connectionState !== 'failed' && _webRtcMediaPeer.connectionState !== 'closed') {
    if (_webRtcMediaNegotiation) await _webRtcMediaNegotiation.catch(() => undefined)
    return
  }
  if (_webRtcMediaPeer) {
    _webRtcMediaPeer.close()
    _webRtcMediaPeer = undefined
    _webRtcMediaPeerSessionId = ''
    _webRtcResourceChannel = undefined
    _webRtcLiveInputMediaReady = false
    _webRtcMediaNegotiation = undefined
  }
  try {
    const pc = new RTCPeerConnection()
    const resources = pc.createDataChannel('resources')
    pc.addTransceiver('video', { direction: 'recvonly' })
    _webRtcMediaPeer = pc
    _webRtcMediaPeerSessionId = sessionId
    _webRtcResourceChannel = resources
    resources.binaryType = 'arraybuffer'
    resources.onmessage = (event) => _handleResourceChannelMessage(event as MessageEvent)
    resources.onopen = () => console.info('[rtd transport] WebRTC resource channel open')
    resources.onclose = () => {
      if (_webRtcResourceChannel === resources) _webRtcResourceChannel = undefined
    }
    pc.ontrack = (event) => {
      if (event.track.kind !== 'video') return
      const stream = event.streams[0] ?? new MediaStream([event.track])
      _webRtcOutputMediaReady = true
      _webRtcOutputMediaStream = stream
      registerDebugStreamSurface('output/media', 'output/media', stream)
      window.dispatchEvent(new CustomEvent('rtd:rtc-output-stream', { detail: { stream } }))
    }
    pc.onconnectionstatechange = () => {
      if (pc.connectionState === 'failed' || pc.connectionState === 'closed') {
        if (_webRtcMediaPeer === pc) {
          _webRtcMediaPeer = undefined
          _webRtcMediaPeerSessionId = ''
          _webRtcLiveInputMediaReady = false
          _webRtcOutputMediaReady = false
          _webRtcOutputMediaStream = undefined
        }
        if (_webRtcResourceChannel === resources) _webRtcResourceChannel = undefined
      }
    }
    await _negotiateWebRtcMediaPeer(pc, sessionId)
    window.dispatchEvent(new CustomEvent('rtd:rtc-media-plane-ready', { detail: { sessionId } }))
  } catch (error) {
    console.warn('WebRTC media/resource plane unavailable; using WebSocket resources', error)
    _webRtcResourceChannel = undefined
    if (_webRtcMediaPeer) {
      _webRtcMediaPeer.close()
      _webRtcMediaPeer = undefined
    }
    _webRtcMediaPeerSessionId = ''
    _webRtcMediaNegotiation = undefined
  }
}

export async function registerWebRtcResourceMediaTrack(track: MediaStreamTrack, meta: Record<string, unknown> = {}) {
  if (!_webRtcPcId || typeof RTCPeerConnection === 'undefined') return false
  const isLiveInputMedia = meta.channel === 'canvas' || meta.channel === 'input' || meta.channel === 'frame'
  try {
    await _ensureWebRtcMediaPlane(_webRtcPcId)
    const pc = _webRtcMediaPeer
    if (!pc) return false
    const streamId = String(meta.stream_id || meta.id || `rtd_media_${Date.now().toString(36)}`)
    const stream = new MediaStream([track])
    pc.addTrack(track, stream)
    await _negotiateWebRtcMediaPeer(pc, _webRtcPcId)
    _sessionSocket?.send(JSON.stringify({ type: 'media_track', stream_id: streamId, track_id: track.id, meta }))
    registerDebugStreamSurface(streamId, String(meta.label || streamId), stream)
    _registerMediaTrackDebugChannel(streamId, String(meta.label || streamId))
    if (isLiveInputMedia) {
      _webRtcLiveInputMediaReady = true
      _webRtcLiveInputMediaRegisteredOnce = true
    }
    return true
  } catch (error) {
    reportError(error instanceof Error ? error.message : String(error))
    return false
  }
}

export function getWebRtcOutputMediaStream() { return _webRtcOutputMediaStream }

function _requestWebRtcSeedRotation() {
  if (!_webRtcPcId) return
  _webRtcSeedRotationPending = true
  _webRtcSeedRotationSerial++
  _webRtcSceneRequestVersion++
  _webRtcSceneDirty = true
  _scheduleWebRtcSceneSend()
}

function _clearWebRtcSeedRotationTimer() {
  if (_webRtcSeedRotationTimer !== undefined) {
    window.clearTimeout(_webRtcSeedRotationTimer)
    _webRtcSeedRotationTimer = undefined
  }
}

function _scheduleWebRtcSeedRotationTimer() {
  _clearWebRtcSeedRotationTimer()
  if (!_webRtcPcId || !$stream.get().isStreaming) return
  const sc = $scene.get()
  if (sc.seedRotationMode !== 'interval') return
  const interval = Math.max(1, Math.round(sc.seedRotationIntervalMs || 1000))
  const lastAt = _webRtcSeedRotationLastAt > 0 ? _webRtcSeedRotationLastAt : performance.now()
  const delay = Math.max(0, interval - (performance.now() - lastAt))
  _webRtcSeedRotationTimer = window.setTimeout(() => {
    _webRtcSeedRotationTimer = undefined
    _maybeRequestWebRtcSeedRotation()
  }, delay)
}

function _maybeRequestWebRtcSeedRotation() {
  if (!_webRtcPcId || !$stream.get().isStreaming) return
  const sc = $scene.get()
  if (sc.seedRotationMode === 'frame') {
    _requestWebRtcSeedRotation()
    return
  }
  if (sc.seedRotationMode !== 'interval') {
    _clearWebRtcSeedRotationTimer()
    _webRtcSeedRotationLastAt = 0
    return
  }
  const interval = Math.max(1, Math.round(sc.seedRotationIntervalMs || 1000))
  const now = performance.now()
  if (_webRtcSeedRotationLastAt <= 0 || now - _webRtcSeedRotationLastAt >= interval) {
    _requestWebRtcSeedRotation()
    return
  }
  _scheduleWebRtcSeedRotationTimer()
}

function _scheduleWebRtcSceneSend() {
  if (!_webRtcInputReady || !_webRtcSceneDirty) return
  if (_webRtcSceneScheduled) return
  _webRtcSceneScheduled = true
  queueMicrotask(() => {
    _webRtcSceneScheduled = false
    void _sendSceneRtc()
  })
}

function _flushWebRtcDirtyScene() {
  if (_webRtcSceneDirty) {
    _scheduleWebRtcSceneSend()
  }
}

function _handleWebRtcTransportBroken(message: string) {
  if (_stoppingWebRtc || !_sessionSocket) return
  reportError(message)
  $stream.setKey('status', 'error')
  stopWebRtc()
}

function _packBinaryEnvelope(header: Record<string, unknown>, payload: ArrayBuffer): ArrayBuffer {
  const headerBytes = new TextEncoder().encode(JSON.stringify(header))
  const out = new Uint8Array(4 + headerBytes.byteLength + payload.byteLength)
  new DataView(out.buffer).setUint32(0, headerBytes.byteLength, false)
  out.set(headerBytes, 4)
  out.set(new Uint8Array(payload), 4 + headerBytes.byteLength)
  return out.buffer
}

function _resourceName(value: string) {
  return value.replace(/[^a-zA-Z0-9._:-]+/g, '_').slice(0, 120) || 'resource'
}

function _dataUrlToBlob(dataUrl: string): Blob | undefined {
  const match = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(dataUrl)
  if (!match) return undefined
  const mime = match[1] || 'application/octet-stream'
  const body = match[2] ? atob(match[3]) : decodeURIComponent(match[3])
  const bytes = new Uint8Array(body.length)
  for (let index = 0; index < body.length; index++) bytes[index] = body.charCodeAt(index)
  return new Blob([bytes], { type: mime })
}

async function _stableJsonHash(value: unknown) {
  const bytes = new TextEncoder().encode(JSON.stringify(value))
  const digest = await crypto.subtle.digest('SHA-256', bytes)
  const prefix = new Uint8Array(digest, 0, 12)
  return Array.from(prefix, (byte) => byte.toString(16).padStart(2, '0')).join('')
}

async function _dataUrlHash(dataUrl: string) {
  const bytes = new TextEncoder().encode(dataUrl)
  const digest = await crypto.subtle.digest('SHA-256', bytes)
  const prefix = new Uint8Array(digest, 0, 12)
  return Array.from(prefix, (byte) => byte.toString(16).padStart(2, '0')).join('')
}

async function _bufferHash(buffer: ArrayBuffer) {
  const digest = await crypto.subtle.digest('SHA-256', buffer)
  const prefix = new Uint8Array(digest, 0, 12)
  return Array.from(prefix, (byte) => byte.toString(16).padStart(2, '0')).join('')
}

function _resourceId(hash: string) {
  return `res_${hash}`
}

function _sceneId(hash: string) {
  return `scene_${hash}`
}

function _blobArrayBuffer(blob: Blob): Promise<ArrayBuffer> {
  const maybeArrayBuffer = (blob as Blob & { arrayBuffer?: () => Promise<ArrayBuffer> }).arrayBuffer
  if (maybeArrayBuffer) return maybeArrayBuffer.call(blob)
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as ArrayBuffer)
    reader.onerror = () => reject(reader.error || new Error('Failed to read blob'))
    reader.readAsArrayBuffer(blob)
  })
}

async function _extractResource(resources: Map<string, SceneResource>, name: string, value: unknown) {
  if (typeof value !== 'string' || !value.startsWith('data:')) return ''
  const debugName = _resourceName(name)
  const cached = _resourceByDataUrl.get(value)
  if (cached) {
    _resourceByDataUrl.delete(value)
    _resourceByDataUrl.set(value, cached)
    resources.set(cached.id, { ...cached, debugName })
    return cached.id
  }
  const hash = await _dataUrlHash(value)
  const blob = _dataUrlToBlob(value)
  if (!blob) return ''
  const id = _resourceId(hash)
  const resource = { id, debugName, blob, mime: blob.type || 'application/octet-stream' }
  _resourceByDataUrl.set(value, resource)
  while (_resourceByDataUrl.size > _RESOURCE_DATA_URL_CACHE_LIMIT) {
    const oldest = _resourceByDataUrl.keys().next().value
    if (oldest === undefined) break
    _resourceByDataUrl.delete(oldest)
  }
  resources.set(id, resource)
  return id
}

async function _extractBlobResource(resources: Map<string, SceneResource>, name: string, value: unknown) {
  if (!(value instanceof Blob)) return ''
  const debugName = _resourceName(name)
  const buffer = await value.arrayBuffer()
  const hash = await _bufferHash(buffer)
  const id = _resourceId(hash)
  resources.set(id, { id, debugName, blob: value, mime: value.type || 'application/octet-stream' })
  return id
}

async function _splitFullFrameIntoScene(packet: Record<string, unknown>) {
  const resources = new Map<string, SceneResource>()
  const imageRef = await _extractResource(resources, 'input.image', packet.image)
  const maskRef = await _extractResource(resources, 'input.mask', packet.mask)
  const signals: Record<string, unknown>[] = []
  if (imageRef) signals.push({ id: 'input.image', type: 'rgba', role: 'input', resource_ref: imageRef })
  if (maskRef) signals.push({ id: 'input.mask', type: 'denoise_mask', role: 'input_mask', resource_ref: maskRef })
  const settings: Record<string, unknown> = { ...packet }
  delete settings.image
  delete settings.mask
  delete settings.layer_conditions
  delete settings.client_frame_id
  delete settings.client_input_id

  const layerConditions = Array.isArray(packet.layer_conditions) ? packet.layer_conditions : []
  const normalizedConditions = []
  for (const [index, raw] of layerConditions.entries()) {
    const cond = { ...(raw as Record<string, unknown>) }
    const layerId = _resourceName(String(cond.layer_id || `layer_${index}`))
    const regionId = _resourceName(String(cond.region_id || cond.name || `region_${index}`))
    const prefix = `layer.${layerId}.${regionId}`
    const imageName = await _extractBlobResource(resources, `${prefix}.image`, cond.image)
      || await _extractResource(resources, `${prefix}.image`, cond.image)
    if (imageName) {
      cond.image_ref = imageName
      signals.push({ id: `${prefix}.image`, type: 'rgba', channel: 'color', layer_id: layerId, region_id: regionId, resource_ref: imageName })
    }
    delete cond.image
    for (const [source, target, type, channel] of [
      ['denoise_mask', 'denoise_mask_ref_name', 'denoise_mask', 'denoise'],
      ['prompt_mask', 'prompt_mask_ref_name', 'prompt_mask', 'prompt'],
      ['cfg_mask', 'cfg_mask_ref_name', 'cfg_mask', 'cfg'],
      ['color_mask', 'color_mask_ref_name', 'rgba_mask', 'color'],
      ['color_image', 'color_image_ref_name', 'rgba', 'color_image'],
      ['controlnet_image', 'controlnet_image_ref_name', 'controlnet_image', 'controlnet'],
    ] as const) {
      const resourceName = await _extractBlobResource(resources, `${prefix}.${source}`, cond[source])
        || await _extractResource(resources, `${prefix}.${source}`, cond[source])
      if (resourceName) {
        cond[target] = resourceName
        signals.push({ id: `${prefix}.${source}`, type, channel, layer_id: layerId, region_id: regionId, resource_ref: resourceName })
      }
      delete cond[source]
    }
    normalizedConditions.push(cond)
  }

  const scene = {
    protocol: 'rtd.scene.v1',
    input: { image_ref: imageRef, mask_ref: maskRef },
    settings,
    layer_conditions: normalizedConditions,
    signals,
  } as Record<string, unknown>
  scene.id = _sceneId(await _stableJsonHash(scene))

  return {
    resources,
    scene,
  }
}

function _layerConditionKey(cond: Record<string, unknown>) {
  return `${String(cond.layer_id || '')}:${String(cond.region_id || cond.name || '')}`
}

async function _layerConditionsMetadataPatch(scene: Record<string, unknown>) {
  const current = Array.isArray(scene.layer_conditions) ? scene.layer_conditions as Record<string, unknown>[] : []
  const currentByKey = new Map(current.map((cond) => [_layerConditionKey(cond), cond]))
  const metadata = await _exportLayerConditionsMetadata()
  if (!Array.isArray(metadata) || metadata.length !== current.length) return undefined
  const next = []
  for (const raw of metadata) {
    const previous = currentByKey.get(_layerConditionKey(raw))
    if (!previous) return undefined
    next.push({
      ...previous,
      ...raw,
      image_ref: previous.image_ref,
      denoise_mask_ref_name: previous.denoise_mask_ref_name,
      prompt_mask_ref_name: previous.prompt_mask_ref_name,
      cfg_mask_ref_name: previous.cfg_mask_ref_name,
      color_mask_ref_name: previous.color_mask_ref_name,
      color_image_ref_name: previous.color_image_ref_name,
      controlnet_image_ref_name: previous.controlnet_image_ref_name,
    })
  }
  return next
}

async function _sendRtcResource(resource: SceneResource): Promise<boolean> {
  if (_webRtcSentResourceIds.has(resource.id)) return true
  try {
    const channel = _rtcResourceChannelOpen() ? _webRtcResourceChannel : _sessionSocket
    if (!channel || !(_rtcResourceChannelOpen() || channel.readyState === WebSocket.OPEN)) return false
    if (channel.bufferedAmount > _RTC_RESOURCE_BUFFER_HIGH_BYTES) {
      _webRtcSceneDirty = true
      _webRtcWaitingForResourceBackpressure = true
      return false
    }
    const payload = await _blobArrayBuffer(resource.blob)
    const requestId = `res_${Date.now().toString(36)}_${Math.random().toString(36).slice(2)}`
    await _sendRtcResourcePayload(channel, resource, requestId, payload)
    _webRtcSentResourceIds.add(resource.id)
    _webRtcWaitingForResourceBackpressure = false
    return true
  } catch (error) {
    console.warn('RTC resource send failed; will retry latest scene', error)
    _webRtcSceneDirty = true
    return false
  }
}

async function _sendRtcResourcePayload(channel: WebSocket | RTCDataChannel, resource: SceneResource, requestId: string, payload: ArrayBuffer) {
  const sendEnvelope = (header: Record<string, unknown>, body: ArrayBuffer) => {
    if (!((channel instanceof WebSocket && channel.readyState === WebSocket.OPEN) || (!(channel instanceof WebSocket) && channel.readyState === 'open'))) throw new Error(`render session resource channel is not open`)
    channel.send(_packBinaryEnvelope(header, body))
  }
  if (payload.byteLength <= _RTC_RESOURCE_CHUNK_BYTES) {
    sendEnvelope({ type: 'resource', request_id: requestId, id: resource.id, name: resource.debugName, mime: resource.mime }, payload)
    return
  }
  const total = Math.ceil(payload.byteLength / _RTC_RESOURCE_CHUNK_BYTES)
  for (let index = 0; index < total; index++) {
    const start = index * _RTC_RESOURCE_CHUNK_BYTES
    const chunk = payload.slice(start, Math.min(payload.byteLength, start + _RTC_RESOURCE_CHUNK_BYTES))
    sendEnvelope({
      type: 'resource_chunk',
      request_id: requestId,
      id: resource.id,
      name: resource.debugName,
      mime: resource.mime,
      index,
      total,
    }, chunk)
  }
}

function _handleResourceChannelMessage(event: MessageEvent) {
  if (typeof event.data !== 'string') return
  const msg = JSON.parse(event.data) as { type?: string; request_id?: string; message?: string }
  if (msg.type === 'error') {
    reportError(msg.message || 'RTC resource channel error')
    return
  }
}

// ── Realtime display FPS — updated once per second, not on every frame ────────

function _updateDisplayFps() {
  const now = performance.now()
  if (_fpsWindowStart === 0) _fpsWindowStart = now
  _fpsCount++
  if (now - _fpsWindowStart >= 1000) {
    $stream.setKey('displayFps', Math.round(_fpsCount * 1000 / (now - _fpsWindowStart)))
    _fpsCount = 0
    _fpsWindowStart = now
  }
}

// ── Inpaint WebSocket ────────────────────────────────────────────────────────

export function startInpaintStream() {
  const ws = openInpaintSocket()
  _socket = ws
  _awaitingFrame = false
  _awaitingFrameSentMs = 0
  _inpaintSeq = 0
  _inpaintLatestInputSeq = 1
  _inpaintAwaitingSeq = 0
  _inpaintSendRequested = true
  $stream.setKey('status', 'connecting')

  ws.onopen = () => {
    $stream.setKey('isStreaming', true)
    $stream.setKey('status', 'streaming')
    _scheduleInpaintFrameSend(0)
    if ($stream.get().scenePanelTab !== 'streamdiffusion') return
    const loop = () => {
      void _sendInpaintFrame()
      if ($stream.get().isStreaming) _streamRaf = window.requestAnimationFrame(loop)
    }
    _streamRaf = window.requestAnimationFrame(loop)
  }

  ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      // Binary frame — caller handles drawing; just signal completion
      _awaitingFrame = false
      _awaitingFrameSentMs = 0
      _inpaintAwaitingSeq = 0
      _updateDisplayFps()
      return
    }
    const msg = JSON.parse(event.data) as StreamMessage
    const responseSeq = typeof msg.client_frame_id === 'number' ? msg.client_frame_id : 0
    const responseInputSeq = typeof msg.client_input_id === 'number' ? msg.client_input_id : responseSeq
    const stale = responseInputSeq > 0 && responseInputSeq < _inpaintLatestInputSeq
    if (msg.error) {
      _awaitingFrame = false
      _awaitingFrameSentMs = 0
      _inpaintAwaitingSeq = 0
      reportError(msg.error)
      $stream.setKey('status', 'error')
      $scene.setKey('selectedModel', _lastWorkingModel)
      $scene.setKey('selectedLoras', [..._lastWorkingLoras])
      return
    }
    if (responseSeq === 0 || responseSeq >= _inpaintAwaitingSeq) {
      _awaitingFrame = false
      _awaitingFrameSentMs = 0
      _inpaintAwaitingSeq = 0
    }
    if (!stale) {
      if (msg.image) {
        $stream.setKey('outputImage', msg.image)
        $stream.setKey('outputImageNonce', ($stream.get().outputImageNonce || 0) + 1)
      }
      $stream.setKey('fps', msg.fps)
      $stream.setKey('latency', msg.latency_ms)
      $stream.setKey('backendMode', msg.mode)
      _lastWorkingModel = $scene.get().selectedModel
      _lastWorkingLoras = [...$scene.get().selectedLoras]
      if (msg.debug_channels) _applyDebugChannels(msg.debug_channels)
    }
    if (_inpaintSendRequested && $stream.get().isStreaming) _scheduleInpaintFrameSend(0)
  }

  ws.onclose = () => {
    _awaitingFrame = false
    _awaitingFrameSentMs = 0
    _inpaintLatestInputSeq = 0
    _inpaintAwaitingSeq = 0
    _inpaintSendRequested = false
    if (_inpaintRetryTimer !== undefined) { window.clearTimeout(_inpaintRetryTimer); _inpaintRetryTimer = undefined }
    $stream.setKey('isStreaming', false)
    $stream.setKey('status', 'offline')
    if (_streamTimer) { window.clearInterval(_streamTimer); _streamTimer = undefined }
    if (_streamRaf) { window.cancelAnimationFrame(_streamRaf); _streamRaf = undefined }
  }

  ws.onerror = () => {
    _awaitingFrame = false
    _awaitingFrameSentMs = 0
    _inpaintAwaitingSeq = 0
    $stream.setKey('status', 'backend unavailable')
  }
}

const _AWAIT_FRAME_TIMEOUT_MS = 8_000
const _INPAINT_STAGING_SEND_INTERVAL_MS = 120
const _INPAINT_MAX_BUFFERED_BYTES = 1_000_000
const _RTC_RESOURCE_CHUNK_BYTES = 64 * 1024
const _RTC_RESOURCE_BUFFER_HIGH_BYTES = 1_000_000
const _RESOURCE_DATA_URL_CACHE_LIMIT = 32
const _CFG_MASK_F32_MIME = 'application/x-rtd-mask-f32'
const _CFG_MASK_F32_MAGIC = 'RTF1'
const _CFG_MASK_F32_MAX = 30

async function _sendInpaintFrame() {
  if (_sendingInpaintFrame) return  // prevent concurrent canvas exports
  const state = $stream.get()
  const streamMode = state.scenePanelTab === 'streamdiffusion'
  const now = performance.now()
  // Watchdog: if server never replied within timeout, unblock so frames resume
  if (!streamMode && _awaitingFrame && _awaitingFrameSentMs > 0 &&
      performance.now() - _awaitingFrameSentMs > _AWAIT_FRAME_TIMEOUT_MS) {
    _awaitingFrame = false
    _awaitingFrameSentMs = 0
    _inpaintAwaitingSeq = 0
  }
  if (!_socket || _socket.readyState !== WebSocket.OPEN) return
  if (streamMode && _socket.bufferedAmount > 0) return
  if (!streamMode && _socket.bufferedAmount > _INPAINT_MAX_BUFFERED_BYTES) { _scheduleInpaintFrameSend(50); return }
  if (!streamMode && !_inpaintSendRequested) return
  if (!streamMode && _awaitingFrame) {
    if (!_inpaintSendRequested) return
    if (now - _inpaintLastSendMs < _INPAINT_STAGING_SEND_INTERVAL_MS) {
      _scheduleInpaintFrameSend(_INPAINT_STAGING_SEND_INTERVAL_MS - (now - _inpaintLastSendMs))
      return
    }
  }
  // Cap at ~30 fps for stream mode — the 60 Hz RAF loop fires faster than the GPU can process
  if (streamMode && now - _inpaintLastSendMs < 33) return
  _sendingInpaintFrame = true
  try {
    const inputSeq = _inpaintLatestInputSeq
    const payload = await _exportFullFrame()
    if (!payload) { _awaitingFrame = false; _awaitingFrameSentMs = 0; return }
    const seq = ++_inpaintSeq
    if (streamMode || inputSeq === _inpaintLatestInputSeq) {
      _inpaintSendRequested = false
    }
    const packet = _fullFramePacket(payload)
    packet.client_frame_id = seq
    packet.client_input_id = streamMode ? seq : inputSeq
    const { scene, resources } = await _splitFullFrameIntoScene(packet)
    packet.scene_id = scene.id
    void _replaceSceneResourceDebugData(resources, scene)
    _inpaintLastSendMs = performance.now()
    if (!streamMode) {
      _awaitingFrame = true
      _awaitingFrameSentMs = _inpaintLastSendMs
      _inpaintAwaitingSeq = seq
    }
    _socket.send(JSON.stringify(packet))
  } finally {
    _sendingInpaintFrame = false
    if (_inpaintSendRequested && $stream.get().isStreaming) _scheduleInpaintFrameSend(0)
  }
}

// ── RTC streaming ─────────────────────────────────────────────────────────────

export async function startWebRtc() {
  if (_sessionSocket || _webRtcPcId) stopWebRtc()
  const lifecycleSeq = ++_webRtcLifecycleSeq
  _webRtcPcId = ''
  _webRtcLastScene = ''
  _webRtcSceneSeq = 0
  $stream.setKey('status', 'connecting')
  try {
    const ws = openRenderSessionSocket()
    _sessionSocket = ws
    ws.onopen = () => {
      if (lifecycleSeq !== _webRtcLifecycleSeq) return
      $stream.setKey('isStreaming', true)
      $stream.setKey('realtimeVideoActive', true)
      $stream.setKey('status', 'streaming')
      _webRtcSceneRequestVersion++
      _webRtcSceneDirty = true
    }
    ws.onmessage = (event) => {
      if (lifecycleSeq !== _webRtcLifecycleSeq) return
      if (event.data instanceof ArrayBuffer) return
      const msg = JSON.parse(String(event.data)) as Record<string, unknown>
      _handleRenderSessionMessage(msg)
    }
    ws.onerror = () => {
      if (lifecycleSeq !== _webRtcLifecycleSeq) return
      _handleWebRtcTransportBroken('render session WebSocket error')
    }
    ws.onclose = () => {
      if (lifecycleSeq !== _webRtcLifecycleSeq || _stoppingWebRtc) return
      _handleWebRtcTransportBroken('render session WebSocket closed')
    }
  } catch (err) {
    reportError(err instanceof Error ? err.message : String(err))
    $stream.setKey('status', 'error')
    stopWebRtc()
  }
}

function _handleRenderSessionMessage(msg: Record<string, unknown>) {
  const type = String(msg.type || '')
  if (type === 'hello') {
    _webRtcPcId = String(msg.session_id || '')
    console.info('[rtd transport]', msg)
    void _ensureWebRtcMediaPlane(_webRtcPcId)
    return
  }
  if (type === 'input_ready') {
    _webRtcInputReady = true
    _maybeRequestWebRtcSeedRotation()
    _flushWebRtcDirtyScene()
    return
  }
  if (type === 'admin_log') {
    console.info('[rtd transport]', msg)
    return
  }
  if (type === 'resource_ack') {
    _handleResourceChannelMessage({ data: JSON.stringify(msg) } as MessageEvent)
    return
  }
  if (type === 'scene_ack') return
  if (type === 'render_done') return
  if (type === 'debug') {
    const channels = msg.channels
    if (channels && typeof channels === 'object') _applyDebugChannels(channels as Record<string, string>)
    return
  }
  if (type === 'frame') {
    if (_webRtcOutputMediaReady) return
    const data = String(msg.data || '')
    if (data) {
      _updateDisplayFps()
      window.dispatchEvent(new CustomEvent('rtd:rtc-frame', { detail: data }))
    }
    return
  }
  if (type === 'status') {
    _handleRtcStatus(msg as Parameters<typeof _handleRtcStatus>[0])
    return
  }
  if (type === 'error') {
    reportError(String(msg.message || 'render session error'))
  }
}

function _handleRtcStatus(s: {
  phase: string; model: string; fps: number; error: string
  build_phase?: string; build_progress?: number; build_message?: string
  compile_requested?: boolean | string
  compile_active?: boolean | string
  compile_unet_compiled?: boolean | string
  compile_triton_available?: boolean | string
  compile_wrapper?: string
  compile_original?: string
}) {
  const compileLabel = _rtcCompileStatusLabel(s)
  if (s.phase === 'error') {
    _flushWebRtcDirtyScene()
    $stream.setKey('backendMode', `✗ ${s.error}${compileLabel}`)
  } else if (s.phase === 'loading') {
    $stream.setKey('backendMode', `⚙ ${s.build_message || `Loading ${s.model || 'model'}…`}${compileLabel}`)
  } else if (s.phase === 'ready') {
    const fps = Number(s.fps || 0)
    $stream.setKey('backendMode', `${fps > 0 ? `${s.model} – ${fps.toFixed(1)} FPS` : (s.model || 'streamdiffusion')}${compileLabel}`)
  }
  const bp = s.build_phase
  if (bp && bp !== 'idle') {
    const taskId = 'sys_stream_build'
    upsertTask({
      task_id: taskId,
      status: bp === 'error' ? 'error' : bp === 'ready' ? 'complete' : 'running',
      phase: bp,
      progress: s.build_progress ?? 0,
      message: s.build_message ?? bp,
      error: bp === 'error' ? (s.build_message ?? null) : null,
      result: undefined,
      layerId: '',
      label: 'Stream session',
      startedAt: Date.now(),
    })
  }
}

function _boolStatus(value: boolean | string | undefined): boolean {
  return value === true || value === 'true' || value === '1'
}

function _rtcCompileStatusLabel(s: {
  phase: string
  compile_requested?: boolean | string
  compile_active?: boolean | string
  compile_unet_compiled?: boolean | string
  compile_triton_available?: boolean | string
  compile_wrapper?: string
  compile_original?: string
}): string {
  const requested = _boolStatus(s.compile_requested)
  if (!requested) return ''
  const available = s.compile_triton_available === undefined || _boolStatus(s.compile_triton_available)
  if (!available) return ' | Triton unavailable'
  const active = _boolStatus(s.compile_active) && _boolStatus(s.compile_unet_compiled)
  if (active) {
    const wrapper = s.compile_wrapper ? ` ${s.compile_wrapper}` : ''
    return ` | Triton ON${wrapper}`
  }
  return s.phase === 'loading' ? ' | Triton compiling' : ' | Triton requested (eager)'
}

async function _requestWebRtcInputMediaFrame() {
  window.dispatchEvent(new CustomEvent('rtd:rtc-input-frame-request'))
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()))
}

function _applyLiveStreamFrameSeed(settings: Record<string, unknown>) {
  if ($stream.get().scenePanelTab !== 'streamdiffusion') return
  const serial = _webRtcSceneSeq + 1
  settings.seed_rotation_serial = serial
  settings.stream_noise_drift = 0.015
}

function _sceneEventsFromPatch(patch: Record<string, unknown>) {
  const events: Record<string, unknown>[] = []
  if (patch.input && typeof patch.input === 'object') {
    events.push({ type: 'input.update', input: patch.input })
  }
  if (patch.settings && typeof patch.settings === 'object') {
    for (const [key, value] of Object.entries(patch.settings as Record<string, unknown>)) {
      events.push(value === null
        ? { type: 'property.unset', path: ['settings', key] }
        : { type: 'property.set', path: ['settings', key], value })
    }
  }
  if (Array.isArray(patch.signals)) {
    for (const signal of patch.signals) {
      if (signal && typeof signal === 'object') events.push({ type: 'signal.update', signal })
    }
  }
  if (Array.isArray(patch.layer_conditions)) {
    events.push({ type: 'layer_conditions.replace', layer_conditions: patch.layer_conditions })
  }
  return events
}

async function _sendSceneRtc() {
  if (!_webRtcPcId) return
  const channel = _sessionSocket
  if (!channel || channel.readyState !== WebSocket.OPEN) return
  if (!_webRtcInputReady || !_webRtcSceneDirty) return
  if (_sendingWebRtcScene) { _webRtcSceneDirty = true; return }
  if (channel.bufferedAmount > _RTC_RESOURCE_BUFFER_HIGH_BYTES) {
    _webRtcSceneDirty = true
    _webRtcWaitingForResourceBackpressure = true
    return
  }
  _webRtcInputReady = false
  _webRtcSceneDirty = false
  _sendingWebRtcScene = true
  const sendVersion = _webRtcSceneRequestVersion
  let sentScene = false
  let sentPatch = false
  const markNewerEditForNextReady = () => {
    if (sendVersion !== _webRtcSceneRequestVersion) _webRtcSceneDirty = true
  }
  try {
    if (_webRtcLastSceneObject && _webRtcLayerConditionsDirty && !(_webRtcResourceRequestVersion > _webRtcResourceSentVersion)) {
      sentPatch = await _sendWebRtcSceneSettingsPatch(false, false, true)
      markNewerEditForNextReady()
      return
    }
    if (_webRtcLastSceneObject && _webRtcLiveInputMediaRegisteredOnce && !_webRtcLiveInputMediaReady) {
      _webRtcInputReady = true
      _webRtcSceneDirty = true
      window.dispatchEvent(new CustomEvent('rtd:rtc-media-plane-ready', { detail: { sessionId: _webRtcPcId, force: true } }))
      return
    }
    if (_webRtcLastSceneObject && _webRtcLiveInputMediaReady) {
      await _requestWebRtcInputMediaFrame()
      sentPatch = await _sendWebRtcSceneSettingsPatch(true, _webRtcResourceRequestVersion > _webRtcResourceSentVersion, _webRtcLayerConditionsDirty)
      markNewerEditForNextReady()
      return
    }
    const payload = await _exportFullFrame(_webRtcPcId)
    if (!payload) { _webRtcInputReady = true; return }
    markNewerEditForNextReady()
    const packet = _fullFramePacket(payload)
    const seedRotationMode = $scene.get().seedRotationMode
    if (_webRtcSeedRotationPending) {
      packet.seed_mode = $scene.get().seedMode
      packet.seed_rotation_serial = _webRtcSeedRotationSerial
      _webRtcSeedRotationLastAt = performance.now()
      _webRtcSeedRotationPending = false
      _scheduleWebRtcSeedRotationTimer()
    } else if (seedRotationMode !== 'off') {
      packet.seed_mode = 'fixed'
      packet.seed_rotation_serial = _webRtcSeedRotationSerial
    }
    const { scene, resources } = await _splitFullFrameIntoScene(packet)
    void _replaceSceneResourceDebugData(resources, scene)
    markNewerEditForNextReady()
    for (const resource of resources.values()) {
      markNewerEditForNextReady()
      const resourceSent = await _sendRtcResource(resource)
      if (!resourceSent) { _webRtcSceneDirty = true; _webRtcInputReady = true; return }
    }
    markNewerEditForNextReady()
    const sceneId = String(scene.id || '')
    if (sceneId && sceneId === _webRtcLastScene) { _webRtcInputReady = true; return }
    _webRtcLastScene = sceneId || JSON.stringify(scene)
    _webRtcLastSceneObject = scene
    channel.send(JSON.stringify({ type: 'scene', id: sceneId, seq: ++_webRtcSceneSeq, scene }))
    _webRtcResourceSentVersion = _webRtcResourceRequestVersion
    sentScene = true
  } catch (err) {
    _webRtcInputReady = true
    _webRtcSceneDirty = true
    reportError(err instanceof Error ? err.message : String(err))
  } finally {
    _sendingWebRtcScene = false
    if (!sentScene && !sentPatch) _webRtcInputReady = true
    if (_webRtcSceneDirty && _webRtcInputReady && !_webRtcWaitingForResourceBackpressure) {
      _scheduleWebRtcSceneSend()
    }
  }
}

async function _sendWebRtcSceneSettingsPatch(forceInputRevision = false, refreshResources = false, includeLayerConditions = false): Promise<boolean> {
  const channel = _sessionSocket
  if (!_webRtcPcId || !channel || channel.readyState !== WebSocket.OPEN) return false
  const scene = _webRtcLastSceneObject
  if (!scene) return false
  try {
    if (refreshResources && (forceInputRevision || includeLayerConditions)) {
      const payload = await _exportFullFrame(_webRtcPcId)
      if (!payload) return false
      const packet = _fullFramePacket(payload)
      if (_webRtcSeedRotationPending) {
        packet.seed_mode = $scene.get().seedMode
        packet.seed_rotation_serial = _webRtcSeedRotationSerial
        _webRtcSeedRotationLastAt = performance.now()
        _webRtcSeedRotationPending = false
        _scheduleWebRtcSeedRotationTimer()
      }
      const { scene: nextScene, resources } = await _splitFullFrameIntoScene(packet)
      void _replaceSceneResourceDebugData(resources, nextScene)
      for (const resource of resources.values()) {
        const resourceSent = await _sendRtcResource(resource)
        if (!resourceSent) return false
      }
      const settings = (nextScene.settings && typeof nextScene.settings === 'object') ? nextScene.settings as Record<string, unknown> : {}
      if (forceInputRevision) {
        settings.input_revision = _webRtcSceneSeq + 1
        _applyLiveStreamFrameSeed(settings)
      }
      scene.input = nextScene.input
      scene.settings = { ...settings }
      scene.layer_conditions = nextScene.layer_conditions
      scene.signals = nextScene.signals
      scene.id = undefined
      _webRtcLastScene = JSON.stringify(scene)
      const seq = ++_webRtcSceneSeq
      const patch = {
        input: nextScene.input,
        settings,
        signals: nextScene.signals,
        layer_conditions: nextScene.layer_conditions,
      } as Record<string, unknown>
      channel.send(JSON.stringify({
        type: 'scene_patch',
        seq,
        events: _sceneEventsFromPatch(patch),
        patch,
      }))
      _webRtcResourceSentVersion = _webRtcResourceRequestVersion
      _webRtcLayerConditionsDirty = false
      return true
    }
    const settings = await _exportSceneSettings()
    if (_webRtcSeedRotationPending) {
      settings.seed_mode = $scene.get().seedMode
      settings.seed_rotation_serial = _webRtcSeedRotationSerial
      _webRtcSeedRotationLastAt = performance.now()
      _webRtcSeedRotationPending = false
      _scheduleWebRtcSeedRotationTimer()
    }
    const currentSettings = (scene.settings && typeof scene.settings === 'object') ? scene.settings as Record<string, unknown> : {}
    if (forceInputRevision) {
      settings.input_revision = _webRtcSceneSeq + 1
      _applyLiveStreamFrameSeed(settings)
    }
    const layerConditions = includeLayerConditions ? await _layerConditionsMetadataPatch(scene) : undefined
    if (includeLayerConditions && !layerConditions) {
      return _sendWebRtcSceneSettingsPatch(forceInputRevision, true, true)
    }
    if (!forceInputRevision && !layerConditions && JSON.stringify(settings) === JSON.stringify(currentSettings)) return false
    scene.settings = { ...settings }
    if (layerConditions) scene.layer_conditions = layerConditions
    scene.id = undefined
    _webRtcLastScene = JSON.stringify(scene)
    const seq = ++_webRtcSceneSeq
    const patch: Record<string, unknown> = { settings }
    if (layerConditions) patch.layer_conditions = layerConditions
    channel.send(JSON.stringify({
      type: 'scene_patch',
      seq,
      events: _sceneEventsFromPatch(patch),
      patch,
    }))
    if (layerConditions) _webRtcLayerConditionsDirty = false
    return true
  } catch (err) {
    reportError(err instanceof Error ? err.message : String(err))
    return false
  }
}

export function stopWebRtc() {
  if (_stoppingWebRtc) return
  _stoppingWebRtc = true
  _webRtcLifecycleSeq++
  if (_frameLoopRaf) { window.cancelAnimationFrame(_frameLoopRaf); _frameLoopRaf = undefined }
  if (_webRtcMediaPeer) { _webRtcMediaPeer.close(); _webRtcMediaPeer = undefined }
  _webRtcMediaPeerSessionId = ''
  _webRtcMediaNegotiation = undefined
  if (_webRtcResourceChannel) { _webRtcResourceChannel.close(); _webRtcResourceChannel = undefined }
  const socket = _sessionSocket
  if (socket) {
    socket.onopen = null
    socket.onmessage = null
    socket.onerror = null
    socket.onclose = null
    if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: 'stop' }))
    socket.close()
  }
  _sessionSocket = undefined
  _webRtcPcId = ''
  _webRtcLastScene = ''
  _webRtcLastSceneObject = undefined
  _sendingWebRtcScene = false
  _webRtcSceneDirty = false
  _webRtcLayerConditionsDirty = false
  _webRtcSceneSeq = 0
  _webRtcSceneRequestVersion = 0
  _webRtcResourceRequestVersion = 0
  _webRtcResourceSentVersion = 0
  _webRtcSceneScheduled = false
  _webRtcInputReady = false
  _clearWebRtcSeedRotationTimer()
  _webRtcSeedRotationLastAt = 0
  _webRtcSeedRotationSerial = 0
  _webRtcSeedRotationPending = false
  _webRtcWaitingForResourceBackpressure = false
  _webRtcLiveInputMediaReady = false
  _webRtcOutputMediaReady = false
  _webRtcOutputMediaStream = undefined
  _webRtcLiveInputMediaRegisteredOnce = false
  _webRtcSentResourceIds.clear()
  _resourceByDataUrl.clear()
  _clearSceneResourceDebugData()
  // Drop the per-session blob ID cache so a fresh session re-uploads any
  // painted mask channels (the server-side store is per-session too).
  clearMaskBlobCache()
  $stream.setKey('isStreaming', false)
  $stream.setKey('realtimeVideoActive', false)
  $stream.setKey('status', 'offline')
  $stream.setKey('displayFps', 0)
  _fpsCount = 0
  _fpsWindowStart = 0
  _stoppingWebRtc = false
}

export function stopStream() {
  if (_socket) { _socket.close(); _socket = undefined }
  stopWebRtc()
}

// ── Debug channels ────────────────────────────────────────────────────────────

const _debugChannelData = new Map<string, string>()
const _sceneResourceDebugData = new Map<string, { resourceId: string; debugName: string; type: string; channelName: string }>()
const _debugSurfaces = new Map<string, DebugSurface>()
const _sceneResourceDebugMedia = new Map<string, { canvas?: HTMLCanvasElement; video?: HTMLVideoElement; url?: string }>()
let _sceneResourceDebugSeq = 0

export function getDebugChannelData() { return _debugChannelData }
export function getSceneResourceDebugData() {
  return new Map([..._sceneResourceDebugData.values()].map((entry) => [entry.channelName, `stream:${entry.resourceId}`]))
}
export function getDebugSurfaces() { return new Map(_debugSurfaces) }

export function registerDebugCanvasSurface(id: string, name: string, canvas: HTMLCanvasElement) {
  const existing = _debugSurfaces.get(id)
  if (existing?.kind === 'canvas' && existing.canvas === canvas && existing.name === name) return
  _debugSurfaces.set(id, { id, name, kind: 'canvas', canvas })
  window.dispatchEvent(new CustomEvent('rtd:debug-surfaces-update'))
}

export function registerDebugStreamSurface(id: string, name: string, stream: MediaStream) {
  const existing = _debugSurfaces.get(id)
  if (existing?.kind === 'stream' && existing.stream === stream && existing.name === name) return
  _debugSurfaces.set(id, { id, name, kind: 'stream', stream })
  window.dispatchEvent(new CustomEvent('rtd:debug-surfaces-update'))
}

function _registerMediaTrackDebugChannel(streamId: string, label: string) {
  const channelName = `input/resource/${streamId} (${_resourceName(label)})`
  _debugChannelData.set(channelName, `stream:${streamId}`)
  const current = $stream.get().debugChannelNames.filter((name) => name !== channelName)
  $stream.setKey('debugChannelNames', [channelName, ...current])
}

export function unregisterDebugSurface(id: string) {
  if (_debugSurfaces.delete(id)) window.dispatchEvent(new CustomEvent('rtd:debug-surfaces-update'))
}

async function _replaceSceneResourceDebugData(resources: Map<string, SceneResource>, scene?: Record<string, unknown>) {
  const seq = ++_sceneResourceDebugSeq
  const imageVideoEntries = new Map<string, SceneResource>()
  for (const resource of resources.values()) {
    if (!resource.mime.startsWith('image/') && !resource.mime.startsWith('video/')) continue
    imageVideoEntries.set(resource.id, resource)
  }
  const nextEntries = new Map<string, { resource: SceneResource; channelName: string }>()
  const signalLabels = _sceneSignalDebugEntries(scene, resources)
  if (signalLabels.size) {
    for (const [key, entry] of signalLabels) nextEntries.set(key, entry)
  } else {
    for (const resource of imageVideoEntries.values()) {
      const channelName = _legacyResourceDebugChannelName(resource)
      if (channelName) nextEntries.set(resource.id, { resource, channelName })
    }
  }
  if (seq !== _sceneResourceDebugSeq) return
  let changed = false
  const oldChannelNames = new Set([..._sceneResourceDebugData.values()].map((entry) => entry.channelName))
  const oldResourceIds = new Set([..._sceneResourceDebugData.values()].map((entry) => entry.resourceId))
  const nextNames = new Set(nextEntries.keys())
  for (const key of _sceneResourceDebugData.keys()) {
    if (!nextNames.has(key)) {
      const oldEntry = _sceneResourceDebugData.get(key)
      if (oldEntry) _debugChannelData.delete(oldEntry.channelName)
      _sceneResourceDebugData.delete(key)
      changed = true
    }
  }
  for (const [key, entry] of nextEntries.entries()) {
    const { resource, channelName } = entry
    const existing = _sceneResourceDebugData.get(key)
    if (existing) {
      if (existing.debugName !== resource.debugName || existing.channelName !== channelName || existing.resourceId !== resource.id) {
        _debugChannelData.delete(existing.channelName)
        _sceneResourceDebugData.set(key, { resourceId: resource.id, debugName: resource.debugName, type: resource.mime, channelName })
        _debugChannelData.set(channelName, `stream:${resource.id}`)
        changed = true
      }
      continue
    }
    _sceneResourceDebugData.set(key, { resourceId: resource.id, debugName: resource.debugName, type: resource.mime, channelName })
    _debugChannelData.set(channelName, `stream:${resource.id}`)
    changed = true
  }
  const nextResourceIds = new Set([...nextEntries.values()].map((entry) => entry.resource.id))
  for (const oldResourceId of oldResourceIds) {
    if (!nextResourceIds.has(oldResourceId)) {
      _deleteSceneResourceDebugMedia(oldResourceId)
      unregisterDebugSurface(oldResourceId)
    }
  }
  const resourcesToRegister = [...new Set([...nextEntries.values()].map((entry) => entry.resource.id))]
    .map((id) => resources.get(id))
    .filter((resource): resource is SceneResource => Boolean(resource))
  await Promise.all(resourcesToRegister.map((resource) => _registerSceneResourceDebugStream(resource)))
  if (changed) {
    const resourceChannels = [..._sceneResourceDebugData.values()].map((entry) => entry.channelName)
    const current = $stream.get().debugChannelNames.filter((name) => !oldChannelNames.has(name) && !name.startsWith('input/resource/'))
    $stream.setKey('debugChannelNames', [...new Set([...resourceChannels, ...current])])
    window.dispatchEvent(new CustomEvent('rtd:scene-resources-update'))
  }
}

function _legacyResourceDebugChannelName(resource: SceneResource) {
  if (resource.debugName.startsWith('input.')) return ''
  return `input/resource/${resource.id} (${resource.debugName})`
}

function _sceneSignalDebugEntries(scene: Record<string, unknown> | undefined, resources: Map<string, SceneResource>) {
  const out = new Map<string, { resource: SceneResource; channelName: string }>()
  const signals = Array.isArray(scene?.signals) ? scene.signals as Record<string, unknown>[] : []
  const promptIndexByLayer = new Map<string, number>()
  const layerNameByRaw = new Map<string, string>()
  for (const signal of signals) {
    const resourceId = String(signal.resource_ref || '')
    const resource = resources.get(resourceId)
    if (!resource) continue
    const label = _signalDebugChannelName(signal, promptIndexByLayer, layerNameByRaw)
    if (!label) continue
    out.set(String(signal.id || label), { resource, channelName: label })
  }
  return out
}

function _signalDebugChannelName(signal: Record<string, unknown>, promptIndexByLayer: Map<string, number>, layerNameByRaw: Map<string, string>) {
  const role = String(signal.role || '')
  if (role === 'input' || role === 'input_mask') return ''
  const rawLayer = _resourceName(String(signal.layer_id || 'layer'))
  const layer = _signalLayerLabel(rawLayer, layerNameByRaw)
  const region = _resourceName(String(signal.region_id || ''))
  const type = String(signal.type || '')
  const channel = String(signal.channel || '')
  if (type === 'rgba' && channel === 'color') return `${layer}/RGBA`
  if (type === 'rgba' && channel === 'color_image') return `${layer}/RGBAImage`
  if (type === 'rgba_mask' || (type === 'prompt_mask' && region.startsWith('inherited'))) return `${layer}/RGBAPrompt`
  if (type === 'cfg_mask' || channel === 'cfg') return `${layer}/CFG`
  if (type === 'denoise_mask' || channel === 'denoise') return `${layer}/Denoise`
  if (type === 'prompt_mask' || channel === 'prompt') {
    const nextIndex = (promptIndexByLayer.get(layer) || 0) + 1
    promptIndexByLayer.set(layer, nextIndex)
    return `${layer}/Prompt${nextIndex}`
  }
  if (type === 'controlnet_image') return `${layer}/ControlNet`
  return ''
}

function _signalLayerLabel(rawLayer: string, layerNameByRaw: Map<string, string>) {
  if (layerNameByRaw.has(rawLayer)) return String(layerNameByRaw.get(rawLayer))
  const normalized = rawLayer.toLowerCase()
  const isUuidLike = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(rawLayer)
  const isHashLike = /^[0-9a-f]{12,}$/i.test(rawLayer)
  const isGeneratedLayerName = /^layer[_-]?\d+$/i.test(rawLayer)
  if (isUuidLike || isHashLike || isGeneratedLayerName || normalized === 'layer') {
    const fallback = `layer${layerNameByRaw.size + 1}`
    layerNameByRaw.set(rawLayer, fallback)
    return fallback
  }
  layerNameByRaw.set(rawLayer, rawLayer)
  return rawLayer
}

async function _registerSceneResourceDebugStream(resource: SceneResource) {
  if (_sceneResourceDebugMedia.has(resource.id) && _debugSurfaces.has(resource.id)) return
  if (resource.mime === _CFG_MASK_F32_MIME) {
    await _registerCfgMaskDebugStream(resource)
    return
  }
  if (resource.mime.startsWith('video/')) {
    await _registerVideoResourceDebugStream(resource)
    return
  }
  const canvas = document.createElement('canvas')
  if (typeof canvas.captureStream !== 'function') return
  const context = canvas.getContext('2d', { alpha: true })
  if (!context) return
  const bitmap = await _imageBitmapFromBlob(resource.blob)
  if (!bitmap) return
  canvas.width = Math.max(1, bitmap.width)
  canvas.height = Math.max(1, bitmap.height)
  context.clearRect(0, 0, canvas.width, canvas.height)
  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height)
  bitmap.close?.()
  const stream = canvas.captureStream(0)
  _sceneResourceDebugMedia.set(resource.id, { canvas })
  registerDebugStreamSurface(resource.id, resource.debugName, stream)
  ;(stream.getVideoTracks()[0] as (MediaStreamTrack & { requestFrame?: () => void }) | undefined)?.requestFrame?.()
}

async function _registerCfgMaskDebugStream(resource: SceneResource) {
  const payload = new Uint8Array(await _blobArrayBuffer(resource.blob))
  if (payload.byteLength < 12) return
  const magic = String.fromCharCode(payload[0], payload[1], payload[2], payload[3])
  if (magic !== _CFG_MASK_F32_MAGIC) return
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength)
  const width = Math.max(1, view.getUint32(4, true))
  const height = Math.max(1, view.getUint32(8, true))
  const expected = 12 + width * height * 4
  if (payload.byteLength < expected) return
  const canvas = document.createElement('canvas')
  if (typeof canvas.captureStream !== 'function') return
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext('2d', { alpha: true })
  if (!context) return
  const image = context.createImageData(width, height)
  const data = image.data
  let offset = 12
  for (let i = 0; i < width * height; i++) {
    const value = view.getFloat32(offset, true)
    offset += 4
    const normalized = Math.max(0, Math.min(1, value / _CFG_MASK_F32_MAX))
    const gray = Math.round(normalized * 255)
    const pixel = i * 4
    data[pixel] = gray
    data[pixel + 1] = gray
    data[pixel + 2] = gray
    data[pixel + 3] = 255
  }
  context.putImageData(image, 0, 0)
  const stream = canvas.captureStream(0)
  _sceneResourceDebugMedia.set(resource.id, { canvas })
  registerDebugStreamSurface(resource.id, resource.debugName, stream)
  ;(stream.getVideoTracks()[0] as (MediaStreamTrack & { requestFrame?: () => void }) | undefined)?.requestFrame?.()
}

async function _registerVideoResourceDebugStream(resource: SceneResource) {
  const video = document.createElement('video')
  const capture = (video as HTMLVideoElement & { captureStream?: () => MediaStream }).captureStream
  if (typeof capture !== 'function') return
  const url = URL.createObjectURL(resource.blob)
  video.muted = true
  video.loop = true
  video.playsInline = true
  video.src = url
  await video.play().catch(() => undefined)
  _sceneResourceDebugMedia.set(resource.id, { video, url })
  registerDebugStreamSurface(resource.id, resource.debugName, capture.call(video))
}

function _deleteSceneResourceDebugMedia(id: string) {
  const item = _sceneResourceDebugMedia.get(id)
  if (!item) return
  for (const track of item.video?.srcObject instanceof MediaStream ? item.video.srcObject.getTracks() : []) track.stop()
  if (item.video) item.video.srcObject = null
  if (item.url) URL.revokeObjectURL(item.url)
  _sceneResourceDebugMedia.delete(id)
}

async function _imageBitmapFromBlob(blob: Blob): Promise<ImageBitmap | undefined> {
  if (typeof createImageBitmap === 'function') {
    try { return await createImageBitmap(blob) } catch { return undefined }
  }
  if (typeof Image === 'undefined' || typeof URL === 'undefined') return undefined
  return new Promise((resolve) => {
    const image = new Image()
    const url = URL.createObjectURL(blob)
    image.onload = () => {
      const canvas = document.createElement('canvas')
      canvas.width = Math.max(1, image.naturalWidth || image.width)
      canvas.height = Math.max(1, image.naturalHeight || image.height)
      const context = canvas.getContext('2d')
      context?.drawImage(image, 0, 0)
      URL.revokeObjectURL(url)
      if (typeof createImageBitmap !== 'function') { resolve(undefined); return }
      createImageBitmap(canvas).then(resolve, () => resolve(undefined))
    }
    image.onerror = () => { URL.revokeObjectURL(url); resolve(undefined) }
    image.src = url
  })
}

function _clearSceneResourceDebugData() {
  _sceneResourceDebugSeq++
  for (const id of _sceneResourceDebugMedia.keys()) {
    unregisterDebugSurface(id)
    _deleteSceneResourceDebugMedia(id)
  }
  const oldChannelNames = new Set([..._sceneResourceDebugData.values()].map((entry) => entry.channelName))
  for (const entry of _sceneResourceDebugData.values()) _debugChannelData.delete(entry.channelName)
  _sceneResourceDebugData.clear()
  $stream.setKey('debugChannelNames', $stream.get().debugChannelNames.filter((name) => !oldChannelNames.has(name) && !name.startsWith('input/resource/')))
  window.dispatchEvent(new CustomEvent('rtd:scene-resources-update'))
}

function _applyDebugChannels(incoming: Record<string, string>) {
  const names = Object.keys(incoming)
  for (const [name, val] of Object.entries(incoming)) {
    _debugChannelData.set(name, val.startsWith('data:image/') ? `stream:${name}` : val)
  }
  const current = $stream.get().debugChannelNames
  const structureChanged = names.some((n) => !current.includes(n))
  if (structureChanged) $stream.setKey('debugChannelNames', [...new Set([...current, ...names])])
  window.dispatchEvent(new CustomEvent('rtd:debug-update', { detail: incoming }))
}
