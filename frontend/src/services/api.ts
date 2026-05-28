import type {
  AssetCatalog,
  GpuCatalog,
  HealthState,
  RendererCapabilities,
  SourceCatalog,
  VideoCapabilities,
} from '../types'

// ── Connection helpers ──────────────────────────────────────────────────────

function backendHost(): string {
  const queryHost = new URLSearchParams(window.location.search).get('backendHost')
  const configuredHost = import.meta.env.VITE_BACKEND_HOST?.trim()
  const host = queryHost || configuredHost || window.location.hostname || '127.0.0.1'
  return host === '0.0.0.0' ? '127.0.0.1' : host
}

function backendPort(): string {
  const queryPort = new URLSearchParams(window.location.search).get('backendPort')
  const configuredPort = import.meta.env.VITE_BACKEND_PORT?.trim()
  const port = queryPort || configuredPort || '8000'
  return /^\d{2,5}$/.test(port) ? port : '8000'
}

function backendProtocol(): string {
  const queryProtocol = new URLSearchParams(window.location.search).get('backendProtocol')
  const configuredProtocol = import.meta.env.VITE_BACKEND_PROTOCOL?.trim()
  const protocol = queryProtocol || configuredProtocol || window.location.protocol.replace(':', '') || 'http'
  return protocol === 'https' ? 'https' : 'http'
}

export function backendUrl(path: string, params?: Record<string, string>): string {
  const url = new URL(path, `${backendProtocol()}://${backendHost()}:${backendPort()}`)
  for (const [key, value] of Object.entries(params ?? {})) url.searchParams.set(key, value)
  return url.toString()
}

export function backendWsUrl(path: string): string {
  const protocol = backendProtocol() === 'https' ? 'wss' : 'ws'
  return `${protocol}://${backendHost()}:${backendPort()}${path}`
}

// ── Assets / catalog ────────────────────────────────────────────────────────

export async function fetchAssets(): Promise<AssetCatalog> {
  const response = await fetch(backendUrl('/assets'))
  if (!response.ok) throw new Error(await response.text())
  return response.json() as Promise<AssetCatalog>
}

export async function fetchHealth(): Promise<HealthState> {
  const response = await fetch(backendUrl('/health'))
  if (!response.ok) throw new Error(await response.text())
  return response.json() as Promise<HealthState>
}

export async function fetchControlNetModels(): Promise<string[]> {
  const response = await fetch(backendUrl('/controlnet/models'))
  if (!response.ok) throw new Error(await response.text())
  return response.json() as Promise<string[]>
}

export async function fetchGpuDevices(): Promise<GpuCatalog> {
  const response = await fetch(backendUrl('/system/gpus'))
  if (!response.ok) throw new Error(await response.text())
  return response.json() as Promise<GpuCatalog>
}

export async function fetchRendererCapabilities(): Promise<RendererCapabilities> {
  const response = await fetch(backendUrl('/renderers/capabilities'))
  if (!response.ok) throw new Error(await response.text())
  return response.json() as Promise<RendererCapabilities>
}

export async function fetchVideoCapabilities(): Promise<VideoCapabilities> {
  const response = await fetch(backendUrl('/video/capabilities'))
  if (!response.ok) throw new Error(await response.text())
  return response.json() as Promise<VideoCapabilities>
}

// ── Sources ─────────────────────────────────────────────────────────────────

export async function fetchSources(root: string): Promise<SourceCatalog> {
  const response = await fetch(backendUrl('/sources', { root }))
  if (!response.ok) throw new Error(await response.text())
  return response.json() as Promise<SourceCatalog>
}

export async function pickSourceRoot(): Promise<{ root: string }> {
  const response = await fetch(backendUrl('/sources/pick-root'), { method: 'POST' })
  if (!response.ok) throw new Error(await response.text())
  return response.json() as Promise<{ root: string }>
}

export function sourceImageUrl(root: string, rel: string): string {
  return backendUrl('/sources/image', { root, rel })
}

// ── Layer generation ─────────────────────────────────────────────────────────

export interface LayerTaskRequest {
  image: string
  mask: string
  prompt: string
  negative_prompt: string
  model_path?: string
  lora_paths?: string[]
  width: number
  height: number
  strength: number
  cfg: number
  steps: number
  sampler?: string
  scheduler?: string
  seed?: number
  variations: number
  device?: string
  layer_conditions?: unknown
  pipeline_nodes?: unknown
  transparent_background?: boolean
  transparent_background_method?: string
  transparent_background_mode?: string
  transparent_background_color?: string
  transparent_background_tolerance?: number
  transparent_alpha_blur?: number
  transparent_alpha_threshold?: number
}

export async function postLayerTask(body: string): Promise<{ task_id: string }> {
  const response = await fetch(backendUrl('/layer/tasks'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body,
  })
  if (!response.ok) throw new Error(await response.text())
  return response.json() as Promise<{ task_id: string }>
}

export function watchLayerTask(taskId: string): WebSocket {
  return new WebSocket(backendWsUrl(`/ws/tasks/${taskId}`))
}

// ── System tasks ─────────────────────────────────────────────────────────────

export async function fetchSystemTasks(): Promise<import('../types').SystemTask[]> {
  const response = await fetch(backendUrl('/system/tasks'))
  if (!response.ok) throw new Error(await response.text())
  return response.json() as Promise<import('../types').SystemTask[]>
}

// ── Inpaint WebSocket ────────────────────────────────────────────────────────

export function openInpaintSocket(): WebSocket {
  const ws = new WebSocket(backendWsUrl('/ws/inpaint'))
  ws.binaryType = 'arraybuffer'
  return ws
}

export function openRenderSessionSocket(): WebSocket {
  const ws = new WebSocket(backendWsUrl('/api/rtc/ws'))
  ws.binaryType = 'arraybuffer'
  return ws
}

// ── RTC streaming ────────────────────────────────────────────────────────────

export async function rtcStart(): Promise<{ pc_id: string }> {
  const resp = await fetch(backendUrl('/api/rtc/start'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: '{}',
  })
  if (!resp.ok) throw new Error(`RTC start failed: ${resp.status} ${await resp.text()}`)
  return resp.json() as Promise<{ pc_id: string }>
}

export async function rtcOffer(
  offer: RTCSessionDescriptionInit,
  sessionId?: string,
  outputTransport: 'video' | 'image' = 'video',
): Promise<{ pc_id: string; answer: RTCSessionDescriptionInit; output_transport?: 'video' | 'image' }> {
  const resp = await fetch(backendUrl('/api/rtc/offer'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ offer, session_id: sessionId, output_transport: outputTransport }),
  })
  if (!resp.ok) throw new Error(`RTC offer failed: ${resp.status} ${await resp.text()}`)
  return resp.json() as Promise<{ pc_id: string; answer: RTCSessionDescriptionInit; output_transport?: 'video' | 'image' }>
}

export function rtcStreamUrl(pcId: string): string {
  return backendUrl(`/api/rtc/${pcId}/stream`)
}

export async function rtcPostFrame(pcId: string, body: string, signal: AbortSignal): Promise<void> {
  await fetch(backendUrl(`/api/rtc/${pcId}/frame`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body,
    signal,
  })
}

export async function rtcPostSettings(pcId: string, payload: string): Promise<void> {
  await fetch(backendUrl(`/api/rtc/${pcId}/settings`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: payload,
  })
}

export function rtcDeletePeer(pcId: string): void {
  void fetch(backendUrl(`/api/rtc/peers/${pcId}`), { method: 'DELETE' }).catch(() => {})
}

/**
 * Legacy fallback: upload a raw PNG (Blob) to the session's binary store over
 * HTTP. The primary stream path installs a WebRTC resources data-channel
 * uploader via mask-blob.service.
 *
 * Use this for per-channel masks (denoise / prompt / cfg / color) instead of
 * inlining base64 in ``layer_conditions`` — saves the ~33 % base64 overhead
 * on every settings update. The returned ID is referenced via
 * ``layer_conditions[*]._mask_ref`` on subsequent settings messages.
 */
export async function rtcUploadBlob(pcId: string, blob: Blob): Promise<{ id: string }> {
  const resp = await fetch(backendUrl(`/api/rtc/${pcId}/blob`), {
    method: 'POST',
    headers: { 'Content-Type': blob.type || 'application/octet-stream' },
    body: blob,
  })
  if (!resp.ok) throw new Error(`RTC blob upload failed: ${resp.status} ${await resp.text()}`)
  return resp.json() as Promise<{ id: string }>
}

// ── Motion / video generation ────────────────────────────────────────────────

export interface MotionTaskRequest {
  name: string
  source_image: string
  prompt: string
  arrival_prompt?: string
  negative_prompt: string
  model_path?: string
  device?: string
  lora_paths?: string[]
  model: string
  motion: string
  region: string
  fps: number
  duration_seconds: number
  loop: string
  intensity?: number
  width: number
  height: number
  arrival_steps?: number
  arrival_strength?: number
  arrival_cfg?: number
  arrival_sampler?: string
  arrival_scheduler?: string
}

export async function postMotionTask(request: MotionTaskRequest): Promise<{ task_id: string }> {
  const response = await fetch(backendUrl('/motion/tasks'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!response.ok) throw new Error(await response.text())
  return response.json() as Promise<{ task_id: string }>
}

export function watchMotionTask(taskId: string): WebSocket {
  return new WebSocket(backendWsUrl(`/ws/motion/${taskId}`))
}

export async function fetchMotionFrameBlob(frameUrl: string): Promise<Blob | null> {
  const response = await fetch(backendUrl(frameUrl, { t: String(Date.now()) }))
  if (!response.ok) return null
  return response.blob()
}

export async function fetchMotionVideoSegment(videoUrl: string): Promise<ArrayBuffer | null> {
  const response = await fetch(backendUrl(videoUrl, { t: String(Date.now()) }))
  if (!response.ok) return null
  return response.arrayBuffer()
}

// ── Generic image fetch ──────────────────────────────────────────────────────

export async function fetchImageBlob(url: string): Promise<Blob> {
  const response = await fetch(url, { cache: 'no-store' })
  if (!response.ok) throw new Error(`${response.status} ${response.statusText || 'image request failed'}`)
  const blob = await response.blob()
  if (!blob.type.startsWith('image/')) throw new Error(`Not an image response (${blob.type || 'unknown type'})`)
  return blob
}

export async function fetchArbitraryBlob(url: string): Promise<Blob> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Could not fetch ${url}: ${response.status}`)
  return response.blob()
}
