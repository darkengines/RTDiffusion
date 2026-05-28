import { $motion } from '../stores/motion.store'
import { $scene } from '../stores/scene.store'
import { $assets } from '../stores/assets.store'
import { $stream } from '../stores/stream.store'
import { reportError } from '../stores/ui.store'
import { loadVideoCapabilities } from './assets.service'
import {
  postMotionTask,
  watchMotionTask as openMotionSocket,
  fetchMotionFrameBlob,
  fetchMotionVideoSegment,
} from './api'
import type { MotionClip, MotionTaskProgress } from '../types'

// ── Non-reactive lifecycle state ──────────────────────────────────────────────

let _motionTaskSocket: WebSocket | undefined
let _motionMediaSource: MediaSource | undefined
let _motionSourceBuffer: SourceBuffer | undefined
let _motionSegmentQueue: ArrayBuffer[] = []
let _motionStreamOpen: Promise<boolean> | undefined
let _motionStreamObjectUrl = ''
let _motionRunId = 0

// ── Video capability guard ────────────────────────────────────────────────────

async function _ensureVideoBackendReady(model: string): Promise<boolean> {
  if (!$assets.get().videoCapabilities) await loadVideoCapabilities()
  const cap = $assets.get().videoCapabilities?.models?.[model]
  if (!cap || cap.configured) return true
  reportError(cap.message)
  return false
}

// ── MediaSource streaming ─────────────────────────────────────────────────────

async function _openVideoStream(): Promise<boolean> {
  if (!('MediaSource' in window)) return false
  const candidates = ['video/mp4; codecs="avc1.42E01E"', 'video/mp4; codecs="avc1.42E01F"', 'video/mp4']
  const mimeType = candidates.find((m) => MediaSource.isTypeSupported(m))
  if (!mimeType) return false
  const ms = new MediaSource()
  _motionMediaSource = ms
  _motionSegmentQueue = []
  _motionStreamObjectUrl = URL.createObjectURL(ms)
  $motion.setKey('activeVideo', _motionStreamObjectUrl)
  $motion.setKey('activeIsStream', true)
  $motion.setKey('streamHasSegments', false)
  _motionStreamOpen = new Promise<boolean>((resolve) => {
    ms.addEventListener('sourceopen', () => {
      if (_motionMediaSource !== ms) return resolve(false)
      try {
        _motionSourceBuffer = ms.addSourceBuffer(mimeType)
        _motionSourceBuffer.addEventListener('updateend', _flushQueue)
        resolve(true)
      } catch { resolve(false) }
    })
    ms.addEventListener('error', () => resolve(false))
  })
  return _motionStreamOpen
}

function _flushQueue() {
  if (!_motionSourceBuffer || !_motionMediaSource || _motionMediaSource.readyState !== 'open' || _motionSourceBuffer.updating) return
  const next = _motionSegmentQueue.shift()
  if (!next) return
  try { _motionSourceBuffer.appendBuffer(next) } catch { _motionSegmentQueue = [] }
}

async function _appendVideoSegment(clip: MotionClip): Promise<boolean> {
  if (!clip.video_url || !_motionMediaSource || !_motionSourceBuffer) return false
  if (_motionStreamOpen && !(await _motionStreamOpen)) return false
  try {
    const buf = await fetchMotionVideoSegment(clip.video_url)
    if (!buf) return false
    _motionSegmentQueue.push(buf)
    $motion.setKey('streamHasSegments', true)
    _flushQueue()
    return true
  } catch { return false }
}

function _disposeVideoStream() {
  _motionSegmentQueue = []
  $motion.setKey('streamHasSegments', false)
  _motionSourceBuffer = undefined
  _motionMediaSource = undefined
  _motionStreamOpen = undefined
  if (_motionStreamObjectUrl) { URL.revokeObjectURL(_motionStreamObjectUrl); _motionStreamObjectUrl = '' }
}

function _finishVideoStream() {
  if (!_motionMediaSource || _motionMediaSource.readyState !== 'open' || _motionSourceBuffer?.updating || _motionSegmentQueue.length) return
  try { _motionMediaSource.endOfStream() } catch { /* already closed */ }
}

// ── Motion task watching ──────────────────────────────────────────────────────

function _watchMotionTask(taskId: string): Promise<MotionClip | undefined> {
  return new Promise<MotionClip | undefined>((resolve, reject) => {
    const ws = openMotionSocket(taskId)
    _motionTaskSocket = ws
    let result: MotionClip | undefined
    const finish = () => resolve(result)
    ws.onmessage = (event) => {
      const progress = JSON.parse(event.data) as MotionTaskProgress
      $motion.setKey('task', progress)
      if (progress.status === 'complete') { result = progress.result ?? undefined; ws.close(); finish() }
      if (progress.status === 'error') { reportError(progress.error ?? progress.message); ws.close(); finish() }
    }
    ws.onerror = () => reject(new Error('Motion task socket error'))
    ws.onclose = () => finish()
  })
}

// ── Last frame extraction ─────────────────────────────────────────────────────

async function _clipLastFrameDataUrl(clip: MotionClip): Promise<string> {
  const frame = clip.frames.at(-1)
  if (!frame) return ''
  const blob = await fetchMotionFrameBlob(frame.url)
  if (!blob) return ''
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result || ''))
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(blob)
  })
}

// ── Play motion result ────────────────────────────────────────────────────────

async function _playResult(clip: MotionClip, useStream: boolean): Promise<void> {
  if (useStream) {
    await _appendVideoSegment(clip)
  } else {
    if (clip.video_url) $motion.setKey('activeVideo', clip.video_url)
    else { const frame = clip.frames.at(-1); if (frame) $motion.setKey('activeFrame', frame.url) }
  }
}

// ── Public API ────────────────────────────────────────────────────────────────

export function clearMotionPlayback() {
  _motionTaskSocket?.close()
  _motionTaskSocket = undefined
  _disposeVideoStream()
  $motion.setKey('activeFrame', '')
  $motion.setKey('activeVideo', '')
  $motion.setKey('activeIsStream', false)
}

export function stopMotionGeneration() {
  _motionRunId += 1
  _motionTaskSocket?.close()
  _motionTaskSocket = undefined
  $motion.setKey('isGenerating', false)
  _finishVideoStream()
  const chunks = $motion.get().chunkIndex
  $stream.setKey('status', chunks ? `stopped after ${chunks} chunk(s)` : 'offline')
}

export async function startMotionGeneration(sourceImage: string, modelOverride?: string, modelPathOverride?: string): Promise<void> {
  const st = $motion.get()
  const scene = $scene.get()
  const model = modelOverride ?? st.model
  if (!sourceImage) { reportError('Create or generate an image before starting the video model.'); return }
  if (!(await _ensureVideoBackendReady(model))) return

  const runId = ++_motionRunId
  const singleClip = false
  clearMotionPlayback()
  $motion.setKey('activeFrame', sourceImage)
  const useStream = singleClip ? false : await _openVideoStream()

  $motion.setKey('isGenerating', true)
  $motion.setKey('chunkIndex', 0)
  $motion.setKey('task', { task_id: 'pending', status: 'queued', phase: 'queued', progress: 0, message: `Waiting for ${model} worker` })

  let currentImage = sourceImage
  try {
    while ($motion.get().isGenerating && _motionRunId === runId) {
      $motion.setKey('chunkIndex', $motion.get().chunkIndex + 1)
      $motion.setKey('task', { task_id: 'pending', status: 'queued', phase: 'queued', progress: 0, message: `Queueing ${model} chunk ${$motion.get().chunkIndex}` })

      const { task_id: taskId } = await postMotionTask({
        name: `${model} stream ${$motion.get().chunkIndex}`,
        source_image: currentImage,
        prompt: st.prompt,
        arrival_prompt: st.arrivalPrompt || undefined,
        negative_prompt: st.negativePrompt,
        model_path: modelPathOverride || scene.selectedModel || undefined,
        device: st.device || undefined,
        lora_paths: [...scene.selectedLoras],
        model,
        motion: 'autoregressive',
        region: 'full',
        fps: st.fps,
        duration_seconds: st.duration,
        loop: 'none',
        intensity: st.intensity,
        width: scene.stageWidth,
        height: scene.stageHeight,
        arrival_steps: Math.max(1, Math.min(32, 24)),
        arrival_strength: Math.max(0, Math.min(0.95, scene.strength || 0.42)),
        arrival_cfg: Math.max(0, Math.min(30, scene.cfg || 4.5)),
      })
      const clip = await _watchMotionTask(taskId)
      if (!clip || !$motion.get().isGenerating || _motionRunId !== runId) break
      await _playResult(clip, useStream)
      if (singleClip) break
      currentImage = (await _clipLastFrameDataUrl(clip)) || currentImage
    }
  } catch (err) {
    reportError(err instanceof Error ? err.message : String(err))
  } finally {
    if (_motionRunId === runId) {
      $motion.setKey('isGenerating', false)
      _motionTaskSocket = undefined
    }
  }
}
