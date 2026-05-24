import { $layer } from '../stores/layer.store'
import { $scene } from '../stores/scene.store'
import { $stream } from '../stores/stream.store'
import { reportError } from '../stores/ui.store'
import { postLayerTask, watchLayerTask as openLayerTaskSocket, fetchSystemTasks } from './api'
import type { GenerationTask, LayerGeneratePayload, LayerItem, LayerTaskProgress, SystemTask } from '../types'

// ── Task helpers ────────────────────────────────────────────────────────────

export function upsertTask(task: GenerationTask) {
  const tasks = $layer.get().generationTasks
  const existing = tasks.find((t) => t.task_id === task.task_id)
  $layer.setKey(
    'generationTasks',
    existing
      ? tasks.map((t) => (t.task_id === task.task_id ? { ...t, ...task } : t))
      : [task, ...tasks].slice(0, 12),
  )
}

function replaceTaskId(localId: string, serverId: string) {
  $layer.setKey(
    'generationTasks',
    $layer.get().generationTasks.map((t) =>
      t.task_id === localId ? { ...t, task_id: serverId, message: 'Task accepted by server' } : t,
    ),
  )
}

// ── System task polling ──────────────────────────────────────────────────────

const _sysTaskSeenAt = new Map<string, number>()
let _pollTimer: number | undefined

function _systemTaskLabel(taskId: string) {
  if (taskId.startsWith('sys_stream')) return 'Stream session'
  return 'System'
}

export function startSystemTaskPolling() {
  if (_pollTimer) return
  _pollTimer = window.setInterval(pollSystemTasks, 3000)
}

export function stopSystemTaskPolling() {
  if (_pollTimer) { window.clearInterval(_pollTimer); _pollTimer = undefined }
}

export async function pollSystemTasks(): Promise<void> {
  try {
    const tasks: SystemTask[] = await fetchSystemTasks()
    const now = Date.now()
    const activeIds = new Set(tasks.map((t) => t.task_id))
    const current = $layer.get().generationTasks.filter((t) => !t.task_id.startsWith('sys_') || activeIds.has(t.task_id))
    $layer.setKey('generationTasks', current)
    for (const task of tasks) {
      const seenAt = _sysTaskSeenAt.get(task.task_id) ?? now
      _sysTaskSeenAt.set(task.task_id, seenAt)
      upsertTask({
        task_id: task.task_id,
        layerId: '',
        label: _systemTaskLabel(task.task_id),
        status: task.status,
        phase: task.phase,
        progress: task.progress,
        message: task.message,
        error: task.error ?? null,
        startedAt: seenAt,
      })
    }
  } catch { /* poll failures are silent */ }
}

// ── Layer task watching ──────────────────────────────────────────────────────

function applyLayerTaskResult(layerId: string, result: LayerGeneratePayload, complete: boolean) {
  const existing = $layer.get().generatedLayers.filter((v) => v.layerId !== layerId)
  const variations = result.variations.map((v) => ({ ...v, layerId, selected: false }))
  $layer.setKey('generatedLayers', [...existing, ...variations])
  $stream.setKey('backendMode', `${result.mode} | layer task ${Math.round(result.latency_ms)} ms${complete ? '' : ' partial'}`)
}

function watchTask(taskId: string, layerId: string): Promise<void> {
  return new Promise<void>((resolve) => {
    let done = false
    const finish = () => { if (!done) { done = true; resolve() } }
    const ws = openLayerTaskSocket(taskId)
    ws.onmessage = (event) => {
      const progress = JSON.parse(event.data) as LayerTaskProgress
      const label = $layer.get().generationTasks.find((t) => t.task_id === taskId)?.label ?? 'Layer generation'
      const startedAt = $layer.get().generationTasks.find((t) => t.task_id === taskId)?.startedAt ?? Date.now()
      upsertTask({ ...progress, layerId, label, startedAt })
      if (progress.result) applyLayerTaskResult(layerId, progress.result, progress.status === 'complete')
      if (progress.status === 'complete') { ws.close(); finish() }
      if (progress.status === 'error') { reportError(progress.error || progress.message); ws.close(); finish() }
    }
    ws.onerror = () => {
      const msg = 'Task monitor connection failed.'
      reportError(msg)
      upsertTask({ task_id: taskId, layerId, label: 'Layer generation', status: 'error', phase: 'monitor', progress: 1, message: msg, error: msg, startedAt: Date.now() })
      finish()
    }
    ws.onclose = () => finish()
  })
}

// ── Layer generation ─────────────────────────────────────────────────────────

export async function generateLayer(layer: LayerItem, canvasImage: string, maskImage: string): Promise<void> {
  const state = $layer.get()
  const scene = $scene.get()
  const localTaskId = `local_${Date.now()}`

  $layer.setKey('isGeneratingLayer', true)
  upsertTask({
    task_id: localTaskId,
    layerId: layer.id,
    label: `${state.variationCount} variation${state.variationCount === 1 ? '' : 's'}`,
    status: 'queued',
    phase: 'queued',
    progress: 0,
    message: 'Submitting…',
    error: null,
    startedAt: Date.now(),
  })

  try {
    const seed = state.generationSeed.trim() ? parseInt(state.generationSeed, 10) : undefined
    const body = JSON.stringify({
      image: canvasImage,
      mask: maskImage,
      prompt: layer.preset.prompt || scene.prompt,
      negative_prompt: layer.preset.negativePrompt || scene.negativePrompt,
      model_path: layer.preset.modelPath || scene.selectedModel || undefined,
      lora_paths: layer.preset.loraPaths.length ? layer.preset.loraPaths : [...scene.selectedLoras],
      width: state.generationWidth,
      height: state.generationHeight,
      strength: layer.preset.strength,
      cfg: layer.preset.layerCfg ?? scene.cfg,
      steps: layer.preset.layerSteps ?? scene.steps,
      sampler: layer.preset.layerSampler ?? undefined,
      scheduler: layer.preset.layerScheduler ?? undefined,
      seed: Number.isFinite(seed) && (seed ?? 0) >= 0 ? seed : undefined,
      variations: state.variationCount,
      device: state.device || undefined,
      transparent_background: state.transparentBackground,
      transparent_background_method: state.transparentMethod,
      transparent_background_mode: state.transparentMode,
      transparent_background_color: state.transparentColor,
      transparent_background_tolerance: state.transparentTolerance,
      transparent_alpha_blur: state.transparentAlphaBlur,
      transparent_alpha_threshold: state.transparentAlphaThreshold,
    })
    const { task_id: taskId } = await postLayerTask(body)
    replaceTaskId(localTaskId, taskId)
    await watchTask(taskId, layer.id)
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    reportError(msg)
    upsertTask({
      task_id: localTaskId,
      layerId: layer.id,
      label: `${state.variationCount} variation${state.variationCount === 1 ? '' : 's'}`,
      status: 'error',
      phase: 'error',
      progress: 1,
      message: msg,
      error: msg,
      startedAt: Date.now(),
    })
  } finally {
    $layer.setKey('isGeneratingLayer', false)
  }
}
