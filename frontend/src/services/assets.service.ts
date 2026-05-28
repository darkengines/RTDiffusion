import { $assets } from '../stores/assets.store'
import { $scene } from '../stores/scene.store'
import {
  fetchAssets,
  fetchControlNetModels,
  fetchGpuDevices,
  fetchHealth,
  fetchRendererCapabilities,
  fetchVideoCapabilities,
} from './api'

export async function loadAllAssets(): Promise<void> {
  $assets.setKey('isLoading', true)
  try {
    const [catalog, health, cnModels] = await Promise.all([
      fetchAssets(),
      fetchHealth(),
      fetchControlNetModels().catch(() => [] as string[]),
    ])
    catalog.video_models ??= []
    $assets.setKey('catalog', catalog)
    const { $stream } = await import('../stores/stream.store')
    $stream.setKey('backendMode', health.mode)
    const scene = $scene.get()
    const preferred = catalog.models.find((m) => m.path === scene.selectedModel)?.path
      ?? catalog.models.find((m) => m.preferred)?.path
      ?? catalog.models[0]?.path
      ?? ''
    $scene.setKey('selectedModel', preferred)
    $assets.setKey('cnLocalModels', cnModels as unknown as Array<{ name: string; path: string; type: string; family: string }>)
  } catch {
    $assets.setKey('catalog', { models: [], loras: [], video_models: [] })
  } finally {
    $assets.setKey('isLoading', false)
  }
}

export async function loadGpuDevices(): Promise<void> {
  try {
    const catalog = await fetchGpuDevices()
    const devices = [
      { id: '', name: `Backend default (${catalog.default})`, kind: 'default', available: true },
      ...catalog.devices,
    ]
    $assets.setKey('gpuDevices', devices)
    const validIds = new Set(devices.map((d) => d.id))
    const { $stream } = await import('../stores/stream.store')
    const { $layer } = await import('../stores/layer.store')
    const { $motion } = await import('../stores/motion.store')
    if (!validIds.has($stream.get().realtimeDevice)) $stream.setKey('realtimeDevice', '')
    if (!validIds.has($layer.get().device)) $layer.setKey('device', '')
    if (!validIds.has($motion.get().device)) $motion.setKey('device', '')
  } catch {
    $assets.setKey('gpuDevices', [{ id: '', name: 'Backend default', kind: 'default', available: true }])
  }
}

export async function loadRendererCapabilities(): Promise<void> {
  try {
    $assets.setKey('rendererCapabilities', await fetchRendererCapabilities())
  } catch {
    $assets.setKey('rendererCapabilities', undefined)
  }
}

export async function loadVideoCapabilities(): Promise<void> {
  try {
    $assets.setKey('videoCapabilities', await fetchVideoCapabilities())
  } catch {
    $assets.setKey('videoCapabilities', undefined)
  }
}
