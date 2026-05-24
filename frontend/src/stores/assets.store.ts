import { map } from 'nanostores'
import type { AssetCatalog, GpuDevice, RendererCapabilities, VideoCapabilities } from '../types'

export interface CnModelEntry {
  name: string
  path: string
  type: string
  family: string
}

export interface AssetsState {
  catalog: AssetCatalog
  gpuDevices: GpuDevice[]
  rendererCapabilities: RendererCapabilities | undefined
  videoCapabilities: VideoCapabilities | undefined
  cnLocalModels: CnModelEntry[]
  isLoading: boolean
}

export const $assets = map<AssetsState>({
  catalog: { models: [], loras: [], video_models: [] },
  gpuDevices: [{ id: '', name: 'Backend default', kind: 'default', available: true }],
  rendererCapabilities: undefined,
  videoCapabilities: undefined,
  cnLocalModels: [],
  isLoading: false,
})
