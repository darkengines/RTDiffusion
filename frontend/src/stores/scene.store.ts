import { map } from 'nanostores'
import type { LatentReuseNoiseMode, RenderStrategy, SeedMode, SeedRotationMode } from '../types'

export interface SceneState {
  prompt: string
  negativePrompt: string
  strength: number
  cfg: number
  steps: number
  sampler: string
  scheduler: string
  seed: string
  seedMode: SeedMode
  seedRotationMode: SeedRotationMode
  seedRotationIntervalMs: number
  selectedModel: string
  selectedLoras: string[]
  stageWidth: number
  stageHeight: number
  renderStrategy: RenderStrategy
  tileOverlap: number
  tileDivisions: number
  layerBboxPadding: number
  reuseLatent: boolean
  reuseLatentDenoise: number
  reuseLatentNoise: number
  reuseLatentNoiseMode: LatentReuseNoiseMode
  transparentBackground: boolean
  transparentMode: string
  transparentColor: string
  transparentTolerance: number
  transparentAlphaBlur: number
  transparentAlphaThreshold: number
}

export const $scene = map<SceneState>({
  prompt: '',
  negativePrompt: '',
  strength: 1.00,
  cfg: 1.0,
  steps: 1,
  sampler: 'euler',
  scheduler: 'simple',
  seed: '',
  seedMode: 'fixed',
  seedRotationMode: 'off',
  seedRotationIntervalMs: 1000,
  selectedModel: '',
  selectedLoras: [],
  stageWidth: 832,
  stageHeight: 1216,
  renderStrategy: 'single',
  tileOverlap: 128,
  tileDivisions: 2,
  layerBboxPadding: 64,
  reuseLatent: false,
  reuseLatentDenoise: 0.24,
  reuseLatentNoise: 0,
  reuseLatentNoiseMode: 'none',
  transparentBackground: false,
  transparentMode: 'border',
  transparentColor: '#ffffff',
  transparentTolerance: 34,
  transparentAlphaBlur: 1.5,
  transparentAlphaThreshold: 10,
})
