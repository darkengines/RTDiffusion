import { map } from 'nanostores'
import type {
  ActiveEntityTarget,
  ControlNetConfig,
  GeneratedLayer,
  GenerationTask,
  LayerItem,
  LayerPanelTab,
  RegionItem,
  RegionShape,
  SelectionRect,
} from '../types'

export interface LayerState {
  layers: LayerItem[]
  selectedLayerId: string
  selectedEntity: ActiveEntityTarget
  layerPanelTab: LayerPanelTab
  regions: RegionItem[]
  selectionRect: SelectionRect | undefined
  generatedLayers: GeneratedLayer[]
  isGeneratingLayer: boolean
  generatorActive: boolean
  variationCount: number
  generationWidth: number
  generationHeight: number
  generationSteps: number
  generationCfg: number
  generationSampler: string
  generationScheduler: string
  generationSeed: string
  device: string
  transparentBackground: boolean
  transparentMethod: string
  transparentMode: string
  transparentColor: string
  transparentTolerance: number
  transparentAlphaBlur: number
  transparentAlphaThreshold: number
  controlNetConfigs: Record<string, ControlNetConfig>
  maskNegated: boolean
  maskAbsolute: boolean
  regionShape: RegionShape
  regionName: string
  generationTasks: GenerationTask[]
}

export const $layer = map<LayerState>({
  layers: [],
  selectedLayerId: '',
  selectedEntity: 'scene',
  layerPanelTab: 'mask',
  regions: [],
  selectionRect: undefined,
  generatedLayers: [],
  isGeneratingLayer: false,
  generatorActive: true,
  variationCount: 4,
  generationWidth: 832,
  generationHeight: 1216,
  generationSteps: 24,
  generationCfg: 5,
  generationSampler: 'euler_ancestral',
  generationScheduler: 'simple',
  generationSeed: '',
  device: '',
  transparentBackground: true,
  transparentMethod: 'auto',
  transparentMode: 'border',
  transparentColor: '#ffffff',
  transparentTolerance: 34,
  transparentAlphaBlur: 1.5,
  transparentAlphaThreshold: 10,
  controlNetConfigs: {},
  maskNegated: false,
  maskAbsolute: false,
  regionShape: 'rope',
  regionName: '',
  generationTasks: [],
})
