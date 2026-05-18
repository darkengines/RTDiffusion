import { Canvas, FabricImage, FabricObject as FabricObjectClass, Path, PencilBrush, Point, Rect, Shadow, Textbox, type FabricObject } from 'fabric'
import { LitElement, css, html, type TemplateResult } from 'lit'
import { customElement, query, state } from 'lit/decorators.js'
import {
  ARCHIVE_ROOT_URL,
  CANVAS_CUSTOM_PROPERTIES,
  archivePathForResource,
  archiveUrlForResource,
  centeredResizeRect,
  isFlushableProjectStorageKey,
  isPaintChildObject,
  normalizeCanvasPaintChildren,
  resourceHashFromArchiveUrl,
  strokeParentLayerId,
  topLevelLayerObjects,
} from './project-state'

type AssetItem = { name: string; path: string; size: number; preferred: boolean }
type AssetCatalog = { models: AssetItem[]; video_models?: AssetItem[]; loras: AssetItem[] }
type HealthState = { status: string; mode: string }
type SourceImage = { name: string; rel: string; size: number; modified: number }
type SourceCatalog = { root: string; images: SourceImage[] }
type LayerConditionMode = 'mask' | 'add' | 'multiply' | 'override' | 'prompt_mix'
type ConditionAggregateOperator = 'replace' | 'add' | 'average' | 'multiply' | 'max'
type LayerPreset = {
  prompt: string
  negativePrompt: string
  modelPath: string
  loraPaths: string[]
  samplingEnabled: boolean
  conditionMode: LayerConditionMode
  conditionWeight: number
  maskOperator: ConditionAggregateOperator
  denoiseOperator: ConditionAggregateOperator
  conditionInnerBlur: number
  conditionOuterBlur: number
  conditionNegate: boolean
  strength: number
  cfg: number
  steps: number
  sampler: string
  scheduler: string
  schedule: RegionSchedule
  scheduleStart: number
  scheduleEnd: number
  autoTag: boolean
  autoTagThreshold: number
  autoTagRefreshFrames: number
}
type LayerItem = {
  id: string
  name: string
  preview: string
  visible: boolean
  opacity: number
  selected: boolean
  preset: LayerPreset
}
type SizePreset = { label: string; width: number; height: number }
type ToolMode = 'brush' | 'eraser' | 'select' | 'region' | 'text' | 'line' | 'rect' | 'fill'
type RegionPoint = { x: number; y: number }
type SelectionRect = { x: number; y: number; width: number; height: number; inverted: boolean; shape?: RegionShape; points?: RegionPoint[] }
type RegionTarget = 'scene' | 'layer'
type RegionShape = 'box' | 'rope' | 'polygon'
type RegionSchedule = 'auto' | 'linear' | 'ease_in' | 'ease_out' | 'ease_in_out'
type MaskScope = 'scene' | `layer:${string}`
type RegionItem = {
  id: string
  target: RegionTarget
  layerId?: string
  color: string
  name: string
  shape: RegionShape
  rect: SelectionRect
  maskStrength: number
  innerBlur: number
  outerBlur: number
  negate: boolean
  inherited: boolean
  prompt: string
  negativePrompt: string
  denoise: number
  maskOperator: ConditionAggregateOperator
  denoiseOperator: ConditionAggregateOperator
  schedule: RegionSchedule
  start: number
  end: number
}
type GeneratedLayer = { image: string; seed: number; selected: boolean; layerId: string }
type LayerPanelTab = 'properties' | 'sampling' | 'mask' | 'source' | 'generator' | 'effects' | 'controlnet' | 'video'
type ControlNetModel = 'openpose' | 'depth' | 'canny' | 'hed' | 'mlsd' | 'scribble' | 'normal' | 'lineart' | 'seg'
type ControlNetConfig = {
  enabled: boolean
  model: ControlNetModel
  scale: number
  startAt: number
  endAt: number
  usePreprocessor: boolean
  useLayerFrame: boolean
  modelPath: string
  explicitImageUrl: string
  sourceImageUrl: string
  cnPreviewHeight: number
  preprocessorParams: {
    cannyLow: number
    cannyHigh: number
    openposeHands: boolean
    openposeFace: boolean
    mlsdThrV: number
    mlsdThrD: number
    linartCoarse: boolean
  }
}
type ViewportTarget = 'input' | 'output'
type ColorSlot = 'primary' | 'secondary'
type EditorDest = 'paint' | 'mask'
type ColorPlannerMode = 'complementary' | 'analogous' | 'triad' | 'tetrad' | 'split' | 'monochrome'
type RgbaColor = { r: number; g: number; b: number; a: number }
type EntityTarget = 'scene' | 'sceneMask' | 'layer'
type ActiveEntityTarget = 'scene' | 'layer'
type ScenePanelTab = 'sdxl' | 'z-image' | 'streamdiffusion' | 'sana' | 'causal-forcing' | 'fastvideo'
type VideoLoopMode = 'loop' | 'ping-pong' | 'once'
type VideoFpsMode = 'fps' | 'sequential'
type SeedMode = 'fixed' | 'random' | 'increment' | 'decrement'
type LatentReuseNoiseMode = 'none' | 'fixed' | 'random'
type StreamRuntimePreset = 'diffusers' | 'lcm-lora-sdxl' | 'sdxl-lightning-4step' | 'sdxl-turbo' | 'sana-sprint-0.6b' | 'sana-sprint-1.6b'
type StreamMotionMode = 'none' | 'sway' | 'orbit' | 'push' | 'zoom'

type MotionClipFrame = { index: number; url: string; duration_ms: number }
type MotionClip = {
  id: string
  name: string
  motion: string
  region: string
  model: string
  prompt: string
  arrival_prompt?: string
  negative_prompt: string
  keyframes?: string[]
  fps: number
  duration_seconds: number
  loop: string
  width: number
  height: number
  frame_count: number
  frames: MotionClipFrame[]
  video_url?: string
  created_at: number
}
type MotionTaskProgress = {
  task_id: string
  status: 'queued' | 'running' | 'complete' | 'error'
  phase: string
  progress: number
  message: string
  preview_frames?: MotionClipFrame[]
  result?: MotionClip | null
  error?: string | null
}
type VideoBackendCapability = { configured: boolean; message: string }
type VideoCapabilities = { models: Record<string, VideoBackendCapability> }
type RendererCapability = {
  configured: boolean
  native: boolean
  streamable: boolean
  realtime: boolean
  runtime: string
  transport: string
  message: string
  requirements?: string[]
  accelerators?: Record<string, boolean>
}
type RendererCapabilities = {
  accelerators?: Record<string, boolean>
  renderers: Partial<Record<ScenePanelTab, RendererCapability>>
  versions?: Record<string, string | null>
}
type GpuDevice = { id: string; name: string; kind: string; available: boolean; memory_total?: number }
type GpuCatalog = { default: string; devices: GpuDevice[] }
type ScenePreset = {
  prompt: string
  negativePrompt: string
  strength: number
  cfg: number
  steps: number
  sampler: string
  scheduler: string
  modelPath: string
  loraPaths: string[]
}

type SamplerPreset = {
  strength?: number
  cfg: number
  steps: number
  sampler: string
  scheduler: string
}
type PickerOption = { label: string; value: string; meta?: string }
type PickerDialog =
  | { kind: 'option'; label: string; value: string; options: PickerOption[]; onSelect: (value: string) => void }
  | { kind: 'lora'; label: string; selectedPaths: string[]; onToggle: (value: string) => void }

type LayerGeneratePayload = { variations: { image: string; seed: number }[]; error?: string | null; mode: string; latency_ms: number }
type LayerTaskProgress = {
  task_id: string
  status: 'queued' | 'running' | 'complete' | 'error'
  phase: string
  progress: number
  message: string
  result?: LayerGeneratePayload | null
  error?: string | null
}
type GenerationTask = LayerTaskProgress & { layerId: string; label: string; startedAt: number }

type EditorSnapshot = {
  canvas: unknown
  mask: string
  regions: RegionItem[]
  stageWidth: number
  stageHeight: number
  selectionRect?: SelectionRect
  selectedLayerId: string
}

type ArchiveResourceEntry = {
  hash: string
  path: string
  url: string
  mediaType: string
  size: number
  storedAt: number
}

type SerializedMask = { scope: MaskScope; url: string }
type SerializedProjectState = {
  version: number
  savedAt: number
  archiveRoot: string
  options: StoredOptions
  canvas: unknown
  masks: SerializedMask[]
  regions: RegionItem[]
  generatedLayers: GeneratedLayer[]
  outputImage: string
  sourceRoot: string
  selectedSourceRel: string
  sourceImages: SourceImage[]
  stageWidth: number
  stageHeight: number
  selectionRect?: SelectionRect
  selectedLayerId: string
  selectedEntity: ActiveEntityTarget
  layerPanelTab: LayerPanelTab
  scenePanelTab: ScenePanelTab
  resources: ArchiveResourceEntry[]
}

type StoredOptions = Partial<{
  version: number
  prompt: string
  negativePrompt: string
  strength: number
  cfg: number
  steps: number
  stageWidth: number
  stageHeight: number
  brushSize: number
  brushHardness: number
  brushColor: string
  secondaryBrushColor: string
  activeColorSlot: ColorSlot
  colorPlannerMode: ColorPlannerMode
  toolMode: ToolMode | 'mask'
  noDefaultMaskLayers: string[]
  inputZoom: number
  outputZoom: number
  inputPaintVisible: boolean
  inputMaskVisible: boolean
  inputActiveMaskOnly: boolean
  editorDest: EditorDest
  selectedModel: string
  selectedLoras: string[]
  layerVariationCount: number
  generationWidth: number
  generationHeight: number
  layerGenerationSteps: number
  layerGenerationCfg: number
  layerGenerationSampler: string
  layerGenerationScheduler: string
  layerGenerationSeed: string
  layerTransparentBackground: boolean
  layerTransparentMethod: string
  layerTransparentMode: string
  layerTransparentColor: string
  layerTransparentTolerance: number
  layerTransparentAlphaBlur: number
  layerTransparentAlphaThreshold: number
  realtimeDevice: string
  layerDevice: string
  videoDevice: string
  sceneSampler: string
  sceneScheduler: string
  sceneSeed: string
  sceneSeedMode: SeedMode
  sceneReuseLatent: boolean
  sceneReuseLatentDenoise: number
  sceneReuseLatentNoise: number
  sceneReuseLatentNoiseMode: LatentReuseNoiseMode
  sceneTransparentBackground: boolean
  sceneTransparentMode: string
  sceneTransparentColor: string
  sceneTransparentTolerance: number
  sceneTransparentAlphaBlur: number
  sceneTransparentAlphaThreshold: number
  streamTimestepIndices: string
  streamFrameBufferSize: number
  streamCfgType: string
  streamSimilarityThreshold: number
  streamMaxSkipFrames: number
  streamRuntimePreset: StreamRuntimePreset
  streamMotionMode: StreamMotionMode
  streamMotionIntensity: number
  streamMotionSpeed: number
  scenePanelTab: ScenePanelTab
  selectedEntity: EntityTarget
  sceneMaskNegated: boolean
  sceneMaskAbsolute: boolean
  regionShape: RegionShape
  generatorActive: boolean
  leftPanelWidth: number
  rightPanelWidth: number
}>

type StreamMessage = {
  image?: string
  fps: number
  latency_ms: number
  mode: string
  content_type?: string
  binary?: boolean
  error?: string | null
  debug_channels?: Record<string, string>
}

type SamplingControlsConfig = {
  prompt: string
  negativePrompt: string
  modelPath?: string
  loraPaths?: string[]
  strength?: number
  cfg: number
  steps: number
  sampler: string
  scheduler: string
  onPrompt: (value: string) => void
  onNegativePrompt: (value: string) => void
  onModelPath?: (value: string) => void
  onToggleLora?: (value: string) => void
  onStrength?: (value: number) => void
  onCfg: (value: number) => void
  onSteps: (value: number) => void
  onSampler: (value: string) => void
  onScheduler: (value: string) => void
}

type GenerationControlsConfig = {
  deviceLabel: string
  deviceValue: string
  onDevice: (value: string) => void
  sampling: SamplingControlsConfig
  beforeSampling?: TemplateResult
  afterSampling?: TemplateResult
}

const SIZE_PRESETS: SizePreset[] = [
  { label: '512 x 512', width: 512, height: 512 },
  { label: '768 x 768', width: 768, height: 768 },
  { label: '832 x 1216', width: 832, height: 1216 },
  { label: '1216 x 832', width: 1216, height: 832 },
  { label: '704 x 1024', width: 704, height: 1024 },
  { label: '1024 x 704', width: 1024, height: 704 },
  { label: '576 x 832', width: 576, height: 832 },
  { label: '832 x 576', width: 832, height: 576 },
]

const SAMPLERS = [
  { label: 'Default', value: '' },
  { label: 'euler', value: 'euler' },
  { label: 'euler_cfg_pp', value: 'euler_cfg_pp' },
  { label: 'euler_ancestral', value: 'euler_ancestral' },
  { label: 'euler_ancestral_cfg_pp', value: 'euler_ancestral_cfg_pp' },
  { label: 'heun', value: 'heun' },
  { label: 'heunpp2', value: 'heunpp2' },
  { label: 'dpm_2', value: 'dpm_2' },
  { label: 'dpm_2_ancestral', value: 'dpm_2_ancestral' },
  { label: 'lms', value: 'lms' },
  { label: 'dpm_fast', value: 'dpm_fast' },
  { label: 'dpm_adaptive', value: 'dpm_adaptive' },
  { label: 'dpmpp_2s_ancestral', value: 'dpmpp_2s_ancestral' },
  { label: 'dpmpp_sde', value: 'dpmpp_sde' },
  { label: 'dpmpp_sde_gpu', value: 'dpmpp_sde_gpu' },
  { label: 'dpmpp_2m', value: 'dpmpp_2m' },
  { label: 'dpmpp_2m_sde', value: 'dpmpp_2m_sde' },
  { label: 'dpmpp_3m_sde', value: 'dpmpp_3m_sde' },
  { label: 'ddpm', value: 'ddpm' },
  { label: 'lcm', value: 'lcm' },
  { label: 'ipndm', value: 'ipndm' },
  { label: 'deis', value: 'deis' },
  { label: 'res_multistep', value: 'res_multistep' },
  { label: 'res_multistep_ancestral', value: 'res_multistep_ancestral' },
  { label: 'gradient_estimation', value: 'gradient_estimation' },
  { label: 'er_sde', value: 'er_sde' },
  { label: 'seeds_2', value: 'seeds_2' },
  { label: 'seeds_3', value: 'seeds_3' },
  { label: 'sa_solver', value: 'sa_solver' },
  { label: 'ddim', value: 'ddim' },
  { label: 'uni_pc', value: 'uni_pc' },
]

const SCHEDULERS = [
  { label: 'simple', value: 'simple' },
  { label: 'sgm_uniform', value: 'sgm_uniform' },
  { label: 'karras', value: 'karras' },
  { label: 'exponential', value: 'exponential' },
  { label: 'ddim_uniform', value: 'ddim_uniform' },
  { label: 'beta', value: 'beta' },
  { label: 'normal', value: 'normal' },
  { label: 'linear_quadratic', value: 'linear_quadratic' },
  { label: 'kl_optimal', value: 'kl_optimal' },
]

const PALETTE = [
  '#000000', '#1f2329', '#3b424c', '#6f7782', '#a9b0ba', '#d8dce2', '#ffffff', '#fff4d7',
  '#6d1f2c', '#b83246', '#f04f65', '#ff8a9a', '#ffd1d8', '#7a2f16', '#c75524', '#f28b38',
  '#ffbf69', '#ffe0ad', '#735200', '#b98300', '#e9b44c', '#ffe066', '#fff3a8', '#4d6219',
  '#7aa12b', '#a8d84f', '#d8f59a', '#123d30', '#1f7a5f', '#35b58f', '#79e0c0', '#c9ffee',
  '#123d44', '#1e6f7a', '#35b7c8', '#83e6f0', '#d2fbff', '#173b70', '#2f6dd1', '#5f8cff',
  '#9ab5ff', '#d8e4ff', '#2f246f', '#5d46c7', '#8a6cff', '#bfaeff', '#ebe4ff', '#551f6d',
  '#9638c7', '#c05cff', '#dea9ff', '#f4ddff', '#6b1f4b', '#bd3f84', '#f266b2', '#ffa6d6',
  '#ffd8ed', '#3c2820', '#704936', '#a66b4b', '#d4976b', '#f1c9a6', '#24413a', '#456d63',
]

const QUICK_PALETTE = [
  '#000000', '#1f2329', '#3b424c', '#6f7782', '#a9b0ba', '#d8dce2', '#ffffff', '#fff4d7',
  '#6d1f2c', '#b83246', '#f04f65', '#ff8a9a', '#f28b38', '#e9b44c', '#35b58f', '#5f8cff',
]

const TOOL_DEFS: Record<ToolMode, { icon: string; label: string; shortcut: string }> = {
  select: { icon: '⤢', label: 'Select / Transform', shortcut: 'Ctrl+1' },
  region: { icon: '⬚', label: 'Mask Region', shortcut: 'Ctrl+2' },
  brush: { icon: '🖌', label: 'Paint', shortcut: 'Ctrl+3' },
  eraser: { icon: '⌦', label: 'Erase', shortcut: 'Ctrl+4' },
  text: { icon: 'Ꭲ', label: 'Text', shortcut: 'Ctrl+5' },
  line: { icon: '╱', label: 'Line', shortcut: 'Ctrl+6' },
  rect: { icon: '▭', label: 'Rectangle', shortcut: 'Ctrl+7' },
  fill: { icon: '▨', label: 'Fill', shortcut: 'Ctrl+8' },
}

const TOOL_SHORTCUTS: Record<string, ToolMode> = {
  '1': 'select',
  '2': 'region',
  '3': 'brush',
  '4': 'eraser',
  '5': 'text',
  '6': 'line',
  '7': 'rect',
  '8': 'fill',
}

const COLOR_PLANNER_MODES: { label: string; value: ColorPlannerMode }[] = [
  { label: 'Complement', value: 'complementary' },
  { label: 'Analogous', value: 'analogous' },
  { label: 'Triad', value: 'triad' },
  { label: 'Tetrad', value: 'tetrad' },
  { label: 'Split', value: 'split' },
  { label: 'Mono', value: 'monochrome' },
]

const OPTIONS_KEY = 'rtdiffusion.options.v1'
const OPTIONS_VERSION = 3
const PROJECT_STATE_KEY = 'rtdiffusion.projectState.v1'
const PROJECT_RESOURCE_PREFIX = 'rtdiffusion.resource.v1.'
const PROJECT_RESOURCE_DB = 'rtdiffusion.resources.v1'
const PROJECT_RESOURCE_STORE = 'resources'
const PROJECT_VERSION = 1
const ARCHIVE_ROOT = ARCHIVE_ROOT_URL
const HISTORY_LIMIT = 60
const SCENE_PRESETS_KEY = 'rtdiffusion.scenePresets.v1'
const SAMPLER_PRESETS_KEY = 'rtdiffusion.samplerPresets.v1'

FabricObjectClass.customProperties = [...new Set([...(FabricObjectClass.customProperties ?? []), ...CANVAS_CUSTOM_PROPERTIES])]

@customElement('rt-diffusion-app')
export class RtDiffusionApp extends LitElement {
  @query('#edit-canvas') private editCanvasElement!: HTMLCanvasElement
  @query('#mask-canvas') private maskCanvasElement!: HTMLCanvasElement
  @query('#mask-preview-canvas') private maskPreviewCanvasElement!: HTMLCanvasElement
  @query('#editor-overlay') private editorOverlayCanvas?: HTMLCanvasElement
  @query('#stream-output-img') private streamOutputImgElement?: HTMLImageElement
  @query('#stream-output-canvas') private streamOutputCanvasElement?: HTMLCanvasElement
  @query('#video-input') private videoInputElement?: HTMLVideoElement
  @query('.stage') private stageElement!: HTMLElement

  @state() private prompt = ''
  @state() private negativePrompt = ''
  @state() private strength = 0.75
  @state() private cfg = 1.5
  @state() private steps = 1
  @state() private stageWidth = 832
  @state() private stageHeight = 1216
  @state() private brushSize = 34
  @state() private brushHardness = 100
  @state() private noDefaultMaskLayers: string[] = []
  @state() private brushColor = '#ffffff'
  @state() private secondaryBrushColor = '#000000'
  @state() private activeColorSlot: ColorSlot = 'primary'
  @state() private colorPlannerMode: ColorPlannerMode = 'complementary'
  @state() private colorPopupOpen = false
  @state() private colorPopupPosition = { x: 92, y: 92 }
  @state() private toolMode: ToolMode = 'brush'
  @state() private isStreaming = false
  @state() private status = 'offline'
  @state() private fps = 0
  @state() private displayFps = 0
  // SANA video generation state
  @state() private sanaVideoModel = '480p'
  @state() private sanaVideoNumFrames = 81
  @state() private sanaVideoGuidance = 6.0
  @state() private sanaVideoSteps = 20
  @state() private sanaVideoTaskId = ''
  @state() private sanaVideoTask: Record<string, any> | null = null
  @state() private sanaSteps = 2
  @state() private latency = 0
  @state() private backendMode = 'mock'
  @state() private outputImage = ''
  @state() private realtimeVideoActive = false
  @state() private assets: AssetCatalog = { models: [], loras: [] }
  @state() private selectedModel = ''
  @state() private selectedLoras = new Set<string>()
  @state() private loraSearch = ''
  @state() private layers: LayerItem[] = []
  @state() private cursor = { x: 0, y: 0, visible: false }
  @state() private inputZoom = 1
  @state() private outputZoom = 1
  @state() private inputPaintVisible = true
  @state() private inputMaskVisible = true
  @state() private inputActiveMaskOnly = false
  @state() private editorDest: EditorDest = 'paint'
  @state() private inputOverlayVisible = true
  @state() private outputOverlayVisible = true
  @state() private errorLog: { id: string; message: string; time: number }[] = []
  @state() private errorPopoverOpen = false
  @state() private selectionRect?: SelectionRect
  @state() private layerVariationCount = 4
  @state() private generationWidth = 832
  @state() private generationHeight = 1216
  @state() private layerGenerationSteps = 24
  @state() private layerGenerationCfg = 5
  @state() private layerGenerationSampler = 'euler_ancestral'
  @state() private layerGenerationScheduler = 'simple'
  @state() private layerGenerationSeed = ''
  @state() private layerTransparentBackground = true
  @state() private layerTransparentMethod = 'auto'
  @state() private layerTransparentMode = 'border'
  @state() private layerTransparentColor = '#ffffff'
  @state() private layerTransparentTolerance = 34
  @state() private layerTransparentAlphaBlur = 1.5
  @state() private layerTransparentAlphaThreshold = 10
  @state() private gpuDevices: GpuDevice[] = [{ id: '', name: 'Backend default', kind: 'default', available: true }]
  @state() private realtimeDevice = ''
  @state() private layerDevice = ''
  @state() private videoDevice = ''
  @state() private sceneSampler = 'euler'
  @state() private sceneScheduler = 'simple'
  @state() private sceneSeed = ''
  @state() private sceneSeedMode: SeedMode = 'fixed'
  @state() private sceneReuseLatent = false
  @state() private sceneReuseLatentDenoise = 0.24
  @state() private sceneReuseLatentNoise = 0
  @state() private sceneReuseLatentNoiseMode: LatentReuseNoiseMode = 'none'
  @state() private sceneReuseLatentRestartNext = false
  @state() private sceneTransparentBackground = false
  @state() private sceneTransparentMode = 'border'
  @state() private sceneTransparentColor = '#ffffff'
  @state() private sceneTransparentTolerance = 34
  @state() private sceneTransparentAlphaBlur = 1.5
  @state() private sceneTransparentAlphaThreshold = 10
  @state() private streamTimestepIndices = '16,32'
  @state() private streamFrameBufferSize = 1
  @state() private streamCfgType = 'self'
  @state() private streamSimilarityThreshold = 0.98
  @state() private streamMaxSkipFrames = 0
  @state() private streamRuntimePreset: StreamRuntimePreset = 'diffusers'
  @state() private streamMotionMode: StreamMotionMode = 'none'
  @state() private streamMotionIntensity = 0.35
  @state() private streamMotionSpeed = 1
  @state() private videoUrl = ''
  @state() private videoLoopMode: VideoLoopMode = 'loop'
  @state() private videoStart = 0
  @state() private videoEnd = 0
  @state() private videoFpsMode: VideoFpsMode = 'fps'
  @state() private videoFps = 30
  @state() private videoDuration = 0
  @state() private generatorActive = true
  @state() private generatedLayers: GeneratedLayer[] = []
  @state() private isGeneratingLayer = false
  @state() private generationTasks: GenerationTask[] = []
    @state() private taskPopoverOpen = false
  @state() private selectedEntity: ActiveEntityTarget = 'scene'
  @state() private layerPanelTab: LayerPanelTab = 'properties'
  @state() private scenePanelTab: ScenePanelTab = 'sdxl'
  @state() private motionPrompt = 'adult woman holding a lollipop, calm gaze at the viewer, subtle natural motion, same identity, same lighting'
  @state() private motionArrivalPrompt = ''
  @state() private motionNegativePrompt = 'identity change, outfit change, camera cut, large motion, deformation, flicker'
  @state() private motionModel = 'causal-forcing-1step'
  @state() private motionFps = 16
  @state() private motionDuration = 2
  @state() private motionIntensity = 1
  @state() private fastVideoModelPath = ''
  @state() private activeMotionFrame = ''
  @state() private activeMotionVideo = ''
  @state() private activeMotionIsStream = false
  @state() private motionStreamHasSegments = false
  @state() private isGeneratingMotion = false
  @state() private motionChunkIndex = 0
  @state() private motionTask?: MotionTaskProgress
  @state() private videoCapabilities?: VideoCapabilities
  @state() private rendererCapabilities?: RendererCapabilities
  @state() private sourceRoot = ''
  @state() private sourceImages: SourceImage[] = []
  @state() private selectedSourceRel = ''
  @state() private isLoadingSources = false
  @state() private pickerDialog?: PickerDialog
  @state() private pickerDialogPosition = { left: 0, top: 0, width: 420, maxHeight: 420 }
  @state() private layerControlNetConfigs = new Map<string, ControlNetConfig>()
  @state() private cnLocalModels: Array<{ name: string; path: string; type: string; family: string }> = []
  @state() private sceneMaskNegated = false
  @state() private sceneMaskAbsolute = false
  @state() private regions: RegionItem[] = []
  @state() private regionName = ''
  @state() private regionShape: RegionShape = 'rope'
  @state() private leftPanelWidth = 220
  @state() private rightPanelWidth = 320
  @state() private projectStatus = 'Local project ready'
  @state() private debugStreamsEnabled = false
  // Only reactive for structure (triggers re-render when channels appear/disappear)
  @state() private debugChannelNames: string[] = []
  @state() private debugMosaicMode = true
  @state() private debugPreviewChannel = ''
  // Non-reactive store for per-frame data — updated directly into DOM elements
  private _debugChannelData: Map<string, string> = new Map()

  private fabricCanvas?: Canvas
  private socket?: WebSocket
  private streamTimer?: number
  private streamRaf?: number
  private webRtcEventSource?: EventSource
  private webRtcPcId = ''
  private webRtcSettingsTimer?: number
  private frameLoopRaf?: number
  private frameAbortController?: AbortController
  private webRtcLastSettings = ''
  private motionTaskSocket?: WebSocket
  private motionMediaSource?: MediaSource
  private motionSourceBuffer?: SourceBuffer
  private motionSegmentQueue: ArrayBuffer[] = []
  private motionStreamOpen?: Promise<boolean>
  private motionStreamObjectUrl = ''
  private motionRunId = 0
  private streamStartedAt = 0
  private streamFeedbackImage?: HTMLImageElement
  private streamFrameContentType = 'image/jpeg'
  private streamVideoMedia?: MediaStream
  private streamSendQueued = false
  private streamTransitionAnimation?: number
  private realtimeFrameTimes: number[] = []
  private maskContext?: CanvasRenderingContext2D
  private maskPreviewContext?: CanvasRenderingContext2D
  private drawing = false
  private awaitingFrame = false
  private lastPointer?: { x: number; y: number }
  private panningTarget?: HTMLElement
  private panStart = { x: 0, y: 0, left: 0, top: 0 }
  private overlayTimers: Partial<Record<ViewportTarget, number>> = {}
  private selectedLayerId = ''
  private lastWorkingModel = ''
  private lastWorkingLoras: string[] = []
  private undoStack: EditorSnapshot[] = []
  private redoStack: EditorSnapshot[] = []
  private isRestoringHistory = false
  private isSyncingLayers = false
  private pendingBrushButton = 0
  private pendingPaintLayerId = ''
  private rightPaintStroke?: { pointerId: number; points: { x: number; y: number }[] }
  private _shapeDrawState: { startX: number; startY: number } | null = null
  private upperCanvasElement?: HTMLCanvasElement
  private optionsLoaded = false
  private maskStrokeMode: 'paint' | 'erase' = 'paint'
  private maskStrokePoints: RegionPoint[] = []
  private maskPreviewTimer?: number
  private lastMaskPreviewAt = 0
  private layerAlphaCoverageCache = new Map<string, { value: number; at: number }>()
  private activeMaskScope?: MaskScope
  private maskCanvases = new Map<MaskScope, HTMLCanvasElement>()
  private layerMaskNegated = new Map<string, boolean>()
  private projectSaveTimer?: number
  private projectSaveInFlight = false
  private projectRestoreInFlight = false
  private numberDrag?: {
    pointerId: number
    startX: number
    startValue: number
    min: number
    max: number
    step: number
    onChange: (value: number) => void
  }
  private colorPopupDrag?: { pointerId: number; startX: number; startY: number; x: number; y: number; target: HTMLElement }
  private panelResize?: { pointerId: number; side: 'left' | 'right'; startX: number; startWidth: number; target: HTMLElement }
  private videoObjectUrl = ''
  private videoReversing = false
  private _videoLayerId: string | null = null
  private _videoOffscreenCanvas: HTMLCanvasElement | null = null
  private _videoRafId: number | null = null

  firstUpdated() {
    this.pruneLegacyResourceStorage()
    this.loadStoredOptions()
    this.sourceRoot = window.localStorage.getItem('rtdiffusion.sourceRoot') ?? ''
    this.setupCanvas()
    this.setupMaskCanvas()
    this.seedCanvas()
    void this.restorePersistedProject()
    this.loadAssets()
    void this.loadVideoCapabilities()
    void this.loadRendererCapabilities()
    void this.loadGpuDevices()
    if (this.sourceRoot) void this.loadSourceImages()
    this.addEventListener('paste', this.handlePaste as EventListener)
    this.addEventListener('keydown', this.handleKeyDown as EventListener)
    this.setAttribute('tabindex', '0')
    this.optionsLoaded = true
  }

  disconnectedCallback() {
    super.disconnectedCallback()
    this.stopStream()
    this.clearMotionPlayback()
    for (const timer of Object.values(this.overlayTimers)) if (timer) window.clearTimeout(timer)
    if (this.projectSaveTimer) window.clearTimeout(this.projectSaveTimer)
    this.detachRightPaintHandlers()
    this.fabricCanvas?.dispose()
    this.removeEventListener('paste', this.handlePaste as EventListener)
    this.removeEventListener('keydown', this.handleKeyDown as EventListener)
  }

  protected updated(changedProperties: Map<PropertyKey, unknown>) {
    if (changedProperties.has('stageWidth') || changedProperties.has('stageHeight') || changedProperties.has('inputZoom')) {
      this.syncCanvasGeometry()
    }
    if (
      changedProperties.has('selectedEntity') ||
      changedProperties.has('scenePanelTab') ||
      changedProperties.has('layerPanelTab') ||
      changedProperties.has('regions') ||
      changedProperties.has('sceneMaskNegated') ||
      changedProperties.has('sceneMaskAbsolute') ||
      changedProperties.has('stageWidth') ||
      changedProperties.has('stageHeight')
    ) {
      this.applyToolMode()
      this.syncMaskScope()
      this.scheduleMaskPreviewRefresh()
    }
    this.persistOptions()
    if (this.shouldPersistProject(changedProperties)) this.scheduleProjectPersist()
  }

  private shouldPersistProject(changedProperties: Map<PropertyKey, unknown>) {
    return [
      'stageWidth',
      'stageHeight',
      'regions',
      'generatedLayers',
      'outputImage',
      'selectionRect',
      'selectedEntity',
      'layerPanelTab',
      'scenePanelTab',
      'sourceRoot',
      'selectedSourceRel',
      'sourceImages',
      'layers',
    ].some((key) => changedProperties.has(key))
  }

  private syncCanvasGeometry() {
    window.requestAnimationFrame(() => {
      this.fabricCanvas?.calcOffset()
      this.fabricCanvas?.requestRenderAll()
    })
  }

  private loadStoredOptions() {
    this.clearLegacyScenePresets()
    try {
      const stored = JSON.parse(window.localStorage.getItem(OPTIONS_KEY) || '{}') as StoredOptions
      const currentOptionsVersion = stored.version === OPTIONS_VERSION
      this.prompt = stored.prompt ?? this.prompt
      this.negativePrompt = stored.negativePrompt ?? this.negativePrompt
      this.strength = this.clamp(stored.strength ?? this.strength, 0, 0.999, this.strength)
      this.cfg = this.clamp(stored.cfg ?? this.cfg, 0, 30, this.cfg)
      this.steps = Math.round(this.clamp(stored.steps ?? this.steps, 1, 32, this.steps))
      this.stageWidth = Math.round(this.clamp(stored.stageWidth ?? this.stageWidth, 256, 1536, this.stageWidth))
      this.stageHeight = Math.round(this.clamp(stored.stageHeight ?? this.stageHeight, 256, 1536, this.stageHeight))
      this.brushSize = Math.round(this.clamp(stored.brushSize ?? this.brushSize, 1, 4096, this.brushSize))
      this.brushHardness = Math.round(this.clamp(stored.brushHardness ?? this.brushHardness, 0, 100, this.brushHardness))
      this.noDefaultMaskLayers = Array.isArray(stored.noDefaultMaskLayers) ? stored.noDefaultMaskLayers : this.noDefaultMaskLayers
      this.brushColor = stored.brushColor ?? this.brushColor
      this.secondaryBrushColor = stored.secondaryBrushColor ?? this.secondaryBrushColor
      this.activeColorSlot = stored.activeColorSlot === 'secondary' ? 'secondary' : 'primary'
      this.colorPlannerMode = stored.colorPlannerMode ?? this.colorPlannerMode
      this.toolMode = currentOptionsVersion && stored.toolMode !== 'mask' ? stored.toolMode ?? this.toolMode : 'brush'
      this.inputZoom = this.clamp(stored.inputZoom ?? this.inputZoom, 0.25, 8, this.inputZoom)
      this.outputZoom = this.clamp(stored.outputZoom ?? this.outputZoom, 0.25, 8, this.outputZoom)
      this.inputPaintVisible = stored.inputPaintVisible ?? this.inputPaintVisible
      this.inputMaskVisible = stored.inputMaskVisible ?? this.inputMaskVisible
      this.inputActiveMaskOnly = stored.inputActiveMaskOnly ?? this.inputActiveMaskOnly
      this.editorDest = stored.editorDest === 'mask' ? 'mask' : 'paint'
      this.selectedModel = stored.selectedModel ?? this.selectedModel
      this.selectedLoras = new Set(stored.selectedLoras ?? [])
      this.layerVariationCount = Math.round(this.clamp(stored.layerVariationCount ?? this.layerVariationCount, 1, 12, this.layerVariationCount))
      this.generationWidth = this.alignGenerationSize(stored.generationWidth ?? this.generationWidth)
      this.generationHeight = this.alignGenerationSize(stored.generationHeight ?? this.generationHeight)
      this.layerGenerationSteps = Math.round(this.clamp(stored.layerGenerationSteps ?? this.layerGenerationSteps, 1, 40, this.layerGenerationSteps))
      this.layerGenerationCfg = this.clamp(stored.layerGenerationCfg ?? this.layerGenerationCfg, 0, 30, this.layerGenerationCfg)
      this.layerGenerationSampler = stored.layerGenerationSampler ?? this.layerGenerationSampler
      this.layerGenerationScheduler = stored.layerGenerationScheduler ?? this.layerGenerationScheduler
      this.layerGenerationSeed = stored.layerGenerationSeed ?? this.layerGenerationSeed
      this.layerTransparentBackground = stored.layerTransparentBackground ?? this.layerTransparentBackground
      this.layerTransparentMethod = this.validTransparentMethod(stored.layerTransparentMethod) ? stored.layerTransparentMethod : this.layerTransparentMethod
      this.layerTransparentMode = stored.layerTransparentMode === 'color' ? 'color' : 'border'
      this.layerTransparentColor = stored.layerTransparentColor ?? this.layerTransparentColor
      this.layerTransparentTolerance = this.clamp(stored.layerTransparentTolerance ?? this.layerTransparentTolerance, 0, 255, this.layerTransparentTolerance)
      this.layerTransparentAlphaBlur = this.clamp(stored.layerTransparentAlphaBlur ?? this.layerTransparentAlphaBlur, 0, 24, this.layerTransparentAlphaBlur)
      this.layerTransparentAlphaThreshold = Math.round(this.clamp(stored.layerTransparentAlphaThreshold ?? this.layerTransparentAlphaThreshold, 0, 255, this.layerTransparentAlphaThreshold))
      this.realtimeDevice = stored.realtimeDevice ?? this.realtimeDevice
      this.layerDevice = stored.layerDevice ?? this.layerDevice
      this.videoDevice = stored.videoDevice ?? this.videoDevice
      this.sceneSampler = stored.sceneSampler ?? this.sceneSampler
      this.sceneScheduler = stored.sceneScheduler ?? this.sceneScheduler
      this.sceneSeed = stored.sceneSeed ?? this.sceneSeed
      this.sceneSeedMode = this.validSeedMode(stored.sceneSeedMode) ? stored.sceneSeedMode : this.sceneSeedMode
      this.sceneReuseLatent = stored.sceneReuseLatent ?? this.sceneReuseLatent
      this.sceneReuseLatentDenoise = this.clamp(stored.sceneReuseLatentDenoise ?? this.sceneReuseLatentDenoise, 0, 0.999, this.sceneReuseLatentDenoise)
      this.sceneReuseLatentNoise = this.clamp(stored.sceneReuseLatentNoise ?? this.sceneReuseLatentNoise, 0, 1, this.sceneReuseLatentNoise)
      this.sceneReuseLatentNoiseMode = this.validLatentReuseNoiseMode(stored.sceneReuseLatentNoiseMode) ? stored.sceneReuseLatentNoiseMode : this.sceneReuseLatentNoiseMode
      this.sceneTransparentBackground = stored.sceneTransparentBackground ?? this.sceneTransparentBackground
      this.sceneTransparentMode = stored.sceneTransparentMode === 'color' ? 'color' : 'border'
      this.sceneTransparentColor = stored.sceneTransparentColor ?? this.sceneTransparentColor
      this.sceneTransparentTolerance = this.clamp(stored.sceneTransparentTolerance ?? this.sceneTransparentTolerance, 0, 255, this.sceneTransparentTolerance)
      this.sceneTransparentAlphaBlur = this.clamp(stored.sceneTransparentAlphaBlur ?? this.sceneTransparentAlphaBlur, 0, 24, this.sceneTransparentAlphaBlur)
      this.sceneTransparentAlphaThreshold = Math.round(this.clamp(stored.sceneTransparentAlphaThreshold ?? this.sceneTransparentAlphaThreshold, 0, 255, this.sceneTransparentAlphaThreshold))
      this.streamTimestepIndices = stored.streamTimestepIndices ?? this.streamTimestepIndices
      this.streamFrameBufferSize = Math.round(this.clamp(stored.streamFrameBufferSize ?? this.streamFrameBufferSize, 1, 4, this.streamFrameBufferSize))
      this.streamCfgType = this.validStreamCfgType(stored.streamCfgType) ? stored.streamCfgType : this.streamCfgType
      this.streamSimilarityThreshold = this.clamp(stored.streamSimilarityThreshold ?? this.streamSimilarityThreshold, 0, 1, this.streamSimilarityThreshold)
      this.streamMaxSkipFrames = Math.round(this.clamp(stored.streamMaxSkipFrames ?? this.streamMaxSkipFrames, 0, 60, this.streamMaxSkipFrames))
      this.streamRuntimePreset = this.validStreamRuntimePreset(stored.streamRuntimePreset) ? stored.streamRuntimePreset : this.streamRuntimePreset
      this.streamMotionMode = this.validStreamMotionMode(stored.streamMotionMode) ? stored.streamMotionMode : this.streamMotionMode
      this.streamMotionIntensity = this.clamp(stored.streamMotionIntensity ?? this.streamMotionIntensity, 0, 2, this.streamMotionIntensity)
      this.streamMotionSpeed = this.clamp(stored.streamMotionSpeed ?? this.streamMotionSpeed, 0.1, 4, this.streamMotionSpeed)
      this.scenePanelTab = this.validScenePanelTab(stored.scenePanelTab) ? stored.scenePanelTab : 'sdxl'
      this.selectedEntity = stored.selectedEntity === 'layer' ? 'layer' : 'scene'
      this.sceneMaskNegated = stored.sceneMaskNegated ?? this.sceneMaskNegated
      this.sceneMaskAbsolute = stored.sceneMaskAbsolute ?? this.sceneMaskAbsolute
      this.regionShape = stored.regionShape ?? this.regionShape
      this.generatorActive = stored.generatorActive ?? this.generatorActive
      this.leftPanelWidth = Math.round(this.clamp(stored.leftPanelWidth ?? this.leftPanelWidth, 144, 420, this.leftPanelWidth))
      this.rightPanelWidth = Math.round(this.clamp(stored.rightPanelWidth ?? this.rightPanelWidth, 220, 520, this.rightPanelWidth))
    } catch {
      window.localStorage.removeItem(OPTIONS_KEY)
    }
  }

  private currentStoredOptions(): StoredOptions {
    return {
      version: OPTIONS_VERSION,
      prompt: this.prompt,
      negativePrompt: this.negativePrompt,
      strength: this.strength,
      cfg: this.cfg,
      steps: this.steps,
      stageWidth: this.stageWidth,
      stageHeight: this.stageHeight,
      brushSize: this.brushSize,
      brushHardness: this.brushHardness,
      noDefaultMaskLayers: this.noDefaultMaskLayers,
      brushColor: this.brushColor,
      secondaryBrushColor: this.secondaryBrushColor,
      activeColorSlot: this.activeColorSlot,
      colorPlannerMode: this.colorPlannerMode,
      toolMode: this.toolMode,
      inputZoom: this.inputZoom,
      outputZoom: this.outputZoom,
      inputPaintVisible: this.inputPaintVisible,
      inputMaskVisible: this.inputMaskVisible,
      inputActiveMaskOnly: this.inputActiveMaskOnly,
      editorDest: this.editorDest,
      selectedModel: this.selectedModel,
      selectedLoras: [...this.selectedLoras],
      layerVariationCount: this.layerVariationCount,
      generationWidth: this.generationWidth,
      generationHeight: this.generationHeight,
      layerGenerationSteps: this.layerGenerationSteps,
      layerGenerationCfg: this.layerGenerationCfg,
      layerGenerationSampler: this.layerGenerationSampler,
      layerGenerationScheduler: this.layerGenerationScheduler,
      layerGenerationSeed: this.layerGenerationSeed,
      layerTransparentBackground: this.layerTransparentBackground,
      layerTransparentMethod: this.layerTransparentMethod,
      layerTransparentMode: this.layerTransparentMode,
      layerTransparentColor: this.layerTransparentColor,
      layerTransparentTolerance: this.layerTransparentTolerance,
      layerTransparentAlphaBlur: this.layerTransparentAlphaBlur,
      layerTransparentAlphaThreshold: this.layerTransparentAlphaThreshold,
      realtimeDevice: this.realtimeDevice,
      layerDevice: this.layerDevice,
      videoDevice: this.videoDevice,
      sceneSampler: this.sceneSampler,
      sceneScheduler: this.sceneScheduler,
      sceneSeed: this.sceneSeed,
      sceneSeedMode: this.sceneSeedMode,
      sceneReuseLatent: this.sceneReuseLatent,
      sceneReuseLatentDenoise: this.sceneReuseLatentDenoise,
      sceneReuseLatentNoise: this.sceneReuseLatentNoise,
      sceneReuseLatentNoiseMode: this.sceneReuseLatentNoiseMode,
      sceneTransparentBackground: this.sceneTransparentBackground,
      sceneTransparentMode: this.sceneTransparentMode,
      sceneTransparentColor: this.sceneTransparentColor,
      sceneTransparentTolerance: this.sceneTransparentTolerance,
      sceneTransparentAlphaBlur: this.sceneTransparentAlphaBlur,
      sceneTransparentAlphaThreshold: this.sceneTransparentAlphaThreshold,
      streamTimestepIndices: this.streamTimestepIndices,
      streamFrameBufferSize: this.streamFrameBufferSize,
      streamCfgType: this.streamCfgType,
      streamSimilarityThreshold: this.streamSimilarityThreshold,
      streamMaxSkipFrames: this.streamMaxSkipFrames,
      streamRuntimePreset: this.streamRuntimePreset,
      streamMotionMode: this.streamMotionMode,
      streamMotionIntensity: this.streamMotionIntensity,
      streamMotionSpeed: this.streamMotionSpeed,
      scenePanelTab: this.scenePanelTab,
      selectedEntity: this.selectedEntity,
      sceneMaskNegated: this.sceneMaskNegated,
      sceneMaskAbsolute: this.sceneMaskAbsolute,
      regionShape: this.regionShape,
      generatorActive: this.generatorActive,
      leftPanelWidth: this.leftPanelWidth,
      rightPanelWidth: this.rightPanelWidth,
    }
  }

  private validScenePanelTab(value: unknown): value is ScenePanelTab {
    return value === 'sdxl' || value === 'z-image' || value === 'streamdiffusion' || value === 'sana' || value === 'causal-forcing' || value === 'fastvideo'
  }

  private validStreamCfgType(value: unknown): value is string {
    return value === 'none' || value === 'self' || value === 'initialize' || value === 'full'
  }

  private validStreamRuntimePreset(value: unknown): value is StreamRuntimePreset {
    return value === 'diffusers' || value === 'lcm-lora-sdxl' || value === 'sdxl-lightning-4step' || value === 'sdxl-turbo'
      || value === 'sana-sprint-0.6b' || value === 'sana-sprint-1.6b'
  }

  private validStreamMotionMode(value: unknown): value is StreamMotionMode {
    return value === 'none' || value === 'sway' || value === 'orbit' || value === 'push' || value === 'zoom'
  }

  private validSeedMode(value: unknown): value is SeedMode {
    return value === 'fixed' || value === 'random' || value === 'increment' || value === 'decrement'
  }

  private validLatentReuseNoiseMode(value: unknown): value is LatentReuseNoiseMode {
    return value === 'none' || value === 'fixed' || value === 'random'
  }

  private validTransparentMethod(value: unknown): value is string {
    return value === 'auto' || value === 'fast' || value === 'layerdiffuse'
  }

  private persistOptions() {
    if (!this.optionsLoaded) return
    if (!this.hasProjectMemoryContent() && !window.localStorage.getItem(PROJECT_STATE_KEY)) {
      window.localStorage.removeItem(OPTIONS_KEY)
      return
    }
    const options = this.currentStoredOptions()
    window.localStorage.setItem(OPTIONS_KEY, JSON.stringify(options))
  }

  private hasProjectMemoryContent() {
    return !!(
      this.layers.length ||
      this.regions.length ||
      this.generatedLayers.length ||
      this.outputImage ||
      this.sourceRoot ||
      this.sourceImages.length ||
      this.selectionRect ||
      this.maskCanvases.size
    )
  }

  private handleKeyDown = (event: KeyboardEvent) => {
    if (this.eventStartedInEditable(event)) return
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
      event.preventDefault()
      void this.undo()
      return
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'y') {
      event.preventDefault()
      void this.redo()
      return
    }
    if (!event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return
    const shortcut = TOOL_SHORTCUTS[event.key.toLowerCase()]
    if (shortcut) {
      event.preventDefault()
      this.setTool(shortcut)
    }
  }

  private eventStartedInEditable(event: Event) {
    return event.composedPath().some((target) => {
      if (!(target instanceof HTMLElement)) return false
      return target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement || target.isContentEditable
    })
  }

  private pushHistory() {
    if (this.isRestoringHistory || !this.fabricCanvas || !this.maskCanvasElement) return
    const snapshot = this.captureSnapshot()
    if (!snapshot) return
    this.undoStack = [...this.undoStack, snapshot].slice(-HISTORY_LIMIT)
    this.redoStack = []
    this.requestUpdate()
  }

  private captureSnapshot(): EditorSnapshot | undefined {
    if (!this.fabricCanvas || !this.maskCanvasElement) return undefined
    const canvas = normalizeCanvasPaintChildren((this.fabricCanvas as Canvas & { toJSON: (properties?: string[]) => unknown }).toJSON([...CANVAS_CUSTOM_PROPERTIES]))
    return {
      canvas,
      mask: this.maskCanvasElement.toDataURL('image/png'),
      regions: this.regions.map((region) => ({ ...region, rect: { ...region.rect, points: region.rect.points ? [...region.rect.points] : undefined } })),
      stageWidth: this.stageWidth,
      stageHeight: this.stageHeight,
      selectionRect: this.selectionRect ? { ...this.selectionRect } : undefined,
      selectedLayerId: this.selectedLayerId,
    }
  }

  private undo = async () => {
    const snapshot = this.undoStack.at(-1)
    if (!snapshot || !this.fabricCanvas) return
    const current = this.captureSnapshot()
    if (current) this.redoStack = [...this.redoStack, current].slice(-HISTORY_LIMIT)
    this.undoStack = this.undoStack.slice(0, -1)
    this.requestUpdate()
    await this.restoreSnapshot(snapshot)
  }

  private redo = async () => {
    const snapshot = this.redoStack.at(-1)
    if (!snapshot || !this.fabricCanvas) return
    const current = this.captureSnapshot()
    if (current) this.undoStack = [...this.undoStack, current].slice(-HISTORY_LIMIT)
    this.redoStack = this.redoStack.slice(0, -1)
    this.requestUpdate()
    await this.restoreSnapshot(snapshot)
  }

  private restoreSnapshot = async (snapshot: EditorSnapshot) => {
    if (!this.fabricCanvas) return
    this.isRestoringHistory = true
    try {
      this.stageWidth = snapshot.stageWidth
      this.stageHeight = snapshot.stageHeight
      this.generationWidth = this.alignGenerationSize(snapshot.stageWidth)
      this.generationHeight = this.alignGenerationSize(snapshot.stageHeight)
      this.fabricCanvas.setDimensions({ width: snapshot.stageWidth, height: snapshot.stageHeight })
      this.maskCanvasElement.width = snapshot.stageWidth
      this.maskCanvasElement.height = snapshot.stageHeight
      this.maskContext = this.maskCanvasElement.getContext('2d') ?? undefined
      await (this.fabricCanvas as Canvas & { loadFromJSON: (json: unknown) => Promise<Canvas> }).loadFromJSON(normalizeCanvasPaintChildren(snapshot.canvas))
      await this.restoreMask(snapshot.mask)
      this.regions = snapshot.regions ?? []
      this.selectionRect = snapshot.selectionRect ? { ...snapshot.selectionRect } : undefined
      const active = snapshot.selectedLayerId ? this.findLayer(snapshot.selectedLayerId) : undefined
      if (active) this.fabricCanvas.setActiveObject(active)
      else this.fabricCanvas.discardActiveObject()
      this.selectedLayerId = snapshot.selectedLayerId
      this.fabricCanvas.requestRenderAll()
    } finally {
      this.isRestoringHistory = false
      this.syncLayers()
      this.syncCanvasGeometry()
    }
  }

  private restoreMask(dataUrl: string) {
    return new Promise<void>((resolve) => {
      const image = new Image()
      image.onload = () => {
        if (!this.maskContext) return resolve()
        this.maskContext.clearRect(0, 0, this.stageWidth, this.stageHeight)
        this.maskContext.drawImage(image, 0, 0, this.stageWidth, this.stageHeight)
        resolve()
      }
      image.onerror = () => resolve()
      image.src = dataUrl
    })
  }

  private scheduleProjectPersist(delay = 900) {
    if (!this.optionsLoaded || this.projectRestoreInFlight) return
    if (this.projectSaveTimer) window.clearTimeout(this.projectSaveTimer)
    this.projectSaveTimer = window.setTimeout(() => void this.saveProjectNow().catch(() => undefined), delay)
  }

  private async saveProjectNow() {
    if (!this.optionsLoaded || this.projectRestoreInFlight || this.projectSaveInFlight || !this.fabricCanvas) return
    this.projectSaveInFlight = true
    try {
      const state = await this.captureSerializableProjectState()
      window.localStorage.setItem(PROJECT_STATE_KEY, JSON.stringify(state))
      this.projectStatus = `Saved ${new Date(state.savedAt).toLocaleTimeString()} · ${state.resources.length} resource${state.resources.length === 1 ? '' : 's'}`
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      this.projectStatus = `Save failed: ${message}`
      this.reportError(message)
      throw error
    } finally {
      this.projectSaveInFlight = false
    }
  }

  private async restorePersistedProject() {
    const serialized = window.localStorage.getItem(PROJECT_STATE_KEY)
    if (!serialized) return
    this.projectRestoreInFlight = true
    try {
      await this.restoreSerializableProjectState(JSON.parse(serialized) as SerializedProjectState)
      this.projectStatus = 'Restored local project'
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      this.projectStatus = `Restore failed: ${message}`
      this.reportError(message)
      throw error
    } finally {
      this.projectRestoreInFlight = false
    }
  }

  private async captureSerializableProjectState(): Promise<SerializedProjectState> {
    if (!this.fabricCanvas) throw new Error('No canvas is available to serialize')
    this.saveCurrentMaskScope()
    const resources = new Map<string, ArchiveResourceEntry>()
    const canvas = normalizeCanvasPaintChildren((this.fabricCanvas as Canvas & { toJSON: (properties?: string[]) => unknown }).toJSON([...CANVAS_CUSTOM_PROPERTIES]))
    const masks = await Promise.all(
      [...this.maskCanvases.entries()].map(async ([scope, canvas]) => ({
        scope,
        url: (await this.storeResourceFromDataUrl(canvas.toDataURL('image/png'), resources)).url,
      })),
    )
    return {
      version: PROJECT_VERSION,
      savedAt: Date.now(),
      archiveRoot: ARCHIVE_ROOT,
      options: this.currentStoredOptions(),
      canvas: await this.archiveObjectResources(canvas, resources),
      masks,
      regions: this.regions.map((region) => ({ ...region, rect: { ...region.rect, points: region.rect.points ? [...region.rect.points] : undefined } })),
      generatedLayers: (await this.archiveObjectResources(this.generatedLayers, resources)) as GeneratedLayer[],
      outputImage: this.outputImage ? await this.archiveResourceUrl(this.outputImage, resources) : '',
      sourceRoot: this.sourceRoot,
      selectedSourceRel: this.selectedSourceRel,
      sourceImages: this.sourceImages.map((image) => ({ ...image })),
      stageWidth: this.stageWidth,
      stageHeight: this.stageHeight,
      selectionRect: this.selectionRect ? { ...this.selectionRect } : undefined,
      selectedLayerId: this.selectedLayerId,
      selectedEntity: this.selectedEntity,
      layerPanelTab: this.layerPanelTab,
      scenePanelTab: this.scenePanelTab,
      resources: [...resources.values()].sort((left, right) => left.path.localeCompare(right.path)),
    }
  }

  private async restoreSerializableProjectState(state: SerializedProjectState) {
    if (!this.fabricCanvas) return
    window.localStorage.setItem(OPTIONS_KEY, JSON.stringify(state.options ?? {}))
    this.loadStoredOptions()
    this.stageWidth = Math.round(this.clamp(state.stageWidth ?? this.stageWidth, 256, 1536, this.stageWidth))
    this.stageHeight = Math.round(this.clamp(state.stageHeight ?? this.stageHeight, 256, 1536, this.stageHeight))
    this.generationWidth = this.alignGenerationSize(this.stageWidth)
    this.generationHeight = this.alignGenerationSize(this.stageHeight)
    this.fabricCanvas.setDimensions({ width: this.stageWidth, height: this.stageHeight })
    this.maskCanvasElement.width = this.stageWidth
    this.maskCanvasElement.height = this.stageHeight
    this.maskContext = this.maskCanvasElement.getContext('2d') ?? undefined
    const canvas = normalizeCanvasPaintChildren(await this.resolveObjectResources(state.canvas))
    await (this.fabricCanvas as Canvas & { loadFromJSON: (json: unknown) => Promise<Canvas> }).loadFromJSON(canvas)
    this.maskCanvases.clear()
    for (const mask of state.masks ?? []) {
      const dataUrl = await this.resolveArchiveUrl(mask.url)
      const canvas = await this.canvasFromDataUrl(dataUrl)
      this.maskCanvases.set(mask.scope, canvas)
    }
    this.activeMaskScope = undefined
    this.regions = state.regions ?? []
    this.generatedLayers = (await this.resolveObjectResources(state.generatedLayers ?? [])) as GeneratedLayer[]
    this.outputImage = state.outputImage ? await this.resolveArchiveUrl(state.outputImage) : ''
    this.sourceRoot = state.sourceRoot ?? ''
    this.selectedSourceRel = state.selectedSourceRel ?? ''
    this.sourceImages = state.sourceImages ?? []
    this.selectionRect = state.selectionRect ? { ...state.selectionRect } : undefined
    this.selectedEntity = state.selectedEntity === 'layer' ? 'layer' : 'scene'
    this.layerPanelTab = state.layerPanelTab === 'sampling' ? 'mask' : state.layerPanelTab ?? this.layerPanelTab
    this.scenePanelTab = state.scenePanelTab ?? this.scenePanelTab
    const active = state.selectedLayerId ? this.findLayer(state.selectedLayerId) : undefined
    if (active) this.fabricCanvas.setActiveObject(active)
    else this.fabricCanvas.discardActiveObject()
    this.selectedLayerId = state.selectedLayerId ?? ''
    this.syncMaskScope()
    this.fabricCanvas.requestRenderAll()
    this.syncLayers()
    this.syncCanvasGeometry()
    this.refreshMaskPreview()
  }

  private async archiveObjectResources(value: unknown, resources: Map<string, ArchiveResourceEntry>): Promise<unknown> {
    if (typeof value === 'string') return this.isSerializableResourceUrl(value) ? this.archiveResourceUrl(value, resources) : value
    if (Array.isArray(value)) return Promise.all(value.map((item) => this.archiveObjectResources(item, resources)))
    if (!value || typeof value !== 'object') return value
    const output: Record<string, unknown> = {}
    for (const [key, item] of Object.entries(value)) output[key] = await this.archiveObjectResources(item, resources)
    return output
  }

  private async resolveObjectResources(value: unknown): Promise<unknown> {
    if (typeof value === 'string') return value.startsWith(ARCHIVE_ROOT) ? this.resolveArchiveUrl(value) : value
    if (Array.isArray(value)) return Promise.all(value.map((item) => this.resolveObjectResources(item)))
    if (!value || typeof value !== 'object') return value
    const output: Record<string, unknown> = {}
    for (const [key, item] of Object.entries(value)) output[key] = await this.resolveObjectResources(item)
    return output
  }

  private isSerializableResourceUrl(value: string) {
    return value.startsWith('data:image/') || value.startsWith('data:video/') || value.startsWith('blob:') || /^https?:\/\//i.test(value) || value.startsWith(ARCHIVE_ROOT)
  }

  private async archiveResourceUrl(url: string, resources: Map<string, ArchiveResourceEntry>) {
    if (url.startsWith(ARCHIVE_ROOT)) {
      const entry = this.resourceEntryForArchiveUrl(url)
      if (entry) resources.set(entry.hash, entry)
      return url
    }
    if (url.startsWith('data:')) return (await this.storeResourceFromDataUrl(url, resources)).url
    const response = await fetch(url)
    if (!response.ok) throw new Error(`Could not archive resource ${url}: ${response.status}`)
    const blob = await response.blob()
    return (await this.storeResourceBytes(new Uint8Array(await blob.arrayBuffer()), blob.type || 'application/octet-stream', resources)).url
  }

  private async storeResourceFromDataUrl(dataUrl: string, resources: Map<string, ArchiveResourceEntry>) {
    const match = /^data:([^;,]+)(?:;[^,]*)?,/.exec(dataUrl)
    return this.storeResourceBytes(await this.dataUrlBytes(dataUrl), match?.[1] || 'application/octet-stream', resources)
  }

  private async storeResourceBytes(bytes: Uint8Array, mediaType: string, resources?: Map<string, ArchiveResourceEntry>) {
    const hash = await this.sha256(bytes)
    const path = archivePathForResource(hash, mediaType)
    const url = archiveUrlForResource(hash, mediaType)
    const entry: ArchiveResourceEntry = { hash, path, url, mediaType, size: bytes.length, storedAt: Date.now() }
    const key = `${PROJECT_RESOURCE_PREFIX}${hash}`
    if (!window.localStorage.getItem(key)) {
      await this.storeResourceData(hash, this.bytesToDataUrl(bytes, mediaType))
      try {
        window.localStorage.setItem(key, JSON.stringify(entry))
      } catch (error) {
        this.pruneLegacyResourceStorage()
        window.localStorage.setItem(key, JSON.stringify(entry))
      }
    }
    resources?.set(hash, entry)
    return entry
  }

  private resourceEntryForArchiveUrl(url: string) {
    const hash = resourceHashFromArchiveUrl(url)
    if (!hash) return undefined
    const stored = window.localStorage.getItem(`${PROJECT_RESOURCE_PREFIX}${hash}`)
    if (!stored) return undefined
    const parsed = JSON.parse(stored) as ArchiveResourceEntry & { dataUrl: string }
    return { hash: parsed.hash, path: parsed.path, url: parsed.url, mediaType: parsed.mediaType, size: parsed.size, storedAt: parsed.storedAt }
  }

  private pruneLegacyResourceStorage() {
    const keys = Array.from({ length: window.localStorage.length }, (_, index) => window.localStorage.key(index)).filter((key): key is string => !!key)
    for (const key of keys) {
      if (!key.startsWith(PROJECT_RESOURCE_PREFIX)) continue
      const value = window.localStorage.getItem(key)
      if (value?.includes('"dataUrl"')) window.localStorage.removeItem(key)
    }
  }

  private async resolveArchiveUrl(url: string) {
    if (!url.startsWith(ARCHIVE_ROOT)) return url
    const hash = resourceHashFromArchiveUrl(url)
    if (!hash) throw new Error(`Invalid archive URL: ${url}`)
    const stored = window.localStorage.getItem(`${PROJECT_RESOURCE_PREFIX}${hash}`)
    if (!stored) throw new Error(`Missing archive resource: ${url}`)
    const parsed = JSON.parse(stored) as { dataUrl?: string }
    if (parsed.dataUrl) return parsed.dataUrl
    const dataUrl = await this.loadResourceData(hash)
    if (!dataUrl) throw new Error(`Missing archive resource data: ${url}`)
    return dataUrl
  }

  private resourceDb() {
    return new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open(PROJECT_RESOURCE_DB, 1)
      request.onupgradeneeded = () => request.result.createObjectStore(PROJECT_RESOURCE_STORE)
      request.onsuccess = () => resolve(request.result)
      request.onerror = () => reject(request.error ?? new Error('Could not open resource database'))
    })
  }

  private async storeResourceData(hash: string, dataUrl: string) {
    const db = await this.resourceDb()
    await new Promise<void>((resolve, reject) => {
      const transaction = db.transaction(PROJECT_RESOURCE_STORE, 'readwrite')
      transaction.objectStore(PROJECT_RESOURCE_STORE).put(dataUrl, hash)
      transaction.oncomplete = () => resolve()
      transaction.onerror = () => reject(transaction.error ?? new Error('Could not store resource data'))
    })
    db.close()
  }

  private async loadResourceData(hash: string) {
    const db = await this.resourceDb()
    try {
      return await new Promise<string | undefined>((resolve, reject) => {
        const transaction = db.transaction(PROJECT_RESOURCE_STORE, 'readonly')
        const request = transaction.objectStore(PROJECT_RESOURCE_STORE).get(hash)
        request.onsuccess = () => resolve(typeof request.result === 'string' ? request.result : undefined)
        request.onerror = () => reject(request.error ?? new Error('Could not load resource data'))
      })
    } finally {
      db.close()
    }
  }

  private async sha256(bytes: Uint8Array) {
    const source = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer
    const digest = await crypto.subtle.digest('SHA-256', source)
    return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, '0')).join('')
  }

  private bytesToDataUrl(bytes: Uint8Array, mediaType: string) {
    let binary = ''
    const chunkSize = 0x8000
    for (let index = 0; index < bytes.length; index += chunkSize) binary += String.fromCharCode(...bytes.slice(index, index + chunkSize))
    return `data:${mediaType};base64,${btoa(binary)}`
  }

  private async canvasFromDataUrl(dataUrl: string) {
    const image = new Image()
    await new Promise<void>((resolve, reject) => {
      image.onload = () => resolve()
      image.onerror = () => reject(new Error('Could not load archived canvas image'))
      image.src = dataUrl
    })
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    canvas.getContext('2d')!.drawImage(image, 0, 0, this.stageWidth, this.stageHeight)
    return canvas
  }

  private exportProject = async () => {
    await this.saveProjectNow()
    const serialized = window.localStorage.getItem(PROJECT_STATE_KEY)
    if (!serialized) return
    const state = JSON.parse(serialized) as SerializedProjectState
    const files: { name: string; bytes: Uint8Array }[] = [
      { name: 'state.json', bytes: new TextEncoder().encode(JSON.stringify(state, null, 2)) },
    ]
    for (const resource of state.resources) {
      const dataUrl = await this.resolveArchiveUrl(resource.url)
      files.push({ name: resource.path, bytes: await this.dataUrlBytes(dataUrl) })
    }
    this.downloadBlob(this.createZip(files), `rtdiffusion-project-${new Date().toISOString().replace(/[:.]/g, '-')}.zip`)
  }

  private importProject = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.zip,application/zip'
    input.onchange = () => {
      const file = input.files?.[0]
      if (file) void this.importProjectFile(file)
    }
    input.click()
  }

  private async importProjectFile(file: File) {
    const entries = this.readZipEntries(new Uint8Array(await file.arrayBuffer()))
    const stateBytes = entries.get('state.json')
    if (!stateBytes) throw new Error('Project archive is missing state.json')
    for (const [name, bytes] of entries) {
      if (!name.startsWith('resources/')) continue
      await this.storeResourceBytes(bytes, this.mediaTypeForPath(name))
    }
    const state = JSON.parse(new TextDecoder().decode(stateBytes)) as SerializedProjectState
    window.localStorage.setItem(PROJECT_STATE_KEY, JSON.stringify(state))
    await this.restoreSerializableProjectState(state)
    this.projectStatus = `Imported ${file.name}`
  }

  private flushProjectMemory = async () => {
    if (!window.confirm('Flush local project memory? Scheduler presets will be kept.')) return
    if (this.projectSaveTimer) {
      window.clearTimeout(this.projectSaveTimer)
      this.projectSaveTimer = undefined
    }
    this.stopStream()
    this.optionsLoaded = false
    for (const key of Object.keys(window.localStorage)) {
      if (isFlushableProjectStorageKey(key)) window.localStorage.removeItem(key)
    }
    this.undoStack = []
    this.redoStack = []
    this.prompt = ''
    this.negativePrompt = ''
    this.selectedLayerId = ''
    this.selectedEntity = 'scene'
    this.layerPanelTab = 'properties'
    this.scenePanelTab = 'sdxl'
    this.selectionRect = undefined
    this.regions = []
    this.generatedLayers = []
    this.outputImage = ''
    this.sourceRoot = ''
    this.sourceImages = []
    this.selectedSourceRel = ''
    this.maskCanvases.clear()
    this.activeMaskScope = undefined
    this.layerMaskNegated.clear()
    this.maskContext?.clearRect(0, 0, this.stageWidth, this.stageHeight)
    this.maskPreviewContext?.clearRect(0, 0, this.stageWidth, this.stageHeight)
    this.fabricCanvas?.clear()
    this.fabricCanvas?.discardActiveObject()
    this.fabricCanvas?.requestRenderAll()
    this.layers = []
    this.projectStatus = 'Flushed local project memory; scheduler presets kept'
    await this.updateComplete
    for (const key of Object.keys(window.localStorage)) {
      if (isFlushableProjectStorageKey(key)) window.localStorage.removeItem(key)
    }
    this.optionsLoaded = true
    this.applyToolMode()
    this.refreshMaskPreview()
  }

  private downloadBlob(blob: Blob, filename: string) {
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = filename
    link.click()
    window.setTimeout(() => URL.revokeObjectURL(link.href), 1000)
  }

  render() {
    const selectedPreset = `${this.stageWidth}x${this.stageHeight}`
    return html`
      <main class="shell" @paste=${this.handlePaste}>
        <header class="topbar">
          <div class="quick-actions">
            <strong>RTDiffusion</strong>
            <button title="Undo" @click=${this.undo} ?disabled=${!this.undoStack.length}>↶ Undo</button>
            <button title="Redo" @click=${this.redo} ?disabled=${!this.redoStack.length}>↷ Redo</button>
            <button title="Import image" @click=${this.loadImage}>▣ Import</button>
            <button title="New layer" @click=${this.addBlankLayer}>＋ Layer</button>
            <button title="Add video layer" @click=${this.addVideoLayer}>▶ Video</button>
            <button title="Export output" @click=${this.downloadOutput}>⇩ Export</button>
            <button title="Save project locally" @click=${() => void this.saveProjectNow()}>▣ Save</button>
            <button title="Export project archive" @click=${this.exportProject}>⇩ Project</button>
            <button title="Import project archive" @click=${this.importProject}>⇧ Project</button>
            <button title="Flush local project memory; keep scheduler presets" @click=${this.flushProjectMemory}>⌧ Flush</button>
            <small>${this.projectStatus}</small>
          </div>
          <div class="tool-options">
            <label class="field inline">
              <span>Viewport</span>
              <select .value=${selectedPreset} @change=${this.selectPreset}>
                ${SIZE_PRESETS.map(
                  (preset) => html`<option value=${`${preset.width}x${preset.height}`} ?selected=${selectedPreset === `${preset.width}x${preset.height}`}>${preset.label}</option>`,
                )}
              </select>
            </label>
            ${this.numberControl('Brush', this.brushSize, 1, 4096, 1, 0, (value) => this.setBrushSize(value))}
            ${this.numberControl('Hardness', this.brushHardness, 0, 100, 1, 0, (value) => this.setBrushHardness(value))}
          </div>
        </header>

        <section
          class="workspace ${this.stageWidth >= this.stageHeight ? 'landscape' : 'portrait'}"
          style="--left-panel-width: ${this.leftPanelWidth}px; --right-panel-width: ${this.rightPanelWidth}px;"
          @pointermove=${this.dragPanelResize}
          @pointerup=${this.endPanelResize}
          @pointercancel=${this.endPanelResize}
        >
          <aside class="panel tools-panel">
            <section class="dock-section">
              <h2>Tools</h2>
              ${this.renderEditorDestination()}
            <div class="tool-row" role="group" aria-label="editor tools">
              ${this.toolButton('select', '↖ Select')}
              ${this.toolButton('region', '▢ Mask')}
              ${this.toolButton('brush', '● Paint')}
              ${this.toolButton('eraser', '⌫ Erase')}
              ${this.toolButton('text', 'T Text')}
              ${this.toolButton('line', '╱ Line')}
              ${this.toolButton('rect', '▭ Rect')}
              ${this.toolButton('fill', '▨ Fill')}
            </div>
            </section>
            <div class="action-row" aria-label="quick actions">
              <button class="tool-icon" title="Add text (T)" aria-label="Add text" @click=${this.addTextLayer}>T</button>
              <button class="tool-icon" title="Delete selection" aria-label="Delete selection" @click=${this.deleteSelection}>⌦</button>
              <button class="tool-icon" title="Clear paint" aria-label="Clear paint" @click=${this.clearActiveLayer}>⌫</button>
              <button class="tool-icon" title="Clear masks" aria-label="Clear masks" @click=${this.clearCurrentMask}>○</button>
              <button class="tool-icon" title="Crop selection" aria-label="Crop selection" @click=${this.cropToSelection}>⌗</button>
              <button class="tool-icon" title="Invert selection" aria-label="Invert selection" @click=${this.invertSelection}>◩</button>
              <button class="tool-icon" title="Clear selection" aria-label="Clear selection" @click=${this.clearSelection}>×</button>
            </div>
            <section class="dock-section">
              <h2>Colors</h2>
              ${this.renderColorPanel()}
            </section>
            <section class="dock-section layers-dock">
              <h2>Entities</h2>
              <div class="scene-entity-row">
                <button class=${this.selectedEntity === 'scene' ? 'scene-entity selected' : 'scene-entity'} @click=${this.selectSceneEntity}>
                  <span>Scene</span>
                  <small>${this.layers.length} layer${this.layers.length === 1 ? '' : 's'}</small>
                </button>
              </div>
              ${this.renderLayerList()}
            </section>
            <button
              class="panel-resizer left-resizer"
              title="Resize tools panel"
              aria-label="Resize tools panel"
              @pointerdown=${(event: PointerEvent) => this.beginPanelResize(event, 'left')}
            ></button>
          </aside>

          <section class="studio">
            <div
              class="viewer"
              @dragover=${this.allowDrop}
              @drop=${this.dropImages}
            >
              <div
                class="canvas-pane work-pane"
                @pointermove=${(event: PointerEvent) => this.showViewportOverlay(event, 'input')}
                @pointerleave=${(event: PointerEvent) => this.leaveViewport(event, 'input')}
              >
                <div class=${this.inputOverlayVisible ? 'viewport-overlay pane-head visible' : 'viewport-overlay pane-head'}>
                  <strong>Input</strong>
                  <span>${this.stageWidth} x ${this.stageHeight}</span>
                  <div class="viewport-toggles">
                    <button class=${this.inputPaintVisible ? 'active' : ''} title="Show or hide paint" @click=${() => (this.inputPaintVisible = !this.inputPaintVisible)}>Paint</button>
                    <button class=${this.inputMaskVisible ? 'active' : ''} title="Show or hide masks" @click=${() => (this.inputMaskVisible = !this.inputMaskVisible)}>Masks</button>
                    <button class=${this.editorDest === 'paint' ? 'active' : ''} title="Send tools to paint" @click=${() => this.setEditorDest('paint')}>To Paint</button>
                    <button class=${this.editorDest === 'mask' ? 'active' : ''} title="Send tools to masks" @click=${() => this.setEditorDest('mask')}>To Masks</button>
                    <button class=${this.inputActiveMaskOnly ? 'active' : ''} title="Show only active color mask" @click=${() => { this.inputActiveMaskOnly = !this.inputActiveMaskOnly; this.scheduleMaskPreviewRefresh(true) }}>Active</button>
                  </div>
                </div>
                <div
                  class="work-surface"
                  @wheel=${(event: WheelEvent) => this.zoomViewport(event, 'input')}
                  @pointerdown=${this.startViewportPan}
                  @pointermove=${(event: PointerEvent) => this.moveViewport(event, 'input')}
                  @pointerup=${this.endViewportPan}
                  @contextmenu=${(event: Event) => event.preventDefault()}
                >
                  <div
                    class=${this.inputPaintVisible ? 'stage' : 'stage paint-hidden'}
                    style="--stage-width: ${this.stageWidth}; --stage-height: ${this.stageHeight}; --zoom: ${this.inputZoom};"
                    @pointermove=${this.trackCursor}
                    @pointerenter=${this.showCursor}
                    @pointerleave=${this.hideCursor}
                  >
                    <canvas id="edit-canvas"></canvas>
                    <canvas id="mask-preview-canvas" class=${this.maskPreviewVisible() ? 'mask-preview visible' : 'mask-preview'}></canvas>
                    <canvas
                      id="mask-canvas"
                      class=${this.maskCanvasActive() ? 'mask active' : 'mask'}
                      @pointerdown=${this.startOverlayStroke}
                      @pointermove=${this.moveOverlayStroke}
                      @pointerup=${this.endOverlayStroke}
                      @pointerleave=${this.endOverlayStroke}
                    ></canvas>
                    <canvas id="editor-overlay"></canvas>
                    ${this.renderSelectionOverlay()}
                    <div
                      class=${this.cursor.visible && (this.toolMode === 'brush' || this.toolMode === 'eraser')
                        ? 'cursor visible'
                        : 'cursor'}
                      style="left: ${(this.cursor.x / this.stageWidth) * 100}%; top: ${(this.cursor.y / this.stageHeight) * 100}%; width: ${(this.brushSize / this.stageWidth) * 100}%; height: ${(this.brushSize / this.stageHeight) * 100}%;"
                    ></div>
                  </div>
                </div>
                <div class=${this.inputOverlayVisible ? 'viewport-overlay pane-foot visible' : 'viewport-overlay pane-foot'}>
                  <span>${this.toolMode}</span>
                </div>
              </div>
              <div
                class="output-pane work-pane"
                @pointermove=${(event: PointerEvent) => this.showViewportOverlay(event, 'output')}
                @pointerleave=${(event: PointerEvent) => this.leaveViewport(event, 'output')}
              >
                <div class=${this.outputOverlayVisible ? 'viewport-overlay pane-head visible' : 'viewport-overlay pane-head'}>
                  <strong>Output</strong>
                  <span>${this.stageWidth} x ${this.stageHeight}</span>
                  <span>${Math.round(this.outputZoom * 100)}%</span>
                </div>
                <div
                  class="work-surface"
                  @wheel=${(event: WheelEvent) => this.zoomViewport(event, 'output')}
                  @pointerdown=${this.startViewportPan}
                  @pointermove=${(event: PointerEvent) => this.moveViewport(event, 'output')}
                  @pointerup=${this.endViewportPan}
                  @contextmenu=${(event: Event) => event.preventDefault()}
                >
                  <div
                    class=${this.outputPreviewClass()}
                    style=${this.outputPreviewStyle()}
                  >
                      ${this.realtimeVideoActive
                      ? html`<img id="stream-output-img" class="stream-output-video" alt="realtime output" /><canvas id="stream-output-canvas" style="display:none" width=${this.stageWidth} height=${this.stageHeight}></canvas>`
                      : this.activeMotionVideo
                      ? html`
                        <video class=${this.activeMotionIsStream && !this.motionStreamHasSegments ? 'stream-pending' : ''} src=${this.activeMotionVideo} autoplay muted playsinline controls ?loop=${!this.activeMotionIsStream}></video>
                          ${this.activeMotionIsStream && !this.motionStreamHasSegments && (this.activeMotionFrame || this.outputImage)
                            ? html`<img class="stream-preview-fallback" src=${this.activeMotionFrame || this.outputImage} alt="latest generated frame" />`
                            : ''}
                        `
                      : this.activeMotionFrame || this.outputImage
                        ? html`<img src=${this.activeMotionFrame || this.outputImage} alt="latest generated frame" />`
                        : html`<div class="empty">Waiting for diffusion</div>`}
                  </div>
                </div>
                <div class=${this.outputOverlayVisible ? 'viewport-overlay pane-foot pane-action-overlay visible' : 'viewport-overlay pane-foot pane-action-overlay'}>
                  <button @click=${this.applyOutput}>Apply output as layer</button>
                  <button
                    class=${this.debugStreamsEnabled ? 'active' : ''}
                    title="Toggle debug stream monitor"
                    @click=${() => { this.debugStreamsEnabled = !this.debugStreamsEnabled; if (!this.debugStreamsEnabled) { this.debugChannelNames = []; this._debugChannelData.clear() } }}
                  >Debug streams</button>
                </div>
                ${this.debugStreamsEnabled ? this.renderDebugOverlay() : ''}
              </div>
            </div>
          </section>

          <aside class="panel inspector">
            <button
              class="panel-resizer right-resizer"
              title="Resize inspector panel"
              aria-label="Resize inspector panel"
              @pointerdown=${(event: PointerEvent) => this.beginPanelResize(event, 'right')}
            ></button>
            <section class="context-panel">
              ${this.renderContextTabs()}
              <div class="context-body">${this.renderContextPanel()}</div>
            </section>

          </aside>
        </section>
        <footer class="status-bar">
          <span>${this.status}</span>
          <span>${this.displayFps > 0 ? this.displayFps.toFixed(1) : this.fps.toFixed(1)} display FPS</span>
          <span>${this.fps.toFixed(1)} gen FPS</span>
          <span>${Math.round(this.latency)} ms</span>
          <span>${this.stageWidth} x ${this.stageHeight}</span>
          <span>${this.backendMode}</span>
          ${this.renderFooterErrors()}
          ${this.renderFooterTaskMonitor()}
        </footer>
        ${this.renderPickerDialog()}
        ${this.renderColorPopup()}
        <video id="video-input" style="display:none" muted playsinline preload="auto" @loadedmetadata=${this.onVideoMetadata} @ended=${this.onVideoEnded}></video>
      </main>
    `
  }

  private toolButton(mode: ToolMode, _label: string) {
    const tool = TOOL_DEFS[mode]
    return html`<button
      class=${this.toolMode === mode ? 'tool-icon active' : 'tool-icon'}
      title=${`${tool.label} (${tool.shortcut})`}
      aria-label=${`${tool.label}, shortcut ${tool.shortcut}`}
      @click=${() => this.setTool(mode)}
    >${tool.icon}</button>`
  }

  private renderEditorDestination() {
    return html`<div class="dest-switch" role="group" aria-label="editor destination">
      <button class=${this.editorDest === 'paint' ? 'active' : ''} @click=${() => this.setEditorDest('paint')}>Paint</button>
      <button class=${this.editorDest === 'mask' ? 'active' : ''} @click=${() => this.setEditorDest('mask')}>Masks</button>
    </div>`
  }

  private outputPreviewClass() {
    const classes = ['preview', 'native']
    if (this.awaitingFrame) classes.push('sampling')
    return classes.join(' ')
  }

  private outputPreviewStyle() {
    return [
      `--stage-width: ${this.stageWidth}`,
      `--stage-height: ${this.stageHeight}`,
      `--zoom: ${this.outputZoom}`,
    ].join('; ')
  }

  private renderColorPanel() {
    const rgba = this.activeRgba()
    return html`<div class="color-panel">
      <div class="color-stack" aria-label="primary and secondary colors">
        <button
          class=${this.activeColorSlot === 'primary' ? 'color-chip primary active' : 'color-chip primary'}
          style="--chip: ${this.brushColor}"
          title="Primary color"
          @click=${() => (this.activeColorSlot = 'primary')}
        ></button>
        <button
          class=${this.activeColorSlot === 'secondary' ? 'color-chip secondary active' : 'color-chip secondary'}
          style="--chip: ${this.secondaryBrushColor}"
          title="Secondary color / right-click color"
          @click=${() => (this.activeColorSlot = 'secondary')}
        ></button>
        <button class="swap-colors" title="Swap colors" @click=${this.swapColors}>⇄</button>
      </div>
      <div class="color-editor">
        <label class="color-native" title="Active color">
          <input type="color" .value=${this.colorInputValue()} @input=${this.pickColor} />
          <input class="hex-input" spellcheck="false" .value=${this.colorTextValue()} @input=${this.pickHexColor} />
        </label>
        <div class="channel-grid rgba-grid">
          ${this.channelInput('R', rgba.r, 0, 255, (value) => this.setActiveRgba({ r: value }))}
          ${this.channelInput('G', rgba.g, 0, 255, (value) => this.setActiveRgba({ g: value }))}
          ${this.channelInput('B', rgba.b, 0, 255, (value) => this.setActiveRgba({ b: value }))}
          ${this.channelInput('A', Math.round(rgba.a * 100), 0, 100, (value) => this.setActiveRgba({ a: value / 100 }))}
        </div>
      </div>
      <div class="palette quick-palette" aria-label="quick color palette">
        ${QUICK_PALETTE.slice(0, 15).map((color) => this.colorSwatch(color, 'Quick color'))}
        <button class="swatch palette-more" title="Open full color palette" aria-label="Open full color palette" @click=${this.openColorPopup}>+</button>
      </div>
    </div>`
  }

  private renderColorPopup() {
    if (!this.colorPopupOpen) return ''
    const rgb = this.activeRgb()
    const hsl = this.activeHsl()
    const rgba = this.activeRgba()
    return html`<section
      class="color-popup"
      style="left: ${this.colorPopupPosition.x}px; top: ${this.colorPopupPosition.y}px;"
      @pointermove=${this.dragColorPopup}
      @pointerup=${this.endColorPopupDrag}
      @pointercancel=${this.endColorPopupDrag}
    >
      <header class="color-popup-head" @pointerdown=${this.beginColorPopupDrag}>
        <strong>Color Palette</strong>
        <button title="Close palette" aria-label="Close color palette" @click=${() => (this.colorPopupOpen = false)}>×</button>
      </header>
      <div class="color-popup-body">
        <div class="popup-color-editor">
          <div class="color-preview-large" style="--chip: ${this.activeColor()}"></div>
          <label class="color-native" title="Active color">
            <input type="color" .value=${this.colorInputValue()} @input=${this.pickColor} />
            <input class="hex-input" spellcheck="false" .value=${this.colorTextValue()} @input=${this.pickHexColor} />
          </label>
          <div class="channel-grid rgba-grid popup-rgba">
            ${this.channelInput('R', rgb.r, 0, 255, (value) => this.setActiveRgba({ r: value }))}
            ${this.channelInput('G', rgb.g, 0, 255, (value) => this.setActiveRgba({ g: value }))}
            ${this.channelInput('B', rgb.b, 0, 255, (value) => this.setActiveRgba({ b: value }))}
            ${this.channelInput('A', Math.round(rgba.a * 100), 0, 100, (value) => this.setActiveRgba({ a: value / 100 }))}
          </div>
          <div class="hsl-sliders">
            ${this.colorSlider('H', hsl.h, 0, 360, 1, (value) => this.setActiveHsl({ h: value }))}
            ${this.colorSlider('S', hsl.s, 0, 100, 1, (value) => this.setActiveHsl({ s: value }))}
            ${this.colorSlider('L', hsl.l, 0, 100, 1, (value) => this.setActiveHsl({ l: value }))}
            ${this.colorSlider('A', Math.round(rgba.a * 100), 0, 100, 1, (value) => this.setActiveRgba({ a: value / 100 }))}
          </div>
        </div>
        <div class="palette full-palette" aria-label="full color palette">
          ${PALETTE.map((color) => this.colorSwatch(this.withActiveAlpha(color), 'Palette color'))}
        </div>
        <div class="planner popup-planner">
          <label class="field inline planner-mode">
            <span>Plan</span>
            <select .value=${this.colorPlannerMode} @change=${(event: Event) => (this.colorPlannerMode = (event.target as HTMLSelectElement).value as ColorPlannerMode)}>
              ${COLOR_PLANNER_MODES.map((mode) => html`<option value=${mode.value}>${mode.label}</option>`)}
            </select>
          </label>
          <div class="planner-row" aria-label="planned harmony colors">
            ${this.plannedColors().map((color) => this.colorSwatch(color, 'Planned color'))}
          </div>
          <div class="planner-row" aria-label="primary to secondary blend">
            ${this.blendColors(this.brushColor, this.secondaryBrushColor, 7).map((color) => this.colorSwatch(color, 'Blend color'))}
          </div>
          <div class="planner-row" aria-label="tints and shades">
            ${this.toneRamp(this.activeColor(), 7).map((color) => this.colorSwatch(color, 'Tone color'))}
          </div>
        </div>
      </div>
    </section>`
  }

  private colorSwatch(color: string, label: string) {
    return html`<button
      class=${this.colorIsActive(color) ? 'swatch active' : 'swatch'}
      style="--swatch: ${color}"
      title=${`${label} ${color}. Left click active slot, right click secondary.`}
      @click=${() => this.setActiveColor(color)}
      @contextmenu=${(event: Event) => this.setSecondaryFromPalette(event, color)}
    ></button>`
  }

  private channelInput(label: string, value: number, min: number, max: number, onChange: (value: number) => void) {
    return html`<label class="channel-input">
      <span>${label}</span>
      <input type="number" min=${String(min)} max=${String(max)} .value=${String(value)} @input=${(event: Event) => onChange(Math.round(this.clamp(Number((event.target as HTMLInputElement).value), min, max, value)))} />
    </label>`
  }

  private colorSlider(label: string, value: number, min: number, max: number, step: number, onChange: (value: number) => void) {
    return html`<label class="color-slider">
      <span>${label}</span>
      <input type="range" min=${min} max=${max} step=${step} .valueAsNumber=${value} @input=${(event: Event) => onChange(Number((event.target as HTMLInputElement).value))} />
      <output>${Math.round(value)}</output>
    </label>`
  }

  private openColorPopup = () => {
    this.colorPopupOpen = true
  }

  private beginColorPopupDrag = (event: PointerEvent) => {
    if ((event.target as HTMLElement).closest('button')) return
    event.preventDefault()
    this.colorPopupDrag = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      x: this.colorPopupPosition.x,
      y: this.colorPopupPosition.y,
      target: event.currentTarget as HTMLElement,
    }
    this.colorPopupDrag.target.setPointerCapture(event.pointerId)
  }

  private dragColorPopup = (event: PointerEvent) => {
    if (!this.colorPopupDrag || event.pointerId !== this.colorPopupDrag.pointerId) return
    const x = this.colorPopupDrag.x + event.clientX - this.colorPopupDrag.startX
    const y = this.colorPopupDrag.y + event.clientY - this.colorPopupDrag.startY
    this.colorPopupPosition = {
      x: Math.round(this.clamp(x, 8, Math.max(8, window.innerWidth - 420), this.colorPopupPosition.x)),
      y: Math.round(this.clamp(y, 8, Math.max(8, window.innerHeight - 420), this.colorPopupPosition.y)),
    }
  }

  private endColorPopupDrag = (event: PointerEvent) => {
    if (!this.colorPopupDrag || event.pointerId !== this.colorPopupDrag.pointerId) return
    this.colorPopupDrag.target.releasePointerCapture?.(event.pointerId)
    this.colorPopupDrag = undefined
  }

  private beginPanelResize(event: PointerEvent, side: 'left' | 'right') {
    if (event.button !== 0) return
    event.preventDefault()
    this.panelResize = {
      pointerId: event.pointerId,
      side,
      startX: event.clientX,
      startWidth: side === 'left' ? this.leftPanelWidth : this.rightPanelWidth,
      target: event.currentTarget as HTMLElement,
    }
    this.panelResize.target.setPointerCapture(event.pointerId)
  }

  private dragPanelResize = (event: PointerEvent) => {
    if (!this.panelResize || event.pointerId !== this.panelResize.pointerId) return
    event.preventDefault()
    const delta = event.clientX - this.panelResize.startX
    if (this.panelResize.side === 'left') {
      this.leftPanelWidth = Math.round(this.clamp(this.panelResize.startWidth + delta, 144, 420, this.leftPanelWidth))
    } else {
      this.rightPanelWidth = Math.round(this.clamp(this.panelResize.startWidth - delta, 220, 520, this.rightPanelWidth))
    }
  }

  private endPanelResize = (event: PointerEvent) => {
    if (!this.panelResize || event.pointerId !== this.panelResize.pointerId) return
    this.panelResize.target.releasePointerCapture?.(event.pointerId)
    this.panelResize = undefined
    this.persistOptions()
  }

  private renderLayerList() {
    return html`<div class="layer-box">
      <div class="layer-list">
        ${this.layers.map(
          (layer) => html`<div class=${layer.selected ? 'layer selected' : 'layer'} @click=${() => this.selectLayer(layer.id)}>
            <img class="layer-thumb" src=${layer.preview} alt="" />
            <span class="layer-name">${layer.name}</span>
            <input
              class="layer-visible"
              type="checkbox"
              title="Layer visibility"
              .checked=${layer.visible}
              @click=${(event: Event) => event.stopPropagation()}
              @change=${() => this.toggleLayer(layer.id)}
            />
            ${this.numberControl('Opacity', layer.opacity, 0, 1, 0.01, 2, (value) => this.setLayerOpacity(layer.id, value), true)}
          </div>`,
        )}
      </div>
      <div class="layer-actions" aria-label="layer actions">
        <button title="New layer" @click=${this.addBlankLayer}>＋</button>
        <button title="Delete layer" @click=${this.deleteActiveLayer}>×</button>
        <button title="Duplicate layer" @click=${this.duplicateActiveLayer}>⧉</button>
        <button title="Clear layer" @click=${this.clearActiveLayer}>⌫</button>
        <button title="Move layer up" @click=${this.moveLayerUp}>↑</button>
        <button title="Move layer down" @click=${this.moveLayerDown}>↓</button>
      </div>
    </div>`
  }

  private renderContextTabs() {
    if (this.selectedEntity === 'scene') {
      return html`<div class="context-tabs">
        <button class=${this.scenePanelTab === 'sdxl' ? 'active' : ''} @click=${() => (this.scenePanelTab = 'sdxl')}>SDXL</button>
        <button class=${this.scenePanelTab === 'z-image' ? 'active' : ''} @click=${() => (this.scenePanelTab = 'z-image')}>Z-Image</button>
        <button class=${this.scenePanelTab === 'streamdiffusion' ? 'active' : ''} @click=${() => (this.scenePanelTab = 'streamdiffusion')}>StreamDiffusion</button>
        <button class=${this.scenePanelTab === 'sana' ? 'active' : ''} @click=${() => (this.scenePanelTab = 'sana')}>SANA</button>
        <button class=${this.scenePanelTab === 'causal-forcing' ? 'active' : ''} @click=${() => (this.scenePanelTab = 'causal-forcing')}>Causal-Forcing</button>
        <button class=${this.scenePanelTab === 'fastvideo' ? 'active' : ''} @click=${() => (this.scenePanelTab = 'fastvideo')}>WAN Video</button>
      </div>`
    }
    const currentLayer = this.selectedLayer()
    return html`<div class="context-tabs">
      <button class=${this.layerPanelTab === 'properties' ? 'active' : ''} @click=${() => (this.layerPanelTab = 'properties')}>Properties</button>
      <button class=${this.layerPanelTab === 'mask' || this.layerPanelTab === 'sampling' ? 'active' : ''} @click=${() => (this.layerPanelTab = 'mask')}>Masks</button>
      <button class=${this.layerPanelTab === 'source' ? 'active' : ''} @click=${() => (this.layerPanelTab = 'source')}>Source</button>
      <button class=${this.layerPanelTab === 'generator' ? 'active' : ''} @click=${() => (this.layerPanelTab = 'generator')}>Generator</button>
      <button class=${this.layerPanelTab === 'effects' ? 'active' : ''} @click=${() => (this.layerPanelTab = 'effects')}>Effects</button>
      <button class=${this.layerPanelTab === 'controlnet' ? 'active' : ''} @click=${() => (this.layerPanelTab = 'controlnet')}>ControlNet</button>
      ${currentLayer && this._isVideoLayer(currentLayer.id) ? html`<button class=${this.layerPanelTab === 'video' ? 'active' : ''} @click=${() => (this.layerPanelTab = 'video')}>Video</button>` : ''}
    </div>`
  }

  private renderContextPanel() {
    if (this.selectedEntity === 'scene') return this.renderScenePanel()
    const layer = this.selectedLayer()
    if (this.selectionRect && !layer) return this.renderRegionProperties()
    if (!layer) return html`<div class="empty layer-empty">Select a layer or mask</div>`
    if (this.layerPanelTab === 'sampling') return this.renderLayerMaskPanel(layer)
    if (this.layerPanelTab === 'mask') return this.renderLayerMaskPanel(layer)
    if (this.layerPanelTab === 'source') return this.renderLayerSource()
    if (this.layerPanelTab === 'generator') return this.renderLayerGenerator(layer)
    if (this.layerPanelTab === 'effects') return this.renderLayerEffects(layer)
    if (this.layerPanelTab === 'controlnet') return this.renderLayerControlNetPanel(layer)
    if (this.layerPanelTab === 'video' && this._isVideoLayer(layer.id)) return this.renderVideoLayerPanel()
    return this.renderLayerProperties(layer)
  }

  private renderScenePanel() {
    if (this.scenePanelTab === 'z-image') return this.renderZImagePanel()
    if (this.scenePanelTab === 'streamdiffusion') return this.renderStreamDiffusionPanel()
    if (this.scenePanelTab === 'sana') return this.renderSanaPanel()
    if (this.scenePanelTab === 'causal-forcing') return this.renderCausalForcingPanel()
    if (this.scenePanelTab === 'fastvideo') return this.renderFastVideoPanel()
    return this.renderSdxlPanel()
  }

  private renderSdxlPanel() {
    return html`<div class="layer-tab scene-tab">
      ${this.renderRendererStatus('sdxl')}
      ${this.renderRegionTimeline()}
      ${this.renderSceneGenerationControls()}
      <div class="button-grid">
        <button class="stream" @click=${this.toggleStream}>${this.isStreaming ? 'Stop scene' : 'Generate scene'}</button>
      </div>
    </div>`
  }

  private renderZImagePanel() {
    return html`<div class="layer-tab scene-tab">
      ${this.renderRendererStatus('z-image')}
      ${this.renderRegionTimeline()}
      ${this.renderSceneGenerationControls()}
      <div class="button-grid">
        <button class="stream" @click=${this.toggleStream}>${this.isStreaming ? 'Stop Z-Image' : 'Start Z-Image'}</button>
      </div>
    </div>`
  }

  private renderSceneGenerationControls() {
    return this.renderGenerationControls({
      deviceLabel: 'Realtime GPU',
      deviceValue: this.realtimeDevice,
      onDevice: (value) => (this.realtimeDevice = value),
      sampling: this.sceneSamplingControls(),
      afterSampling: html`${this.renderRealtimeSeedAndReuseControls()}${this.renderRealtimeTransparencyControls()}`,
    })
  }

  private sceneSamplingControls(): SamplingControlsConfig {
    return {
      prompt: this.prompt,
      negativePrompt: this.negativePrompt,
      modelPath: this.selectedModel,
      loraPaths: [...this.selectedLoras],
      strength: this.strength,
      cfg: this.cfg,
      steps: this.steps,
      sampler: this.sceneSampler,
      scheduler: this.sceneScheduler,
      onPrompt: (value) => this.setScenePresetValue('prompt', value),
      onNegativePrompt: (value) => this.setScenePresetValue('negativePrompt', value),
      onModelPath: this.setSceneModel,
      onToggleLora: this.toggleSceneLora,
      onStrength: (value) => this.setSceneSamplerValue('strength', this.clamp(value, 0, 0.999, this.strength)),
      onCfg: (value) => this.setSceneSamplerValue('cfg', this.clamp(value, 0, 30, this.cfg)),
      onSteps: (value) => this.setSceneSamplerValue('steps', Math.round(this.clamp(value, 1, 32, this.steps))),
      onSampler: (value) => this.setSceneSamplerValue('sampler', value),
      onScheduler: (value) => this.setSceneSamplerValue('scheduler', value),
    }
  }

  private renderGenerationControls(config: GenerationControlsConfig) {
    return html`${config.beforeSampling ?? ''}
      ${this.renderDeviceSelect(config.deviceLabel, config.deviceValue, config.onDevice)}
      ${this.renderSamplingControls(config.sampling)}
      ${config.afterSampling ?? ''}`
  }

  private renderRealtimeSeedAndReuseControls() {
    return html`<section class="advanced-sampling">
      <label class="field inline">
        <span>Seed mode</span>
        <select .value=${this.sceneSeedMode} @change=${(event: Event) => {
          const value = (event.target as HTMLSelectElement).value
          if (this.validSeedMode(value)) this.sceneSeedMode = value
        }}>
          <option value="fixed">Fixed</option>
          <option value="random">Random</option>
          <option value="increment">Increment</option>
          <option value="decrement">Decrement</option>
        </select>
      </label>
      <label class="field inline">
        <span>Seed</span>
        <input type="number" min="0" max="4294967295" step="1" placeholder="auto" .value=${this.sceneSeed} @input=${(event: Event) => (this.sceneSeed = (event.target as HTMLInputElement).value)} />
      </label>
      <label class="field inline">
        <span>Reuse latent</span>
        <input type="checkbox" .checked=${this.sceneReuseLatent} @change=${(event: Event) => {
          this.sceneReuseLatent = (event.target as HTMLInputElement).checked
          if (!this.sceneReuseLatent) this.sceneReuseLatentRestartNext = false
        }} />
      </label>
      ${this.numberControl('Reuse denoise', this.sceneReuseLatentDenoise, 0, 0.999, 0.01, 2, (value) => (this.sceneReuseLatentDenoise = this.clamp(value, 0, 0.999, this.sceneReuseLatentDenoise)))}
      <label class="field inline">
        <span>Perturbator</span>
        <select .value=${this.sceneReuseLatentNoiseMode} ?disabled=${!this.sceneReuseLatent} @change=${(event: Event) => {
          const value = (event.target as HTMLSelectElement).value
          if (this.validLatentReuseNoiseMode(value)) this.sceneReuseLatentNoiseMode = value
        }}>
          <option value="none">None</option>
          <option value="fixed">Fixed noise</option>
          <option value="random">Random noise</option>
        </select>
      </label>
      ${this.numberControl('Perturb noise', this.sceneReuseLatentNoise, 0, 1, 0.01, 2, (value) => (this.sceneReuseLatentNoise = this.clamp(value, 0, 1, this.sceneReuseLatentNoise)))}
      <div class="button-grid">
        <button type="button" ?disabled=${!this.sceneReuseLatent} @click=${this.restartLatentReuse}>Restart reuse</button>
      </div>
    </section>`
  }

  private renderRealtimeTransparencyControls() {
    return this.renderTransparencyControls({
      enabled: this.sceneTransparentBackground,
      onEnabled: (value) => (this.sceneTransparentBackground = value),
      label: 'Transparent background',
      mode: this.sceneTransparentMode,
      onMode: (value) => (this.sceneTransparentMode = value),
      color: this.sceneTransparentColor,
      onColor: (value) => (this.sceneTransparentColor = value),
      tolerance: this.sceneTransparentTolerance,
      onTolerance: (value) => (this.sceneTransparentTolerance = value),
      alphaBlur: this.sceneTransparentAlphaBlur,
      onAlphaBlur: (value) => (this.sceneTransparentAlphaBlur = value),
      alphaThreshold: this.sceneTransparentAlphaThreshold,
      onAlphaThreshold: (value) => (this.sceneTransparentAlphaThreshold = value),
    })
  }

  private renderLayerTransparencyControls() {
    return this.renderTransparencyControls({
      enabled: this.layerTransparentBackground,
      onEnabled: (value) => (this.layerTransparentBackground = value),
      label: 'Transparent variation',
      method: this.layerTransparentMethod,
      onMethod: (value) => (this.layerTransparentMethod = value),
      mode: this.layerTransparentMode,
      onMode: (value) => (this.layerTransparentMode = value),
      color: this.layerTransparentColor,
      onColor: (value) => (this.layerTransparentColor = value),
      tolerance: this.layerTransparentTolerance,
      onTolerance: (value) => (this.layerTransparentTolerance = value),
      alphaBlur: this.layerTransparentAlphaBlur,
      onAlphaBlur: (value) => (this.layerTransparentAlphaBlur = value),
      alphaThreshold: this.layerTransparentAlphaThreshold,
      onAlphaThreshold: (value) => (this.layerTransparentAlphaThreshold = value),
    })
  }

  private renderTransparencyControls(config: {
    enabled: boolean
    onEnabled: (value: boolean) => void
    label: string
    method?: string
    onMethod?: (value: string) => void
    mode: string
    onMode: (value: string) => void
    color: string
    onColor: (value: string) => void
    tolerance: number
    onTolerance: (value: number) => void
    alphaBlur: number
    onAlphaBlur: (value: number) => void
    alphaThreshold: number
    onAlphaThreshold: (value: number) => void
  }) {
    return html`<section class="advanced-sampling">
      <label class="check-field">
        <input type="checkbox" .checked=${config.enabled} @change=${(event: Event) => config.onEnabled((event.target as HTMLInputElement).checked)} />
        <span>${config.label}</span>
      </label>
      ${config.enabled
        ? html`
            ${config.onMethod
              ? html`<label class="field inline">
                  <span>Alpha tech</span>
                  <select .value=${config.method ?? 'auto'} @change=${(event: Event) => config.onMethod?.((event.target as HTMLSelectElement).value)}>
                    <option value="auto">Auto VRAM</option>
                    <option value="fast">Fast alpha</option>
                    <option value="layerdiffuse">LayerDiffuse</option>
                  </select>
                </label>`
              : ''}
            <label class="field inline">
              <span>Alpha source</span>
              <select .value=${config.mode} @change=${(event: Event) => config.onMode((event.target as HTMLSelectElement).value)}>
                <option value="border">Sample border</option>
                <option value="color">Key color</option>
              </select>
            </label>
            ${config.mode === 'color'
              ? html`<label class="field inline"><span>Key color</span><input type="color" .value=${config.color} @input=${(event: Event) => config.onColor((event.target as HTMLInputElement).value)} /></label>`
              : ''}
            ${this.numberControl('Tolerance', config.tolerance, 0, 255, 1, 0, (value) => config.onTolerance(this.clamp(value, 0, 255, config.tolerance)))}
            ${this.numberControl('Alpha blur', config.alphaBlur, 0, 24, 0.5, 1, (value) => config.onAlphaBlur(this.clamp(value, 0, 24, config.alphaBlur)))}
            ${this.numberControl('Alpha floor', config.alphaThreshold, 0, 255, 1, 0, (value) => config.onAlphaThreshold(Math.round(this.clamp(value, 0, 255, config.alphaThreshold))))}
          `
        : ''}
    </section>`
  }

  private renderStreamDiffusionPanel() {
    const capability = this.rendererCapability('streamdiffusion')
    const native = capability?.native === true
    return html`<div class="layer-tab scene-tab">
      ${this.renderRendererStatus('streamdiffusion')}
      ${this.renderRegionTimeline()}
      <div class="empty" style="font-size:0.82em;line-height:1.4;padding:4px 2px 0;opacity:0.75">
        StreamDiffusion runs inference on every frame at 10–30 fps. Draw on the canvas to guide the output.
        Works with any SDXL model — pick a preset that matches your model, set a prompt, and click Start.
      </div>
      ${this.renderSamplingControls({
        prompt: this.prompt,
        negativePrompt: this.negativePrompt,
        modelPath: this.selectedModel,
        loraPaths: [...this.selectedLoras],
        strength: this.strength,
        cfg: this.cfg,
        steps: this.steps,
        sampler: this.sceneSampler,
        scheduler: this.sceneScheduler,
        onPrompt: (value) => this.setScenePresetValue('prompt', value),
        onNegativePrompt: (value) => this.setScenePresetValue('negativePrompt', value),
        onModelPath: this.setSceneModel,
        onToggleLora: this.toggleSceneLora,
        onStrength: (value) => this.setSceneSamplerValue('strength', this.clamp(value, 0, 0.999, this.strength)),
        onCfg: (value) => this.setSceneSamplerValue('cfg', this.clamp(value, 0, 30, this.cfg)),
        onSteps: (value) => this.setSceneSamplerValue('steps', Math.round(this.clamp(value, 1, 32, this.steps))),
        onSampler: (value) => this.setSceneSamplerValue('sampler', value),
        onScheduler: (value) => this.setSceneSamplerValue('scheduler', value),
      })}
      <section class="advanced-sampling">
        <label class="field inline" title="Choose a preset matching your model type. Each preset auto-tunes steps, strength and timesteps for that model family.">
          <span>Model preset</span>
          <select .value=${this.streamRuntimePreset} @change=${(event: Event) => this.applyStreamRuntimePreset((event.target as HTMLSelectElement).value)}>
            <option value="diffusers">SDXL (any model, slower)</option>
            <option value="lcm-lora-sdxl">SDXL + LCM LoRA (fast)</option>
            <option value="sdxl-lightning-4step">SDXL Lightning 4-step</option>
            <option value="sdxl-turbo">SDXL Turbo (fastest)</option>
          </select>
        </label>
        <label class="field inline" title="Which denoising steps StreamDiffusion runs each frame. Auto-set by the preset — only change if you know what you're doing.">
          <span>Timestep indices</span>
          <input type="text" .value=${this.streamTimestepIndices} @input=${(event: Event) => (this.streamTimestepIndices = (event.target as HTMLInputElement).value)} />
        </label>
        <div title="Number of latent frames queued before decoding. 1 = fastest / lowest quality, 4 = smoother but adds latency.">
          ${this.numberControl('Frame buffer', this.streamFrameBufferSize, 1, 4, 1, 0, (value) => (this.streamFrameBufferSize = Math.round(this.clamp(value, 1, 4, this.streamFrameBufferSize))))}
        </div>
        <label class="field inline" title="How guidance is applied each frame. 'Self-negative RCFG' is fastest and recommended for realtime use.">
          <span>Guidance mode</span>
          <select .value=${this.streamCfgType} @change=${(event: Event) => (this.streamCfgType = (event.target as HTMLSelectElement).value)}>
            <option value="none">None (unconditioned)</option>
            <option value="self">Self-negative RCFG (recommended)</option>
            <option value="initialize">Onetime-negative RCFG</option>
            <option value="full">Full CFG (slowest)</option>
          </select>
        </label>
        <div title="Skip re-generating a frame when the scene hasn't changed enough. 0 = always generate, 0.98 = skip very similar frames (saves GPU).">
          ${this.numberControl('Frame skip threshold', this.streamSimilarityThreshold, 0, 1, 0.01, 2, (value) => (this.streamSimilarityThreshold = this.clamp(value, 0, 1, this.streamSimilarityThreshold)))}
        </div>
        <div title="Maximum number of consecutive frames to skip. 0 = never skip.">
          ${this.numberControl('Max consecutive skips', this.streamMaxSkipFrames, 0, 60, 1, 0, (value) => (this.streamMaxSkipFrames = Math.round(this.clamp(value, 0, 60, this.streamMaxSkipFrames))))}
        </div>
        <label class="field inline" title="MotionStream: auto-animates the AI output with a camera-like motion, creating a living/dreamlike effect. Set to None if you just want still inpainting.">
          <span>Auto-motion</span>
          <select .value=${this.streamMotionMode} @change=${(event: Event) => {
            const value = (event.target as HTMLSelectElement).value
            if (this.validStreamMotionMode(value)) this.streamMotionMode = value
          }}>
            <option value="none">None (static)</option>
            <option value="sway">Sway (gentle side-to-side)</option>
            <option value="orbit">Orbit (circular drift)</option>
            <option value="push">Push (forward/back pan)</option>
            <option value="zoom">Zoom (slow pulse)</option>
          </select>
        </label>
        <div title="How much the motion transform shifts the frame each cycle. 0 = no motion, 1 = full intensity.">
          ${this.numberControl('Motion intensity', this.streamMotionIntensity, 0, 2, 0.01, 2, (value) => (this.streamMotionIntensity = this.clamp(value, 0, 2, this.streamMotionIntensity)))}
        </div>
        <div title="How fast the auto-motion cycle repeats (cycles per second).">
          ${this.numberControl('Motion speed', this.streamMotionSpeed, 0.1, 4, 0.05, 2, (value) => (this.streamMotionSpeed = this.clamp(value, 0.1, 4, this.streamMotionSpeed)))}
        </div>
        <div class="empty">Runtime: ${capability?.runtime ?? 'checking'}${native ? '' : ' (fallback — install StreamDiffusion for native speed)'}</div>
      </section>
      <div class="button-grid">
        <button class="stream" @click=${this.toggleStream}
          ?disabled=${this.status === 'connecting'}>
          ${this.status === 'connecting' ? 'Connecting…'
            : this.isStreaming ? 'Stop stream'
            : native ? 'Start StreamDiffusion' : 'Start (fallback mode)'}
        </button>
        ${this.isStreaming ? html`<div class="stream-status-line">
          ${this.status === 'connecting' ? '⏳ Connecting…'
            : (this.displayFps === 0 && this.fps === 0) ? '⚙ Loading model…'
            : `● Streaming – ${(this.displayFps || this.fps).toFixed(0)} FPS`}
        </div>` : ''}
      </div>

    </div>`
  }

  private renderCausalForcingPanel() {
    const capability = this.videoCapability(this.motionModel)
    return html`<div class="layer-tab scene-tab">
      ${this.renderRendererStatus('causal-forcing')}
      <label class="field">
        <span>Prompt</span>
        <textarea rows="3" .value=${this.motionPrompt} @input=${(event: Event) => (this.motionPrompt = (event.target as HTMLTextAreaElement).value)}></textarea>
      </label>
      <label class="field">
        <span>Target prompt</span>
        <textarea rows="3" .value=${this.motionArrivalPrompt} @input=${(event: Event) => (this.motionArrivalPrompt = (event.target as HTMLTextAreaElement).value)}></textarea>
      </label>
      <label class="field">
        <span>Motion negative</span>
        <textarea rows="2" .value=${this.motionNegativePrompt} @input=${(event: Event) => (this.motionNegativePrompt = (event.target as HTMLTextAreaElement).value)}></textarea>
      </label>
      <label class="field inline">
        <span>Model</span>
        <select .value=${this.motionModel} @change=${(event: Event) => (this.motionModel = (event.target as HTMLSelectElement).value)}>
          <option value="causal-forcing" ?disabled=${this.videoCapabilityDisabled('causal-forcing')}>Causal Forcing I2V</option>
          <option value="causal-forcing-2step" ?disabled=${this.videoCapabilityDisabled('causal-forcing-2step')}>Causal Forcing++ 2-step</option>
          <option value="causal-forcing-1step" ?disabled=${this.videoCapabilityDisabled('causal-forcing-1step')}>Causal Forcing++ 1-step</option>
          <option value="krea-realtime-video" ?disabled=${this.videoCapabilityDisabled('krea-realtime-video')}>Krea Realtime Video 14B</option>
        </select>
      </label>
      ${capability && !capability.configured ? html`<div class="empty">${capability.message}</div>` : ''}
      <div class="resolution-grid">
        ${this.numberControl('FPS', this.motionFps, 4, 60, 1, 0, (value) => (this.motionFps = Math.round(this.clamp(value, 4, 60, this.motionFps))))}
        ${this.numberControl('Chunk seconds', this.motionDuration, 0.25, 12, 0.25, 2, (value) => (this.motionDuration = this.clamp(value, 0.25, 12, this.motionDuration)))}
      </div>
      ${this.numberControl('Intensity', this.motionIntensity, 0, 3, 0.05, 2, (value) => (this.motionIntensity = this.clamp(value, 0, 3, this.motionIntensity)))}
      <div class="button-grid">
        <button class="stream" @click=${this.toggleCausalStream} ?disabled=${!this.videoCapabilityReady(this.motionModel) && !this.isGeneratingMotion}>${this.isGeneratingMotion ? `Stop chunk loop ${this.motionChunkIndex ? this.motionChunkIndex : ''}` : 'Start chunk loop'}</button>
      </div>
      <div class="empty">The current Causal-Forcing adapter runs upstream CLI chunks and reloads the model per chunk. A true continuous stream needs a persistent model worker.</div>
      ${this.motionTask ? this.renderMotionTaskProgress(this.motionTask) : ''}
    </div>`
  }

  private renderFastVideoPanel() {
    const capability = this.videoCapability('fastvideo')
    const videoModels = this.videoModelAssets()
    return html`<div class="layer-tab scene-tab">
      ${this.renderRendererStatus('fastvideo')}
      <label class="field">
        <span>Prompt</span>
        <textarea rows="3" .value=${this.motionPrompt} @input=${(event: Event) => (this.motionPrompt = (event.target as HTMLTextAreaElement).value)}></textarea>
      </label>
      <label class="field">
        <span>Negative prompt</span>
        <textarea rows="2" .value=${this.motionNegativePrompt} @input=${(event: Event) => (this.motionNegativePrompt = (event.target as HTMLTextAreaElement).value)}></textarea>
      </label>
      <label class="field inline">
        <span>WAN model</span>
        <select .value=${this.fastVideoModelPath} @change=${(event: Event) => (this.fastVideoModelPath = (event.target as HTMLSelectElement).value)}>
          <option value="">Auto from FastVideo/WAN config</option>
          ${videoModels.map((model) => html`<option value=${model.path}>${model.name}</option>`)}
        </select>
      </label>
      ${this.renderDeviceSelect('Video GPU', this.videoDevice, (value) => (this.videoDevice = value))}
      <div class="resolution-grid">
        ${this.numberControl('FPS', this.motionFps, 4, 60, 1, 0, (value) => (this.motionFps = Math.round(this.clamp(value, 4, 60, this.motionFps))))}
        ${this.numberControl('Seconds', this.motionDuration, 0.25, 12, 0.25, 2, (value) => (this.motionDuration = this.clamp(value, 0.25, 12, this.motionDuration)))}
      </div>
      <div class="button-grid">
        <button class="stream" @click=${this.startFastVideoStream} ?disabled=${this.isGeneratingMotion}>Generate WAN clip</button>
      </div>
      <div class="empty">${capability && !capability.configured ? capability.message : 'WAN/FastVideo is offline clip generation. RTDiffusion uses native WAN 2.2 high/low components when both are configured; RTD_FASTVIDEO_COMMAND remains an external runner override.'}</div>
      ${this.motionTask ? this.renderMotionTaskProgress(this.motionTask) : ''}
    </div>`
  }

  private renderVideoLayerPanel() {
    const hasVideo = !!this.videoUrl
    const end = this.videoEnd > 0 ? this.videoEnd : this.videoDuration
    return html`<div class="layer-tab">
      <div class="button-grid">
        <button @click=${this.pickVideoFile}>Choose video…</button>
        ${hasVideo ? html`<button @click=${this.clearVideoFile}>Clear</button>` : ''}
      </div>
      ${hasVideo
        ? html`<div class="empty" style="font-size:11px;margin-top:4px">Duration: ${end.toFixed(2)}s · ${this.videoDuration.toFixed(2)}s total</div>`
        : html`<div class="empty" style="color:#888;font-size:12px">No video selected. Choose a file to project video onto this layer.</div>`}
      <label class="field inline">
        <span>Loop</span>
        <select .value=${this.videoLoopMode} @change=${(e: Event) => this.setVideoLoopMode((e.target as HTMLSelectElement).value as VideoLoopMode)}>
          <option value="loop">Loop</option>
          <option value="ping-pong">Ping-pong</option>
          <option value="once">Once</option>
        </select>
      </label>
      <div class="resolution-grid">
        ${this.numberControl('Start (s)', this.videoStart, 0, this.videoDuration || 9999, 0.01, 2, (v) => { this.videoStart = this.clamp(v, 0, this.videoDuration, this.videoStart); this.syncVideoTime() })}
        ${this.numberControl('End (s)', this.videoEnd, 0, this.videoDuration || 9999, 0.01, 2, (v) => { this.videoEnd = this.clamp(v, 0, this.videoDuration, this.videoEnd || this.videoDuration) })}
      </div>
      <label class="field inline">
        <span>Frame mode</span>
        <select .value=${this.videoFpsMode} @change=${(e: Event) => (this.videoFpsMode = (e.target as HTMLSelectElement).value as VideoFpsMode)}>
          <option value="fps">Real-time playback</option>
          <option value="sequential">Sequential (1 frame per send)</option>
        </select>
      </label>
      ${this.videoFpsMode === 'fps'
        ? this.numberControl('Playback FPS', this.videoFps, 1, 120, 1, 0, (v) => { this.videoFps = Math.round(this.clamp(v, 1, 120, this.videoFps)); this.syncVideoPlaybackRate() })
        : this.numberControl('Frame step (s)', 1 / this.videoFps, 0.001, 10, 0.001, 3, (v) => { this.videoFps = Math.round(this.clamp(1 / Math.max(v, 0.001), 1, 120, this.videoFps)) })}
    </div>`
  }

  private pickVideoFile = async () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = 'video/*'
    await new Promise<void>((resolve) => {
      input.onchange = () => {
        const file = input.files?.[0]
        if (file) {
          this._stopVideoRaf()
          this.videoInputElement?.pause()
          if (this.videoObjectUrl) URL.revokeObjectURL(this.videoObjectUrl)
          this.videoObjectUrl = URL.createObjectURL(file)
          this.videoUrl = this.videoObjectUrl
          this.videoReversing = false
          const vid = this.videoInputElement
          if (vid) {
            vid.src = this.videoObjectUrl
            vid.currentTime = this.videoStart
            vid.loop = this.videoLoopMode === 'loop'
            if (this.videoFpsMode === 'fps') void vid.play()
          }
          this._startVideoRaf()
        }
        resolve()
      }
      input.click()
    })
  }

  private clearVideoFile = () => {
    this._stopVideoRaf()
    this.videoInputElement?.pause()
    if (this.videoObjectUrl) URL.revokeObjectURL(this.videoObjectUrl)
    this.videoObjectUrl = ''
    this.videoUrl = ''
    this.videoDuration = 0
    const vid = this.videoInputElement
    if (vid) vid.src = ''
    this._videoOffscreenCanvas = null
    // Remove the video fabric object from canvas
    if (this._videoLayerId) {
      const obj = this.findLayer(this._videoLayerId)
      if (obj) {
        this.pushHistory()
        this.fabricCanvas?.remove(obj as FabricObject)
        this.fabricCanvas?.requestRenderAll()
        this.syncLayers()
      }
      this._videoLayerId = null
    }
  }

  private onVideoMetadata = () => {
    const vid = this.videoInputElement
    if (!vid) return
    this.videoDuration = vid.duration
    vid.currentTime = this.videoStart
    this.syncVideoPlaybackRate()
  }

  private onVideoEnded = () => {
    const vid = this.videoInputElement
    if (!vid) return
    if (this.videoLoopMode === 'loop') {
      vid.currentTime = this.videoStart
      void vid.play()
    } else if (this.videoLoopMode === 'ping-pong') {
      this.videoReversing = !this.videoReversing
      vid.currentTime = this.videoReversing ? (this.videoEnd > 0 ? this.videoEnd : this.videoDuration) : this.videoStart
      void vid.play()
    }
    // 'once' mode: just stop — video stays at last frame
  }

  private setVideoLoopMode(mode: VideoLoopMode) {
    this.videoLoopMode = mode
    const vid = this.videoInputElement
    if (!vid) return
    vid.loop = mode === 'loop'
  }

  private syncVideoTime() {
    const vid = this.videoInputElement
    if (vid) vid.currentTime = this.videoStart
  }

  private syncVideoPlaybackRate() {
    const vid = this.videoInputElement
    if (!vid) return
    // Adjust playbackRate to match desired FPS relative to native video FPS
    // We can't know the native FPS, so just set currentTime-based approach:
    // For FPS mode, play at the requested rate by adjusting playbackRate
    vid.playbackRate = 1.0  // default; FPS control happens via capture timing
  }

  private addVideoLayer = async () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = 'video/*'
    await new Promise<void>((resolve) => {
      input.onchange = async () => {
        const file = input.files?.[0]
        if (!file) { resolve(); return }
        if (this.videoObjectUrl) URL.revokeObjectURL(this.videoObjectUrl)
        this.videoObjectUrl = URL.createObjectURL(file)
        this.videoUrl = this.videoObjectUrl
        this.videoReversing = false

        const vid = this.videoInputElement
        if (!vid) { resolve(); return }
        vid.src = this.videoObjectUrl
        vid.currentTime = this.videoStart

        // Wait for metadata so we know dimensions
        await new Promise<void>((res) => {
          if (vid.readyState >= 1) { res(); return }
          vid.onloadedmetadata = () => res()
        })
        this.videoDuration = vid.duration

        // Create off-screen canvas sized to the stage
        const offscreen = document.createElement('canvas')
        offscreen.width = this.stageWidth
        offscreen.height = this.stageHeight
        this._videoOffscreenCanvas = offscreen

        // Draw first frame before creating FabricImage
        if (vid.readyState >= 2) {
          const ctx = offscreen.getContext('2d')!
          const vw = vid.videoWidth || this.stageWidth
          const vh = vid.videoHeight || this.stageHeight
          const scale = Math.min(offscreen.width / vw, offscreen.height / vh)
          const sw = vw * scale
          const sh = vh * scale
          ctx.drawImage(vid, (offscreen.width - sw) / 2, (offscreen.height - sh) / 2, sw, sh)
        }

        // Create FabricImage from the off-screen canvas (objectCaching=false so it re-reads on each render)
        const img = new FabricImage(offscreen, {
          objectCaching: false,
          left: 0,
          top: 0,
          scaleX: 1,
          scaleY: 1,
        }) as unknown as FabricObject & { __isVideoLayer?: boolean; id?: string }
        img.__isVideoLayer = true

        this.pushHistory()
        this.fabricCanvas?.add(img as unknown as FabricObject)
        this.fabricCanvas?.setActiveObject(img as unknown as FabricObject)
        this.fabricCanvas?.requestRenderAll()
        this.syncLayers()

        // Record the layer id
        await this.updateComplete
        const vid2LayerId = this.layers.find((l) => {
          const obj = this.findLayer(l.id)
          return obj && (obj as FabricObject & { __isVideoLayer?: boolean }).__isVideoLayer
        })?.id ?? null
        this._videoLayerId = vid2LayerId

        // Start playback and rAF loop
        vid.loop = this.videoLoopMode === 'loop'
        if (this.videoFpsMode === 'fps') void vid.play()
        this._startVideoRaf()

        // Select the video tab in the layer panel
        this.layerPanelTab = 'video'
        resolve()
      }
      input.click()
    })
  }

  private _isVideoLayer(layerId: string): boolean {
    const obj = this.findLayer(layerId)
    return !!(obj && (obj as FabricObject & { __isVideoLayer?: boolean }).__isVideoLayer)
  }

  private _startVideoRaf() {
    if (this._videoRafId != null) return
    const step = () => {
      const vid = this.videoInputElement
      const canvas = this._videoOffscreenCanvas
      if (vid && canvas && vid.readyState >= 2) {
        const ctx = canvas.getContext('2d')!
        const vw = vid.videoWidth || canvas.width
        const vh = vid.videoHeight || canvas.height
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        const scale = Math.min(canvas.width / vw, canvas.height / vh)
        const sw = vw * scale
        const sh = vh * scale
        ctx.drawImage(vid, (canvas.width - sw) / 2, (canvas.height - sh) / 2, sw, sh)
        this.fabricCanvas?.requestRenderAll()
      }
      this._videoRafId = requestAnimationFrame(step)
    }
    this._videoRafId = requestAnimationFrame(step)
  }

  private _stopVideoRaf() {
    if (this._videoRafId != null) {
      cancelAnimationFrame(this._videoRafId)
      this._videoRafId = null
    }
  }

  private advanceVideoSequential() {
    const vid = this.videoInputElement
    if (!vid || !this.videoUrl) return
    const end = this.videoEnd > 0 ? this.videoEnd : this.videoDuration
    const step = 1 / this.videoFps
    if (!this.videoReversing) {
      vid.currentTime = Math.min(vid.currentTime + step, end)
      if (vid.currentTime >= end - step * 0.5) {
        if (this.videoLoopMode === 'loop') {
          vid.currentTime = this.videoStart
        } else if (this.videoLoopMode === 'ping-pong') {
          this.videoReversing = true
        }
      }
    } else {
      vid.currentTime = Math.max(vid.currentTime - step, this.videoStart)
      if (vid.currentTime <= this.videoStart + step * 0.5) {
        this.videoReversing = false
      }
    }
  }

  private renderRegionTimeline(currentLayer?: LayerItem) {
    const activeLayers = this.layers.filter((layer) => layer.visible && (!currentLayer || layer.preset.samplingEnabled || layer.id === currentLayer.id))
    const layerSchedules = this.autoLayerSchedules(activeLayers)
    const rows = activeLayers.flatMap((layer, index) => {
        const timing = layerSchedules.get(layer.id) ?? { start: 0, end: 1, schedule: 'auto' as RegionSchedule, active: true }
        if (!timing.active) return [{ id: layer.id, label: `${layer.name} · occluded`, color: this.layerRegionColor(layer.id), timing }]
        return this.layerRegionSpecs(layer).map((region) => ({
          id: region.id,
          label: currentLayer?.id === layer.id ? (region.name || layer.name) : `${layer.name}${layer.preset.samplingEnabled ? '' : ' · off'}`,
          color: region.color || this.hslToHex((index * 41 + 212) % 360, 70, 54),
          timing: this.resolvedRegionTiming(region, timing),
        }))
      })
    return html`<section class="schedule-timeline">
      <div class="timeline-head"><strong>Mask schedule</strong><span>0</span><span>0.5</span><span>1</span></div>
      ${rows.length
        ? rows.map((row) => {
            const start = Math.round(row.timing.start * 100)
            const width = Math.max(2, Math.round((row.timing.end - row.timing.start) * 100))
            return html`<div class=${row.timing.active ? 'timeline-row' : 'timeline-row muted'}>
              <span>${row.label}</span>
              <div class="timeline-track">
                <i style=${`left: ${start}%; width: ${width}%; background: ${row.color};`}></i>
              </div>
            </div>`
          })
        : html`<div class="timeline-empty">No scheduled masks</div>`}
    </section>`
  }

  private renderLayerMaskPanel(layer: LayerItem) {
    const hasExplicit = this.regions.some((r) => r.target === 'layer' && r.layerId === layer.id)
    const inheritedHidden = this.noDefaultMaskLayers.includes(layer.id)
    const showInherited = !inheritedHidden && !hasExplicit
    return html`<div class="layer-tab mask-tab layer-mask-tab">
      ${this.renderRegionTimeline(layer)}
      ${showInherited ? html`<section class="region-card inherited-region-card">
        <div class="region-head">
          <button class="region-color region-color-button" style=${`--region-color: ${this.layerRegionColor(layer.id)}`} title="Use layer mask color" @click=${() => this.useLayerRegionColor(layer.id)}></button>
          <input .value=${layer.name} disabled />
          <button title="Remove default mask" @click=${() => this.hideInheritedMask(layer.id)}>×</button>
        </div>
        <div class="region-meta">Inherited from layer alpha · alpha mask</div>
        <label class="check-field" title="Use WD Eva02 tagger to auto-detect prompts from this layer each frame">
          <input type="checkbox" .checked=${layer.preset.autoTag} @change=${(event: Event) => this.updateLayerPreset(layer.id, { autoTag: (event.target as HTMLInputElement).checked })} />
          <span>Auto-tag prompt (WD Eva02)</span>
        </label>
        ${layer.preset.autoTag ? html`
          <div class="region-meta" style="color:#8fa;font-size:11px">Prompt is detected automatically from the layer area each frame. The text prompt below is ignored.</div>
          ${this.numberControl('Threshold', layer.preset.autoTagThreshold, 0, 1, 0.01, 2, (v) => this.updateLayerPreset(layer.id, { autoTagThreshold: this.clamp(v, 0, 1, layer.preset.autoTagThreshold) }))}
          ${this.numberControl('Refresh frames', layer.preset.autoTagRefreshFrames, 1, 60, 1, 0, (v) => this.updateLayerPreset(layer.id, { autoTagRefreshFrames: Math.round(this.clamp(v, 1, 60, layer.preset.autoTagRefreshFrames)) }))}
        ` : html`
          <label class="field">
            <span>Prompt +</span>
            <textarea rows="2" .value=${layer.preset.prompt} @input=${(event: Event) => this.updateLayerPreset(layer.id, { prompt: (event.target as HTMLTextAreaElement).value })}></textarea>
          </label>
          <label class="field">
            <span>Prompt -</span>
            <textarea rows="2" .value=${layer.preset.negativePrompt} @input=${(event: Event) => this.updateLayerPreset(layer.id, { negativePrompt: (event.target as HTMLTextAreaElement).value })}></textarea>
          </label>
        `}
        <label class="check-field">
          <input type="checkbox" .checked=${layer.preset.conditionNegate} @change=${(event: Event) => this.updateLayerPreset(layer.id, { conditionNegate: (event.target as HTMLInputElement).checked })} />
          <span>Negate mask</span>
        </label>
        <label class="field inline">
          <span>Mode</span>
          <select .value=${layer.preset.conditionMode} @change=${(event: Event) => this.updateLayerPreset(layer.id, { conditionMode: (event.target as HTMLSelectElement).value as LayerConditionMode })}>
            <option value="mask">Mask</option>
            <option value="override">Override</option>
            <option value="add">Add</option>
            <option value="multiply">Multiply</option>
            <option value="prompt_mix">Prompt mix</option>
          </select>
        </label>
        ${this.numberControl('Mask', layer.preset.conditionWeight, 0, 4, 0.05, 2, (value) =>
          this.updateLayerPreset(layer.id, { conditionWeight: this.clamp(value, 0, 4, layer.preset.conditionWeight) }))}
        ${this.aggregateOperatorControl('Mask mix', layer.preset.maskOperator, (value) =>
          this.updateLayerPreset(layer.id, { maskOperator: value }))}
        ${this.numberControl('Denoise', layer.preset.strength, 0, 0.999, 0.01, 2, (value) =>
          this.updateLayerPreset(layer.id, { strength: this.clamp(value, 0, 0.999, layer.preset.strength) }))}
        ${this.aggregateOperatorControl('Denoise mix', layer.preset.denoiseOperator, (value) =>
          this.updateLayerPreset(layer.id, { denoiseOperator: value }))}
        <label class="field inline">
          <span>Schedule</span>
          <select .value=${layer.preset.schedule} @change=${(event: Event) => this.updateLayerPreset(layer.id, { schedule: (event.target as HTMLSelectElement).value as RegionSchedule })}>
            <option value="auto">Auto</option>
            <option value="linear">Linear</option>
            <option value="ease_in">Ease in</option>
            <option value="ease_out">Ease out</option>
            <option value="ease_in_out">Ease in/out</option>
          </select>
        </label>
        ${layer.preset.schedule === 'auto'
          ? html`<div class="region-meta">Auto start/end resolves from layer order, opacity, and transparency.</div>`
          : html`<div class="interval-row">
              ${this.numberControl('Start', layer.preset.scheduleStart, 0, 1, 0.01, 2, (value) =>
                this.updateLayerPreset(layer.id, { scheduleStart: this.clamp(value, 0, layer.preset.scheduleEnd, layer.preset.scheduleStart) }))}
              ${this.numberControl('End', layer.preset.scheduleEnd, 0, 1, 0.01, 2, (value) =>
                this.updateLayerPreset(layer.id, { scheduleEnd: this.clamp(value, layer.preset.scheduleStart, 1, layer.preset.scheduleEnd) }))}
            </div>`}
        ${this.numberControl('Inner blur', layer.preset.conditionInnerBlur, 0, 160, 1, 0, (value) =>
          this.updateLayerPreset(layer.id, { conditionInnerBlur: Math.round(this.clamp(value, 0, 160, layer.preset.conditionInnerBlur)) }))}
        ${this.numberControl('Outer blur', layer.preset.conditionOuterBlur, 0, 160, 1, 0, (value) =>
          this.updateLayerPreset(layer.id, { conditionOuterBlur: Math.round(this.clamp(value, 0, 160, layer.preset.conditionOuterBlur)) }))}
      </section>` : ''}
      <div class="mask-actions">
        <button @click=${() => this.addDefaultMask(layer)}>↩ Add default mask</button>
      </div>
      ${this.renderRegionsPanel('layer', layer.id)}
    </div>`
  }

  private layerRegionSpecs(layer: LayerItem) {
    const explicit = this.regions.filter((region) => region.target === 'layer' && region.layerId === layer.id)
    if (explicit.length) return explicit
    if (this.noDefaultMaskLayers.includes(layer.id)) return []
    return [this.inheritedLayerRegion(layer)]
  }

  private inheritedLayerRegion(layer: LayerItem): RegionItem {
    return {
      id: `inherited:${layer.id}`,
      target: 'layer',
      layerId: layer.id,
      color: this.layerRegionColor(layer.id),
      name: layer.name,
      shape: 'box',
      rect: { x: 0, y: 0, width: this.stageWidth, height: this.stageHeight, inverted: false, shape: 'box' },
      maskStrength: layer.preset.conditionWeight,
      innerBlur: layer.preset.conditionInnerBlur,
      outerBlur: layer.preset.conditionOuterBlur,
      negate: layer.preset.conditionNegate,
      inherited: true,
      prompt: layer.preset.prompt,
      negativePrompt: layer.preset.negativePrompt,
      denoise: layer.preset.strength,
      maskOperator: layer.preset.maskOperator,
      denoiseOperator: layer.preset.denoiseOperator,
      schedule: layer.preset.schedule,
      start: layer.preset.scheduleStart,
      end: layer.preset.scheduleEnd,
    }
  }

  private layerRegionColor(layerId: string) {
    const index = Math.max(0, this.layers.findIndex((layer) => layer.id === layerId))
    return this.hslToHex((index * 47 + 205) % 360, 78, 54)
  }


  private hideInheritedMask(layerId: string) {
    if (!this.noDefaultMaskLayers.includes(layerId))
      this.noDefaultMaskLayers = [...this.noDefaultMaskLayers, layerId]
    this.scheduleProjectPersist()
  }

  private addDefaultMask(layer: LayerItem) {
    const region: RegionItem = {
      id: crypto.randomUUID(),
      target: 'layer',
      layerId: layer.id,
      color: this.nextRegionColor(),
      name: `${layer.name} default`,
      shape: 'box',
      rect: { x: 0, y: 0, width: this.stageWidth, height: this.stageHeight, inverted: false, shape: 'box' },
      maskStrength: layer.preset.conditionWeight,
      innerBlur: layer.preset.conditionInnerBlur,
      outerBlur: layer.preset.conditionOuterBlur,
      negate: layer.preset.conditionNegate,
      inherited: true,
      prompt: layer.preset.prompt,
      negativePrompt: layer.preset.negativePrompt,
      denoise: layer.preset.strength,
      maskOperator: layer.preset.maskOperator,
      denoiseOperator: layer.preset.denoiseOperator,
      schedule: layer.preset.schedule,
      start: layer.preset.scheduleStart,
      end: layer.preset.scheduleEnd,
    }
    this.noDefaultMaskLayers = this.noDefaultMaskLayers.filter((id) => id !== layer.id)
    this.regions = [region, ...this.regions]
    this.scheduleProjectPersist()
  }

  private renderSamplingControls(config: {
    prompt: string
    negativePrompt: string
    modelPath?: string
    loraPaths?: string[]
    strength?: number
    cfg: number
    steps: number
    sampler: string
    scheduler: string
    onPrompt: (value: string) => void
    onNegativePrompt: (value: string) => void
    onModelPath?: (value: string) => void
    onToggleLora?: (value: string) => void
    onStrength?: (value: number) => void
    onCfg: (value: number) => void
    onSteps: (value: number) => void
    onSampler: (value: string) => void
    onScheduler: (value: string) => void
  }) {
    return html`<div class="sampling-controls">
      <label class="field">
        <span>Prompt</span>
        <textarea rows="4" .value=${config.prompt} @input=${(event: Event) => config.onPrompt((event.target as HTMLTextAreaElement).value)}></textarea>
      </label>
      <label class="field">
        <span>Negative prompt</span>
        <textarea rows="3" .value=${config.negativePrompt} @input=${(event: Event) => config.onNegativePrompt((event.target as HTMLTextAreaElement).value)}></textarea>
      </label>
      ${config.onModelPath ? this.renderOptionPicker(
        'Model',
        config.modelPath ?? this.selectedModel,
        this.assets.models.map((model) => ({ label: `${model.preferred ? '* ' : ''}${model.name}`, value: model.path, meta: this.formatBytes(model.size) })),
        (value) => config.onModelPath?.(value),
      ) : ''}
      <div class="sampling-grid">
        ${config.onStrength ? this.numberControl('Denoise', config.strength ?? this.strength, 0, 0.999, 0.01, 2, config.onStrength) : ''}
        ${this.numberControl('CFG', config.cfg, 0, 30, 0.1, 1, config.onCfg)}
        ${this.numberControl('Steps', config.steps, 1, 40, 1, 0, config.onSteps)}
      </div>
      ${this.renderOptionPicker('sampler_name', config.sampler, SAMPLERS, config.onSampler)}
      ${this.renderOptionPicker('scheduler', config.scheduler, SCHEDULERS, config.onScheduler)}
      ${config.onToggleLora ? this.renderLoraPicker(config.loraPaths ?? [], config.onToggleLora) : ''}
    </div>`
  }

  private renderOptionPicker(label: string, value: string, options: PickerOption[], onSelect: (value: string) => void) {
    const selected = options.find((option) => option.value === value) ?? options[0]
    return html`<div class="popup-picker field">
      <button class="picker-trigger" type="button" @click=${(event: MouseEvent) => this.openOptionDialog(event, label, value, options, onSelect)}>
        <strong>${selected?.label ?? 'Default'}</strong><span>${this.displayPickerLabel(label)}</span>
      </button>
    </div>`
  }

  private renderLoraPicker(selectedPaths: string[], onToggle: (value: string) => void) {
    const selectedNames = this.assets.loras.filter((lora) => selectedPaths.includes(lora.path)).map((lora) => lora.name)
    return html`<div class="popup-picker lora-picker field">
      <button class="picker-trigger" type="button" @click=${(event: MouseEvent) => this.openLoraDialog(event, selectedPaths, onToggle)}>
        <strong>${selectedNames.length ? selectedNames.slice(0, 2).join(', ') : 'None'}</strong><span>${selectedNames.length > 2 ? `+${selectedNames.length - 2} LoRAs` : 'LoRAs'}</span>
      </button>
    </div>`
  }

  private renderPickerDialog() {
    const dialog = this.pickerDialog
    if (!dialog) return ''
    const title = dialog.kind === 'option' ? dialog.label : 'LoRAs'
    return html`<dialog
      class="picker-dialog"
      style=${`--picker-left: ${this.pickerDialogPosition.left}px; --picker-top: ${this.pickerDialogPosition.top}px; --picker-width: ${this.pickerDialogPosition.width}px; --picker-max-height: ${this.pickerDialogPosition.maxHeight}px;`}
      @cancel=${this.closePickerDialog}
      @close=${this.closePickerDialog}
      @click=${this.closePickerBackdrop}
    >
      <section class="picker-dialog-panel" @click=${(event: Event) => event.stopPropagation()}>
        <div class="picker-dialog-head">
          <strong>${title}</strong>
          <button type="button" title="Close" @click=${this.closePickerDialog}>×</button>
        </div>
        ${dialog.kind === 'option' ? this.renderOptionDialogBody(dialog) : this.renderLoraDialogBody(dialog)}
      </section>
    </dialog>`
  }

  private renderOptionDialogBody(dialog: Extract<PickerDialog, { kind: 'option' }>) {
    return html`<div class="picker-list dialog-picker-list">
      ${dialog.options.map((option) => html`<button
        class=${option.value === dialog.value ? 'picker-option selected' : 'picker-option'}
        type="button"
        title=${option.label}
        @click=${() => this.choosePickerOption(option.value, dialog.onSelect)}
      >
        <span>${option.label}</span>
        ${option.meta ? html`<small>${option.meta}</small>` : ''}
      </button>`)}
    </div>`
  }

  private renderLoraDialogBody(dialog: Extract<PickerDialog, { kind: 'lora' }>) {
    return html`<input class="search" placeholder="Search LoRAs" .value=${this.loraSearch} @input=${(event: Event) => (this.loraSearch = (event.target as HTMLInputElement).value)} />
      <div class="picker-list dialog-picker-list lora-list">
        ${this.visibleLoras(dialog.selectedPaths).map((lora) => html`<label class=${dialog.selectedPaths.includes(lora.path) ? 'lora checked' : 'lora'} title=${lora.name}>
          <input type="checkbox" .checked=${dialog.selectedPaths.includes(lora.path)} @change=${() => this.toggleDialogLora(lora.path, dialog)} />
          <span>${lora.name}</span>
        </label>`)}
      </div>`
  }

  private openOptionDialog(event: MouseEvent, label: string, value: string, options: PickerOption[], onSelect: (value: string) => void) {
    this.positionPickerDialog(event.currentTarget as HTMLElement, label)
    this.pickerDialog = { kind: 'option', label, value, options, onSelect }
    void this.showPickerDialog()
  }

  private openLoraDialog(event: MouseEvent, selectedPaths: string[], onToggle: (value: string) => void) {
    this.positionPickerDialog(event.currentTarget as HTMLElement, 'LoRAs')
    this.pickerDialog = { kind: 'lora', label: 'LoRAs', selectedPaths: [...selectedPaths], onToggle }
    void this.showPickerDialog()
  }

  private displayPickerLabel(label: string) {
    if (label === 'sampler_name') return 'Sampler'
    return label.charAt(0).toUpperCase() + label.slice(1)
  }

  private positionPickerDialog(anchor: HTMLElement, label: string) {
    const rect = anchor.getBoundingClientRect()
    const viewportPadding = 8
    const gap = 6
    const availableWidth = window.innerWidth - viewportPadding * 2
    const preferredWidth = label === 'Model' ? 680 : label === 'LoRAs' ? 640 : 420
    const width = Math.min(availableWidth, Math.max(rect.width, preferredWidth))
    const left = Math.min(Math.max(viewportPadding, rect.left), Math.max(viewportPadding, window.innerWidth - width - viewportPadding))
    const availableBelow = window.innerHeight - rect.bottom - viewportPadding - gap
    const availableAbove = rect.top - viewportPadding - gap
    const openBelow = availableBelow >= 260 || availableBelow >= availableAbove
    const maxHeight = Math.max(220, Math.min(620, openBelow ? availableBelow : availableAbove))
    const top = openBelow ? rect.bottom + gap : Math.max(viewportPadding, rect.top - maxHeight - gap)
    this.pickerDialogPosition = { left, top, width, maxHeight }
  }

  private async showPickerDialog() {
    await this.updateComplete
    const dialog = this.renderRoot.querySelector('.picker-dialog') as HTMLDialogElement | null
    if (dialog && !dialog.open) dialog.showModal()
  }

  private choosePickerOption(value: string, onSelect: (value: string) => void) {
    onSelect(value)
    this.closePickerDialog()
  }

  private toggleDialogLora(value: string, dialog: Extract<PickerDialog, { kind: 'lora' }>) {
    dialog.onToggle(value)
    const selectedPaths = dialog.selectedPaths.includes(value)
      ? dialog.selectedPaths.filter((path) => path !== value)
      : [...dialog.selectedPaths, value]
    this.pickerDialog = { ...dialog, selectedPaths }
  }

  private closePickerBackdrop = (event: MouseEvent) => {
    if (event.target === event.currentTarget) this.closePickerDialog()
  }

  private closePickerDialog = () => {
    const dialog = this.renderRoot.querySelector('.picker-dialog') as HTMLDialogElement | null
    if (dialog?.open) dialog.close()
    this.pickerDialog = undefined
  }

  private formatBytes(bytes: number) {
    if (!Number.isFinite(bytes) || bytes <= 0) return ''
    const units = ['B', 'KB', 'MB', 'GB', 'TB']
    let value = bytes
    let unit = 0
    while (value >= 1024 && unit < units.length - 1) {
      value /= 1024
      unit += 1
    }
    return `${value >= 10 || unit === 0 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`
  }

  private renderLayerSource() {
    const selected = this.sourceImages.find((image) => image.rel === this.selectedSourceRel) ?? this.sourceImages[0]
    return html`<div class="layer-tab source-tab">
      <label class="field">
        <span>Data root</span>
        <input
          placeholder="Path to image folder"
          .value=${this.sourceRoot}
          @change=${(event: Event) => this.setSourceRoot((event.target as HTMLInputElement).value)}
        />
      </label>
      <div class="button-grid">
        <button @click=${this.pickSourceRoot}>Choose folder</button>
        <button @click=${this.loadSourceImages} ?disabled=${this.isLoadingSources || !this.sourceRoot.trim()}>
          ${this.isLoadingSources ? 'Loading...' : 'Scan'}
        </button>
      </div>
      <div class="button-grid">
        <button @click=${() => this.setSourceRoot('')}>Clear</button>
      </div>
      ${selected
        ? html`<div class="source-preview">
            <img src=${this.sourceImageUrl(selected.rel)} alt=${selected.name} />
          </div>`
        : html`<div class="source-preview empty">Choose a folder to preview sprites</div>`}
      <select class="source-select" size="7" .value=${selected?.rel ?? ''} @change=${(event: Event) => this.selectSourceImage((event.target as HTMLSelectElement).value)}>
        ${this.sourceImages.map((image) => html`<option value=${image.rel}>${image.rel}</option>`)}
      </select>
      <button @click=${() => selected && this.applySourceImageToSelectedLayer(selected)} ?disabled=${!selected}>Use source image on selected layer</button>
    </div>`
  }

  private renderLayerProperties(layer: LayerItem) {
    return html`<div class="layer-tab">
      ${this.renderLayerTransform(layer)}
      ${this.numberControl('Opacity', layer.opacity, 0, 1, 0.01, 2, (value) => this.setLayerOpacity(layer.id, value))}
      <label class="check-field">
        <input type="checkbox" .checked=${layer.preset.samplingEnabled} @change=${(event: Event) => this.updateSelectedLayerPreset({ samplingEnabled: (event.target as HTMLInputElement).checked })} />
        <span>Use layer as condition</span>
      </label>
      ${this.numberControl('Condition strength', layer.preset.conditionWeight, 0, 4, 0.05, 2, (value) => this.updateSelectedLayerPreset({ conditionWeight: this.clamp(value, 0, 4, layer.preset.conditionWeight) }))}
      <div class="empty">Explicit masks define conditioning areas. Without explicit masks, layer alpha is used.</div>
    </div>`
  }

  private renderLayerTransform(layer: LayerItem) {
    const object = this.findLayer(layer.id)
    if (!object) return ''
    const transform = this.layerTransform(object)
    return html`<section class="transform-panel">
      <div class="transform-head"><strong>Transform</strong><span>center anchor</span></div>
      <div class="transform-grid">
        ${this.numberControl('Center X', transform.centerX, 0, this.stageWidth, 1, 0, (value) => this.setLayerCenter(layer.id, value, transform.centerY))}
        ${this.numberControl('Center Y', transform.centerY, 0, this.stageHeight, 1, 0, (value) => this.setLayerCenter(layer.id, transform.centerX, value))}
        ${this.numberControl('Width', transform.width, 1, this.stageWidth * 4, 1, 0, (value) => this.setLayerSize(layer.id, value, transform.height))}
        ${this.numberControl('Height', transform.height, 1, this.stageHeight * 4, 1, 0, (value) => this.setLayerSize(layer.id, transform.width, value))}
        ${this.numberControl('Scale', transform.scale, 1, 400, 1, 0, (value) => this.setLayerScale(layer.id, value))}
        ${this.numberControl('Angle', transform.angle, -180, 180, 1, 0, (value) => this.setLayerAngle(layer.id, value))}
      </div>
      <div class="button-grid">
        <button @click=${() => this.centerLayer(layer.id)}>Center</button>
        <button @click=${() => this.fitLayerToViewport(layer.id)}>Fit viewport</button>
      </div>
    </section>`
  }

  private renderSelectionTransform(selection: SelectionRect) {
    const centerX = Math.round(selection.x + selection.width / 2)
    const centerY = Math.round(selection.y + selection.height / 2)
    return html`<section class="transform-panel">
      <div class="transform-head"><strong>Transform</strong><span>center anchor</span></div>
      <div class="transform-grid">
        ${this.numberControl('Center X', centerX, 0, this.stageWidth, 1, 0, (value) => this.setSelectionTransform({ centerX: value }))}
        ${this.numberControl('Center Y', centerY, 0, this.stageHeight, 1, 0, (value) => this.setSelectionTransform({ centerY: value }))}
        ${this.numberControl('Width', selection.width, 1, this.stageWidth, 1, 0, (value) => this.setSelectionTransform({ width: value }))}
        ${this.numberControl('Height', selection.height, 1, this.stageHeight, 1, 0, (value) => this.setSelectionTransform({ height: value }))}
      </div>
      <div class="button-grid">
        <button @click=${() => this.setSelectionTransform({ centerX: this.stageWidth / 2, centerY: this.stageHeight / 2 })}>Center</button>
        <button @click=${() => this.setSelectionTransform({ x: 0, y: 0, width: this.stageWidth, height: this.stageHeight })}>Full viewport</button>
      </div>
    </section>`
  }

  private renderLayerGenerator(layer: LayerItem) {
    const variations = this.generatedLayers.filter((variation) => variation.layerId === layer.id)
    return html`<div class="layer-tab generator-tab">
      <div class="generator-controls">
        ${this.renderLayerGenerationControls(layer)}
      </div>
      <div class="generator-actions">
        <div class="resolution-grid">
          ${this.numberControl('Gen W', this.generationWidth, 256, 1536, 64, 0, (value) => this.setGenerationWidth(value))}
          ${this.numberControl('Gen H', this.generationHeight, 256, 1536, 64, 0, (value) => this.setGenerationHeight(value))}
        </div>
        <div class="generator-action-row">
          <button @click=${this.useViewportResolution}>Use viewport</button>
          ${this.numberControl('Variations', this.layerVariationCount, 1, 12, 1, 0, (value) => (this.layerVariationCount = Math.round(value)))}
        </div>
        <button @click=${this.generatePromptLayer} ?disabled=${this.isGeneratingLayer}>
          ${this.isGeneratingLayer ? 'Generating...' : 'Generate variations'}
        </button>
      </div>
      <section class="variation-dock">
        ${variations.length
          ? html`<div class="variation-list">
              ${variations.map(
                (variation, index) => html`<button class=${variation.selected ? 'variation selected' : 'variation'} @click=${() => this.placeGeneratedLayer(variation)}>
                  <span class="variation-label">${index + 1}</span>
                  <img src=${variation.image} alt=${`seed ${variation.seed}`} />
                  <span class="variation-seed">Seed ${variation.seed}</span>
                </button>`,
              )}
            </div>`
          : html`<div class="variation-empty empty">No variations yet</div>`}
      </section>
    </div>`
  }

  private renderLayerGenerationControls(layer: LayerItem) {
    return this.renderGenerationControls({
      deviceLabel: 'Variation GPU',
      deviceValue: this.layerDevice,
      onDevice: (value) => (this.layerDevice = value),
      beforeSampling: html`
        <label class="check-field">
          <input type="checkbox" .checked=${this.generatorActive} @change=${(event: Event) => this.setGeneratorActive((event.target as HTMLInputElement).checked)} />
          <span>Active: selected variation drives this layer</span>
        </label>
      `,
      sampling: this.layerSamplingControls(layer),
      afterSampling: html`
        ${this.renderLayerTransparencyControls()}
        <label class="field inline">
          <span>Seed</span>
          <input placeholder="random" .value=${this.layerGenerationSeed} @input=${(event: Event) => (this.layerGenerationSeed = (event.target as HTMLInputElement).value)} />
        </label>
      `,
    })
  }

  private layerSamplingControls(layer: LayerItem): SamplingControlsConfig {
    return {
      prompt: layer.preset.prompt,
      negativePrompt: layer.preset.negativePrompt,
      modelPath: this.selectedModel,
      loraPaths: [...this.selectedLoras],
      cfg: this.layerGenerationCfg,
      steps: this.layerGenerationSteps,
      sampler: this.layerGenerationSampler,
      scheduler: this.layerGenerationScheduler,
      onPrompt: (value) => this.updateSelectedLayerPreset({ prompt: value }),
      onNegativePrompt: (value) => this.updateSelectedLayerPreset({ negativePrompt: value }),
      onModelPath: this.setSceneModel,
      onToggleLora: this.toggleSceneLora,
      onCfg: (value) => this.setLayerSamplerValue('cfg', this.clamp(value, 0, 30, this.layerGenerationCfg), this.selectedModel),
      onSteps: (value) => this.setLayerSamplerValue('steps', Math.round(this.clamp(value, 1, 40, this.layerGenerationSteps)), this.selectedModel),
      onSampler: (value) => this.setLayerSamplerValue('sampler', value, this.selectedModel),
      onScheduler: (value) => this.setLayerSamplerValue('scheduler', value, this.selectedModel),
    }
  }

  private renderFooterTaskMonitor() {
    const pending = this.generationTasks.filter((task) => task.status === 'queued' || task.status === 'running')
    const queued = pending.filter((task) => task.status === 'queued' || task.message === 'Waiting for GPU worker')
    const running = pending.length - queued.length
    const newest = this.generationTasks[0]
    const progress = pending.length
      ? pending.reduce((sum, task) => sum + (task.progress || 0), 0) / pending.length
      : newest?.progress ?? 0
    const percent = Math.round(progress * 100)
    return html`<div class="footer-task-host">
      <button class="footer-task-button" @click=${() => (this.taskPopoverOpen = !this.taskPopoverOpen)}>
        <span>Tasks ${pending.length}</span>
        <span>Queued ${queued.length}</span>
        <span>Running ${running}</span>
        <span>${percent}%</span>
      </button>
      <div class="footer-task-bar" aria-label=${`Global task progress ${percent}%`}><span style="width: ${percent}%"></span></div>
      ${this.taskPopoverOpen
        ? html`<div class="task-popover">
            <div class="task-popover-head">
              <strong>Tasks</strong>
              <button @click=${() => (this.taskPopoverOpen = false)}>×</button>
            </div>
            ${this.generationTasks.length
              ? html`<div class="task-popover-list">${this.generationTasks.map((task) => this.renderTaskProgress(task))}</div>`
              : html`<div class="task-empty">No tasks yet</div>`}
          </div>`
        : ''}
    </div>`
  }

  private _isTextChannel(name: string): boolean {
    return name.startsWith('tagger:')
  }

  private _applyDebugChannels(incoming: Record<string, string>) {
    const names = Object.keys(incoming)
    for (const [name, val] of Object.entries(incoming)) this._debugChannelData.set(name, val)

    const root = this.renderRoot as ShadowRoot
    for (const name of names) {
      const val = incoming[name]
      if (val.startsWith('data:')) {
        const el = root.querySelector(`img[data-channel="${CSS.escape(name)}"]`) as HTMLImageElement | null
        if (el) el.src = val
      } else {
        const el = root.querySelector(`[data-channel="${CSS.escape(name)}"]`) as HTMLElement | null
        if (el) el.textContent = val
      }
    }
    const currentNames = this.debugChannelNames
    const structureChanged = names.length !== currentNames.length || names.some((n, i) => n !== currentNames[i])
    if (structureChanged) this.debugChannelNames = [...new Set([...currentNames, ...names])]
  }

  private renderDebugChannel(name: string) {
    const val = this._debugChannelData.get(name) ?? ''
    if (this._isTextChannel(name)) {
      // Text channel: render tags as chips
      const tags = val ? val.split(',').map((t) => t.trim()).filter(Boolean) : []
      return html`<div class="debug-mosaic-cell debug-text-cell" title=${name}>
        <div class="debug-tag-content" data-channel=${name}>
          ${tags.length
            ? tags.map((tag) => html`<span class="debug-tag-chip">${tag}</span>`)
            : html`<span class="debug-tag-empty">waiting…</span>`}
        </div>
        <span class="debug-cell-label">${name}</span>
      </div>`
    }
    return html`<div
      class="debug-mosaic-cell"
      title=${name}
      @click=${() => { this.debugPreviewChannel = name; this.debugMosaicMode = false }}
    >
      <img data-channel=${name} src=${val} alt=${name} />
      <span class="debug-cell-label">${name}</span>
    </div>`
  }

  private renderDebugOverlay() {
    const names = this.debugChannelNames
    if (!names.length) {
      return html`<div class="debug-overlay debug-overlay-empty">
        <span>Debug streams: waiting for frames…</span>
      </div>`
    }

    if (!this.debugMosaicMode && this.debugPreviewChannel) {
      const val = this._debugChannelData.get(this.debugPreviewChannel) ?? ''
      return html`<div class="debug-overlay debug-overlay-single">
        <div class="debug-overlay-toolbar">
          <button @click=${() => { this.debugMosaicMode = true }}>← Mosaic</button>
          <span class="debug-channel-label">${this.debugPreviewChannel}</span>
        </div>
        ${this._isTextChannel(this.debugPreviewChannel)
          ? html`<div class="debug-preview-text" data-channel=${this.debugPreviewChannel}>
              ${(val || '').split(',').map((t) => t.trim()).filter(Boolean).map((tag) =>
                html`<span class="debug-tag-chip">${tag}</span>`
              )}
            </div>`
          : html`<img class="debug-preview-full" data-channel=${this.debugPreviewChannel} src=${val} alt=${this.debugPreviewChannel} />`
        }
      </div>`
    }

    return html`<div class="debug-overlay debug-overlay-mosaic">
      <div class="debug-overlay-toolbar">
        <span>${names.length} channel${names.length === 1 ? '' : 's'}</span>
        <button @click=${() => { this.debugStreamsEnabled = false; this.debugChannelNames = []; this._debugChannelData.clear() }}>Close</button>
      </div>
      <div class="debug-mosaic-grid">
        ${names.map((name) => this.renderDebugChannel(name))}
      </div>
    </div>`
  }

  private renderFooterErrors() {
    const count = this.errorLog.length
    return html`<div class="footer-error-host">
      <button
        class=${count ? 'footer-error-button active' : 'footer-error-button'}
        title=${count ? 'Show errors' : 'No errors'}
        @click=${() => (this.errorPopoverOpen = !this.errorPopoverOpen)}
      >${count} error${count === 1 ? '' : 's'}</button>
      ${this.errorPopoverOpen
        ? html`<div class="error-popover">
            <div class="error-popover-head">
              <strong>Errors</strong>
              <button @click=${this.clearErrors}>Clear</button>
            </div>
            ${count
              ? html`<div class="error-list">${this.errorLog.map((item) => html`<button class="error-item">
                  <span>${new Date(item.time).toLocaleTimeString()}</span>
                  <p>${item.message}</p>
                </button>`)}</div>`
              : html`<div class="task-empty">No errors</div>`}
          </div>`
        : ''}
    </div>`
  }

  private reportError(message: string) {
    const text = message.trim()
    if (!text) return
    this.errorLog = [{ id: crypto.randomUUID(), message: text, time: Date.now() }, ...this.errorLog].slice(0, 12)
  }

  private clearErrors = () => {
    this.errorLog = []
    this.errorPopoverOpen = false
  }

  private renderTaskProgress(task: GenerationTask) {
    const percent = Math.round((task.progress || 0) * 100)
    return html`<div class=${task.status === 'error' ? 'task-item error-task' : 'task-item'}>
      <div class="task-row">
        <strong>${task.label}</strong>
        <span>${task.status}</span>
      </div>
      <div class="task-bar" aria-label=${`${percent}%`}><span style="width: ${percent}%"></span></div>
      <div class="task-row task-detail">
        <span>${task.phase}</span>
        <span>${percent}%</span>
      </div>
      <p>${task.error || task.message}</p>
    </div>`
  }

  private renderMotionTaskProgress(task: MotionTaskProgress) {
    const percent = Math.round((task.progress || 0) * 100)
    const previewFrames = task.preview_frames || task.result?.frames || []
    return html`<div class=${task.status === 'error' ? 'task-item error-task' : 'task-item'}>
      <div class="task-row">
        <strong>Motion</strong>
        <span>${task.status}</span>
      </div>
      <div class="task-bar" aria-label=${`Motion generation ${percent}%`}><span style="width: ${percent}%"></span></div>
      <div class="task-row task-detail">
        <span>${task.phase}</span>
        <span>${percent}%</span>
      </div>
      <p>${task.error || task.message}</p>
      ${previewFrames.length
        ? html`<div class="motion-frame-strip">
            ${previewFrames.slice(-6).map((frame) => html`<img src=${this.backendUrl(frame.url, { t: String(Date.now()) })} alt=${`frame ${frame.index + 1}`} />`)}
          </div>`
        : ''}
    </div>`
  }

  private setSourceRoot(root: string) {
    this.sourceRoot = root.trim()
    this.selectedSourceRel = ''
    this.sourceImages = []
    if (this.sourceRoot) window.localStorage.setItem('rtdiffusion.sourceRoot', this.sourceRoot)
    else window.localStorage.removeItem('rtdiffusion.sourceRoot')
  }

  private loadSourceImages = async () => {
    if (!this.sourceRoot.trim()) return
    this.isLoadingSources = true
    try {
      const response = await fetch(this.backendUrl('/sources', { root: this.sourceRoot }))
      if (!response.ok) throw new Error(await response.text())
      const catalog = (await response.json()) as SourceCatalog
      this.sourceRoot = catalog.root
      window.localStorage.setItem('rtdiffusion.sourceRoot', catalog.root)
      this.sourceImages = catalog.images
      this.selectedSourceRel = catalog.images[0]?.rel ?? ''
    } catch (error) {
      this.reportError(error instanceof Error ? error.message : String(error))
      this.sourceImages = []
      this.selectedSourceRel = ''
    } finally {
      this.isLoadingSources = false
    }
  }

  private loadVideoCapabilities = async () => {
    try {
      const response = await fetch(this.backendUrl('/video/capabilities'))
      if (!response.ok) return
      this.videoCapabilities = (await response.json()) as VideoCapabilities
    } catch {
      this.videoCapabilities = undefined
    }
  }

  private loadRendererCapabilities = async () => {
    try {
      const response = await fetch(this.backendUrl('/renderers/capabilities'))
      if (!response.ok) return
      this.rendererCapabilities = (await response.json()) as RendererCapabilities
    } catch {
      this.rendererCapabilities = undefined
    }
  }

  private loadGpuDevices = async () => {
    try {
      const response = await fetch(this.backendUrl('/system/gpus'))
      if (!response.ok) return
      const catalog = (await response.json()) as GpuCatalog
      const devices = [{ id: '', name: `Backend default (${catalog.default})`, kind: 'default', available: true }, ...catalog.devices]
      this.gpuDevices = devices
      const valid = new Set(devices.map((device) => device.id))
      if (!valid.has(this.realtimeDevice)) this.realtimeDevice = ''
      if (!valid.has(this.layerDevice)) this.layerDevice = ''
      if (!valid.has(this.videoDevice)) this.videoDevice = ''
    } catch {
      this.gpuDevices = [{ id: '', name: 'Backend default', kind: 'default', available: true }]
    }
  }

  private renderDeviceSelect(label: string, value: string, onChange: (value: string) => void) {
    return html`<label class="field inline">
      <span>${label}</span>
      <select .value=${value} @change=${(event: Event) => onChange((event.target as HTMLSelectElement).value)}>
        ${this.gpuDevices.map((device) => html`<option value=${device.id}>${device.name}${device.memory_total ? ` (${this.formatBytes(device.memory_total)})` : ''}</option>`)}
      </select>
    </label>`
  }

  private pickSourceRoot = async () => {
    this.isLoadingSources = true
    try {
      const response = await fetch(this.backendUrl('/sources/pick-root'), { method: 'POST' })
      if (!response.ok) throw new Error(await response.text())
      const result = (await response.json()) as { root: string }
      if (!result.root) return
      this.setSourceRoot(result.root)
      await this.loadSourceImages()
    } catch (error) {
      this.reportError(error instanceof Error ? error.message : String(error))
    } finally {
      this.isLoadingSources = false
    }
  }

  private selectSourceImage(rel: string) {
    this.selectedSourceRel = rel
  }

  private sourceImageUrl(rel: string) {
    return this.backendUrl('/sources/image', { root: this.sourceRoot, rel })
  }

  private async applySourceImageToSelectedLayer(source: SourceImage) {
    if (!this.fabricCanvas) return
    try {
      const layer = this.selectedLayer()
      if (!layer) throw new Error('Select a layer before applying a source image.')
      const dataUrl = await this.fetchImageDataUrl(this.sourceImageUrl(source.rel), source.name)
      const target = this.findLayer(layer.id)
      if (!target || !this.fabricCanvas) throw new Error('Selected layer is no longer available.')
      const oldId = (target as FabricObject & { __uid?: string }).__uid
      const preset = this.normalizeLayerPreset((target as FabricObject & { __preset?: LayerPreset }).__preset ?? this.currentPreset())
      const bounds = this.layerBounds(target)
      const image = await FabricImage.fromURL(dataUrl)
      const targetWidth = target.type === 'rect' && bounds.width >= this.stageWidth ? Math.min(image.width ?? bounds.width, this.stageWidth) : bounds.width
      const targetHeight = target.type === 'rect' && bounds.height >= this.stageHeight ? Math.min(image.height ?? bounds.height, this.stageHeight) : bounds.height
      image.set({ name: source.name, left: bounds.x, top: bounds.y, originX: 'left', originY: 'top' })
      image.scaleToWidth(targetWidth)
      image.scaleToHeight(targetHeight)
      this.styleTransformControls(image)
      ;(image as FabricObject & { __uid?: string; __preset?: LayerPreset }).__uid = oldId
      ;(image as FabricObject & { __preset?: LayerPreset }).__preset = preset
      this.pushHistory()
      this.replaceLayerObject(layer.id, target, image)
    } catch (error) {
      this.reportError(`Source image load failed: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  private renderRegionsPanel(target: RegionTarget, layerId?: string) {
    const targetRegions = this.regions.filter((region) => region.target === target && (target === 'scene' || region.layerId === layerId))
    const selection = this.selectionRect ? this.normalizedSelection() : undefined
    return html`<div class="layer-tab regions-tab">
      <label class="field">
        <span>Mask name</span>
        <input placeholder="face, hands, background" .value=${this.regionName} @input=${(event: Event) => (this.regionName = (event.target as HTMLInputElement).value)} />
      </label>
      <label class="field inline">
        <span>Shape</span>
        <select .value=${this.regionShape} @change=${(event: Event) => this.setRegionShape((event.target as HTMLSelectElement).value as RegionShape)}>
          <option value="rope">Rope</option>
          <option value="polygon">Polygon</option>
          <option value="box">Box</option>
        </select>
      </label>
      <div class="button-grid">
        <button @click=${() => this.addRegionFromSelection(target, layerId)} ?disabled=${!selection}>Add selected mask</button>
        <button @click=${this.clearSelection} ?disabled=${!selection}>Clear selection</button>
      </div>
      <div class="empty">Set destination to Masks and use Mask on the input to mark an area. Rope drags a freehand outline; polygon adds vertices by clicking.</div>
      ${targetRegions.length
        ? html`<div class="region-list">
            ${targetRegions.map(
              (region) => html`<section class="region-card">
                <div class="region-head">
                  <input class="region-color" type="color" .value=${region.color} title="Use mask color" @click=${() => this.useRegionColor(region.color)} @input=${(event: Event) => this.setRegionColor(region.id, (event.target as HTMLInputElement).value)} />
                  <input .value=${region.name} @input=${(event: Event) => this.updateRegion(region.id, { name: (event.target as HTMLInputElement).value })} />
                  <button title="Delete mask" @click=${() => this.deleteRegion(region.id)}>×</button>
                </div>
                <div class="region-meta">${region.color} · ${region.shape} · ${Math.round(region.rect.width)} x ${Math.round(region.rect.height)} · ${region.inherited ? 'inherited' : 'edited'}</div>
                <label class="field">
                  <span>Prompt +</span>
                  <textarea rows="2" .value=${region.prompt} @input=${(event: Event) => this.updateRegion(region.id, { prompt: (event.target as HTMLTextAreaElement).value, inherited: false })}></textarea>
                </label>
                <label class="field">
                  <span>Prompt -</span>
                  <textarea rows="2" .value=${region.negativePrompt} @input=${(event: Event) => this.updateRegion(region.id, { negativePrompt: (event.target as HTMLTextAreaElement).value, inherited: false })}></textarea>
                </label>
                <label class="check-field">
                  <input type="checkbox" .checked=${region.negate} @change=${(event: Event) => this.updateRegion(region.id, { negate: (event.target as HTMLInputElement).checked })} />
                  <span>Negate mask</span>
                </label>
                <label class="field inline">
                  <span>Schedule</span>
                  <select .value=${region.schedule} @change=${(event: Event) => this.updateRegion(region.id, { schedule: (event.target as HTMLSelectElement).value as RegionSchedule, inherited: false })}>
                    <option value="auto">Auto</option>
                    <option value="linear">Linear</option>
                    <option value="ease_in">Ease in</option>
                    <option value="ease_out">Ease out</option>
                    <option value="ease_in_out">Ease in/out</option>
                  </select>
                </label>
                ${this.numberControl('Mask', region.maskStrength, 0, 1, 0.01, 2, (value) => this.updateRegion(region.id, { maskStrength: this.clamp(value, 0, 1, region.maskStrength) }))}
                ${this.aggregateOperatorControl('Mask mix', region.maskOperator, (value) => this.updateRegion(region.id, { maskOperator: value, inherited: false }))}
                ${this.numberControl('Denoise', region.denoise, 0, 0.999, 0.01, 2, (value) => this.updateRegion(region.id, { denoise: this.clamp(value, 0, 0.999, region.denoise), inherited: false }))}
                ${this.aggregateOperatorControl('Denoise mix', region.denoiseOperator, (value) => this.updateRegion(region.id, { denoiseOperator: value, inherited: false }))}
                ${region.schedule === 'auto'
                  ? html`<div class="region-meta">Auto start/end resolves from layer order, opacity, and transparency.</div>`
                  : html`<div class="interval-row">
                      ${this.numberControl('Start', region.start, 0, 1, 0.01, 2, (value) => this.updateRegion(region.id, { start: this.clamp(value, 0, region.end, region.start), inherited: false }))}
                      ${this.numberControl('End', region.end, 0, 1, 0.01, 2, (value) => this.updateRegion(region.id, { end: this.clamp(value, region.start, 1, region.end), inherited: false }))}
                    </div>`}
                ${this.numberControl('Inner blur', region.innerBlur, 0, 160, 1, 0, (value) => this.updateRegion(region.id, { innerBlur: Math.round(this.clamp(value, 0, 160, region.innerBlur)) }))}
                ${this.numberControl('Outer blur', region.outerBlur, 0, 160, 1, 0, (value) => this.updateRegion(region.id, { outerBlur: Math.round(this.clamp(value, 0, 160, region.outerBlur)) }))}
                <button @click=${() => this.restoreRegionInheritance(region.id)}>Restore inherited</button>
              </section>`,
            )}
          </div>`
        : html`<div class="variation-empty empty">No named masks yet</div>`}
    </div>`
  }

  private renderLayerEffects(layer: LayerItem) {
    return html`<div class="layer-tab">
      <button @click=${this.clearActiveLayer}>Clear selected layer</button>
      <button @click=${this.duplicateActiveLayer}>Duplicate selected layer</button>
      <button @click=${this.deleteActiveLayer}>Delete selected layer</button>
      ${this.numberControl('Opacity', layer.opacity, 0, 1, 0.01, 2, (value) => this.setLayerOpacity(layer.id, value))}
      <div class="empty">Effect slots will apply to the selected layer only.</div>
    </div>`
  }

  private getControlNetConfig(layerId: string): ControlNetConfig {
    return this.layerControlNetConfigs.get(layerId) ?? {
      enabled: false,
      model: 'canny',
      scale: 1.0,
      startAt: 0.0,
      endAt: 1.0,
      usePreprocessor: true,
      useLayerFrame: false,
      modelPath: '',
      explicitImageUrl: '',
      sourceImageUrl: '',
      cnPreviewHeight: 160,
      preprocessorParams: {
        cannyLow: 100, cannyHigh: 200,
        openposeHands: true, openposeFace: true,
        mlsdThrV: 0.1, mlsdThrD: 0.1,
        linartCoarse: false,
      },
    }
  }

  private updateControlNetConfig(layerId: string, patch: Partial<ControlNetConfig>) {
    const current = this.getControlNetConfig(layerId)
    const updated = { ...current, ...patch }
    this.layerControlNetConfigs = new Map(this.layerControlNetConfigs).set(layerId, updated)
  }

  private updateControlNetParams(layerId: string, patch: Partial<ControlNetConfig['preprocessorParams']>) {
    const current = this.getControlNetConfig(layerId)
    this.updateControlNetConfig(layerId, { preprocessorParams: { ...current.preprocessorParams, ...patch } })
  }

  private pickControlNetImage = async (layerId: string, field: 'explicitImageUrl' | 'sourceImageUrl') => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = 'image/*'
    await new Promise<void>((resolve) => {
      input.onchange = () => {
        const file = input.files?.[0]
        if (file) {
          const reader = new FileReader()
          reader.onload = () => {
            this.updateControlNetConfig(layerId, { [field]: reader.result as string })
          }
          reader.readAsDataURL(file)
        }
        resolve()
      }
      input.click()
    })
  }

  private renderLayerControlNetPanel(layer: LayerItem) {
    const cfg = this.getControlNetConfig(layer.id)
    const up = (patch: Partial<ControlNetConfig>) => this.updateControlNetConfig(layer.id, patch)
    const upP = (patch: Partial<ControlNetConfig['preprocessorParams']>) => this.updateControlNetParams(layer.id, patch)
    const hasExplicit = !!cfg.explicitImageUrl
    const hasSource = !!cfg.sourceImageUrl
    const modelHasParams: Record<ControlNetModel, boolean> = {
      canny: true, openpose: true, mlsd: true, lineart: true,
      depth: false, hed: false, scribble: false, normal: false, seg: false,
    }
    return html`<div class="layer-tab source-tab controlnet-panel">
      <label class="check-field">
        <input type="checkbox" .checked=${cfg.enabled} @change=${(e: Event) => up({ enabled: (e.target as HTMLInputElement).checked })} />
        <span>Enable ControlNet</span>
      </label>
      ${!cfg.enabled ? html`<div class="empty" style="font-size:11px">Enable to add ControlNet conditioning to this layer.</div>` : html`
        <label class="field inline">
          <span>Model</span>
          <select .value=${cfg.model} @change=${(e: Event) => up({ model: (e.target as HTMLSelectElement).value as ControlNetModel })}>
            <option value="canny">Canny edges</option>
            <option value="openpose">OpenPose (body)</option>
            <option value="depth">Depth (MiDaS)</option>
            <option value="hed">HED soft edges</option>
            <option value="mlsd">MLSD lines</option>
            <option value="scribble">Scribble</option>
            <option value="normal">Normal maps</option>
            <option value="lineart">Lineart</option>
            <option value="seg">Segmentation</option>
          </select>
        </label>
        <label class="field inline">
          <span>Model file</span>
          <select .value=${cfg.modelPath} @change=${(e: Event) => up({ modelPath: (e.target as HTMLSelectElement).value })}>
            <option value="">Auto (best for pipeline)</option>
            ${this.cnLocalModels.map(m => html`<option value=${m.path} ?selected=${cfg.modelPath === m.path}>${m.name}</option>`)}
          </select>
        </label>
        <label class="field inline" style="margin-top:2px">
          <span>Custom path</span>
          <input placeholder="HuggingFace ID or override path" .value=${cfg.modelPath}
            @change=${(e: Event) => up({ modelPath: (e.target as HTMLInputElement).value })} />
        </label>
        ${this.numberControl('Scale', cfg.scale, 0, 2, 0.05, 2, (v) => up({ scale: this.clamp(v, 0, 2, cfg.scale) }))}
        <div class="resolution-grid">
          ${this.numberControl('Start', cfg.startAt, 0, 1, 0.01, 2, (v) => up({ startAt: this.clamp(v, 0, 1, cfg.startAt) }))}
          ${this.numberControl('End', cfg.endAt, 0, 1, 0.01, 2, (v) => up({ endAt: this.clamp(v, 0, 1, cfg.endAt) }))}
        </div>

        <div class="section-header" style="font-weight:600;margin:8px 0 4px">Control image</div>
        <label class="check-field">
          <input type="checkbox" .checked=${cfg.useLayerFrame} @change=${(e: Event) => up({ useLayerFrame: (e.target as HTMLInputElement).checked })} />
          <span>Use current layer frame (live)</span>
        </label>
        ${cfg.useLayerFrame ? html`
          <div class="empty" style="font-size:11px;color:#888;margin:4px 0">The layer's current rendered frame is sent each frame as the control image.</div>
        ` : html`
          <label class="check-field">
            <input type="checkbox" .checked=${cfg.usePreprocessor} @change=${(e: Event) => up({ usePreprocessor: (e.target as HTMLInputElement).checked })} />
            <span>Auto-preprocess source image</span>
          </label>
          ${cfg.usePreprocessor ? html`
            <div class="button-grid">
              <button @click=${() => this.pickControlNetImage(layer.id, 'sourceImageUrl')}>Choose raw image…</button>
              ${hasSource ? html`<button @click=${() => up({ sourceImageUrl: '' })}>Clear</button>` : ''}
            </div>
            <div class="source-preview" style="height:${cfg.cnPreviewHeight}px;resize:vertical;overflow:hidden;min-height:80px">
              ${hasSource
                ? html`<img src=${cfg.sourceImageUrl} alt="source for preprocessing" style="max-width:100%;max-height:100%;object-fit:contain" />`
                : html`<span class="empty" style="color:#888;font-size:12px">No image — canvas layer will be used</span>`}
            </div>
          ` : html`
            <div class="button-grid">
              <button @click=${() => this.pickControlNetImage(layer.id, 'explicitImageUrl')}>Choose control image…</button>
              ${hasExplicit ? html`<button @click=${() => up({ explicitImageUrl: '' })}>Clear</button>` : ''}
            </div>
            <div class="source-preview" style="height:${cfg.cnPreviewHeight}px;resize:vertical;overflow:hidden;min-height:80px">
              ${hasExplicit
                ? html`<img src=${cfg.explicitImageUrl} alt="control image" style="max-width:100%;max-height:100%;object-fit:contain" />`
                : html`<span class="empty" style="color:#888;font-size:12px">No image — canvas layer will be used as control</span>`}
            </div>
          `}
        `}

        ${modelHasParams[cfg.model] ? html`
          <div class="section-header" style="font-weight:600;margin:8px 0 4px">Preprocessor options</div>
          ${cfg.model === 'canny' ? html`
            <div class="resolution-grid">
              ${this.numberControl('Low threshold', cfg.preprocessorParams.cannyLow, 0, 255, 1, 0, (v) => upP({ cannyLow: Math.round(this.clamp(v, 0, 255, cfg.preprocessorParams.cannyLow)) }))}
              ${this.numberControl('High threshold', cfg.preprocessorParams.cannyHigh, 0, 255, 1, 0, (v) => upP({ cannyHigh: Math.round(this.clamp(v, 0, 255, cfg.preprocessorParams.cannyHigh)) }))}
            </div>
          ` : ''}
          ${cfg.model === 'openpose' ? html`
            <label class="check-field">
              <input type="checkbox" .checked=${cfg.preprocessorParams.openposeHands} @change=${(e: Event) => upP({ openposeHands: (e.target as HTMLInputElement).checked })} />
              <span>Detect hands</span>
            </label>
            <label class="check-field">
              <input type="checkbox" .checked=${cfg.preprocessorParams.openposeFace} @change=${(e: Event) => upP({ openposeFace: (e.target as HTMLInputElement).checked })} />
              <span>Detect face</span>
            </label>
          ` : ''}
          ${cfg.model === 'mlsd' ? html`
            <div class="resolution-grid">
              ${this.numberControl('thr_v', cfg.preprocessorParams.mlsdThrV, 0, 10, 0.01, 2, (v) => upP({ mlsdThrV: this.clamp(v, 0, 10, cfg.preprocessorParams.mlsdThrV) }))}
              ${this.numberControl('thr_d', cfg.preprocessorParams.mlsdThrD, 0, 10, 0.01, 2, (v) => upP({ mlsdThrD: this.clamp(v, 0, 10, cfg.preprocessorParams.mlsdThrD) }))}
            </div>
          ` : ''}
          ${cfg.model === 'lineart' ? html`
            <label class="check-field">
              <input type="checkbox" .checked=${cfg.preprocessorParams.linartCoarse} @change=${(e: Event) => upP({ linartCoarse: (e.target as HTMLInputElement).checked })} />
              <span>Coarse lineart</span>
            </label>
          ` : ''}
        ` : ''}
      `}
    </div>`
  }

  private setGeneratorActive(active: boolean) {
    this.generatorActive = active
    if (!active) return
    const layer = this.selectedLayer()
    const selectedVariation = layer ? this.generatedLayers.find((variation) => variation.layerId === layer.id && variation.selected) : undefined
    if (selectedVariation) this.placeGeneratedLayer(selectedVariation)
  }

  private renderRegionProperties() {
    const selection = this.normalizedSelection()
    return html`<div class="layer-tab">
      <h2>Mask</h2>
      <span>${selection.width} x ${selection.height}</span>
      ${this.renderSelectionTransform(selection)}
      <button @click=${this.cropToSelection}>Crop to mask</button>
      <button @click=${this.deleteSelection}>Delete pixels in mask</button>
      <button @click=${this.invertSelection}>Invert mask</button>
      <button @click=${this.clearSelection}>Clear mask</button>
    </div>`
  }

  private renderSelectionOverlay() {
    if (!this.selectionRect) return ''
    if (this.selectionRect.points?.length) {
      const points = this.selectionRect.points.map((point) => `${(point.x / this.stageWidth) * 100},${(point.y / this.stageHeight) * 100}`).join(' ')
      return html`<svg class="selection-path" viewBox="0 0 100 100" preserveAspectRatio="none">
        <polygon points=${points} vector-effect="non-scaling-stroke"></polygon>
      </svg>`
    }
    const { x, y, width, height, inverted } = this.normalizedSelection()
    return html`<div
      class=${inverted ? 'selection-box inverted' : 'selection-box'}
      style="left: ${(x / this.stageWidth) * 100}%; top: ${(y / this.stageHeight) * 100}%; width: ${(width / this.stageWidth) * 100}%; height: ${(height / this.stageHeight) * 100}%;"
    ></div>`
  }

  private numberControl(
    label: string,
    value: number,
    min: number,
    max: number,
    step: number,
    decimals: number,
    onChange: (value: number) => void,
    compact = false,
  ) {
    const formatted = decimals > 0 ? value.toFixed(decimals) : String(Math.round(value))
    const dragStep = step
    return html`<div class=${compact ? 'number-field embedded' : 'number-field'}>
      <div
        class="number-control"
        title="Drag left/right to adjust"
        @wheel=${(event: WheelEvent) => this.wheelNumber(event, value, min, max, dragStep, onChange)}
        @pointerdown=${(event: PointerEvent) => this.beginNumberDrag(event, value, min, max, dragStep, onChange)}
        @pointermove=${this.dragNumber}
        @pointerup=${this.endNumberDrag}
        @pointercancel=${this.endNumberDrag}
      >
        <button type="button" class="scrub-arrow" title="Decrease" @click=${(event: MouseEvent) => this.nudgeNumber(event, value, min, max, -step, onChange)}>&lt;</button>
        <span class="scrub-name">${label}</span>
        <input
          type="number"
          min=${String(min)}
          max=${String(max)}
          step=${step}
          .value=${formatted}
          @keydown=${(event: KeyboardEvent) => event.stopPropagation()}
          @input=${(event: Event) => onChange(Number((event.target as HTMLInputElement).value))}
        />
        <button type="button" class="scrub-arrow" title="Increase" @click=${(event: MouseEvent) => this.nudgeNumber(event, value, min, max, step, onChange)}>&gt;</button>
      </div>
    </div>`
  }

  private aggregateOperatorControl(
    label: string,
    value: ConditionAggregateOperator,
    onChange: (value: ConditionAggregateOperator) => void,
  ) {
    return html`<label class="field inline">
      <span>${label}</span>
      <select .value=${value} @change=${(event: Event) => onChange((event.target as HTMLSelectElement).value as ConditionAggregateOperator)}>
        <option value="replace">Replace</option>
        <option value="add">Add</option>
        <option value="average">Average</option>
        <option value="multiply">Multiply</option>
        <option value="max">Max</option>
      </select>
    </label>`
  }

  private beginNumberDrag(
    event: PointerEvent,
    value: number,
    min: number,
    max: number,
    step: number,
    onChange: (value: number) => void,
  ) {
    if (event.button !== 0) return
    const control = event.currentTarget as HTMLElement
    control.querySelector('input')?.focus()
    const target = event.target as HTMLElement
    if (target instanceof HTMLButtonElement) return
    if (target instanceof HTMLInputElement) return
    event.preventDefault()
    this.numberDrag = { pointerId: event.pointerId, startX: event.clientX, startValue: value, min, max, step, onChange }
    control.setPointerCapture?.(event.pointerId)
  }

  private dragNumber = (event: PointerEvent) => {
    if (!this.numberDrag || event.pointerId !== this.numberDrag.pointerId) return
    const multiplier = event.shiftKey ? 5 : event.altKey ? 0.1 : 1
    const delta = (event.clientX - this.numberDrag.startX) * this.numberDrag.step * multiplier
    const value = this.roundToStep(this.numberDrag.startValue + delta, this.numberDrag.step)
    this.numberDrag.onChange(this.clamp(value, this.numberDrag.min, this.numberDrag.max, this.numberDrag.startValue))
  }

  private endNumberDrag = (event: PointerEvent) => {
    if (!this.numberDrag || event.pointerId !== this.numberDrag.pointerId) return
    ;(event.currentTarget as HTMLElement).releasePointerCapture?.(event.pointerId)
    this.numberDrag = undefined
  }

  private wheelNumber(
    event: WheelEvent,
    value: number,
    min: number,
    max: number,
    step: number,
    onChange: (value: number) => void,
  ) {
    event.preventDefault()
    this.nudgeNumber(event, value, min, max, event.deltaY > 0 ? -step : step, onChange)
  }

  private nudgeNumber(
    event: MouseEvent | WheelEvent,
    value: number,
    min: number,
    max: number,
    delta: number,
    onChange: (value: number) => void,
  ) {
    const multiplier = event.shiftKey ? 10 : event.altKey ? 0.1 : 1
    onChange(this.clamp(this.roundToStep(value + delta * multiplier, Math.abs(delta)), min, max, value))
  }

  private roundToStep(value: number, step: number) {
    const decimals = Math.max(0, Math.ceil(-Math.log10(step)))
    const quantized = Math.round(value / step) * step
    return Number(quantized.toFixed(decimals + 1))
  }

  private setupCanvas() {
    this.fabricCanvas = new Canvas(this.editCanvasElement, {
      width: this.stageWidth,
      height: this.stageHeight,
      selection: true,
      fireRightClick: true,
      stopContextMenu: true,
      selectionColor: 'rgba(47, 155, 255, 0.14)',
      selectionBorderColor: '#00e5ff',
      selectionLineWidth: 2,
      selectionDashArray: [7, 4],
    })
    const syncIfReady = () => {
      if (!this.isRestoringHistory && !this.isSyncingLayers && !this.pendingPaintLayerId) this.syncLayers()
    }
    this.fabricCanvas.on('selection:created', syncIfReady)
    this.fabricCanvas.on('selection:updated', syncIfReady)
    this.fabricCanvas.on('selection:cleared', syncIfReady)
    this.fabricCanvas.on('object:added', syncIfReady)
    this.fabricCanvas.on('object:removed', syncIfReady)
    this.fabricCanvas.on('object:modified', syncIfReady)
    this.fabricCanvas.on('before:transform', () => this.pushHistory())
    this.fabricCanvas.on('mouse:down', (event) => {
      if (this._isShapeTool()) {
        if (this.editorDest !== 'paint' || !this.selectedLayerId) return
        const pt = this.fabricCanvas!.getScenePoint(event.e)
        if (this.toolMode === 'fill') {
          this.pushHistory()
          this.executeFillTool(pt.x, pt.y)
        } else {
          this.pushHistory()
          this._shapeDrawState = { startX: pt.x, startY: pt.y }
        }
        return
      }
      if (!this.fabricCanvas?.isDrawingMode) return
      this.pushHistory()
      this.pendingPaintLayerId = this.selectedLayerId
      this.pendingBrushButton = this.pointerButton(event.e)
      this.configureFabricBrush(this.pendingBrushButton)
    })
    this.fabricCanvas.on('mouse:up', (event) => {
      if (this._isShapeTool() && this._shapeDrawState) {
        const pt = this.fabricCanvas!.getScenePoint(event.e)
        this.clearEditorOverlay()
        this.finalizeShape(this._shapeDrawState.startX, this._shapeDrawState.startY, pt.x, pt.y)
        this._shapeDrawState = null
        return
      }
      this.pendingBrushButton = 0
      if (!this.rightPaintStroke) window.setTimeout(() => (this.pendingPaintLayerId = ''), 0)
      this.configureFabricBrush(0)
      this.queueLiveStreamFrame()
    })
    this.fabricCanvas.on('mouse:move', (event) => {
      if (this._isShapeTool() && this._shapeDrawState) {
        const pt = this.fabricCanvas!.getScenePoint(event.e)
        this.drawShapePreview(this._shapeDrawState.startX, this._shapeDrawState.startY, pt.x, pt.y)
        return
      }
      if (this.fabricCanvas?.isDrawingMode && this.pendingPaintLayerId) this.queueLiveStreamFrame()
    })
    this.fabricCanvas.on('path:created', (event) => {
      this.attachStrokeToLayer(event.path as FabricObject, this.pendingPaintLayerId, this.isEraseStroke(this.pendingBrushButton))
      // Synchronous re-render so the destination-out path is committed to the lower
      // canvas before the next RAF-queued frame export fires.
      this.fabricCanvas?.renderAll()
      this.queueLiveStreamFrame()
    })
    const brush = new PencilBrush(this.fabricCanvas)
    brush.color = this.brushColor
    brush.width = this.brushSize
    this.fabricCanvas.freeDrawingBrush = brush
    this.attachRightPaintHandlers()
    this.setTool(this.toolMode)
  }

  private attachRightPaintHandlers() {
    const upperCanvas = (this.fabricCanvas as (Canvas & { upperCanvasEl?: HTMLCanvasElement }) | undefined)?.upperCanvasEl
    if (!upperCanvas || this.upperCanvasElement === upperCanvas) return
    this.detachRightPaintHandlers()
    this.upperCanvasElement = upperCanvas
    upperCanvas.addEventListener('pointerdown', this.startRightPaintStroke, true)
    upperCanvas.addEventListener('pointermove', this.moveRightPaintStroke, true)
    upperCanvas.addEventListener('pointerup', this.endRightPaintStroke, true)
    upperCanvas.addEventListener('pointercancel', this.cancelRightPaintStroke, true)
    upperCanvas.addEventListener('contextmenu', this.preventPaintContextMenu, true)
  }

  private detachRightPaintHandlers() {
    if (!this.upperCanvasElement) return
    this.upperCanvasElement.removeEventListener('pointerdown', this.startRightPaintStroke, true)
    this.upperCanvasElement.removeEventListener('pointermove', this.moveRightPaintStroke, true)
    this.upperCanvasElement.removeEventListener('pointerup', this.endRightPaintStroke, true)
    this.upperCanvasElement.removeEventListener('pointercancel', this.cancelRightPaintStroke, true)
    this.upperCanvasElement.removeEventListener('contextmenu', this.preventPaintContextMenu, true)
    this.upperCanvasElement = undefined
  }

  private setupMaskCanvas() {
    this.maskCanvasElement.width = this.stageWidth
    this.maskCanvasElement.height = this.stageHeight
    this.maskContext = this.maskCanvasElement.getContext('2d') ?? undefined
    this.maskPreviewCanvasElement.width = this.stageWidth
    this.maskPreviewCanvasElement.height = this.stageHeight
    this.maskPreviewContext = this.maskPreviewCanvasElement.getContext('2d') ?? undefined
  }

  private seedCanvas() {
    this.fabricCanvas?.clear()
    this.fabricCanvas?.requestRenderAll()
    this.syncLayers()
  }

  private async loadAssets() {
    try {
      const [assetsResponse, healthResponse, cnModelsResponse] = await Promise.all([
        fetch(this.backendUrl('/assets')),
        fetch(this.backendUrl('/health')),
        fetch(this.backendUrl('/controlnet/models')),
      ])
      this.assets = (await assetsResponse.json()) as AssetCatalog
      this.assets.video_models = this.assets.video_models ?? []
      const health = (await healthResponse.json()) as HealthState
      this.backendMode = health.mode
      const storedModel = this.assets.models.find((model) => model.path === this.selectedModel)?.path
      this.selectedModel = storedModel || this.assets.models.find((model) => model.preferred)?.path || this.assets.models[0]?.path || ''
      const videoModels = this.videoModelAssets()
      const storedVideoModel = videoModels.find((model) => model.path === this.fastVideoModelPath)?.path
      this.fastVideoModelPath = storedVideoModel || ''
      this.applySamplerPresetToScene(this.selectedModel)
      this.applySamplerPresetToLayer(this.selectedModel)
      try {
        this.cnLocalModels = (await cnModelsResponse.json()) as typeof this.cnLocalModels
      } catch { /* non-fatal */ }
    } catch {
      this.assets = { models: [], video_models: [], loras: [] }
    }
  }

  private setTool(mode: ToolMode) {
    this.toolMode = mode
    if (mode === 'text') this.addTextLayer()
    this.applyToolMode()
    this.clearEditorOverlay()
  }

  private setEditorDest(dest: EditorDest) {
    this.editorDest = dest
    this.applyToolMode()
    this.syncMaskScope()
    this.scheduleMaskPreviewRefresh(true)
  }

  private applyToolMode() {
    if (!this.fabricCanvas) return
    const selectedLayer = this.selectedLayerId ? this.findLayer(this.selectedLayerId) : undefined
    const activeObject = this.fabricCanvas.getActiveObject()
    const activeIsPaintChild = activeObject ? isPaintChildObject(activeObject as FabricObject & { __paintChild?: boolean; __parentLayerId?: string }) : false
    if (this.editorDest === 'mask' && activeObject) this.fabricCanvas.discardActiveObject()
    if (this.editorDest === 'paint' && selectedLayer && activeObject !== selectedLayer && !activeIsPaintChild) {
      this.fabricCanvas.setActiveObject(selectedLayer)
    }
    this.fabricCanvas.isDrawingMode =
      this.editorDest === 'paint' && (this.toolMode === 'brush' || this.toolMode === 'eraser') && this.selectedEntity === 'layer' && !!this.selectedLayerId
    this.fabricCanvas.selection = this.toolMode === 'select'
    this.configureFabricBrush(0)
  }

  private maskPreviewVisible() {
    return this.inputMaskVisible && this.currentMaskScope() !== undefined
  }

  private maskCanvasEditable() {
    return this.editorDest === 'mask' && this.currentMaskScope() !== undefined && (this.toolMode === 'brush' || this.toolMode === 'eraser')
  }

  private maskCanvasActive() {
    return this.inputMaskVisible && this.editorDest === 'mask' && (this.toolMode === 'region' || this.maskCanvasEditable())
  }

  private currentMaskScope(): MaskScope | undefined {
    if (this.selectedEntity === 'scene') return 'scene'
    if (this.selectedEntity === 'layer' && this.selectedLayerId) return `layer:${this.selectedLayerId}`
    return undefined
  }

  private pointerButton(event: Event) {
    const pointer = event as PointerEvent | MouseEvent
    return pointer.button === 2 ? 2 : 0
  }

  private isEraseStroke(button = this.pendingBrushButton) {
    return this.toolMode === 'eraser' || button === 2
  }

  private configureFabricBrush(button = 0) {
    if (!this.fabricCanvas?.freeDrawingBrush) return
    const color = button === 2 ? this.secondaryBrushColor : this.brushColor
    const fillColor = this.isEraseStroke(button) ? '#000000' : color
    this.fabricCanvas.freeDrawingBrush.color = fillColor
    this.fabricCanvas.freeDrawingBrush.width = this.brushSize
    // Soft brush: shadow with same color creates a blurred halo that softens edges
    if (this.brushHardness < 100) {
      const blur = ((100 - this.brushHardness) / 100) * this.brushSize * 0.8
      this.fabricCanvas.freeDrawingBrush.shadow = new Shadow({ color: fillColor, blur, offsetX: 0, offsetY: 0 })
    } else {
      this.fabricCanvas.freeDrawingBrush.shadow = null
    }
  }

  private _isShapeTool() {
    return this.toolMode === 'line' || this.toolMode === 'rect' || this.toolMode === 'fill'
  }

  private drawShapePreview(x1: number, y1: number, x2: number, y2: number) {
    this.drawEditorOverlay((ctx) => {
      ctx.strokeStyle = this.brushColor
      ctx.fillStyle = this.brushColor
      ctx.lineWidth = this.brushSize
      ctx.lineCap = 'round'
      if (this.toolMode === 'line') {
        ctx.beginPath()
        ctx.moveTo(x1, y1)
        ctx.lineTo(x2, y2)
        ctx.stroke()
      } else if (this.toolMode === 'rect') {
        const rx = Math.min(x1, x2), ry = Math.min(y1, y2)
        const rw = Math.abs(x2 - x1), rh = Math.abs(y2 - y1)
        ctx.strokeRect(rx, ry, rw, rh)
      }
    })
  }

  private finalizeShape(x1: number, y1: number, x2: number, y2: number) {
    if (!this.fabricCanvas || !this.selectedLayerId) return
    let obj: FabricObject
    const color = this.brushColor
    const w = this.brushSize
    if (this.toolMode === 'line') {
      obj = new Path(`M ${x1} ${y1} L ${x2} ${y2}`, {
        stroke: color, fill: '', strokeWidth: w, strokeLineCap: 'round',
        selectable: false, evented: false, objectCaching: false,
      }) as unknown as FabricObject
    } else {
      const rx = Math.min(x1, x2), ry = Math.min(y1, y2)
      const rw = Math.abs(x2 - x1), rh = Math.abs(y2 - y1)
      obj = new Rect({
        left: rx, top: ry, width: rw, height: rh,
        fill: color, stroke: '', strokeWidth: 0,
        selectable: false, evented: false, objectCaching: false,
      }) as unknown as FabricObject
    }
    this.fabricCanvas.add(obj)
    this.attachStrokeToLayer(obj, this.selectedLayerId, false)
    this.fabricCanvas.renderAll()
    this.queueLiveStreamFrame()
  }

  private executeFillTool(clickX: number, clickY: number) {
    if (!this.fabricCanvas || !this.selectedLayerId) return
    const fc = this.fabricCanvas as Canvas & { lowerCanvasEl: HTMLCanvasElement }
    const ctx = fc.lowerCanvasEl.getContext('2d', { willReadFrequently: true })
    if (!ctx) return
    const cw = fc.lowerCanvasEl.width, ch = fc.lowerCanvasEl.height
    const sx = cw / this.stageWidth, sy = ch / this.stageHeight
    const px = Math.round(clickX * sx), py = Math.round(clickY * sy)
    const imageData = ctx.getImageData(0, 0, cw, ch)
    const d = imageData.data
    const tidx = (py * cw + px) * 4
    const tr = d[tidx], tg = d[tidx + 1], tb = d[tidx + 2], ta = d[tidx + 3]
    const fill = this._parseHexColor(this.brushColor)
    if (!fill) return
    const tol = 32
    const same = (i: number) =>
      Math.abs(d[i] - tr) <= tol && Math.abs(d[i + 1] - tg) <= tol &&
      Math.abs(d[i + 2] - tb) <= tol && Math.abs(d[i + 3] - ta) <= tol
    const out = new Uint8ClampedArray(cw * ch * 4)
    const vis = new Uint8Array(cw * ch)
    const stack = [py * cw + px]
    vis[py * cw + px] = 1
    while (stack.length) {
      const pos = stack.pop()!
      const x = pos % cw, y = (pos / cw) | 0
      const i = pos * 4
      out[i] = fill.r; out[i + 1] = fill.g; out[i + 2] = fill.b; out[i + 3] = 255
      for (const [dx, dy] of [[-1, 0], [1, 0], [0, -1], [0, 1]] as [number, number][]) {
        const nx = x + dx, ny = y + dy
        if (nx < 0 || nx >= cw || ny < 0 || ny >= ch) continue
        const ni = ny * cw + nx
        if (vis[ni] || !same(ni * 4)) continue
        vis[ni] = 1; stack.push(ni)
      }
    }
    const tmp = document.createElement('canvas')
    tmp.width = cw; tmp.height = ch
    tmp.getContext('2d')!.putImageData(new ImageData(out, cw, ch), 0, 0)
    const img = new FabricImage(tmp, {
      left: 0, top: 0, scaleX: this.stageWidth / cw, scaleY: this.stageHeight / ch,
      selectable: false, evented: false, objectCaching: false,
    }) as unknown as FabricObject
    this.fabricCanvas.add(img)
    this.attachStrokeToLayer(img, this.selectedLayerId, false)
    this.fabricCanvas.renderAll()
    this.queueLiveStreamFrame()
  }

  private _parseHexColor(hex: string) {
    const m = hex.match(/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i)
    if (!m) return null
    return { r: parseInt(m[1], 16), g: parseInt(m[2], 16), b: parseInt(m[3], 16) }
  }

  private canRightPaint() {
    return this.editorDest === 'paint' && !!this.fabricCanvas?.isDrawingMode && (this.toolMode === 'brush' || this.toolMode === 'eraser') && this.selectedEntity === 'layer' && !!this.selectedLayerId
  }

  private preventPaintContextMenu = (event: MouseEvent) => {
    if (!this.canRightPaint()) return
    event.preventDefault()
    event.stopImmediatePropagation()
  }

  private startRightPaintStroke = (event: PointerEvent) => {
    if (event.button !== 2 || !this.canRightPaint() || !this.upperCanvasElement) return
    event.preventDefault()
    event.stopImmediatePropagation()
    this.pushHistory()
    this.pendingPaintLayerId = this.selectedLayerId
    this.pendingBrushButton = 2
    this.configureFabricBrush(2)
    const point = this.canvasPointFromPointer(event)
    this.rightPaintStroke = { pointerId: event.pointerId, points: [point] }
    this.upperCanvasElement.setPointerCapture(event.pointerId)
    this.renderRightPaintPreview()
  }

  private moveRightPaintStroke = (event: PointerEvent) => {
    if (!this.rightPaintStroke || this.rightPaintStroke.pointerId !== event.pointerId) return
    event.preventDefault()
    event.stopImmediatePropagation()
    const point = this.canvasPointFromPointer(event)
    const last = this.rightPaintStroke.points.at(-1)
    if (!last || Math.hypot(point.x - last.x, point.y - last.y) >= 1.5) {
      this.rightPaintStroke = { ...this.rightPaintStroke, points: [...this.rightPaintStroke.points, point] }
      this.renderRightPaintPreview()
      this.queueLiveStreamFrame()
    }
  }

  private endRightPaintStroke = (event: PointerEvent) => {
    if (!this.rightPaintStroke || this.rightPaintStroke.pointerId !== event.pointerId) return
    event.preventDefault()
    event.stopImmediatePropagation()
    this.finishRightPaintStroke()
  }

  private cancelRightPaintStroke = (event: PointerEvent) => {
    if (!this.rightPaintStroke || this.rightPaintStroke.pointerId !== event.pointerId) return
    event.preventDefault()
    event.stopImmediatePropagation()
    this.clearRightPaintPreview()
    this.rightPaintStroke = undefined
    this.pendingPaintLayerId = ''
    this.pendingBrushButton = 0
    this.configureFabricBrush(0)
  }

  private finishRightPaintStroke() {
    if (!this.fabricCanvas || !this.rightPaintStroke) return
    const points = this.rightPaintStroke.points
    this.clearRightPaintPreview()
    if (this.upperCanvasElement?.hasPointerCapture(this.rightPaintStroke.pointerId)) {
      this.upperCanvasElement.releasePointerCapture(this.rightPaintStroke.pointerId)
    }
    const strokePath = this.pathFromPoints(points)
    const path = new Path(strokePath, {
      fill: '',
      stroke: '#000000',
      strokeWidth: this.brushSize,
      strokeLineCap: 'round',
      strokeLineJoin: 'round',
      selectable: false,
      evented: false,
      name: 'Paint stroke',
    })
    path.set({ globalCompositeOperation: 'destination-out' } as Partial<FabricObject>)
    const parentLayerId = this.pendingPaintLayerId
    this.fabricCanvas.add(path)
    this.attachStrokeToLayer(path, parentLayerId, true)
    this.fabricCanvas.requestRenderAll()
    this.rightPaintStroke = undefined
    this.pendingPaintLayerId = ''
    this.pendingBrushButton = 0
    this.configureFabricBrush(0)
    this.queueLiveStreamFrame()
  }

  private canvasPointFromPointer(event: PointerEvent) {
    const rect = this.editCanvasElement.getBoundingClientRect()
    return {
      x: this.clamp(((event.clientX - rect.left) / rect.width) * this.stageWidth, 0, this.stageWidth, 0),
      y: this.clamp(((event.clientY - rect.top) / rect.height) * this.stageHeight, 0, this.stageHeight, 0),
    }
  }

  private pathFromPoints(points: { x: number; y: number }[]) {
    const first = points[0] ?? { x: 0, y: 0 }
    if (points.length < 2) return `M ${first.x} ${first.y} L ${first.x + 0.01} ${first.y}`
    return points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`).join(' ')
  }

  private renderRightPaintPreview() {
    if (!this.fabricCanvas || !this.rightPaintStroke) return
    const canvas = this.fabricCanvas as Canvas & { contextTop: CanvasRenderingContext2D; clearContext: (context: CanvasRenderingContext2D) => void }
    const context = canvas.contextTop
    canvas.clearContext(context)
    const points = this.rightPaintStroke.points
    const first = points[0]
    if (!first) return
    context.save()
    context.strokeStyle = '#000000'
    context.lineWidth = this.brushSize
    context.lineCap = 'round'
    context.lineJoin = 'round'
    context.beginPath()
    context.moveTo(first.x, first.y)
    for (const point of points.slice(1)) context.lineTo(point.x, point.y)
    if (points.length === 1) context.lineTo(first.x + 0.01, first.y)
    context.stroke()
    context.restore()
  }

  private clearRightPaintPreview() {
    const canvas = this.fabricCanvas as (Canvas & { contextTop?: CanvasRenderingContext2D; clearContext?: (context: CanvasRenderingContext2D) => void }) | undefined
    if (canvas?.contextTop && canvas.clearContext) canvas.clearContext(canvas.contextTop)
  }

  private setBrushColor(color: string) {
    this.brushColor = this.normalizeColor(color) ?? this.brushColor
    this.configureFabricBrush(this.pendingBrushButton)
    if (this.inputActiveMaskOnly) this.scheduleMaskPreviewRefresh(true)
  }

  private setSecondaryBrushColor(color: string) {
    this.secondaryBrushColor = this.normalizeColor(color) ?? this.secondaryBrushColor
    this.configureFabricBrush(this.pendingBrushButton)
  }

  private activeColor() {
    return this.activeColorSlot === 'primary' ? this.brushColor : this.secondaryBrushColor
  }

  private colorInputValue() {
    return this.rgbToHex(this.activeRgba())
  }

  private colorTextValue() {
    const color = this.activeRgba()
    return color.a < 1 ? this.rgbaToHex8(color) : this.rgbToHex(color)
  }

  private colorIsActive(color: string) {
    const active = this.activeRgba()
    const candidate = this.parseColor(color)
    return !!candidate && active.r === candidate.r && active.g === candidate.g && active.b === candidate.b && Math.round(active.a * 100) === Math.round(candidate.a * 100)
  }

  private setActiveColor(color: string) {
    const nextColor = this.normalizeColor(color)
    if (!nextColor) return
    if (this.activeColorSlot === 'primary') this.setBrushColor(nextColor)
    else this.setSecondaryBrushColor(nextColor)
  }

  private useLayerRegionColor(layerId: string) {
    this.activeColorSlot = 'primary'
    this.setBrushColor(this.layerRegionColor(layerId))
    this.setSecondaryBrushColor('rgba(0, 0, 0, 0)')
    this.setTool('brush')
  }

  private useRegionColor(color: string) {
    this.activeColorSlot = 'primary'
    this.setBrushColor(color)
    this.setSecondaryBrushColor('rgba(0, 0, 0, 0)')
    this.setTool('brush')
  }

  private setSecondaryFromPalette(event: Event, color: string) {
    event.preventDefault()
    this.setSecondaryBrushColor(color)
  }

  private swapColors = () => {
    const primary = this.brushColor
    this.brushColor = this.secondaryBrushColor
    this.secondaryBrushColor = primary
    this.configureFabricBrush(this.pendingBrushButton)
    if (this.inputActiveMaskOnly) this.scheduleMaskPreviewRefresh(true)
  }

  private pickHexColor = (event: Event) => {
    const color = this.normalizeColor((event.target as HTMLInputElement).value)
    if (color) this.setActiveColor(color)
  }

  private activeRgb() {
    const { r, g, b } = this.activeRgba()
    return { r, g, b }
  }

  private activeRgba() {
    return this.parseColor(this.activeColor()) ?? { r: 255, g: 255, b: 255, a: 1 }
  }

  private activeHsl() {
    return this.rgbToHsl(this.activeRgb())
  }

  private setActiveRgba(patch: Partial<RgbaColor>) {
    const next = { ...this.activeRgba(), ...patch }
    this.setActiveColor(this.rgbaToCss(next))
  }

  private setActiveHsl(patch: Partial<{ h: number; s: number; l: number }>) {
    const hsl = { ...this.activeHsl(), ...patch }
    this.setActiveColor(this.withActiveAlpha(this.hslToHex(hsl.h, hsl.s, hsl.l)))
  }

  private withActiveAlpha(color: string) {
    const parsed = this.parseColor(color)
    if (!parsed) return color
    return this.rgbaToCss({ ...parsed, a: this.activeRgba().a })
  }

  private plannedColors() {
    const hsl = this.activeHsl()
    const wheel = (offset: number, saturation = hsl.s, lightness = hsl.l) => this.withActiveAlpha(this.hslToHex(hsl.h + offset, saturation, lightness))
    if (this.colorPlannerMode === 'analogous') return [-36, -18, 0, 18, 36, 54].map((offset) => wheel(offset))
    if (this.colorPlannerMode === 'triad') return [0, 120, 240, 120, 240, 0].map((offset, index) => wheel(offset, hsl.s, index > 2 ? Math.min(92, hsl.l + 18) : hsl.l))
    if (this.colorPlannerMode === 'tetrad') return [0, 60, 180, 240, 30, 210].map((offset) => wheel(offset))
    if (this.colorPlannerMode === 'split') return [0, 150, 210, 30, 180, 330].map((offset) => wheel(offset))
    if (this.colorPlannerMode === 'monochrome') return [18, 30, 42, 54, 66, 78].map((lightness) => wheel(0, hsl.s, lightness))
    return [0, 180, 150, 210, 0, 180].map((offset, index) => wheel(offset, hsl.s, index > 3 ? Math.min(90, hsl.l + 18) : hsl.l))
  }

  private blendColors(startColor: string, endColor: string, count: number) {
    const start = this.parseColor(startColor) ?? { r: 255, g: 255, b: 255, a: 1 }
    const end = this.parseColor(endColor) ?? { r: 0, g: 0, b: 0, a: 1 }
    return Array.from({ length: count }, (_, index) => {
      const mix = count === 1 ? 0 : index / (count - 1)
      return this.rgbaToCss({
        r: Math.round(start.r + (end.r - start.r) * mix),
        g: Math.round(start.g + (end.g - start.g) * mix),
        b: Math.round(start.b + (end.b - start.b) * mix),
        a: start.a + (end.a - start.a) * mix,
      })
    })
  }

  private toneRamp(color: string, count: number) {
    const parsed = this.parseColor(color) ?? { r: 255, g: 255, b: 255, a: 1 }
    const hsl = this.rgbToHsl(parsed)
    return Array.from({ length: count }, (_, index) => this.rgbaToCss({ ...this.parseColor(this.hslToHex(hsl.h, hsl.s, 12 + index * (76 / Math.max(1, count - 1))))!, a: parsed.a }))
  }

  private normalizeColor(color: string) {
    const parsed = this.parseColor(color)
    return parsed ? this.rgbaToCss(parsed) : undefined
  }

  private parseColor(color: string): RgbaColor | undefined {
    const trimmed = color.trim()
    const short = /^#?([0-9a-f]{3})$/i.exec(trimmed)
    if (short) {
      const expanded = short[1].split('').map((value) => value + value).join('')
      return this.hexChannels(expanded, 'ff')
    }
    const shortAlpha = /^#?([0-9a-f]{4})$/i.exec(trimmed)
    if (shortAlpha) {
      const expanded = shortAlpha[1].split('').map((value) => value + value).join('')
      return this.hexChannels(expanded.slice(0, 6), expanded.slice(6, 8))
    }
    const full = /^#?([0-9a-f]{6})$/i.exec(trimmed)
    if (full) return this.hexChannels(full[1], 'ff')
    const fullAlpha = /^#?([0-9a-f]{8})$/i.exec(trimmed)
    if (fullAlpha) return this.hexChannels(fullAlpha[1].slice(0, 6), fullAlpha[1].slice(6, 8))
    const rgba = /^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*([\d.]+))?\s*\)$/i.exec(trimmed)
    if (!rgba) return undefined
    return {
      r: Math.round(this.clamp(Number(rgba[1]), 0, 255, 0)),
      g: Math.round(this.clamp(Number(rgba[2]), 0, 255, 0)),
      b: Math.round(this.clamp(Number(rgba[3]), 0, 255, 0)),
      a: this.clamp(rgba[4] === undefined ? 1 : Number(rgba[4]), 0, 1, 1),
    }
  }

  private hexChannels(rgb: string, alpha: string): RgbaColor {
    return {
      r: Number.parseInt(rgb.slice(0, 2), 16),
      g: Number.parseInt(rgb.slice(2, 4), 16),
      b: Number.parseInt(rgb.slice(4, 6), 16),
      a: Number.parseInt(alpha, 16) / 255,
    }
  }

  private rgbToHex(rgb: { r: number; g: number; b: number }) {
    const channel = (value: number) => Math.round(this.clamp(value, 0, 255, 0)).toString(16).padStart(2, '0')
    return `#${channel(rgb.r)}${channel(rgb.g)}${channel(rgb.b)}`
  }

  private rgbaToHex8(color: RgbaColor) {
    const alpha = Math.round(this.clamp(color.a, 0, 1, 1) * 255).toString(16).padStart(2, '0')
    return `${this.rgbToHex(color)}${alpha}`
  }

  private rgbaToCss(color: RgbaColor) {
    const normalized = {
      r: Math.round(this.clamp(color.r, 0, 255, 0)),
      g: Math.round(this.clamp(color.g, 0, 255, 0)),
      b: Math.round(this.clamp(color.b, 0, 255, 0)),
      a: this.clamp(color.a, 0, 1, 1),
    }
    return normalized.a >= 0.995 ? this.rgbToHex(normalized) : `rgba(${normalized.r}, ${normalized.g}, ${normalized.b}, ${Number(normalized.a.toFixed(3))})`
  }

  private sameRgb(first: string, second: string) {
    const firstColor = this.parseColor(first)
    const secondColor = this.parseColor(second)
    return !!firstColor && !!secondColor && firstColor.r === secondColor.r && firstColor.g === secondColor.g && firstColor.b === secondColor.b
  }

  private rgbToHsl(rgb: { r: number; g: number; b: number }) {
    const red = rgb.r / 255
    const green = rgb.g / 255
    const blue = rgb.b / 255
    const max = Math.max(red, green, blue)
    const min = Math.min(red, green, blue)
    const lightness = (max + min) / 2
    if (max === min) return { h: 0, s: 0, l: Math.round(lightness * 100) }
    const delta = max - min
    const saturation = lightness > 0.5 ? delta / (2 - max - min) : delta / (max + min)
    let hue = 0
    if (max === red) hue = (green - blue) / delta + (green < blue ? 6 : 0)
    if (max === green) hue = (blue - red) / delta + 2
    if (max === blue) hue = (red - green) / delta + 4
    return { h: Math.round(hue * 60), s: Math.round(saturation * 100), l: Math.round(lightness * 100) }
  }

  private hslToHex(hue: number, saturation: number, lightness: number) {
    const normalizedHue = ((hue % 360) + 360) % 360
    const sat = this.clamp(saturation, 0, 100, 0) / 100
    const light = this.clamp(lightness, 0, 100, 50) / 100
    const chroma = (1 - Math.abs(2 * light - 1)) * sat
    const x = chroma * (1 - Math.abs(((normalizedHue / 60) % 2) - 1))
    const match = light - chroma / 2
    const [red, green, blue] = normalizedHue < 60
      ? [chroma, x, 0]
      : normalizedHue < 120
        ? [x, chroma, 0]
        : normalizedHue < 180
          ? [0, chroma, x]
          : normalizedHue < 240
            ? [0, x, chroma]
            : normalizedHue < 300
              ? [x, 0, chroma]
              : [chroma, 0, x]
    return this.rgbToHex({ r: (red + match) * 255, g: (green + match) * 255, b: (blue + match) * 255 })
  }

  private setBrushSize(value: number) {
    this.brushSize = Math.round(this.clamp(value, 1, 4096, this.brushSize))
    this.configureFabricBrush(this.pendingBrushButton)
  }

  private setBrushHardness(value: number) {
    this.brushHardness = Math.round(this.clamp(value, 0, 100, this.brushHardness))
    this.configureFabricBrush(this.pendingBrushButton)
  }

  private setGenerationWidth(value: number) {
    this.generationWidth = this.alignGenerationSize(value)
  }

  private setGenerationHeight(value: number) {
    this.generationHeight = this.alignGenerationSize(value)
  }

  private useViewportResolution = () => {
    this.generationWidth = this.alignGenerationSize(this.stageWidth)
    this.generationHeight = this.alignGenerationSize(this.stageHeight)
  }

  private clamp(value: number, min: number, max: number, fallback: number) {
    if (!Number.isFinite(value)) return fallback
    return Math.max(min, Math.min(max, value))
  }

  private normalizeAggregateOperator(value: unknown, fallback: ConditionAggregateOperator): ConditionAggregateOperator {
    return ['replace', 'add', 'average', 'multiply', 'max'].includes(String(value)) ? value as ConditionAggregateOperator : fallback
  }

  private pickColor = (event: Event) => {
    this.setActiveColor(this.withActiveAlpha((event.target as HTMLInputElement).value))
  }

  private currentPreset(): LayerPreset {
    return {
      prompt: this.prompt,
      negativePrompt: this.negativePrompt,
      modelPath: this.selectedModel,
      loraPaths: [...this.selectedLoras],
      samplingEnabled: true,
      conditionMode: 'mask',
      conditionWeight: 1,
      maskOperator: 'add',
      denoiseOperator: 'replace',
      conditionInnerBlur: 0,
      conditionOuterBlur: 0,
      conditionNegate: false,
      strength: this.strength,
      cfg: this.layerGenerationCfg,
      steps: this.layerGenerationSteps,
      sampler: this.layerGenerationSampler,
      scheduler: this.layerGenerationScheduler,
      schedule: 'auto',
      scheduleStart: 0,
      scheduleEnd: 1,
      autoTag: false,
      autoTagThreshold: 0.35,
      autoTagRefreshFrames: 5,
    }
  }

  private normalizeLayerPreset(preset: Partial<LayerPreset> = {}): LayerPreset {
    return {
      prompt: preset.prompt ?? this.prompt,
      negativePrompt: preset.negativePrompt ?? this.negativePrompt,
      modelPath: preset.modelPath ?? this.selectedModel,
      loraPaths: [...(preset.loraPaths ?? this.selectedLoras)],
      samplingEnabled: preset.samplingEnabled ?? true,
      conditionMode: preset.conditionMode ?? 'mask',
      conditionWeight: this.clamp(preset.conditionWeight ?? 1, 0, 4, 1),
      maskOperator: this.normalizeAggregateOperator(preset.maskOperator, 'add'),
      denoiseOperator: this.normalizeAggregateOperator(preset.denoiseOperator, 'replace'),
      conditionInnerBlur: Math.round(this.clamp(preset.conditionInnerBlur ?? 0, 0, 160, 0)),
      conditionOuterBlur: Math.round(this.clamp(preset.conditionOuterBlur ?? 0, 0, 160, 0)),
      conditionNegate: preset.conditionNegate ?? false,
      strength: this.clamp(preset.strength ?? this.strength, 0, 0.999, this.strength),
      cfg: this.clamp(preset.cfg ?? this.layerGenerationCfg, 0, 30, this.layerGenerationCfg),
      steps: Math.round(this.clamp(preset.steps ?? this.layerGenerationSteps, 1, 40, this.layerGenerationSteps)),
      sampler: preset.sampler ?? this.layerGenerationSampler,
      scheduler: preset.scheduler ?? this.layerGenerationScheduler,
      schedule: ['auto', 'linear', 'ease_in', 'ease_out', 'ease_in_out'].includes(preset.schedule ?? '') ? (preset.schedule as RegionSchedule) : 'auto',
      scheduleStart: this.clamp(preset.scheduleStart ?? 0, 0, 1, 0),
      scheduleEnd: this.clamp(preset.scheduleEnd ?? 1, 0, 1, 1),
      autoTag: preset.autoTag ?? false,
      autoTagThreshold: this.clamp(preset.autoTagThreshold ?? 0.35, 0, 1, 0.35),
      autoTagRefreshFrames: Math.round(this.clamp(preset.autoTagRefreshFrames ?? 5, 1, 60, 5)),
    }
  }

  private presetForNewLayer(): LayerPreset {
    const objects = this.fabricCanvas?.getObjects() ?? []
    const active = this.fabricCanvas?.getActiveObject()
    const activeIndex = active ? objects.indexOf(active) : objects.length
    const source = objects[Math.max(0, activeIndex - 1)] as (FabricObject & { __preset?: LayerPreset }) | undefined
    return this.normalizeLayerPreset(source?.__preset ?? this.currentPreset())
  }

  private attachPreset(object: FabricObject, preset = this.presetForNewLayer()) {
    ;(object as FabricObject & { __preset?: LayerPreset }).__preset = this.normalizeLayerPreset(preset)
    this.styleTransformControls(object)
  }

  private styleTransformControls(object: FabricObject) {
    object.set({
      borderColor: '#00e5ff',
      cornerColor: '#f3f1ec',
      cornerStrokeColor: '#050507',
      cornerStyle: 'circle',
      cornerSize: 15,
      transparentCorners: false,
      borderScaleFactor: 2,
      padding: 2,
    } as Partial<FabricObject>)
  }

  private attachStrokeToLayer(object: FabricObject, layerId = this.pendingPaintLayerId || this.selectedLayerId, erase = this.isEraseStroke()) {
    const withMeta = object as FabricObject & { __uid?: string; __paintChild?: boolean; __parentLayerId?: string }
    const parentLayerId = strokeParentLayerId(layerId, this.selectedLayerId, withMeta.__uid)
    if (!parentLayerId) {
      this.pendingPaintLayerId = ''
      this.syncLayers()
      return
    }
    withMeta.__paintChild = true
    withMeta.__parentLayerId = parentLayerId
    if (erase) {
      // Clip the erase stroke to the parent layer's current bounding box so it doesn't bleed into other layers.
      const parentObj = this.findLayer(parentLayerId)
      if (parentObj) {
        const b = parentObj.getBoundingRect()
        const clip = new Rect({
          left: b.left + b.width / 2,
          top: b.top + b.height / 2,
          width: b.width,
          height: b.height,
          absolutePositioned: true,
        })
        object.clipPath = clip
      }
      object.set({ globalCompositeOperation: 'destination-out' } as Partial<FabricObject>)
    }
    object.set({
      name: 'Paint stroke',
      selectable: false,
      evented: false,
      hasControls: false,
      hasBorders: false,
      lockMovementX: true,
      lockMovementY: true,
      lockRotation: true,
      lockScalingX: true,
      lockScalingY: true,
    } as Partial<FabricObject>)
    this.selectedEntity = 'layer'
    this.selectedLayerId = parentLayerId
    const parent = this.findLayer(parentLayerId)
    if (parent) this.fabricCanvas?.setActiveObject(parent)
    this.pendingPaintLayerId = ''
    this.syncLayers()
  }

  private selectPreset(event: Event) {
    const [width, height] = (event.target as HTMLSelectElement).value.split('x').map(Number)
    this.pushHistory()
    this.resizeStage(width, height)
  }

  private resizeStage(width: number, height: number) {
    this.stageWidth = Math.max(256, Math.min(1536, width || this.stageWidth))
    this.stageHeight = Math.max(256, Math.min(1536, height || this.stageHeight))
    this.generationWidth = this.alignGenerationSize(this.stageWidth)
    this.generationHeight = this.alignGenerationSize(this.stageHeight)
    this.fabricCanvas?.setDimensions({ width: this.stageWidth, height: this.stageHeight })
    this.maskCanvasElement.width = this.stageWidth
    this.maskCanvasElement.height = this.stageHeight
    this.maskPreviewCanvasElement.width = this.stageWidth
    this.maskPreviewCanvasElement.height = this.stageHeight
    if (this.editorOverlayCanvas) {
      this.editorOverlayCanvas.width = this.stageWidth
      this.editorOverlayCanvas.height = this.stageHeight
    }
    this.clearMask()
    this.resizeStoredMasks()
    this.fabricCanvas?.requestRenderAll()
  }

  // ── Editor overlay — draw tool widgets (guides, handles, outlines) that
  //    are NEVER composited into the inference input canvas export.
  //    Call clearEditorOverlay() to wipe, or drawEditorOverlay(fn) to
  //    paint fresh content.  Future tools that need on-canvas UI should
  //    use these instead of adding objects to the fabric canvas.
  protected clearEditorOverlay() {
    const canvas = this.editorOverlayCanvas
    if (!canvas) return
    canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height)
  }

  protected drawEditorOverlay(fn: (ctx: CanvasRenderingContext2D, w: number, h: number) => void) {
    const canvas = this.editorOverlayCanvas
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    fn(ctx, this.stageWidth, this.stageHeight)
  }

  private startOverlayStroke(event: PointerEvent) {
    if (event.button !== 0 && event.button !== 2) return
    event.preventDefault()
    event.stopPropagation()
    if (this.toolMode === 'region') {
      if (event.button !== 0) return
      this.pushHistory()
      const point = this.maskPoint(event)
      if (this.regionShape === 'polygon') {
        const points = [...(this.selectionRect?.shape === 'polygon' ? this.selectionRect.points ?? [] : []), point]
        this.selectionRect = this.selectionFromPoints(points, 'polygon')
        return
      }
      this.maskCanvasElement.setPointerCapture(event.pointerId)
      this.drawing = true
      this.lastPointer = point
      this.selectionRect = this.regionShape === 'rope'
        ? this.selectionFromPoints([point], 'rope')
        : { x: point.x, y: point.y, width: 0, height: 0, inverted: false, shape: 'box' }
      return
    }
    if (!this.maskCanvasEditable() || !this.maskContext) return
    this.syncMaskScope()
    this.pushHistory()
    this.maskStrokeMode = this.toolMode === 'eraser' || event.button === 2 ? 'erase' : 'paint'
    this.maskCanvasElement.setPointerCapture(event.pointerId)
    this.drawing = true
    this.lastPointer = this.maskPoint(event)
    this.maskStrokePoints = [this.lastPointer]
    this.drawMaskPoint(this.lastPointer)
    this.scheduleMaskPreviewRefresh()
    this.queueLiveStreamFrame()
  }

  private moveOverlayStroke(event: PointerEvent) {
    event.stopPropagation()
    this.trackCursor(event)
    if (!this.drawing && (event.buttons & 3) !== 0 && this.maskCanvasEditable() && this.maskContext) {
      try {
        this.maskCanvasElement.setPointerCapture(event.pointerId)
      } catch {
        // Pointer capture is best-effort; drawing still works without it.
      }
      this.pushHistory()
      this.maskStrokeMode = this.toolMode === 'eraser' || (event.buttons & 2) !== 0 ? 'erase' : 'paint'
      this.drawing = true
      this.lastPointer = this.maskPoint(event)
      this.maskStrokePoints = [this.lastPointer]
      this.drawMaskPoint(this.lastPointer)
      this.scheduleMaskPreviewRefresh()
      this.queueLiveStreamFrame()
      return
    }
    if (this.toolMode === 'region' && this.drawing && this.lastPointer) {
      const point = this.maskPoint(event)
      if (this.regionShape === 'rope') {
        const points = [...(this.selectionRect?.points ?? [])]
        const lastPoint = points.at(-1)
        if (!lastPoint || Math.hypot(point.x - lastPoint.x, point.y - lastPoint.y) > 4) {
          points.push(point)
          this.selectionRect = this.selectionFromPoints(points, 'rope')
        }
        return
      }
      this.selectionRect = {
        x: this.lastPointer.x,
        y: this.lastPointer.y,
        width: point.x - this.lastPointer.x,
        height: point.y - this.lastPointer.y,
        inverted: this.selectionRect?.inverted ?? false,
        shape: 'box',
      }
      return
    }
    if (!this.drawing || !this.maskContext || !this.lastPointer) return
    const point = this.maskPoint(event)
    const context = this.maskContext
    context.save()
    context.globalCompositeOperation = this.maskStrokeMode === 'erase' ? 'destination-out' : 'source-over'
    context.strokeStyle = this.maskStrokeMode === 'erase' ? 'rgba(0, 0, 0, 1)' : this.brushColor
    context.lineCap = 'round'
    context.lineJoin = 'round'
    context.lineWidth = this.brushSize
    context.beginPath()
    context.moveTo(this.lastPointer.x, this.lastPointer.y)
    context.lineTo(point.x, point.y)
    context.stroke()
    context.restore()
    this.lastPointer = point
    const previousPoint = this.maskStrokePoints.at(-1)
    if (!previousPoint || Math.hypot(point.x - previousPoint.x, point.y - previousPoint.y) > 2) this.maskStrokePoints = [...this.maskStrokePoints, point]
    this.scheduleMaskPreviewRefresh()
    this.queueLiveStreamFrame()
  }

  private endOverlayStroke(event: PointerEvent) {
    event.stopPropagation()
    if (this.drawing && this.maskCanvasElement.hasPointerCapture(event.pointerId)) this.maskCanvasElement.releasePointerCapture(event.pointerId)
    if (this.drawing) this.createRegionFromMaskStroke()
    this.drawing = false
    this.lastPointer = undefined
    this.maskStrokePoints = []
    this.saveCurrentMaskScope()
    this.scheduleMaskPreviewRefresh(true)
    this.scheduleProjectPersist()
    this.queueLiveStreamFrame()
  }

  private normalizedSelection() {
    const selection = this.selectionRect ?? { x: 0, y: 0, width: this.stageWidth, height: this.stageHeight, inverted: false }
    const x = Math.max(0, Math.min(this.stageWidth, selection.width < 0 ? selection.x + selection.width : selection.x))
    const y = Math.max(0, Math.min(this.stageHeight, selection.height < 0 ? selection.y + selection.height : selection.y))
    const right = Math.max(0, Math.min(this.stageWidth, selection.width < 0 ? selection.x : selection.x + selection.width))
    const bottom = Math.max(0, Math.min(this.stageHeight, selection.height < 0 ? selection.y : selection.y + selection.height))
    return { x, y, width: Math.max(1, right - x), height: Math.max(1, bottom - y), inverted: selection.inverted }
  }

  private setSelectionTransform(patch: Partial<SelectionRect> & { centerX?: number; centerY?: number }) {
    if (!this.selectionRect) return
    this.pushHistory()
    const previous = this.normalizedSelection()
    const requested = centeredResizeRect(previous, patch)
    const width = Math.min(this.stageWidth, Math.max(1, requested.width))
    const height = Math.min(this.stageHeight, Math.max(1, requested.height))
    const x = this.clamp(requested.x, 0, this.stageWidth - width, previous.x)
    const y = this.clamp(requested.y, 0, this.stageHeight - height, previous.y)
    const next: SelectionRect = { x, y, width, height, inverted: this.selectionRect.inverted, shape: this.selectionRect.shape }
    const points = this.selectionRect.points
    if (points?.length) {
      const scaleX = width / Math.max(1, previous.width)
      const scaleY = height / Math.max(1, previous.height)
      next.points = points.map((point) => ({
        x: x + (point.x - previous.x) * scaleX,
        y: y + (point.y - previous.y) * scaleY,
      }))
    }
    this.selectionRect = next
  }

  private selectionFromPoints(points: RegionPoint[], shape: RegionShape): SelectionRect {
    const xs = points.map((point) => point.x)
    const ys = points.map((point) => point.y)
    const x = Math.max(0, Math.min(...xs))
    const y = Math.max(0, Math.min(...ys))
    const right = Math.min(this.stageWidth, Math.max(...xs))
    const bottom = Math.min(this.stageHeight, Math.max(...ys))
    return { x, y, width: Math.max(1, right - x), height: Math.max(1, bottom - y), inverted: false, shape, points }
  }

  private clearSelection = () => {
    this.pushHistory()
    this.selectionRect = undefined
  }

  private setRegionShape(shape: RegionShape) {
    this.regionShape = shape
    this.clearSelection()
  }

  private invertSelection = () => {
    if (!this.selectionRect) return
    this.pushHistory()
    this.selectionRect = { ...this.selectionRect, inverted: !this.selectionRect.inverted }
  }

  private deleteSelection = async () => {
    if (!this.selectionRect || !this.fabricCanvas) return
    this.pushHistory()
    const selection = this.normalizedSelection()
    const image = await this.canvasImage()
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    const context = canvas.getContext('2d')!
    context.drawImage(image, 0, 0)
    context.globalCompositeOperation = 'destination-out'
    if (selection.inverted) {
      context.fillRect(0, 0, this.stageWidth, selection.y)
      context.fillRect(0, selection.y + selection.height, this.stageWidth, this.stageHeight - selection.y - selection.height)
      context.fillRect(0, selection.y, selection.x, selection.height)
      context.fillRect(selection.x + selection.width, selection.y, this.stageWidth - selection.x - selection.width, selection.height)
    } else {
      context.fillRect(selection.x, selection.y, selection.width, selection.height)
    }
    await this.replaceCanvasWithDataUrl(canvas.toDataURL('image/png'), 'Selection edit')
  }

  private cropToSelection = async () => {
    if (!this.selectionRect || !this.fabricCanvas) return
    const selection = this.normalizedSelection()
    if (selection.inverted) return
    this.pushHistory()
    const image = await this.canvasImage()
    const canvas = document.createElement('canvas')
    canvas.width = selection.width
    canvas.height = selection.height
    canvas.getContext('2d')!.drawImage(image, selection.x, selection.y, selection.width, selection.height, 0, 0, selection.width, selection.height)
    this.resizeStage(selection.width, selection.height)
    await this.replaceCanvasWithDataUrl(canvas.toDataURL('image/png'), 'Cropped selection')
    this.clearSelection()
  }

  private canvasImage() {
    return new Promise<HTMLImageElement>((resolve) => {
      const image = new Image()
      image.onload = () => resolve(image)
      image.src = this.exportCanvasImage(true)
    })
  }

  private replaceCanvasWithDataUrl(dataUrl: string, name: string) {
    return FabricImage.fromURL(dataUrl).then((image) => {
      this.fabricCanvas?.clear()
      image.set({ left: 0, top: 0, originX: 'left', originY: 'top', name })
      this.attachPreset(image)
      this.fabricCanvas?.add(image)
      this.fabricCanvas?.setActiveObject(image)
      this.fabricCanvas?.requestRenderAll()
      this.syncLayers()
    })
  }

  private drawMaskPoint(point: { x: number; y: number }) {
    if (!this.maskContext) return
    this.maskContext.save()
    this.maskContext.globalCompositeOperation = this.maskStrokeMode === 'erase' ? 'destination-out' : 'source-over'
    this.maskContext.fillStyle = this.maskStrokeMode === 'erase' ? 'rgba(0, 0, 0, 1)' : this.brushColor
    this.maskContext.beginPath()
    this.maskContext.arc(point.x, point.y, this.brushSize / 2, 0, Math.PI * 2)
    this.maskContext.fill()
    this.maskContext.restore()
  }

  private createRegionFromMaskStroke() {
    const scope = this.currentMaskScope()
    if (!scope?.startsWith('layer:') || this.maskStrokeMode !== 'paint' || this.maskStrokePoints.length === 0) return
    const color = this.normalizeColor(this.brushColor)
    const parsed = color ? this.parseColor(color) : undefined
    if (!color || !parsed || parsed.a <= 0.001) return
    const layerId = scope.replace('layer:', '')
    if (this.sameRgb(color, this.layerRegionColor(layerId))) return
    const layer = this.layers.find((item) => item.id === layerId)
    const xs = this.maskStrokePoints.map((point) => point.x)
    const ys = this.maskStrokePoints.map((point) => point.y)
    const pad = Math.max(1, this.brushSize / 2)
    const x = this.clamp(Math.min(...xs) - pad, 0, this.stageWidth, 0)
    const y = this.clamp(Math.min(...ys) - pad, 0, this.stageHeight, 0)
    const right = this.clamp(Math.max(...xs) + pad, 0, this.stageWidth, this.stageWidth)
    const bottom = this.clamp(Math.max(...ys) + pad, 0, this.stageHeight, this.stageHeight)
    const rect = { x, y, width: Math.max(1, right - x), height: Math.max(1, bottom - y), inverted: false, shape: 'box' as RegionShape }
    const existing = this.regions.find((region) => region.target === 'layer' && region.layerId === layerId && this.sameRgb(region.color, color))
    if (existing) {
      const existingRect = this.normalizedRegionRect(existing.rect)
      const unionX = Math.min(existingRect.x, rect.x)
      const unionY = Math.min(existingRect.y, rect.y)
      const unionRight = Math.max(existingRect.x + existingRect.width, rect.x + rect.width)
      const unionBottom = Math.max(existingRect.y + existingRect.height, rect.y + rect.height)
      this.updateRegion(existing.id, {
        rect: { x: unionX, y: unionY, width: unionRight - unionX, height: unionBottom - unionY, inverted: false, shape: 'box' },
        maskStrength: Math.max(existing.maskStrength, parsed.a),
      })
      return
    }
    const region: RegionItem = {
      id: crypto.randomUUID(),
      target: 'layer',
      layerId,
      color: this.rgbToHex(parsed),
      name: `${layer?.name ?? 'Layer'} ${this.rgbToHex(parsed)}`,
      shape: 'box',
      rect,
      maskStrength: 1,
      innerBlur: 0,
      outerBlur: 0,
      negate: false,
      inherited: false,
      prompt: layer?.preset.prompt ?? '',
      negativePrompt: layer?.preset.negativePrompt ?? '',
      denoise: layer?.preset.strength ?? this.strength,
      maskOperator: layer?.preset.maskOperator ?? 'add',
      denoiseOperator: layer?.preset.denoiseOperator ?? 'replace',
      schedule: 'auto',
      start: 0,
      end: 1,
    }
    this.regions = [region, ...this.regions]
  }

  private maskPoint(event: PointerEvent) {
    const rect = this.stageElement.getBoundingClientRect()
    const x = ((event.clientX - rect.left) / rect.width) * this.stageWidth
    const y = ((event.clientY - rect.top) / rect.height) * this.stageHeight
    return {
      x: this.clamp(x, 0, this.stageWidth, 0),
      y: this.clamp(y, 0, this.stageHeight, 0),
    }
  }

  private showCursor = () => {
    this.cursor = { ...this.cursor, visible: true }
  }

  private hideCursor = () => {
    this.cursor = { ...this.cursor, visible: false }
  }

  private trackCursor = (event: PointerEvent) => {
    const rect = this.stageElement.getBoundingClientRect()
    const x = ((event.clientX - rect.left) / rect.width) * this.stageWidth
    const y = ((event.clientY - rect.top) / rect.height) * this.stageHeight
    this.cursor = {
      x: this.clamp(x, 0, this.stageWidth, this.cursor.x),
      y: this.clamp(y, 0, this.stageHeight, this.cursor.y),
      visible: true,
    }
  }

  private syncMaskScope() {
    const nextScope = this.currentMaskScope()
    if (nextScope === this.activeMaskScope) return
    this.saveCurrentMaskScope()
    this.activeMaskScope = nextScope
    this.loadMaskScope(nextScope)
  }

  private saveCurrentMaskScope() {
    if (!this.activeMaskScope || !this.maskCanvasElement) return
    const canvas = this.maskCanvasForScope(this.activeMaskScope)
    const context = canvas.getContext('2d')!
    context.clearRect(0, 0, this.stageWidth, this.stageHeight)
    context.drawImage(this.maskCanvasElement, 0, 0, this.stageWidth, this.stageHeight)
  }

  private loadMaskScope(scope?: MaskScope) {
    if (!this.maskContext) return
    this.maskContext.clearRect(0, 0, this.stageWidth, this.stageHeight)
    if (!scope) return
    this.maskContext.drawImage(this.maskCanvasForScope(scope), 0, 0, this.stageWidth, this.stageHeight)
  }

  private maskCanvasForScope(scope: MaskScope) {
    const existing = this.maskCanvases.get(scope)
    if (existing) return existing
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    this.maskCanvases.set(scope, canvas)
    return canvas
  }

  private resizeStoredMasks() {
    for (const [scope, source] of this.maskCanvases) {
      const resized = document.createElement('canvas')
      resized.width = this.stageWidth
      resized.height = this.stageHeight
      resized.getContext('2d')!.drawImage(source, 0, 0, this.stageWidth, this.stageHeight)
      this.maskCanvases.set(scope, resized)
    }
    this.activeMaskScope = undefined
    this.syncMaskScope()
    this.scheduleMaskPreviewRefresh()
  }

  private clearMaskScope(scope: MaskScope) {
    this.syncMaskScope()
    this.pushHistory()
    const context = this.maskCanvasForScope(scope).getContext('2d')!
    context.clearRect(0, 0, this.stageWidth, this.stageHeight)
    if (this.activeMaskScope === scope) this.loadMaskScope(scope)
    this.refreshMaskPreview()
    this.scheduleProjectPersist()
  }

  private refreshMaskPreview() {
    if (!this.maskPreviewContext) return
    this.lastMaskPreviewAt = performance.now()
    this.maskPreviewContext.clearRect(0, 0, this.stageWidth, this.stageHeight)
    const scope = this.currentMaskScope()
    if (!scope) return
    this.maskPreviewContext.drawImage(this.composeMaskPreviewCanvas(scope), 0, 0)
  }

  private scheduleMaskPreviewRefresh(force = false) {
    if (force) {
      if (this.maskPreviewTimer) {
        window.clearTimeout(this.maskPreviewTimer)
        this.maskPreviewTimer = undefined
      }
      this.refreshMaskPreview()
      return
    }
    if (this.maskPreviewTimer) return
    const delay = Math.max(0, 100 - (performance.now() - this.lastMaskPreviewAt))
    this.maskPreviewTimer = window.setTimeout(() => {
      this.maskPreviewTimer = undefined
      this.refreshMaskPreview()
    }, delay)
  }

  private composeMaskPreviewCanvas(scope: MaskScope) {
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    const context = canvas.getContext('2d')!
    this.drawPaintedMaskPreview(context, scope)
    if (scope === 'scene') {
      this.drawRegionMaskPreview(context, 'scene')
      if (!this.sceneMaskAbsolute) {
        for (const layer of this.layers) {
          if (!this.inputActiveMaskOnly || this.sameRgb(this.layerRegionColor(layer.id), this.brushColor)) this.drawLayerAlphaMaskPreview(context, layer.id)
          this.drawPaintedMaskPreview(context, `layer:${layer.id}`)
          this.drawRegionMaskPreview(context, 'layer', layer.id)
        }
      }
    } else {
      const layerId = scope.replace('layer:', '')
      if (!this.inputActiveMaskOnly || this.sameRgb(this.layerRegionColor(layerId), this.brushColor)) this.drawLayerAlphaMaskPreview(context, layerId)
      this.drawRegionMaskPreview(context, 'layer', layerId)
    }
    return canvas
  }

  private drawLayerAlphaMaskPreview(context: CanvasRenderingContext2D, layerId: string) {
    const object = this.findLayer(layerId)
    if (!object) return
    const alphaCanvas = document.createElement('canvas')
    alphaCanvas.width = this.stageWidth
    alphaCanvas.height = this.stageHeight
    const alphaContext = alphaCanvas.getContext('2d')!
    this.drawLayerCondition(alphaContext, object, layerId)
    const color = this.hexToRgb(this.layerRegionColor(layerId))
    alphaContext.save()
    alphaContext.globalCompositeOperation = 'source-in'
    alphaContext.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, 0.42)`
    alphaContext.fillRect(0, 0, this.stageWidth, this.stageHeight)
    alphaContext.restore()
    context.drawImage(alphaCanvas, 0, 0)
  }

  private drawPaintedMaskPreview(context: CanvasRenderingContext2D, scope: MaskScope) {
    const painted = document.createElement('canvas')
    painted.width = this.stageWidth
    painted.height = this.stageHeight
    const source = this.activeMaskScope === scope ? this.maskCanvasElement : this.maskCanvasForScope(scope)
    painted.getContext('2d')!.drawImage(source, 0, 0)
    if (this.inputActiveMaskOnly) this.keepOnlyActiveMaskColor(painted)
    context.drawImage(painted, 0, 0)
  }

  private keepOnlyActiveMaskColor(canvas: HTMLCanvasElement) {
    const active = this.parseColor(this.brushColor)
    const context = canvas.getContext('2d')!
    const imageData = context.getImageData(0, 0, this.stageWidth, this.stageHeight)
    for (let index = 0; index < imageData.data.length; index += 4) {
      const alpha = imageData.data[index + 3]
      const matches = active && alpha > 0 && Math.abs(imageData.data[index] - active.r) <= 3 && Math.abs(imageData.data[index + 1] - active.g) <= 3 && Math.abs(imageData.data[index + 2] - active.b) <= 3
      if (!matches) imageData.data[index + 3] = 0
    }
    context.putImageData(imageData, 0, 0)
  }

  private drawRegionMaskPreview(context: CanvasRenderingContext2D, target: RegionTarget, layerId?: string) {
    for (const region of this.regions.filter((item) => item.target === target && (target === 'scene' || item.layerId === layerId))) {
      if (this.inputActiveMaskOnly && !this.sameRgb(region.color, this.brushColor)) continue
      if (target === 'layer' && layerId && this.hasLayerRegionMaskPixels(layerId, region)) continue
      context.save()
      context.globalCompositeOperation = 'source-over'
      context.filter = region.outerBlur ? `blur(${region.outerBlur}px)` : 'none'
      context.fillStyle = this.hexToRgba(region.color, region.maskStrength * 0.58)
      this.fillRegionShape(context, region, this.normalizedRegionRect(region.rect))
      if (region.innerBlur) {
        context.filter = `blur(${region.innerBlur}px)`
        context.fillStyle = this.hexToRgba(region.color, region.maskStrength * 0.38)
        this.fillRegionShape(context, region, this.normalizedRegionRect(region.rect))
      }
      context.restore()
    }
  }

  private hexToRgba(hex: string, alpha: number) {
    const color = this.hexToRgb(hex)
    return `rgba(${color.r},${color.g},${color.b},${this.clamp(alpha, 0, 1, 0.5)})`
  }

  private hexToRgb(hex: string) {
    const value = hex.replace('#', '')
    return {
      r: parseInt(value.slice(0, 2), 16) || 255,
      g: parseInt(value.slice(2, 4), 16) || 78,
      b: parseInt(value.slice(4, 6), 16) || 99,
    }
  }

  private _maskCanvasHasContent(scope: MaskScope): boolean {
    const canvas = this.activeMaskScope === scope ? this.maskCanvasElement : this.maskCanvases.get(scope)
    if (!canvas) return false
    const data = canvas.getContext('2d')?.getImageData(0, 0, canvas.width, canvas.height).data
    if (!data) return false
    for (let i = 3; i < data.length; i += 4) if (data[i] > 0) return true
    return false
  }

  private drawLayerAlphaMask(context: CanvasRenderingContext2D, layerId: string) {
    const object = this.findLayer(layerId)
    if (!object) return
    const temp = document.createElement('canvas')
    temp.width = this.stageWidth
    temp.height = this.stageHeight
    const tctx = temp.getContext('2d')!
    this.drawLayerCondition(tctx, object, layerId)
    const imageData = tctx.getImageData(0, 0, this.stageWidth, this.stageHeight)
    for (let i = 0; i < imageData.data.length; i += 4) {
      imageData.data[i] = 255
      imageData.data[i + 1] = 255
      imageData.data[i + 2] = 255
    }
    tctx.putImageData(imageData, 0, 0)
    context.drawImage(temp, 0, 0)
  }

  private composeSceneMaskCanvas() {
    this.saveCurrentMaskScope()

    // Auto-selective: if any explicit content exists, switch to black background so regions
    // and painted masks define WHERE generation happens (white = generate, black = preserve).
    // With no explicit masks, keep white background = full img2img stylization of the whole canvas.
    const hasSceneRegions = this.regions.some((r) => r.target === 'scene')
    const hasLayerRegions = !this.sceneMaskAbsolute && this.regions.some((r) => r.target === 'layer')
    const hasScenePaint = this._maskCanvasHasContent('scene')
    const hasLayerPaint = !this.sceneMaskAbsolute && this.layers.some((l) => this._maskCanvasHasContent(`layer:${l.id}`))
    const hasVisibleLayers = !this.sceneMaskAbsolute && this.layers.some((l) => l.visible && !this.noDefaultMaskLayers.includes(l.id))
    const autoSelective = hasSceneRegions || hasLayerRegions || hasScenePaint || hasLayerPaint || hasVisibleLayers
    const useBlack = this.sceneMaskNegated || autoSelective

    const canvas = this.blankMaskCanvas(useBlack ? '#000' : '#fff')
    const context = canvas.getContext('2d')!
    this.drawPaintedMask(context, 'scene')
    this.drawRegionMasks(context, 'scene')
    if (!this.sceneMaskAbsolute) {
      for (const layer of this.layers) {
        if (useBlack) {
          // With black background, include layer visual alpha as "generate here" area
          // unless the layer has explicit regions (those take precedence) or is hidden from default mask.
          const hasExplicit = this.regions.some((r) => r.target === 'layer' && r.layerId === layer.id)
          const isHidden = this.noDefaultMaskLayers.includes(layer.id)
          if (!hasExplicit && !isHidden) {
            this.drawLayerAlphaMask(context, layer.id)
          }
        }
        this.drawPaintedMask(context, `layer:${layer.id}`, this.layerMaskNegated.get(layer.id) ?? false)
        this.drawRegionMasks(context, 'layer', layer.id)
      }
    }
    return canvas
  }

  private blankMaskCanvas(fillStyle: string) {
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    const context = canvas.getContext('2d')!
    context.fillStyle = fillStyle
    context.fillRect(0, 0, this.stageWidth, this.stageHeight)
    return canvas
  }

  private clearMask = () => {
    this.maskContext?.clearRect(0, 0, this.stageWidth, this.stageHeight)
  }

  private clearCurrentMask = () => {
    this.clearMaskScope(this.currentMaskScope() ?? 'scene')
  }

  private loadImage = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = 'image/*'
    input.multiple = true
    input.onchange = () => Array.from(input.files ?? []).forEach((file) => this.addImageFile(file))
    input.click()
  }

  private async addImageFile(file: File) {
    const url = await this.fileToDataUrl(file)
    this.pushHistory()
    FabricImage.fromURL(url).then((image) => {
      this.fitImage(image)
      image.set({ name: file.name })
      this.attachPreset(image)
      this.fabricCanvas?.add(image)
      this.fabricCanvas?.setActiveObject(image)
      this.fabricCanvas?.requestRenderAll()
      this.syncLayers()
    })
  }

  private fileToDataUrl(file: File) {
    return this.blobToDataUrl(file, file.name)
  }

  private async fetchImageDataUrl(url: string, name: string) {
    const response = await fetch(url, { cache: 'no-store' })
    if (!response.ok) throw new Error(`${response.status} ${response.statusText || 'image request failed'}`)
    const blob = await response.blob()
    if (!blob.type.startsWith('image/')) throw new Error(`${name} is not an image response (${blob.type || 'unknown type'})`)
    return this.blobToDataUrl(blob, name)
  }

  private blobToDataUrl(blob: Blob, name: string) {
    return new Promise<string>((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(String(reader.result ?? ''))
      reader.onerror = () => reject(reader.error ?? new Error(`Could not read ${name}`))
      reader.readAsDataURL(blob)
    })
  }

  private handlePaste = (event: ClipboardEvent) => {
    const items = Array.from(event.clipboardData?.items ?? [])
    for (const item of items) {
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile()
        if (file) {
          event.preventDefault()
          void this.addImageFile(file)
        }
      }
    }
  }

  private fitImage(image: FabricImage) {
    const scale = Math.min(this.stageWidth / image.width!, this.stageHeight / image.height!, 1)
    image.scale(scale)
    image.set({ left: this.stageWidth / 2, top: this.stageHeight / 2, originX: 'center', originY: 'center' })
  }

  private addBlankLayer = () => {
    this.pushHistory()
    const rect = new Rect({
      left: 0,
      top: 0,
      width: this.stageWidth,
      height: this.stageHeight,
      fill: 'rgba(255,255,255,0)',
      name: `Layer ${this.layers.length + 1}`,
    })
    this.attachPreset(rect)
    this.fabricCanvas?.add(rect)
    this.fabricCanvas?.setActiveObject(rect)
    this.syncLayers()
  }

  private addTextLayer = () => {
    this.pushHistory()
    const text = new Textbox('Text', {
      left: this.stageWidth * 0.2,
      top: this.stageHeight * 0.2,
      width: 260,
      fill: this.brushColor,
      fontSize: 48,
      name: `Text ${this.layers.length + 1}`,
    })
    this.attachPreset(text)
    this.fabricCanvas?.add(text)
    this.fabricCanvas?.setActiveObject(text)
    this.fabricCanvas?.requestRenderAll()
    this.syncLayers()
  }

  private syncLayers() {
    if (this.isSyncingLayers) return
    this.isSyncingLayers = true
    try {
    const active = this.fabricCanvas?.getActiveObject()
    if (active && !isPaintChildObject(active as FabricObject & { __paintChild?: boolean; __parentLayerId?: string })) {
      this.selectedEntity = 'layer'
      this.selectedLayerId = String((active as FabricObject & { __uid?: string }).__uid ?? this.assignLayerId(active))
    }
    const layerObjects = topLevelLayerObjects([...(this.fabricCanvas?.getObjects() ?? [])] as (FabricObject & { __paintChild?: boolean; __parentLayerId?: string })[])
    if (this.selectedEntity === 'layer' && this.selectedLayerId && !layerObjects.some((object) => (object as FabricObject & { __uid?: string }).__uid === this.selectedLayerId)) {
      this.selectedLayerId = ''
    }
    this.layers = topLevelLayerObjects([...(this.fabricCanvas?.getObjects() ?? [])] as (FabricObject & { __paintChild?: boolean; __parentLayerId?: string })[])
      .reverse()
      .map((object, index) => {
        this.styleTransformControls(object)
        const id = String((object as FabricObject & { __uid?: string }).__uid ?? this.assignLayerId(object))
        return {
          id,
          name: object.get('name') || `Layer ${index + 1}`,
          preview: this.layerPreview(object),
          visible: object.visible ?? true,
          opacity: object.opacity ?? 1,
          selected: id === this.selectedLayerId,
          preset: this.normalizeLayerPreset((object as FabricObject & { __preset?: LayerPreset }).__preset ?? this.currentPreset()),
        }
      })
    this.applyToolMode()
    this.syncMaskScope()
    this.scheduleMaskPreviewRefresh()
    } finally {
      this.isSyncingLayers = false
    }
  }

  private layerPreview(object: FabricObject) {
    try {
      return (object as FabricObject & { toDataURL?: (options: { format: string; multiplier: number }) => string }).toDataURL?.({
        format: 'png',
        multiplier: 0.12,
      }) ?? this.transparentPreview()
    } catch {
      return this.transparentPreview()
    }
  }

  private transparentPreview() {
    const canvas = document.createElement('canvas')
    canvas.width = 64
    canvas.height = 64
    return canvas.toDataURL('image/png')
  }

  private assignLayerId(object: FabricObject) {
    const withId = object as FabricObject & { __uid?: string }
    withId.__uid = withId.__uid || crypto.randomUUID()
    return withId.__uid
  }

  private findLayer(id: string) {
    return (this.fabricCanvas?.getObjects() ?? []).find(
      (object) => (object as FabricObject & { __uid?: string }).__uid === id,
    )
  }

  private selectLayer(id: string) {
    const object = this.findLayer(id)
    if (!object) return
    this.selectedEntity = 'layer'
    this.selectedLayerId = id
    this.fabricCanvas?.setActiveObject(object)
    this.applyToolMode()
    this.syncMaskScope()
    this.refreshMaskPreview()
    this.fabricCanvas?.requestRenderAll()
    this.syncLayers()
  }

  private selectSceneEntity = () => {
    this.selectedEntity = 'scene'
    this.selectedLayerId = ''
    this.fabricCanvas?.discardActiveObject()
    this.applyToolMode()
    this.syncMaskScope()
    this.refreshMaskPreview()
    this.fabricCanvas?.requestRenderAll()
    this.syncLayers()
  }

  private toggleLayer(id: string) {
    const object = this.findLayer(id)
    if (!object) return
    this.pushHistory()
    object.set('visible', !object.visible)
    for (const child of this.childrenForLayer(id)) child.set('visible', object.visible)
    this.fabricCanvas?.requestRenderAll()
    this.syncLayers()
  }

  private setLayerOpacity(id: string, opacity: number) {
    const object = this.findLayer(id)
    if (!object) return
    opacity = this.clamp(opacity, 0, 1, object.opacity ?? 1)
    this.pushHistory()
    object.set('opacity', opacity)
    for (const child of this.childrenForLayer(id)) child.set('opacity', opacity)
    this.fabricCanvas?.requestRenderAll()
    this.syncLayers()
  }

  private layerTransform(object: FabricObject) {
    const bounds = object.getBoundingRect()
    const center = object.getCenterPoint()
    return {
      centerX: Math.round(center.x),
      centerY: Math.round(center.y),
      width: Math.max(1, Math.round(bounds.width || 1)),
      height: Math.max(1, Math.round(bounds.height || 1)),
      scale: Math.max(1, Math.round((((object.scaleX ?? 1) + (object.scaleY ?? 1)) / 2) * 100)),
      angle: Math.round(object.angle ?? 0),
    }
  }

  private setLayerCenter(id: string, centerX: number, centerY: number) {
    const object = this.findLayer(id)
    if (!object) return
    this.pushHistory()
    object.setPositionByOrigin(new Point(this.clamp(centerX, 0, this.stageWidth, centerX), this.clamp(centerY, 0, this.stageHeight, centerY)), 'center', 'center')
    this.afterLayerTransform(object)
  }

  private setLayerSize(id: string, width: number, height: number) {
    const object = this.findLayer(id)
    if (!object) return
    this.pushHistory()
    const center = object.getCenterPoint()
    object.scaleToWidth(this.clamp(width, 1, this.stageWidth * 4, width))
    object.scaleToHeight(this.clamp(height, 1, this.stageHeight * 4, height))
    object.setPositionByOrigin(center, 'center', 'center')
    this.afterLayerTransform(object)
  }

  private setLayerScale(id: string, scalePercent: number) {
    const object = this.findLayer(id)
    if (!object) return
    this.pushHistory()
    const center = object.getCenterPoint()
    const scale = this.clamp(scalePercent, 1, 400, scalePercent) / 100
    object.set({ scaleX: scale, scaleY: scale })
    object.setPositionByOrigin(center, 'center', 'center')
    this.afterLayerTransform(object)
  }

  private setLayerAngle(id: string, angle: number) {
    const object = this.findLayer(id)
    if (!object) return
    this.pushHistory()
    const center = object.getCenterPoint()
    object.rotate(this.clamp(angle, -180, 180, angle))
    object.setPositionByOrigin(center, 'center', 'center')
    this.afterLayerTransform(object)
  }

  private centerLayer(id: string) {
    this.setLayerCenter(id, this.stageWidth / 2, this.stageHeight / 2)
  }

  private fitLayerToViewport(id: string) {
    const object = this.findLayer(id)
    if (!object) return
    const bounds = object.getBoundingRect()
    const ratio = Math.min(this.stageWidth / Math.max(1, bounds.width), this.stageHeight / Math.max(1, bounds.height), 1)
    this.setLayerSize(id, Math.max(1, Math.round(bounds.width * ratio)), Math.max(1, Math.round(bounds.height * ratio)))
    this.centerLayer(id)
  }

  private afterLayerTransform(object: FabricObject) {
    object.setCoords()
    this.fabricCanvas?.setActiveObject(object)
    this.fabricCanvas?.requestRenderAll()
    this.syncLayers()
  }

  private childrenForLayer(id: string) {
    return (this.fabricCanvas?.getObjects() ?? []).filter(
      (object) => (object as FabricObject & { __parentLayerId?: string }).__parentLayerId === id,
    )
  }

  private layerBlocks() {
    const objects = (this.fabricCanvas?.getObjects() ?? []) as (FabricObject & { __uid?: string; __parentLayerId?: string; __paintChild?: boolean })[]
    const seen = new Set<FabricObject>()
    return topLevelLayerObjects(objects).map((object) => {
      const id = String(object.__uid ?? this.assignLayerId(object))
      const blockObjects = [object, ...objects.filter((candidate) => candidate.__parentLayerId === id)]
        .filter((candidate) => {
          if (seen.has(candidate)) return false
          seen.add(candidate)
          return true
        })
      return {
        id,
        object,
        objects: blockObjects,
      }
    })
  }

  private removeLayerState(id: string) {
    for (const region of this.regions.filter((item) => item.target === 'layer' && item.layerId === id)) this.clearMaskDrawForRegion(region)
    this.regions = this.regions.filter((region) => region.target !== 'layer' || region.layerId !== id)
    this.maskCanvases.delete(`layer:${id}`)
    if (this.activeMaskScope === `layer:${id}`) this.activeMaskScope = undefined
    this.generatedLayers = this.generatedLayers.filter((variation) => variation.layerId !== id)
    this.generationTasks = this.generationTasks.filter((task) => task.layerId !== id)
  }

  private rebuildLayerBlocks(blocks: { id: string; object: FabricObject; objects: FabricObject[] }[], activeId = this.selectedLayerId) {
    if (!this.fabricCanvas) return
    const uniqueObjects = [...new Set(blocks.flatMap((block) => block.objects))]
    this.isRestoringHistory = true
    for (const object of uniqueObjects) this.fabricCanvas.remove(object)
    for (const block of blocks) for (const object of block.objects) this.fabricCanvas.add(object)
    this.isRestoringHistory = false
    const activeBlock = blocks.find((block) => block.id === activeId) ?? blocks.at(-1)
    if (activeBlock) {
      this.selectedEntity = 'layer'
      this.selectedLayerId = activeBlock.id
      this.fabricCanvas.setActiveObject(activeBlock.object)
    } else {
      this.selectedLayerId = ''
      this.fabricCanvas.discardActiveObject()
    }
    this.fabricCanvas.renderAll()
    this.syncLayers()
  }

  private replaceLayerObject(layerId: string, target: FabricObject, replacement: FabricObject) {
    if (!this.fabricCanvas) return
    const blocks = this.layerBlocks()
    const block = blocks.find((item) => item.id === layerId)
    if (!block) return
    const removedObjects = new Set(this.fabricCanvas.getObjects().filter((object) => {
      const meta = object as FabricObject & { __uid?: string; __parentLayerId?: string }
      return object === target || meta.__parentLayerId === layerId
    }))
    for (const item of blocks) item.objects = item.objects.filter((object) => !removedObjects.has(object))
    ;(replacement as FabricObject & { __uid?: string; __paintChild?: boolean; __parentLayerId?: string }).__uid = layerId
    ;(replacement as FabricObject & { __paintChild?: boolean; __parentLayerId?: string }).__paintChild = false
    ;(replacement as FabricObject & { __parentLayerId?: string }).__parentLayerId = undefined
    block.object = replacement
    block.objects = [replacement]
    this.isRestoringHistory = true
    for (const object of removedObjects) this.fabricCanvas.remove(object)
    this.isRestoringHistory = false
    this.rebuildLayerBlocks(blocks, layerId)
  }

  private selectedLayer() {
    return this.layers.find((layer) => layer.selected)
  }

  private updateSelectedLayerPreset(patch: Partial<LayerPreset>) {
    const active = this.fabricCanvas?.getActiveObject() as (FabricObject & { __uid?: string }) | undefined
    const layerId = this.selectedLayerId || (active ? String(active.__uid ?? '') : '')
    if (layerId) this.updateLayerPreset(layerId, patch)
  }

  private updateLayerPreset(layerId: string, patch: Partial<LayerPreset>) {
    const object = this.findLayer(layerId) as (FabricObject & { __preset?: LayerPreset }) | undefined
    if (!object) return
    const preset = this.normalizeLayerPreset({ ...(object.__preset ?? this.currentPreset()), ...patch })
    object.__preset = preset
    this.layers = this.layers.map((layer) => (layer.id === layerId ? { ...layer, preset } : layer))
  }

  private addRegionFromSelection(target: RegionTarget, layerId?: string) {
    if (!this.selectionRect) return
    const normalized = this.normalizedSelection()
    const layer = layerId ? this.layers.find((item) => item.id === layerId) : undefined
    const region: RegionItem = {
      id: crypto.randomUUID(),
      target,
      layerId: target === 'layer' ? layerId : undefined,
      color: layerId ? this.layerRegionColor(layerId) : this.nextRegionColor(),
      name: this.regionName.trim() || `${layer?.name ?? (target === 'scene' ? 'Scene' : 'Layer')} mask ${this.regions.length + 1}`,
      shape: this.selectionRect.shape ?? 'box',
      rect: { ...normalized, shape: this.selectionRect.shape ?? 'box', points: this.selectionRect.points ? [...this.selectionRect.points] : undefined },
      maskStrength: 1,
      innerBlur: 0,
      outerBlur: 24,
      negate: false,
      inherited: true,
      prompt: layer?.preset.prompt ?? '',
      negativePrompt: layer?.preset.negativePrompt ?? '',
      denoise: layer?.preset.strength ?? this.strength,
      maskOperator: layer?.preset.maskOperator ?? 'add',
      denoiseOperator: layer?.preset.denoiseOperator ?? 'replace',
      schedule: 'auto',
      start: 0,
      end: 1,
    }
    this.regions = [region, ...this.regions]
    this.regionName = ''
  }

  private updateRegion(id: string, patch: Partial<RegionItem>) {
    this.regions = this.regions.map((region) => (region.id === id ? { ...region, ...patch } : region))
  }

  private setRegionColor(id: string, color: string) {
    const region = this.regions.find((item) => item.id === id)
    const next = this.normalizeColor(color)
    if (!region || !next) return
    this.recolorMaskDrawForRegion(region, next)
    this.updateRegion(id, { color: next, inherited: false })
    this.loadMaskScope(this.activeMaskScope)
    this.scheduleMaskPreviewRefresh(true)
    this.scheduleProjectPersist()
  }

  private restoreRegionInheritance(id: string) {
    const region = this.regions.find((item) => item.id === id)
    const layer = region?.layerId ? this.layers.find((item) => item.id === region.layerId) : undefined
    this.updateRegion(id, {
      inherited: true,
      prompt: layer?.preset.prompt ?? '',
      negativePrompt: layer?.preset.negativePrompt ?? '',
      denoise: this.strength,
      maskOperator: layer?.preset.maskOperator ?? 'add',
      denoiseOperator: layer?.preset.denoiseOperator ?? 'replace',
      schedule: 'auto',
      start: 0,
      end: 1,
    })
  }

  private nextRegionColor() {
    const hue = (this.regions.length * 47 + 205) % 360
    return this.hslToHex(hue, 78, 54)
  }

  private deleteRegion(id: string) {
    const region = this.regions.find((item) => item.id === id)
    if (region) this.clearMaskDrawForRegion(region)
    this.regions = this.regions.filter((item) => item.id !== id)
    this.loadMaskScope(this.activeMaskScope)
    this.scheduleMaskPreviewRefresh(true)
    this.scheduleProjectPersist()
  }

  private clearMaskDrawForRegion(region: RegionItem) {
    if (region.target !== 'layer' || !region.layerId) return
    this.saveCurrentMaskScope()
    const target = this.parseColor(region.color)
    if (!target) return
    const scope: MaskScope = `layer:${region.layerId}`
    const canvas = this.maskCanvasForScope(scope)
    const context = canvas.getContext('2d')!
    const imageData = context.getImageData(0, 0, this.stageWidth, this.stageHeight)
    let changed = false
    for (let index = 0; index < imageData.data.length; index += 4) {
      const alpha = imageData.data[index + 3]
      const matches = alpha > 0 && Math.abs(imageData.data[index] - target.r) <= 3 && Math.abs(imageData.data[index + 1] - target.g) <= 3 && Math.abs(imageData.data[index + 2] - target.b) <= 3
      if (matches) {
        imageData.data[index + 3] = 0
        changed = true
      }
    }
    if (changed) context.putImageData(imageData, 0, 0)
  }

  private recolorMaskDrawForRegion(region: RegionItem, color: string) {
    if (region.target !== 'layer' || !region.layerId) return
    this.saveCurrentMaskScope()
    const from = this.parseColor(region.color)
    const to = this.parseColor(color)
    if (!from || !to) return
    const canvas = this.maskCanvasForScope(`layer:${region.layerId}`)
    const context = canvas.getContext('2d')!
    const imageData = context.getImageData(0, 0, this.stageWidth, this.stageHeight)
    let changed = false
    for (let index = 0; index < imageData.data.length; index += 4) {
      const alpha = imageData.data[index + 3]
      const matches = alpha > 0 && Math.abs(imageData.data[index] - from.r) <= 3 && Math.abs(imageData.data[index + 1] - from.g) <= 3 && Math.abs(imageData.data[index + 2] - from.b) <= 3
      if (matches) {
        imageData.data[index] = to.r
        imageData.data[index + 1] = to.g
        imageData.data[index + 2] = to.b
        changed = true
      }
    }
    if (changed) context.putImageData(imageData, 0, 0)
  }

  private clearActiveLayer = () => {
    const object = this.selectedLayerId ? this.findLayer(this.selectedLayerId) : this.fabricCanvas?.getActiveObject()
    if (!object) return
    this.pushHistory()
    const id = (object as FabricObject & { __uid?: string }).__uid
    if (id) for (const child of this.childrenForLayer(id)) this.fabricCanvas?.remove(child)
    if (object.type === 'textbox' || object.type === 'rect') object.set({ opacity: 0 })
    else object.set({ opacity: 0 })
    this.fabricCanvas?.discardActiveObject()
    this.fabricCanvas?.requestRenderAll()
    this.syncLayers()
  }

  private duplicateActiveLayer = async () => {
    const object = this.fabricCanvas?.getActiveObject()
    if (!object) return
    this.pushHistory()
    const clone = await object.clone()
    clone.set({ left: (object.left ?? 0) + 12, top: (object.top ?? 0) + 12, name: `${object.get('name') || 'Layer'} copy` })
    this.attachPreset(clone, { ...((object as FabricObject & { __preset?: LayerPreset }).__preset ?? this.currentPreset()) })
    this.fabricCanvas?.add(clone)
    this.fabricCanvas?.setActiveObject(clone)
    this.fabricCanvas?.requestRenderAll()
    this.syncLayers()
  }

  private deleteActiveLayer = () => {
    if (!this.fabricCanvas) return
    const blocks = this.layerBlocks()
    const activeId = this.selectedLayerId || ((this.fabricCanvas.getActiveObject() as FabricObject & { __uid?: string } | undefined)?.__uid ?? '')
    const index = blocks.findIndex((block) => block.id === activeId)
    if (index < 0) return
    this.pushHistory()
    const [removed] = blocks.splice(index, 1)
    const removedObjects = new Set(this.fabricCanvas.getObjects().filter((object) => {
      const meta = object as FabricObject & { __uid?: string; __parentLayerId?: string }
      return meta.__uid === removed.id || meta.__parentLayerId === removed.id
    }))
    for (const block of blocks) block.objects = block.objects.filter((object) => !removedObjects.has(object))
    this.removeLayerState(removed.id)
    this.isRestoringHistory = true
    for (const object of removedObjects) this.fabricCanvas.remove(object)
    this.isRestoringHistory = false
    const nextActive = blocks[Math.min(index, blocks.length - 1)]?.id ?? blocks.at(-1)?.id ?? ''
    this.rebuildLayerBlocks(blocks, nextActive)
  }

  private moveLayerUp = () => {
    const blocks = this.layerBlocks()
    const index = blocks.findIndex((block) => block.id === this.selectedLayerId)
    if (index < 0 || index >= blocks.length - 1) return
    this.pushHistory()
    const [block] = blocks.splice(index, 1)
    blocks.splice(index + 1, 0, block)
    this.rebuildLayerBlocks(blocks, block.id)
  }

  private moveLayerDown = () => {
    const blocks = this.layerBlocks()
    const index = blocks.findIndex((block) => block.id === this.selectedLayerId)
    if (index <= 0) return
    this.pushHistory()
    const [block] = blocks.splice(index, 1)
    blocks.splice(index - 1, 0, block)
    this.rebuildLayerBlocks(blocks, block.id)
  }

  private generatePromptLayer = async () => {
    const layer = this.selectedLayer()
    const object = layer ? this.findLayer(layer.id) : undefined
    if (!layer || !object) {
      this.reportError('Select a layer before generating variations.')
      return
    }
    this.isGeneratingLayer = true
    const localTaskId = `local-${crypto.randomUUID()}`
    this.upsertGenerationTask({
      task_id: localTaskId,
      layerId: layer.id,
      label: `${this.layerVariationCount} variation${this.layerVariationCount === 1 ? '' : 's'}`,
      status: 'queued',
      phase: 'submitting',
      progress: 0,
      message: 'Submitting generation task',
      startedAt: Date.now(),
    })
    try {
      const body = JSON.stringify({
        prompt: layer.preset.prompt,
        negative_prompt: layer.preset.negativePrompt,
        model_path: this.selectedModel || undefined,
        device: this.layerDevice || undefined,
        lora_paths: [...this.selectedLoras],
        width: this.generationWidth,
        height: this.generationHeight,
        variations: this.layerVariationCount,
        steps: this.layerGenerationSteps,
        cfg: this.layerGenerationCfg,
        sampler: this.layerGenerationSampler || undefined,
        scheduler: this.layerGenerationScheduler || undefined,
        transparent_background: this.layerTransparentBackground,
        transparent_background_method: this.layerTransparentMethod,
        transparent_background_mode: this.layerTransparentMode,
        transparent_background_color: this.layerTransparentColor,
        transparent_background_tolerance: this.layerTransparentTolerance,
        transparent_alpha_blur: this.layerTransparentAlphaBlur,
        transparent_alpha_threshold: this.layerTransparentAlphaThreshold,
        seed: this.parseOptionalSeed(),
      })
      const response = await fetch(this.backendUrl('/layer/tasks'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
      })
      if (!response.ok) throw new Error(await response.text())
      const { task_id: taskId } = (await response.json()) as { task_id: string }
      this.replaceGenerationTaskId(localTaskId, taskId)
      await this.watchLayerTask(taskId, layer.id)
    } catch (error) {
      this.reportError(error instanceof Error ? error.message : String(error))
      this.upsertGenerationTask({
        task_id: localTaskId,
        layerId: layer.id,
        label: `${this.layerVariationCount} variation${this.layerVariationCount === 1 ? '' : 's'}`,
        status: 'error',
        phase: 'error',
        progress: 1,
        message: error instanceof Error ? error.message : String(error),
        error: error instanceof Error ? error.message : String(error),
        startedAt: Date.now(),
      })
    } finally {
      this.isGeneratingLayer = false
    }
  }

  private upsertGenerationTask(task: GenerationTask) {
    const existing = this.generationTasks.find((item) => item.task_id === task.task_id)
    this.generationTasks = existing
      ? this.generationTasks.map((item) => (item.task_id === task.task_id ? { ...item, ...task } : item))
      : [task, ...this.generationTasks].slice(0, 8)
  }

  private replaceGenerationTaskId(localTaskId: string, taskId: string) {
    this.generationTasks = this.generationTasks.map((task) => (task.task_id === localTaskId ? { ...task, task_id: taskId, message: 'Task accepted by server' } : task))
  }

  private watchLayerTask(taskId: string, layerId: string) {
    return new Promise<void>((resolve) => {
      let settled = false
      const finish = () => {
        if (!settled) {
          settled = true
          resolve()
        }
      }
      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
      const socket = new WebSocket(`${protocol}://${window.location.hostname}:8000/ws/tasks/${taskId}`)
      socket.onmessage = (event) => {
        const progress = JSON.parse(event.data) as LayerTaskProgress
        const label = this.generationTasks.find((task) => task.task_id === taskId)?.label ?? 'Layer generation'
        const startedAt = this.generationTasks.find((task) => task.task_id === taskId)?.startedAt ?? Date.now()
        this.upsertGenerationTask({ ...progress, layerId, label, startedAt })
        this.status = `${progress.phase} ${Math.round(progress.progress * 100)}%`
        if (progress.result) {
          this.applyLayerTaskResult(layerId, progress.result, progress.status === 'complete')
        }
        if (progress.status === 'complete') {
          socket.close()
          finish()
        }
        if (progress.status === 'error') {
          this.reportError(progress.error || progress.message)
          socket.close()
          finish()
        }
      }
      socket.onerror = () => {
        const message = 'Task monitor connection failed.'
        this.reportError(message)
        this.upsertGenerationTask({
          task_id: taskId,
          layerId,
          label: 'Layer generation',
          status: 'error',
          phase: 'monitor',
          progress: 1,
          message,
          error: message,
          startedAt: Date.now(),
        })
        finish()
      }
      socket.onclose = () => finish()
    })
  }

  private applyLayerTaskResult(layerId: string, result: LayerGeneratePayload, complete: boolean) {
    this.generatedLayers = [
      ...this.generatedLayers.filter((variation) => variation.layerId !== layerId),
      ...result.variations.map((variation) => ({ ...variation, layerId, selected: false })),
    ]
    this.backendMode = `${result.mode} | layer task ${Math.round(result.latency_ms)} ms${complete ? '' : ' partial'}`
  }

  private parseOptionalSeed() {
    const seed = Number.parseInt(this.layerGenerationSeed.trim(), 10)
    return Number.isFinite(seed) && seed >= 0 ? seed : undefined
  }

  private placeGeneratedLayer(variation: GeneratedLayer) {
    if (!this.fabricCanvas) return
    const target = this.findLayer(variation.layerId)
    if (!target) return
    const bounds = this.layerBounds(target)
    const preset = (target as FabricObject & { __preset?: LayerPreset }).__preset ?? this.currentPreset()
    const name = target.get('name') || `Prompt layer ${variation.seed}`
    const oldId = (target as FabricObject & { __uid?: string }).__uid
    this.generatedLayers = this.generatedLayers.map((item) => ({ ...item, selected: item.layerId === variation.layerId && item.seed === variation.seed }))
    if (!this.generatorActive) return
    this.pushHistory()
    FabricImage.fromURL(variation.image).then((image) => {
      image.set({ name, left: bounds.x, top: bounds.y, originX: 'left', originY: 'top' })
      image.scaleToWidth(bounds.width)
      image.scaleToHeight(bounds.height)
      this.attachPreset(image)
      ;(image as FabricObject & { __uid?: string; __preset?: LayerPreset }).__uid = oldId
      ;(image as FabricObject & { __preset?: LayerPreset }).__preset = preset
      this.replaceLayerObject(variation.layerId, target, image)
    })
  }

  private layerBounds(object: FabricObject) {
    const bounds = object.getBoundingRect()
    return {
      x: Math.max(0, Math.round(bounds.left || 0)),
      y: Math.max(0, Math.round(bounds.top || 0)),
      width: Math.max(64, Math.round(bounds.width || this.stageWidth)),
      height: Math.max(64, Math.round(bounds.height || this.stageHeight)),
    }
  }

  private alignGenerationSize(value: number) {
    return Math.max(256, Math.min(1536, Math.round(value / 64) * 64 || 256))
  }

  private setSceneModel = (modelPath: string) => {
    this.selectedModel = modelPath
    this.applySamplerPresetToScene(modelPath)
    this.applySamplerPresetToLayer(modelPath)
    this.persistOptions()
    if (this.isStreaming) this.stopStream()
    this.backendMode = `model queued: ${this.assets.models.find((model) => model.path === this.selectedModel)?.name ?? 'custom'}`
  }

  private setScenePresetValue<K extends keyof ScenePreset>(key: K, value: ScenePreset[K]) {
    if (key === 'prompt') this.prompt = value as string
    if (key === 'negativePrompt') this.negativePrompt = value as string
    if (key === 'strength') this.strength = value as number
    if (key === 'cfg') this.cfg = value as number
    if (key === 'steps') this.steps = value as number
    if (key === 'sampler') this.sceneSampler = value as string
    if (key === 'scheduler') this.sceneScheduler = value as string
  }

  private setSceneSamplerValue<K extends keyof SamplerPreset>(key: K, value: SamplerPreset[K]) {
    if (key === 'strength') this.setScenePresetValue('strength', value as number)
    if (key === 'cfg') this.setScenePresetValue('cfg', value as number)
    if (key === 'steps') this.setScenePresetValue('steps', value as number)
    if (key === 'sampler') this.setScenePresetValue('sampler', value as string)
    if (key === 'scheduler') this.setScenePresetValue('scheduler', value as string)
    this.saveSamplerPresetForModel(this.selectedModel, 'scene', this.currentSceneSamplerPreset())
  }

  private setLayerSamplerValue<K extends keyof SamplerPreset>(key: K, value: SamplerPreset[K], modelPath: string) {
    if (key === 'cfg') this.layerGenerationCfg = value as number
    if (key === 'steps') this.layerGenerationSteps = value as number
    if (key === 'sampler') this.layerGenerationSampler = value as string
    if (key === 'scheduler') this.layerGenerationScheduler = value as string
    this.saveSamplerPresetForModel(modelPath, 'layer', this.currentLayerSamplerPreset())
  }

  private applySamplerPresetToScene(modelPath: string) {
    const preset = this.samplerPresetForModel(modelPath, 'scene')
    if (!preset) return
    this.strength = this.clamp(preset.strength ?? this.strength, 0, 0.999, this.strength)
    this.cfg = this.clamp(preset.cfg ?? this.cfg, 0, 30, this.cfg)
    this.steps = Math.round(this.clamp(preset.steps ?? this.steps, 1, 40, this.steps))
    this.sceneSampler = preset.sampler ?? this.sceneSampler
    this.sceneScheduler = preset.scheduler ?? this.sceneScheduler
  }

  private applySamplerPresetToLayer(modelPath: string) {
    const preset = this.samplerPresetForModel(modelPath, 'layer')
    if (!preset) return
    this.layerGenerationCfg = this.clamp(preset.cfg ?? this.layerGenerationCfg, 0, 30, this.layerGenerationCfg)
    this.layerGenerationSteps = Math.round(this.clamp(preset.steps ?? this.layerGenerationSteps, 1, 40, this.layerGenerationSteps))
    this.layerGenerationSampler = preset.sampler ?? this.layerGenerationSampler
    this.layerGenerationScheduler = preset.scheduler ?? this.layerGenerationScheduler
  }

  private currentSceneSamplerPreset(): SamplerPreset {
    return {
      strength: this.strength,
      cfg: this.cfg,
      steps: this.steps,
      sampler: this.sceneSampler,
      scheduler: this.sceneScheduler,
    }
  }

  private currentLayerSamplerPreset(): SamplerPreset {
    return {
      cfg: this.layerGenerationCfg,
      steps: this.layerGenerationSteps,
      sampler: this.layerGenerationSampler,
      scheduler: this.layerGenerationScheduler,
    }
  }

  private saveSamplerPresetForModel(modelPath: string, scope: 'scene' | 'layer', preset: SamplerPreset) {
    if (!modelPath) return
    const presets = this.readSamplerPresets()
    presets[`${modelPath}::${scope}`] = preset
    window.localStorage.setItem(SAMPLER_PRESETS_KEY, JSON.stringify(presets))
  }

  private samplerPresetForModel(modelPath: string, scope: 'scene' | 'layer') {
    if (!modelPath) return undefined
    const presets = this.readSamplerPresets()
    return presets[`${modelPath}::${scope}`] ?? (scope === 'scene' ? presets[modelPath] : undefined)
  }

  private readSamplerPresets(): Record<string, SamplerPreset> {
    try {
      return JSON.parse(window.localStorage.getItem(SAMPLER_PRESETS_KEY) || '{}') as Record<string, SamplerPreset>
    } catch {
      return {}
    }
  }

  private clearLegacyScenePresets() {
    window.localStorage.removeItem(SCENE_PRESETS_KEY)
  }

  private visibleLoras(selectedPaths = [...this.selectedLoras]) {
    const query = this.loraSearch.trim().toLowerCase()
    return this.assets.loras.filter(
      (lora) => selectedPaths.includes(lora.path) || !query || lora.name.toLowerCase().includes(query),
    )
  }

  private toggleSceneLora = (path: string) => {
    const next = new Set(this.selectedLoras)
    if (next.has(path)) next.delete(path)
    else next.add(path)
    this.selectedLoras = next
    if (this.isStreaming) this.stopStream()
  }

  private applyStreamRuntimePreset(value: string) {
    if (!this.validStreamRuntimePreset(value)) return
    if (this.isStreaming) this.stopStream()
    this.streamRuntimePreset = value
    if (value === 'diffusers') {
      this.selectedLoras = new Set([...this.selectedLoras].filter((path) => !this.isStreamPresetLora(path)))
      return
    }
    if (value === 'lcm-lora-sdxl') {
      this.applyStreamLoraPreset(this.findLoraAsset(/lcm.*sdxl|lcm-lora-sdxl/i) || 'D:/comfyui/comfy/models/loras/lcm-lora-sdxl/pytorch_lora_weights.safetensors')
      this.sceneSampler = 'lcm'
      this.sceneScheduler = 'simple'
      this.steps = 4
      this.cfg = 1.2
      this.strength = 0.7
      // [16,32] → t≈680,360: strong stylisation. [32,45] was t≈360,100 (too conservative — output ≈ input).
      this.streamTimestepIndices = '16,32'
      return
    }
    if (value === 'sdxl-lightning-4step') {
      this.applyStreamLoraPreset(this.findLoraAsset(/lightning.*4step|sdxl_lightning_4step/i) || 'D:/comfyui/comfy/models/loras/SDXL-Lightning/sdxl_lightning_4step_lora.safetensors')
      this.sceneScheduler = 'euler'
      this.steps = 4
      this.cfg = 1.0
      this.strength = 0.6
      this.streamTimestepIndices = '16,32'
      return
    }
    if (value === 'sana-sprint-0.6b' || value === 'sana-sprint-1.6b') {
      // SANA-Sprint: set model path to HuggingFace ID — backend auto-detects and routes to sana_pipeline
      this.selectedModel = value === 'sana-sprint-0.6b'
        ? 'Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers'
        : 'Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers'
      // Remove SDXL LoRAs (SANA uses its own text encoder, not CLIP)
      this.selectedLoras = new Set([...this.selectedLoras].filter((path) => !this.isStreamPresetLora(path)))
      this.cfg = 4.5           // SANA default guidance scale
      this.strength = 0.7      // img2img strength
      this.sanaSteps = 2       // Sprint runs in 2 steps
      return
    }
    const turbo = this.findSdxlTurboModel() || 'D:/comfyui/comfy/models/diffusers/sdxl-turbo'
    this.selectedModel = turbo
    this.selectedLoras = new Set([...this.selectedLoras].filter((path) => !this.isStreamPresetLora(path)))
    this.sceneSampler = 'euler'
    this.sceneScheduler = 'simple'
    this.steps = 1
    this.cfg = 1.0
    this.strength = 0.5
    // SDXL Turbo: higher indices work fine since it's a distilled model
    this.streamTimestepIndices = '32,45'
  }

  private applyStreamLoraPreset(path: string) {
    const next = new Set([...this.selectedLoras].filter((item) => !this.isStreamPresetLora(item)))
    next.add(path)
    this.selectedLoras = next
  }

  private isStreamPresetLora(path: string) {
    return /lcm-lora-sdxl|pytorch_lora_weights\.safetensors|sdxl-lightning|sdxl_lightning/i.test(path.replace(/\\/g, '/'))
  }

  private findLoraAsset(pattern: RegExp) {
    return this.assets.loras.find((asset) => pattern.test(`${asset.name} ${asset.path}`))?.path
  }

  private findModelAsset(pattern: RegExp) {
    return this.assets.models.find((asset) => pattern.test(`${asset.name} ${asset.path}`))?.path
  }

  private findSdxlTurboModel() {
    const normalized = (asset: AssetItem) => asset.path.replace(/\\/g, '/').toLowerCase()
    return this.assets.models.find((asset) => normalized(asset).endsWith('/sdxl-turbo') || normalized(asset).endsWith('/sd_xl_turbo'))?.path
      ?? this.assets.models.find((asset) => asset.name.toLowerCase() === 'sdxl-turbo')?.path
      ?? this.findModelAsset(/stabilityai\/sdxl-turbo/i)
  }

  private allowDrop = (event: DragEvent) => {
    event.preventDefault()
  }

  private dropImages = (event: DragEvent) => {
    event.preventDefault()
    Array.from(event.dataTransfer?.files ?? [])
      .filter((file) => file.type.startsWith('image/'))
      .forEach((file) => this.addImageFile(file))
  }

  private zoomViewport(event: WheelEvent, target: ViewportTarget) {
    event.preventDefault()
    if (event.altKey) {
      const delta = event.deltaY > 0 ? -3 : 3
      this.brushSize = Math.max(1, Math.min(4096, this.brushSize + delta))
      this.configureFabricBrush()
      return
    }
    this.showViewportOverlay(event, target)
    const currentZoom = target === 'input' ? this.inputZoom : this.outputZoom
    const multiplier = event.deltaY > 0 ? 0.9 : 1.1
    const nextZoom = Math.max(0.25, Math.min(8, Number((currentZoom * multiplier).toFixed(3))))
    if (target === 'input') this.inputZoom = nextZoom
    else this.outputZoom = nextZoom
  }

  private startViewportPan = (event: PointerEvent) => {
    if (event.button !== 1 && !(event.pointerType === 'touch' && this.toolMode === 'select')) return
    event.preventDefault()
    const wrap = event.currentTarget as HTMLElement
    this.panningTarget = wrap
    this.panStart = { x: event.clientX, y: event.clientY, left: wrap.scrollLeft, top: wrap.scrollTop }
    wrap.setPointerCapture(event.pointerId)
  }

  private moveViewport(event: PointerEvent, target: ViewportTarget) {
    this.showViewportOverlay(event, target)
    this.moveViewportPan(event)
  }

  private moveViewportPan(event: PointerEvent) {
    const wrap = event.currentTarget as HTMLElement
    if (this.panningTarget !== wrap) return
    wrap.scrollLeft = this.panStart.left - (event.clientX - this.panStart.x)
    wrap.scrollTop = this.panStart.top - (event.clientY - this.panStart.y)
  }

  private showViewportOverlay(_event: Event, target: ViewportTarget) {
    if (target === 'input') this.inputOverlayVisible = true
    else this.outputOverlayVisible = true
    const existingTimer = this.overlayTimers[target]
    if (existingTimer) window.clearTimeout(existingTimer)
    this.overlayTimers[target] = window.setTimeout(() => {
      if (target === 'input') this.inputOverlayVisible = false
      else this.outputOverlayVisible = false
    }, 2000)
  }

  private leaveViewport(event: PointerEvent, target: ViewportTarget) {
    this.endViewportPan(event)
    const existingTimer = this.overlayTimers[target]
    if (existingTimer) window.clearTimeout(existingTimer)
    this.overlayTimers[target] = undefined
    if (target === 'input') this.inputOverlayVisible = false
    else this.outputOverlayVisible = false
  }

  private endViewportPan = (event: PointerEvent) => {
    const wrap = this.panningTarget
    if (!wrap) return
    if (wrap.hasPointerCapture(event.pointerId)) wrap.releasePointerCapture(event.pointerId)
    this.panningTarget = undefined
  }

  private toggleStream = () => {
    if (this.isStreaming) this.stopStream()
    else void this.startStream()
  }

  private async startStream() {
    this.clearMotionPlayback()
    this.realtimeVideoActive = true
    this.streamStartedAt = performance.now()
    this.streamFeedbackImage = undefined
    this.realtimeFrameTimes = []
    this.displayFps = 0
    await this.updateComplete
    if (this.scenePanelTab === 'streamdiffusion' || this.scenePanelTab === 'sana') {
      await this.startWebRtc()
      return
    }
    this.setupRealtimeVideoOutput()
    this.socket = new WebSocket(this.backendWsUrl('/ws/inpaint'))
    this.socket.binaryType = 'arraybuffer'
    this.status = 'connecting'
    this.socket.onopen = () => {
      this.isStreaming = true
      this.status = 'streaming'
      const loop = () => {
        this.sendFrame()
        if (this.isStreaming) this.streamRaf = window.requestAnimationFrame(loop)
      }
      this.streamRaf = window.requestAnimationFrame(loop)
    }
    this.socket.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        void this.drawRealtimeBinaryFrame(event.data)
        return
      }
      const message = JSON.parse(event.data) as StreamMessage
      if (message.error) {
        this.awaitingFrame = false
        this.reportError(message.error)
        this.status = 'error'
        this.selectedModel = this.lastWorkingModel || this.selectedModel
        this.selectedLoras = new Set(this.lastWorkingLoras)
        return
      }
      this.streamFrameContentType = message.content_type || 'image/jpeg'
      if (message.image) {
        this.awaitingFrame = false
        this.outputImage = message.image
        this.streamFeedbackImage = this.loadedImageElement(message.image)
      }
      this.fps = message.fps
      this.latency = message.latency_ms
      this.backendMode = message.mode
      this.lastWorkingModel = this.selectedModel
      this.lastWorkingLoras = [...this.selectedLoras]
      if (message.debug_channels && Object.keys(message.debug_channels).length > 0) {
        this._applyDebugChannels(message.debug_channels as Record<string, string>)
      }
    }
    this.socket.onclose = () => {
      this.awaitingFrame = false
      this.isStreaming = false
      this.status = 'offline'
      if (this.streamTimer) window.clearInterval(this.streamTimer)
      if (this.streamRaf) { window.cancelAnimationFrame(this.streamRaf); this.streamRaf = undefined }
    }
    this.socket.onerror = () => {
      this.awaitingFrame = false
      this.status = 'backend unavailable'
    }
  }

  private async startWebRtc() {
    this.webRtcPcId = ''
    this.status = 'connecting'

    try {
      const resp = await fetch(this.backendUrl('/api/rtc/start'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}',
      })
      if (!resp.ok) throw new Error(`RTC start failed: ${resp.status} ${await resp.text()}`)
      const data = await resp.json() as { pc_id: string }
      this.webRtcPcId = data.pc_id

      // Open SSE stream for output frames
      const es = new EventSource(this.backendUrl(`/api/rtc/${data.pc_id}/stream`))
      this.webRtcEventSource = es
      es.onmessage = (e) => {
        const img = this.streamOutputImgElement
        if (img) img.src = e.data as string
        this.updateRealtimeDisplayFps()
      }
      es.onerror = () => {
        // EventSource reconnects automatically; only log when session is still active
        if (!this.webRtcPcId) return
        console.debug('SSE error — EventSource will retry automatically')
      }
      es.addEventListener('status', (e) => {
        const s = JSON.parse((e as MessageEvent).data) as { phase: string; model: string; fps: number; error: string }
        if (s.phase === 'error') {
          this.backendMode = `✗ ${s.error}`
        } else if (s.phase === 'loading') {
          this.backendMode = `⚙ Loading ${s.model || 'model'}…`
        } else if (s.phase === 'ready') {
          this.backendMode = s.fps > 0 ? `${s.model} – ${s.fps.toFixed(1)} FPS` : (s.model || 'streamdiffusion')
        }
      })
      es.addEventListener('debug', (e) => {
        this._applyDebugChannels(JSON.parse((e as MessageEvent).data) as Record<string, string>)
      })

      // Start feeding canvas frames and settings via HTTP POST
      this.isStreaming = true
      this.status = 'streaming'

      void this.sendSettingsHttp()
      void this.sendFrameHttp()

      this.webRtcSettingsTimer = window.setInterval(() => void this.sendSettingsHttp(), 200)

      let lastFrameMs = 0
      const frameLoop = () => {
        if (!this.webRtcPcId) return
        this.frameLoopRaf = window.requestAnimationFrame(frameLoop)
        const now = performance.now()
        if (now - lastFrameMs >= 33) {
          lastFrameMs = now
          void this.sendFrameHttp()
        }
      }
      this.frameLoopRaf = window.requestAnimationFrame(frameLoop)
    } catch (err) {
      this.reportError(err instanceof Error ? err.message : String(err))
      this.status = 'error'
      this.stopWebRtc()
    }
  }

  private async sendFrameHttp() {
    if (!this.webRtcPcId) return
    this.frameAbortController?.abort()
    this.frameAbortController = new AbortController()
    if (this._videoLayerId && this.videoFpsMode === 'sequential') {
      this.advanceVideoSequential()
    }
    const image = this.exportRealtimeCanvasImage('image/jpeg')
    if (!image) return
    const layerFrames = this.exportLayerFrames()
    try {
      await fetch(this.backendUrl(`/api/rtc/${this.webRtcPcId}/frame`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image,
          transform: this.currentStreamMotionTransform() ?? null,
          layers: Object.keys(layerFrames).length ? layerFrames : undefined,
        }),
        signal: this.frameAbortController.signal,
      })
    } catch { /* aborted or network error — skip frame */ }
  }

  private async sendSettingsHttp() {
    if (!this.webRtcPcId) return
    const payload = JSON.stringify({
      prompt: this.prompt,
      negative_prompt: this.negativePrompt,
      mask: this.exportRealtimeMask(),
      cfg: this.cfg,
      strength: this.strength,
      steps: this.steps,
      stream_timestep_indices: this.streamTimestepIndexList(),
      stream_frame_buffer_size: this.streamFrameBufferSize,
      stream_cfg_type: this.streamCfgType,
      model_path: this.selectedModel || undefined,
      device: this.realtimeDevice || undefined,
      lora_paths: [...this.selectedLoras],
      width: this.stageWidth,
      height: this.stageHeight,
      sana_steps: this.sanaSteps,
      layer_conditions: this.exportLayerConditions(),
      pipeline_nodes: this.exportPipelineNodes(),
      debug_streams: this.debugStreamsEnabled,
    })
    if (payload === this.webRtcLastSettings) return
    this.webRtcLastSettings = payload
    try {
      await fetch(this.backendUrl(`/api/rtc/${this.webRtcPcId}/settings`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: payload,
      })
    } catch { /* ignore — next interval will retry */ }
  }

  private stopWebRtc() {
    if (this.webRtcSettingsTimer) { window.clearInterval(this.webRtcSettingsTimer); this.webRtcSettingsTimer = undefined }
    if (this.frameLoopRaf) { window.cancelAnimationFrame(this.frameLoopRaf); this.frameLoopRaf = undefined }
    this.frameAbortController?.abort()
    this.frameAbortController = undefined
    this.webRtcEventSource?.close()
    this.webRtcEventSource = undefined
    const pcId = this.webRtcPcId
    if (pcId) void fetch(this.backendUrl(`/api/rtc/peers/${pcId}`), { method: 'DELETE' }).catch(() => {})
    this.webRtcPcId = ''
    this.webRtcLastSettings = ''
    this.isStreaming = false
    this.realtimeVideoActive = false
    this.status = 'offline'
    this.displayFps = 0
    this.realtimeFrameTimes = []
  }

  private stopStream() {
    if (this.webRtcPcId) { this.stopWebRtc(); return }
    if (this.streamTimer) window.clearInterval(this.streamTimer)
    if (this.streamRaf) { window.cancelAnimationFrame(this.streamRaf); this.streamRaf = undefined }
    if (this.streamTransitionAnimation) window.cancelAnimationFrame(this.streamTransitionAnimation)
    this.streamTransitionAnimation = undefined
    this.streamSendQueued = false
    this.socket?.close()
    this.awaitingFrame = false
    this.isStreaming = false
    this.status = 'offline'
    this.realtimeVideoActive = false
    this.displayFps = 0
    this.realtimeFrameTimes = []
    this.streamVideoMedia?.getTracks().forEach((track) => track.stop())
    this.streamVideoMedia = undefined
    if (this.streamOutputImgElement) {
      const prev = this.streamOutputImgElement.dataset.blobUrl
      if (prev) { URL.revokeObjectURL(prev); delete this.streamOutputImgElement.dataset.blobUrl }
      this.streamOutputImgElement.src = ''
    }
  }

  private setupRealtimeVideoOutput() {
    const canvas = this.streamOutputCanvasElement
    if (!canvas) return
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    this.streamVideoMedia?.getTracks().forEach((track) => track.stop())
    this.streamVideoMedia = canvas.captureStream(Math.max(1, Math.min(60, Math.round(1000 / 28))))
  }

  private async drawRealtimeBinaryFrame(buffer: ArrayBuffer) {
    try {
      const blob = new Blob([buffer], { type: this.streamFrameContentType })

      // Update the visible <img> element directly via blob URL (SDXL/Z-Image path).
      // The SSE path sets img.src to a data URL; we mirror that here for WebSocket binary frames.
      const imgEl = this.streamOutputImgElement
      if (imgEl) {
        const url = URL.createObjectURL(blob)
        const prev = imgEl.dataset.blobUrl
        imgEl.src = url
        imgEl.dataset.blobUrl = url
        if (prev) URL.revokeObjectURL(prev)
      }

      // Also draw to the off-screen canvas (supports captureStream if ever wired up)
      const canvas = this.streamOutputCanvasElement
      if (canvas) {
        const bitmap = await createImageBitmap(blob)
        const context = canvas.getContext('2d')!
        this.drawRealtimeFrameWithTransition(context, canvas, bitmap)
        bitmap.close()
      }

      this.updateRealtimeDisplayFps()
    } catch (error) {
      this.reportError(error instanceof Error ? error.message : String(error))
    } finally {
      this.awaitingFrame = false
    }
  }

  private drawRealtimeFrameWithTransition(context: CanvasRenderingContext2D, canvas: HTMLCanvasElement, bitmap: ImageBitmap) {
    if (this.streamTransitionAnimation) {
      window.cancelAnimationFrame(this.streamTransitionAnimation)
      this.streamTransitionAnimation = undefined
    }
    context.clearRect(0, 0, canvas.width, canvas.height)
    context.drawImage(bitmap, 0, 0, canvas.width, canvas.height)
  }

  private updateRealtimeDisplayFps() {
    const now = performance.now()
    this.realtimeFrameTimes.push(now)
    const windowStart = now - 2000
    while (this.realtimeFrameTimes.length && this.realtimeFrameTimes[0] < windowStart) this.realtimeFrameTimes.shift()
    const elapsed = this.realtimeFrameTimes.length > 1 ? this.realtimeFrameTimes[this.realtimeFrameTimes.length - 1] - this.realtimeFrameTimes[0] : 0
    this.displayFps = elapsed > 0 ? ((this.realtimeFrameTimes.length - 1) * 1000) / elapsed : 0
  }

  private toggleCausalStream = () => {
    if (this.isGeneratingMotion) this.stopMotionGeneration()
    else void this.startCausalStream()
  }

  private startCausalStream = async (modelOverride?: string, modelPathOverride?: string) => {
    let sourceImage = this.outputImage || this.exportCanvasImage(true)
    const model = modelOverride || this.motionModel
    if (!sourceImage) {
      this.reportError('Create or generate an image before starting the video model.')
      return
    }
    if (!(await this.ensureVideoBackendReady(model))) return
    const runId = ++this.motionRunId
    const singleClip = model === 'fastvideo'
    const separateVideoDevice = Boolean(this.videoDevice && this.realtimeDevice && this.videoDevice !== this.realtimeDevice)
    if (!separateVideoDevice) this.stopStream()
    this.clearMotionPlayback()
    this.activeMotionFrame = sourceImage
    const useVideoStream = singleClip ? false : await this.openMotionVideoStream()
    this.isGeneratingMotion = true
    this.motionChunkIndex = 0
    this.motionTask = {
      task_id: 'pending',
      status: 'queued',
      phase: 'queued',
      progress: 0,
        message: singleClip ? `Waiting for ${model} worker` : `Waiting for ${model} chunk worker`,
    }
    try {
      while (this.isGeneratingMotion && this.motionRunId === runId) {
        this.motionChunkIndex += 1
        this.motionTask = {
          task_id: 'pending',
          status: 'queued',
          phase: 'queued',
          progress: 0,
          message: singleClip ? `Queueing ${model} clip` : `Queueing ${model} chunk ${this.motionChunkIndex}`,
        }
        const response = await fetch(this.backendUrl('/motion/tasks'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: `${model} stream ${this.motionChunkIndex}`,
            source_image: sourceImage,
            prompt: this.motionPrompt,
            arrival_prompt: this.motionArrivalPrompt,
            negative_prompt: this.motionNegativePrompt,
            model_path: modelPathOverride || this.selectedModel || undefined,
            device: this.videoDevice || undefined,
            lora_paths: [...this.selectedLoras],
            model,
            motion: 'autoregressive',
            region: 'full',
            fps: this.motionFps,
            duration_seconds: this.motionDuration,
            loop: 'none',
            intensity: this.motionIntensity,
            width: this.stageWidth,
            height: this.stageHeight,
            arrival_steps: Math.max(1, Math.min(32, this.layerGenerationSteps)),
            arrival_strength: Math.max(0, Math.min(0.95, this.strength || 0.42)),
            arrival_cfg: Math.max(0, Math.min(30, this.layerGenerationCfg || 4.5)),
            arrival_sampler: this.layerGenerationSampler,
            arrival_scheduler: this.layerGenerationScheduler,
          }),
        })
        if (!response.ok) throw new Error(await response.text())
        const { task_id: taskId } = (await response.json()) as { task_id: string }
        const clip = await this.watchMotionTask(taskId)
        if (!clip || !this.isGeneratingMotion || this.motionRunId !== runId) break
        await this.playMotionResult(clip, useVideoStream)
        if (singleClip) break
        sourceImage = (await this.motionClipLastFrameDataUrl(clip)) || sourceImage
      }
    } catch (error) {
      this.reportError(error instanceof Error ? error.message : String(error))
    } finally {
      if (this.motionRunId === runId) {
        this.isGeneratingMotion = false
        this.motionTaskSocket = undefined
      }
    }
  }

  private async ensureVideoBackendReady(model: string) {
    if (!this.videoCapabilities) await this.loadVideoCapabilities()
    const capability = this.videoCapability(model)
    if (!capability || capability.configured) return true
    this.reportError(capability.message)
    return false
  }

  private videoCapability(model: string) {
    return this.videoCapabilities?.models?.[model]
  }

  private rendererCapability(renderer: ScenePanelTab) {
    return this.rendererCapabilities?.renderers?.[renderer]
  }

  private renderRendererStatus(renderer: ScenePanelTab) {
    const capability = this.rendererCapability(renderer)
    if (!capability) return html`<div class="empty">Renderer status pending</div>`
    const state = capability.configured ? (capability.native ? 'native' : 'fallback') : 'missing'
    const flags = [capability.streamable ? 'streamable' : '', capability.realtime ? 'realtime' : '', capability.transport].filter(Boolean).join(' / ')
    const accelerators = capability.accelerators ? Object.entries(capability.accelerators).filter(([, enabled]) => enabled).map(([name]) => name).join(', ') : ''
    return html`<div class="empty">${state} | ${capability.runtime} | ${flags}${accelerators ? ` | ${accelerators}` : ''}<br />${capability.message}</div>`
  }

  private videoCapabilityReady(model: string) {
    return this.videoCapability(model)?.configured ?? true
  }

  private videoCapabilityDisabled(model: string) {
    return this.videoCapability(model)?.configured === false
  }

  private watchMotionTask(taskId: string) {
    return new Promise<MotionClip | undefined>((resolve, reject) => {
      const socket = new WebSocket(this.backendWsUrl(`/ws/motion/${taskId}`))
      this.motionTaskSocket = socket
      let result: MotionClip | undefined
      const finish = () => resolve(result)
      socket.onmessage = (event) => {
        const progress = JSON.parse(event.data) as MotionTaskProgress
        this.motionTask = progress
        this.status = `${progress.phase} ${Math.round(progress.progress * 100)}%`
        const previewFrame = progress.preview_frames?.[progress.preview_frames.length - 1]
        if (previewFrame) {
          this.activeMotionFrame = this.backendUrl(previewFrame.url, { t: String(Date.now()) })
        }
        if (progress.result) {
          result = progress.result
        }
        if (progress.status === 'complete') {
          socket.close()
          finish()
        }
        if (progress.status === 'error') {
          socket.close()
          reject(new Error(progress.error || progress.message))
        }
      }
      socket.onerror = () => {
        reject(new Error('Motion task monitor connection failed.'))
      }
      socket.onclose = () => {
        if (this.motionTaskSocket === socket) this.motionTaskSocket = undefined
      }
    })
  }

  private async playMotionResult(clip: MotionClip, appendToStream = false) {
    if (appendToStream && clip.video_url && (await this.appendMotionClipToVideoStream(clip))) return
    this.activeMotionFrame = ''
    this.activeMotionVideo = ''
    this.activeMotionIsStream = false
    if (clip.video_url) {
      this.activeMotionVideo = this.backendUrl(clip.video_url, { t: String(Date.now()) })
      return
    }
    const frame = clip.frames[0]
    if (frame) this.activeMotionFrame = this.backendUrl(frame.url)
  }

  private async motionClipLastFrameDataUrl(clip: MotionClip) {
    const frame = clip.frames.at(-1)
    if (!frame) return ''
    const response = await fetch(this.backendUrl(frame.url, { t: String(Date.now()) }))
    if (!response.ok) return ''
    const blob = await response.blob()
    return await new Promise<string>((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(String(reader.result || ''))
      reader.onerror = () => reject(reader.error || new Error('Could not read generated frame.'))
      reader.readAsDataURL(blob)
    })
  }

  private startFastVideoStream = () => void this.startCausalStream('fastvideo', this.fastVideoModelPath || undefined)

  private renderSanaPanel() {
    const task = this.sanaVideoTask
    const busy = task !== null && task.status !== 'complete' && task.status !== 'error'
    const done = task?.status === 'complete'
    const percent = Math.round((task?.progress || 0) * 100)
    return html`<div class="layer-tab scene-tab">
      <label class="field">
        <span>Prompt</span>
        <textarea rows="3" .value=${this.prompt} @input=${(e: Event) => (this.prompt = (e.target as HTMLTextAreaElement).value)}></textarea>
      </label>
      <label class="field">
        <span>Negative prompt</span>
        <textarea rows="2" .value=${this.negativePrompt} @input=${(e: Event) => (this.negativePrompt = (e.target as HTMLTextAreaElement).value)}></textarea>
      </label>

      <div class="section-header" style="font-weight:600;margin:8px 0 4px">SANA Sprint — Realtime</div>
      <label class="field inline">
        <span>Model</span>
        <select .value=${this.streamRuntimePreset.startsWith('sana') ? this.streamRuntimePreset : 'sana-sprint-0.6b'}
          @change=${(e: Event) => this.applyStreamRuntimePreset((e.target as HTMLSelectElement).value)}>
          <option value="sana-sprint-0.6b">SANA Sprint 0.6B (fastest, ~7-10 FPS)</option>
          <option value="sana-sprint-1.6b">SANA Sprint 1.6B (quality, ~3-5 FPS)</option>
        </select>
      </label>
      ${this.numberControl('Steps', this.sanaSteps, 1, 8, 1, 0, (v) => (this.sanaSteps = Math.round(this.clamp(v, 1, 8, this.sanaSteps))))}
      ${this.numberControl('Strength', this.strength, 0, 0.999, 0.01, 2, (v) => this.setSceneSamplerValue('strength', this.clamp(v, 0, 0.999, this.strength)))}
      ${this.numberControl('CFG', this.cfg, 0, 20, 0.1, 1, (v) => this.setSceneSamplerValue('cfg', this.clamp(v, 0, 20, this.cfg)))}
      <div class="button-grid">
        <button class="stream" @click=${this.toggleStream} ?disabled=${this.status === 'connecting'}>
          ${this.status === 'connecting' ? 'Connecting…' : this.isStreaming ? 'Stop SANA Sprint' : 'Start SANA Sprint'}
        </button>
        ${this.isStreaming ? html`<div class="stream-status-line">
          ${this.status === 'connecting' ? '⏳ Connecting…'
            : (this.displayFps === 0 && this.fps === 0) ? '⚙ Loading model…'
            : `● ${this.backendMode || 'Streaming'}`}
        </div>` : ''}
      </div>

      <div class="section-header" style="font-weight:600;margin:12px 0 4px">SANA Video — Offline Generation</div>
      <label class="field inline">
        <span>Model</span>
        <select .value=${this.sanaVideoModel} @change=${(e: Event) => (this.sanaVideoModel = (e.target as HTMLSelectElement).value)}>
          <option value="480p">SANA-Video 2B 480p</option>
          <option value="720p">SANA-Video 2B 720p (HQ)</option>
          <option value="longlive">SANA-Video LongLive 480p (streaming chunks)</option>
          <option value="wm">SANA-WM 2.6B (6-DoF camera control)</option>
        </select>
      </label>
      <div class="resolution-grid">
        ${this.numberControl('Frames', this.sanaVideoNumFrames, 17, 201, 4, 0, (v) => (this.sanaVideoNumFrames = Math.round(this.clamp(v, 17, 201, this.sanaVideoNumFrames))))}
        ${this.numberControl('Steps', this.sanaVideoSteps, 1, 50, 1, 0, (v) => (this.sanaVideoSteps = Math.round(this.clamp(v, 1, 50, this.sanaVideoSteps))))}
        ${this.numberControl('Guidance', this.sanaVideoGuidance, 1, 20, 0.5, 1, (v) => (this.sanaVideoGuidance = this.clamp(v, 1, 20, this.sanaVideoGuidance)))}
      </div>
      <div class="button-grid">
        <button class="stream" @click=${this.startSanaVideoTask} ?disabled=${busy}>
          ${busy ? `Generating… ${percent}%` : 'Generate SANA video'}
        </button>
      </div>
      ${task ? html`
        <div class=${task.status === 'error' ? 'task-item error-task' : 'task-item'}>
          <div class="task-row">
            <strong>SANA video</strong>
            <span>${task.status}</span>
          </div>
          <div class="task-bar" aria-label=${`SANA video ${percent}%`}><span style="width:${percent}%"></span></div>
          <div class="task-row task-detail"><span>${task.phase || task.message || ''}</span><span>${percent}%</span></div>
          ${task.error ? html`<p style="color:var(--error,#f55)">${task.error}</p>` : ''}
          ${done && task.download_url ? html`<a href=${this.backendUrl(task.download_url)} target="_blank" download>
            Download ${this.sanaVideoModel} video
          </a>` : ''}
        </div>` : ''}
    </div>`
  }

  private startSanaVideoTask = async () => {
    const canvasImg = this.exportCanvasImage(true, 'image/jpeg')
    if (!canvasImg) return
    this.sanaVideoTask = { status: 'queued', progress: 0, phase: 'Queuing…' }
    try {
      const resp = await fetch(this.backendUrl('/sana/video/tasks'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: this.sanaVideoModel,
          prompt: this.prompt,
          negative_prompt: this.negativePrompt,
          image: canvasImg,
          num_frames: this.sanaVideoNumFrames,
          guidance_scale: this.sanaVideoGuidance,
          num_inference_steps: this.sanaVideoSteps,
          width: this.stageWidth,
          height: this.stageHeight,
        }),
      })
      if (!resp.ok) throw new Error(await resp.text())
      const data = await resp.json() as { task_id: string }
      this.sanaVideoTaskId = data.task_id
      void this.pollSanaVideoTask(data.task_id)
    } catch (err) {
      this.sanaVideoTask = { status: 'error', progress: 0, error: String(err) }
    }
  }

  private pollSanaVideoTask = async (taskId: string) => {
    while (true) {
      await new Promise((r) => setTimeout(r, 1500))
      if (this.sanaVideoTaskId !== taskId) return
      try {
        const resp = await fetch(this.backendUrl(`/sana/video/tasks/${taskId}`))
        if (!resp.ok) break
        const data = await resp.json() as Record<string, any>
        this.sanaVideoTask = data
        if (data.status === 'complete' || data.status === 'done' || data.status === 'error') return
      } catch {
        break
      }
    }
  }

  private videoModelAssets() {
    return this.assets.video_models?.length ? this.assets.video_models : this.assets.models.filter((model) => /wan|fastvideo|ltx|hunyuan|cogvideo/i.test(`${model.name} ${model.path}`))
  }

  private stopMotionGeneration() {
    this.motionRunId += 1
    this.motionTaskSocket?.close()
    this.motionTaskSocket = undefined
    this.isGeneratingMotion = false
    this.finishMotionVideoStream()
    this.status = this.motionChunkIndex ? `stopped after ${this.motionChunkIndex} chunk(s)` : 'offline'
  }

  private clearMotionPlayback() {
    this.motionTaskSocket?.close()
    this.motionTaskSocket = undefined
    this.disposeMotionVideoStream()
    this.activeMotionFrame = ''
    this.activeMotionVideo = ''
    this.activeMotionIsStream = false
  }

  private motionVideoMimeType() {
    const candidates = [
      'video/mp4; codecs="avc1.42E01E"',
      'video/mp4; codecs="avc1.42E01F"',
      'video/mp4',
    ]
    return candidates.find((mime) => MediaSource.isTypeSupported(mime)) || ''
  }

  private async openMotionVideoStream() {
    if (!('MediaSource' in window)) return false
    const mimeType = this.motionVideoMimeType()
    if (!mimeType) return false
    const mediaSource = new MediaSource()
    this.motionMediaSource = mediaSource
    this.motionSegmentQueue = []
    this.motionStreamObjectUrl = URL.createObjectURL(mediaSource)
    this.activeMotionVideo = this.motionStreamObjectUrl
    this.activeMotionIsStream = true
    this.motionStreamHasSegments = false
    this.motionStreamOpen = new Promise<boolean>((resolve) => {
      mediaSource.addEventListener('sourceopen', () => {
        if (this.motionMediaSource !== mediaSource) {
          resolve(false)
          return
        }
        try {
          const sourceBuffer = mediaSource.addSourceBuffer(mimeType)
          sourceBuffer.mode = 'sequence'
          sourceBuffer.addEventListener('updateend', () => this.flushMotionSegmentQueue())
          sourceBuffer.addEventListener('error', () => this.reportError('Video stream buffer failed.'))
          this.motionSourceBuffer = sourceBuffer
          resolve(true)
        } catch (error) {
          this.reportError(error instanceof Error ? error.message : String(error))
          resolve(false)
        }
      }, { once: true })
    })
    return await this.motionStreamOpen
  }

  private async appendMotionClipToVideoStream(clip: MotionClip) {
    if (!clip.video_url || !this.motionMediaSource || !this.motionSourceBuffer) return false
    if (this.motionStreamOpen && !(await this.motionStreamOpen)) return false
    try {
      const response = await fetch(this.backendUrl(clip.video_url, { t: String(Date.now()) }))
      if (!response.ok) return false
      this.motionSegmentQueue.push(await response.arrayBuffer())
      this.motionStreamHasSegments = true
      this.flushMotionSegmentQueue()
      return true
    } catch {
      return false
    }
  }

  private flushMotionSegmentQueue() {
    const sourceBuffer = this.motionSourceBuffer
    const mediaSource = this.motionMediaSource
    if (!sourceBuffer || !mediaSource || mediaSource.readyState !== 'open' || sourceBuffer.updating) return
    const next = this.motionSegmentQueue.shift()
    if (!next) return
    try {
      sourceBuffer.appendBuffer(next)
    } catch (error) {
      this.motionSegmentQueue = []
      this.reportError(error instanceof Error ? error.message : String(error))
    }
  }

  private finishMotionVideoStream() {
    const mediaSource = this.motionMediaSource
    if (!mediaSource || mediaSource.readyState !== 'open' || this.motionSourceBuffer?.updating || this.motionSegmentQueue.length) return
    try {
      mediaSource.endOfStream()
    } catch {
      // The browser may already have closed the MediaSource.
    }
  }

  private disposeMotionVideoStream() {
    this.motionSegmentQueue = []
    this.motionStreamHasSegments = false
    this.motionSourceBuffer = undefined
    this.motionMediaSource = undefined
    this.motionStreamOpen = undefined
    if (this.motionStreamObjectUrl) {
      URL.revokeObjectURL(this.motionStreamObjectUrl)
      this.motionStreamObjectUrl = ''
    }
  }

  private backendHost() {
    return window.location.hostname || '127.0.0.1'
  }

  private backendPort() {
    const queryPort = new URLSearchParams(window.location.search).get('backendPort')
    const port = queryPort || '8000'
    if (!queryPort) window.localStorage.removeItem('rtdiffusion.backendPort')
    return /^\d{2,5}$/.test(port) ? port : '8000'
  }

  private backendUrl(path: string, params?: Record<string, string>) {
    const url = new URL(path, `${window.location.protocol}//${this.backendHost()}:${this.backendPort()}`)
    for (const [key, value] of Object.entries(params ?? {})) url.searchParams.set(key, value)
    return url.toString()
  }

  private backendWsUrl(path: string) {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    return `${protocol}://${this.backendHost()}:${this.backendPort()}${path}`
  }

  private restartLatentReuse = () => {
    if (!this.sceneReuseLatent) return
    this.sceneReuseLatentRestartNext = true
    this.backendMode = 'latent reuse restart queued'
  }

  private sendFrame() {
    const streamMode = this.scenePanelTab === 'streamdiffusion'
    // StreamDiffusion: drop frame if socket not drained — prevents stale frame queue buildup.
    // Other renderers: backpressure (don't send until previous response arrives).
    if (!streamMode && this.awaitingFrame) return
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN || !this.fabricCanvas) return
    if (streamMode && this.socket.bufferedAmount > 0) return
    if (!streamMode) this.awaitingFrame = true
    const stockRenderer = this.scenePanelTab === 'sdxl' || this.scenePanelTab === 'z-image'
    const restartLatentReuse = stockRenderer && this.sceneReuseLatent && this.sceneReuseLatentRestartNext
    this.socket.send(
      JSON.stringify({
        prompt: this.prompt,
        response_mode: 'binary',
        negative_prompt: this.negativePrompt,
        model_path: this.selectedModel || undefined,
        device: this.realtimeDevice || undefined,
        lora_paths: [...this.selectedLoras],
        strength: this.strength,
        cfg: this.cfg,
        steps: this.steps,
        sampler: this.sceneSampler || undefined,
        scheduler: this.sceneScheduler || undefined,
        seed: stockRenderer ? this.sceneSeedValue() : undefined,
        seed_mode: stockRenderer ? this.sceneSeedMode : 'fixed',
        reuse_previous_latent: stockRenderer ? this.sceneReuseLatent : false,
        latent_reuse_denoise: stockRenderer ? this.sceneReuseLatentDenoise : 0.24,
        latent_reuse_noise: stockRenderer ? this.sceneReuseLatentNoise : 0,
        latent_reuse_noise_mode: stockRenderer ? this.sceneReuseLatentNoiseMode : 'none',
        latent_reuse_restart: restartLatentReuse,
        stream_diffusion: this.scenePanelTab === 'streamdiffusion',
        stream_direct: this.scenePanelTab === 'streamdiffusion',
        stream_timestep_indices: this.streamTimestepIndexList(),
        stream_frame_buffer_size: this.streamFrameBufferSize,
        stream_cfg_type: this.streamCfgType,
        stream_similarity_threshold: this.streamSimilarityThreshold,
        stream_max_skip_frames: this.streamMaxSkipFrames,
        transparent_background: stockRenderer ? this.sceneTransparentBackground : false,
        transparent_background_mode: this.sceneTransparentMode,
        transparent_background_color: this.sceneTransparentColor,
        transparent_background_tolerance: this.sceneTransparentTolerance,
        transparent_alpha_blur: this.sceneTransparentAlphaBlur,
        transparent_alpha_threshold: this.sceneTransparentAlphaThreshold,
        width: this.stageWidth,
        height: this.stageHeight,
        image: this.exportRealtimeCanvasImage(),
        mask: this.exportRealtimeMask(),
        layer_conditions: this.exportLayerConditions(),
        debug_streams: this.debugStreamsEnabled,
      }),
    )
    if (restartLatentReuse) this.sceneReuseLatentRestartNext = false
  }

  private queueLiveStreamFrame() {
    if (!this.isStreaming || this.streamSendQueued) return
    this.streamSendQueued = true
    window.requestAnimationFrame(() => {
      this.streamSendQueued = false
      this.sendFrame()
    })
  }

  private sceneSeedValue() {
    const value = Math.round(Number(this.sceneSeed))
    if (!Number.isFinite(value) || value < 0) return undefined
    return Math.min(4294967295, value)
  }

  private streamTimestepIndexList() {
    const values = this.streamTimestepIndices
      .split(',')
      .map((part) => Math.round(Number(part.trim())))
      .filter((value) => Number.isFinite(value) && value >= 0 && value <= 999)
    return values.length ? values.slice(0, 16) : [32, 45]
  }

  private exportCanvasImage(hideSelection = false, format = 'image/png') {
    if (!this.fabricCanvas) return ''
    const exportImage = () => this.exportCompositedFabricCanvas(format)
    return hideSelection ? this.withSelectionHidden(exportImage) : exportImage()
  }

  private exportCompositedFabricCanvas(format = 'image/png') {
    const lowerCanvas = this.fabricCanvas?.getElement()
    if (!lowerCanvas) return ''
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    const context = canvas.getContext('2d')!
    context.drawImage(lowerCanvas, 0, 0, this.stageWidth, this.stageHeight)
    const upperCanvas = (this.fabricCanvas as (Canvas & { upperCanvasEl?: HTMLCanvasElement }) | undefined)?.upperCanvasEl
    if (upperCanvas && this.pendingPaintLayerId) {
      context.save()
      context.globalCompositeOperation = this.isEraseStroke(this.pendingBrushButton) ? 'destination-out' : 'source-over'
      context.drawImage(upperCanvas, 0, 0, this.stageWidth, this.stageHeight)
      context.restore()
    }
    return canvas.toDataURL(format, 0.88)
  }

  private exportRealtimeCanvasImage(format = 'image/jpeg') {
    const transform = this.currentStreamMotionTransform()
    // Never call withSelectionHidden during streaming — it emits renderAll() twice per frame
    // which disrupts Fabric.js drag state and cancels in-progress layer moves.
    if (!transform) return this.exportCompositedFabricCanvas(format)
    return this.exportTransformedRealtimeSource(transform, format)
  }

  private exportTransformedRealtimeSource(transform: [number, number, number, number, number, number], format = 'image/png') {
    const source = this.outputImage || this.exportCompositedFabricCanvas()
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    const context = canvas.getContext('2d')!
    const videoCanvas = this.streamOutputCanvasElement
    const image = this.streamFeedbackImage?.complete && this.streamFeedbackImage.naturalWidth > 0 ? this.streamFeedbackImage : source ? this.loadedImageElement(source) : undefined
    const drawable = videoCanvas && videoCanvas.width > 0 && videoCanvas.height > 0 ? videoCanvas : image
    if (!drawable) return this.exportCompositedFabricCanvas(format)
    if (drawable instanceof HTMLImageElement && (!drawable.complete || drawable.naturalWidth <= 0)) return source || this.exportCompositedFabricCanvas(format)
    context.setTransform(...transform)
    context.drawImage(drawable, 0, 0, this.stageWidth, this.stageHeight)
    context.resetTransform()
    return canvas.toDataURL(format, 0.88)
  }

  private loadedImageElement(source: string) {
    const image = new Image()
    image.decoding = 'sync'
    image.src = source
    return image
  }

  private currentStreamMotionTransform(): [number, number, number, number, number, number] | undefined {
    if (this.scenePanelTab !== 'streamdiffusion' || this.streamMotionMode === 'none' || this.streamMotionIntensity <= 0) return undefined
    const elapsedSeconds = Math.max(0, (performance.now() - this.streamStartedAt) / 1000)
    const phase = elapsedSeconds * this.streamMotionSpeed * Math.PI * 2
    const intensity = this.streamMotionIntensity
    const swayX = this.stageWidth * 0.026 * intensity
    const swayY = this.stageHeight * 0.018 * intensity
    let dx = 0
    let dy = 0
    let scale = 1
    let angle = 0
    if (this.streamMotionMode === 'sway') {
      dx = Math.sin(phase) * swayX
      dy = Math.sin(phase * 1.7) * swayY
      angle = Math.sin(phase * 0.7) * 0.006 * intensity
    } else if (this.streamMotionMode === 'orbit') {
      dx = Math.cos(phase) * swayX
      dy = Math.sin(phase) * swayY
      angle = Math.sin(phase) * 0.004 * intensity
    } else if (this.streamMotionMode === 'push') {
      dx = Math.sin(phase) * swayX * 1.6
      scale = 1 + (0.008 + 0.006 * Math.sin(phase * 0.5)) * intensity
    } else if (this.streamMotionMode === 'zoom') {
      scale = 1 + (0.014 + 0.01 * Math.sin(phase)) * intensity
    }
    const centerX = this.stageWidth / 2
    const centerY = this.stageHeight / 2
    const cos = Math.cos(angle) * scale
    const sin = Math.sin(angle) * scale
    return [cos, sin, -sin, cos, centerX + dx - cos * centerX + sin * centerY, centerY + dy - sin * centerX - cos * centerY]
  }

  private withSelectionHidden<T>(callback: () => T): T {
    if (!this.fabricCanvas) return callback()
    const fc = this.fabricCanvas as unknown as { _activeObject: unknown; _currentTransform: unknown }
    const active = fc._activeObject
    if (!active) return callback()
    // If a drag/transform is currently in progress, skip hiding the selection.
    // Manipulating _activeObject mid-transform causes Fabric.js to lose the dragged
    // object reference and cancels the move — making continuous layer dragging impossible.
    if (fc._currentTransform) return callback()
    // Bypass the public API (which emits selection:cleared/created) to avoid
    // disrupting in-progress drag transforms during streaming.
    fc._activeObject = undefined
    this.fabricCanvas.renderAll()
    try {
      return callback()
    } finally {
      fc._activeObject = active
      this.fabricCanvas.renderAll()
    }
  }

  private exportLayerConditions() {
    const objects = this.fabricCanvas?.getObjects() ?? []
    const conditions = []
    const activeLayers = this.layers.filter((layer) => layer.visible && layer.preset.samplingEnabled)
    const schedules = this.autoLayerSchedules(activeLayers)
    for (const layer of activeLayers) {
      const object = objects.find((candidate) => (candidate as FabricObject & { __uid?: string }).__uid === layer.id)
      if (!object) continue
      const timing = schedules.get(layer.id) ?? { start: 0, end: 1, schedule: 'auto' as RegionSchedule, active: true }
      if (!timing.active) continue
      for (const region of this.layerRegionSpecs(layer)) {
        const prompt = region.inherited ? layer.preset.prompt : region.prompt
        const negativePrompt = region.inherited ? layer.preset.negativePrompt : region.negativePrompt
        const regionTiming = this.resolvedRegionTiming(region, timing)
        const cn = this.layerControlNetConfigs.get(layer.id)
        const cnPayload = cn?.enabled ? {
          controlnet_model: cn.model,
          controlnet_model_path: cn.modelPath || undefined,
          controlnet_scale: cn.scale,
          controlnet_start_at: cn.startAt,
          controlnet_end_at: cn.endAt,
          controlnet_use_layer_frame: cn.useLayerFrame,
          controlnet_image: cn.useLayerFrame ? undefined : cn.usePreprocessor ? (cn.sourceImageUrl || undefined) : (cn.explicitImageUrl || undefined),
          controlnet_preprocessor: cn.useLayerFrame ? cn.usePreprocessor : cn.usePreprocessor,
          controlnet_preprocessor_params: {
            canny_low: cn.preprocessorParams.cannyLow,
            canny_high: cn.preprocessorParams.cannyHigh,
            openpose_hands: cn.preprocessorParams.openposeHands,
            openpose_face: cn.preprocessorParams.openposeFace,
            mlsd_thr_v: cn.preprocessorParams.mlsdThrV,
            mlsd_thr_d: cn.preprocessorParams.mlsdThrD,
            lineart_coarse: cn.preprocessorParams.linartCoarse,
          },
        } : {}
        conditions.push({
          layer_id: layer.id,
          name: region.name || layer.name,
          prompt: layer.preset.autoTag ? '' : prompt,
          negative_prompt: negativePrompt,
          auto_tag: layer.preset.autoTag,
          auto_tag_threshold: layer.preset.autoTagThreshold,
          auto_tag_refresh_frames: layer.preset.autoTagRefreshFrames,
          image: this.exportLayerRegionConditionImage(object, layer.id, region),
          weight: region.inherited ? layer.preset.conditionWeight : region.maskStrength,
          mode: layer.preset.conditionMode,
          mask_operator: region.inherited ? layer.preset.maskOperator : region.maskOperator,
          denoise_operator: region.inherited ? layer.preset.denoiseOperator : region.denoiseOperator,
          denoise: region.inherited ? layer.preset.strength : region.denoise,
          schedule: regionTiming.schedule,
          schedule_start: regionTiming.start,
          schedule_end: regionTiming.end,
          primary_input: this._isVideoLayer(layer.id),
          ...cnPayload,
        })
      }
    }
    return conditions
  }

  private exportPipelineNodes() {
    const nodes: object[] = []
    const activeLayers = this.layers.filter((layer) => layer.visible && layer.preset.samplingEnabled)
    for (const layer of activeLayers) {
      if (layer.preset.autoTag) {
        nodes.push({
          id: `tagger_${layer.id}`,
          type: 'tagger',
          layer_id: layer.id,
          tagger_model: 'wd-eva02-large-v3',
          tagger_threshold: layer.preset.autoTagThreshold,
          tagger_refresh_frames: layer.preset.autoTagRefreshFrames,
        })
      }
      const cn = this.layerControlNetConfigs.get(layer.id)
      if (cn?.enabled && cn.useLayerFrame) {
        nodes.push({
          id: `cn_from_layer_${layer.id}`,
          type: 'cn_from_layer',
          layer_id: layer.id,
          cn_model: cn.model,
          cn_model_path: cn.modelPath || undefined,
          cn_scale: cn.scale,
          cn_start: cn.startAt,
          cn_end: cn.endAt,
          cn_preprocessor_params: {
            canny_low: cn.preprocessorParams.cannyLow,
            canny_high: cn.preprocessorParams.cannyHigh,
            openpose_hands: cn.preprocessorParams.openposeHands,
            openpose_face: cn.preprocessorParams.openposeFace,
            mlsd_thr_v: cn.preprocessorParams.mlsdThrV,
            mlsd_thr_d: cn.preprocessorParams.mlsdThrD,
            lineart_coarse: cn.preprocessorParams.linartCoarse,
          },
        })
      }
    }
    return nodes
  }

  private exportLayerFrames(): Record<string, string> {
    const frames: Record<string, string> = {}
    const activeLayers = this.layers.filter((layer) => layer.visible && layer.preset.samplingEnabled)
    const needsLayerFrames = activeLayers.some((l) =>
      l.preset.autoTag ||
      this._isVideoLayer(l.id) ||
      (this.layerControlNetConfigs.get(l.id)?.enabled && this.layerControlNetConfigs.get(l.id)?.useLayerFrame)
    )
    if (!needsLayerFrames) return frames

    const objects = this.fabricCanvas?.getObjects() ?? []
    for (const layer of activeLayers) {
      if (this._isVideoLayer(layer.id)) {
        // Read directly from the offscreen canvas to avoid Fabric.js async rendering race
        const src = this._videoOffscreenCanvas
        if (!src) continue
        const canvas = document.createElement('canvas')
        canvas.width = this.stageWidth
        canvas.height = this.stageHeight
        const ctx = canvas.getContext('2d')!
        ctx.drawImage(src, 0, 0, canvas.width, canvas.height)
        frames[layer.id] = canvas.toDataURL('image/jpeg', 0.90)
        continue
      }
      const object = objects.find((candidate) => (candidate as FabricObject & { __uid?: string }).__uid === layer.id)
      if (!object) continue
      const canvas = document.createElement('canvas')
      canvas.width = this.stageWidth
      canvas.height = this.stageHeight
      const ctx = canvas.getContext('2d')!
      this.drawLayerCondition(ctx, object, layer.id)
      frames[layer.id] = canvas.toDataURL('image/jpeg', 0.82)
    }
    return frames
  }

  private autoLayerSchedules(layers: LayerItem[]) {
    const schedules = new Map<string, { start: number; end: number; schedule: RegionSchedule; active: boolean }>()
    if (!layers.length) return schedules
    const sorted = [...layers].sort((first, second) => this.layerStackIndex(first.id) - this.layerStackIndex(second.id))
    const topOpaque = [...sorted].reverse().find((layer) => {
      const object = this.findLayer(layer.id)
      return (object?.opacity ?? layer.opacity) >= 0.98 && this.layerAlphaCoverage(layer.id) >= 0.92
    })
    if (topOpaque) {
      for (const layer of layers) {
        const hasExplicitMasks = this.regions.some((region) => region.target === 'layer' && region.layerId === layer.id)
        schedules.set(layer.id, { start: 0, end: 1, schedule: 'auto', active: layer.id === topOpaque.id || hasExplicitMasks })
      }
      return schedules
    }
    const overlap = 0.08
    const span = 1 / sorted.length
    sorted.forEach((layer, index) => {
      const start = Math.max(0, index * span - (index ? overlap : 0))
      const end = Math.min(1, (index + 1) * span + (index < sorted.length - 1 ? overlap : 0))
      schedules.set(layer.id, { start, end, schedule: 'auto', active: true })
    })
    return schedules
  }

  private layerStackIndex(layerId: string) {
    const objects = this.fabricCanvas?.getObjects() ?? []
    const index = objects.findIndex((candidate) => (candidate as FabricObject & { __uid?: string }).__uid === layerId)
    return index < 0 ? this.layers.findIndex((layer) => layer.id === layerId) : index
  }

  private layerAlphaCoverage(layerId: string) {
    const cached = this.layerAlphaCoverageCache.get(layerId)
    const now = performance.now()
    if (cached && now - cached.at < 750) return cached.value
    const object = this.findLayer(layerId)
    if (!object) return 0
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    const context = canvas.getContext('2d')!
    this.drawLayerCondition(context, object, layerId)
    const data = context.getImageData(0, 0, this.stageWidth, this.stageHeight).data
    let alpha = 0
    for (let index = 3; index < data.length; index += 4) alpha += data[index]
    const value = alpha / Math.max(1, this.stageWidth * this.stageHeight * 255)
    this.layerAlphaCoverageCache.set(layerId, { value, at: now })
    return value
  }

  private resolvedRegionTiming(region: RegionItem, fallback?: { start: number; end: number; schedule: RegionSchedule; active: boolean }) {
    if (region.schedule !== 'auto') return { start: region.start, end: region.end, schedule: region.schedule }
    return { start: fallback?.start ?? 0, end: fallback?.end ?? 1, schedule: 'auto' as RegionSchedule, active: fallback?.active ?? true }
  }

  private drawLayerCondition(context: CanvasRenderingContext2D, object: FabricObject, layerId: string) {
    const source = this.fabricCanvas?.getElement()
    if (!source) return
    const isTransforming = !!(this.fabricCanvas as unknown as { _currentTransform?: unknown })._currentTransform
    if (isTransforming) {
      context.drawImage(source, 0, 0, this.stageWidth, this.stageHeight)
      return
    }
    const wasSyncingLayers = this.isSyncingLayers
    this.isSyncingLayers = true
    const visibility = new Map<FabricObject, boolean | undefined>()
    const active = this.fabricCanvas?.getActiveObject()
    try {
      for (const candidate of this.fabricCanvas?.getObjects() ?? []) {
        visibility.set(candidate, candidate.visible)
        candidate.set('visible', candidate === object || (candidate as FabricObject & { __parentLayerId?: string }).__parentLayerId === layerId)
      }
      this.fabricCanvas?.discardActiveObject()
      this.fabricCanvas?.renderAll()
      context.drawImage(source, 0, 0, this.stageWidth, this.stageHeight)
    } finally {
      for (const [candidate, visible] of visibility) candidate.set('visible', visible)
      if (active) this.fabricCanvas?.setActiveObject(active)
      this.fabricCanvas?.renderAll()
      this.isSyncingLayers = wasSyncingLayers
    }
  }

  private exportLayerRegionConditionImage(object: FabricObject, layerId: string, region: RegionItem) {
    if (region.id.startsWith('inherited:')) return this.exportInheritedLayerConditionImage(object, layerId, region)
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    const context = canvas.getContext('2d')!
    this.drawLayerCondition(context, object, layerId)
    const regionMask = this.layerRegionConditionMaskCanvas(layerId, region)
    const imageData = context.getImageData(0, 0, this.stageWidth, this.stageHeight)
    const maskContext = regionMask.getContext('2d')!
    const maskData = maskContext.getImageData(0, 0, this.stageWidth, this.stageHeight).data
    for (let index = 0; index < imageData.data.length; index += 4) {
      const alpha = maskData[index + 3]
      imageData.data[index + 3] = alpha
    }
    context.putImageData(imageData, 0, 0)
    context.globalCompositeOperation = 'source-over'
    return canvas.toDataURL('image/png')
  }

  private exportInheritedLayerConditionImage(object: FabricObject, layerId: string, region: RegionItem) {
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    const context = canvas.getContext('2d')!
    this.drawLayerCondition(context, object, layerId)
    const imageData = context.getImageData(0, 0, this.stageWidth, this.stageHeight)
    const mask = document.createElement('canvas')
    mask.width = this.stageWidth
    mask.height = this.stageHeight
    const maskContext = mask.getContext('2d')!
    const maskImage = maskContext.createImageData(this.stageWidth, this.stageHeight)
    for (let index = 0; index < imageData.data.length; index += 4) {
      const alpha = region.negate ? 255 - imageData.data[index + 3] : imageData.data[index + 3]
      maskImage.data[index] = 255
      maskImage.data[index + 1] = 255
      maskImage.data[index + 2] = 255
      maskImage.data[index + 3] = alpha
    }
    maskContext.putImageData(maskImage, 0, 0)
    const blur = Math.max(region.innerBlur, region.outerBlur)
    const finalMask = document.createElement('canvas')
    finalMask.width = this.stageWidth
    finalMask.height = this.stageHeight
    const finalMaskContext = finalMask.getContext('2d')!
    finalMaskContext.filter = blur ? `blur(${blur}px)` : 'none'
    finalMaskContext.drawImage(mask, 0, 0)
    const maskData = finalMaskContext.getImageData(0, 0, this.stageWidth, this.stageHeight).data
    for (let index = 0; index < imageData.data.length; index += 4) imageData.data[index + 3] = maskData[index + 3]
    context.putImageData(imageData, 0, 0)
    return canvas.toDataURL('image/png')
  }

  private layerRegionConditionMaskCanvas(layerId: string, region: RegionItem) {
    const extracted = this.layerRegionColorMaskCanvas(layerId, region, false)
    if (extracted.hasPixels) return extracted.canvas
    const fallback = document.createElement('canvas')
    fallback.width = this.stageWidth
    fallback.height = this.stageHeight
    const context = fallback.getContext('2d')!
    const rect = this.normalizedRegionRect(region.rect)
    context.save()
    context.filter = region.outerBlur ? `blur(${region.outerBlur}px)` : 'none'
    context.fillStyle = region.negate ? 'rgba(0,0,0,1)' : 'rgba(255,255,255,1)'
    this.fillRegionShape(context, region, rect)
    if (region.innerBlur) {
      context.filter = `blur(${region.innerBlur}px)`
      context.fillStyle = region.negate ? '#000' : '#fff'
      this.fillRegionShape(context, region, rect)
    }
    context.restore()
    return fallback
  }

  private hasLayerRegionMaskPixels(layerId: string, region: RegionItem) {
    return this.layerRegionColorMaskCanvas(layerId, region).hasPixels
  }

  private layerRegionColorMaskCanvas(layerId: string, region: RegionItem, applyStrength = true) {
    const canvas = document.createElement('canvas')
    canvas.width = this.stageWidth
    canvas.height = this.stageHeight
    const context = canvas.getContext('2d')!
    const source = this.activeMaskScope === `layer:${layerId}` ? this.maskCanvasElement : this.maskCanvasForScope(`layer:${layerId}`)
    context.drawImage(source, 0, 0)
    const imageData = context.getImageData(0, 0, this.stageWidth, this.stageHeight)
    const target = this.parseColor(region.color)
    let hasPixels = false
    if (!target) {
      context.clearRect(0, 0, this.stageWidth, this.stageHeight)
      return { canvas, hasPixels }
    }
    for (let index = 0; index < imageData.data.length; index += 4) {
      const alpha = imageData.data[index + 3]
      const matches = alpha > 0 && Math.abs(imageData.data[index] - target.r) <= 3 && Math.abs(imageData.data[index + 1] - target.g) <= 3 && Math.abs(imageData.data[index + 2] - target.b) <= 3
      if (matches) {
        hasPixels = true
        const nextAlpha = Math.round(alpha * (applyStrength ? region.maskStrength : 1))
        imageData.data[index] = 255
        imageData.data[index + 1] = 255
        imageData.data[index + 2] = 255
        imageData.data[index + 3] = nextAlpha
      } else {
        imageData.data[index] = 0
        imageData.data[index + 1] = 0
        imageData.data[index + 2] = 0
        imageData.data[index + 3] = 0
      }
    }
    context.putImageData(imageData, 0, 0)
    return { canvas, hasPixels }
  }

  private async dataUrlBytes(dataUrl: string) {
    const response = await fetch(dataUrl)
    return new Uint8Array(await response.arrayBuffer())
  }

  private exportMask() {
    const canvas = this.composeSceneMaskCanvas()
    const context = canvas.getContext('2d')!
    const imageData = context.getImageData(0, 0, this.stageWidth, this.stageHeight)
    for (let index = 0; index < imageData.data.length; index += 4) {
      const value = Math.max(imageData.data[index], imageData.data[index + 1], imageData.data[index + 2])
      imageData.data[index] = value
      imageData.data[index + 1] = value
      imageData.data[index + 2] = value
      imageData.data[index + 3] = 255
    }
    context.putImageData(imageData, 0, 0)
    return canvas.toDataURL('image/png')
  }

  private exportRealtimeMask() {
    const transform = this.currentStreamMotionTransform()
    if (!transform) return this.exportMask()
    const canvas = this.composeSceneMaskCanvas()
    const context = canvas.getContext('2d')!
    const imageData = context.getImageData(0, 0, this.stageWidth, this.stageHeight)
    let hasMask = false
    for (let index = 0; index < imageData.data.length; index += 4) {
      const value = Math.max(imageData.data[index], imageData.data[index + 1], imageData.data[index + 2])
      if (value > 8) hasMask = true
      imageData.data[index] = value
      imageData.data[index + 1] = value
      imageData.data[index + 2] = value
      imageData.data[index + 3] = 255
    }
    if (!hasMask) {
      for (let index = 0; index < imageData.data.length; index += 4) {
        imageData.data[index] = 255
        imageData.data[index + 1] = 255
        imageData.data[index + 2] = 255
        imageData.data[index + 3] = 255
      }
    }
    context.putImageData(imageData, 0, 0)
    const transformed = document.createElement('canvas')
    transformed.width = this.stageWidth
    transformed.height = this.stageHeight
    const transformedContext = transformed.getContext('2d')!
    transformedContext.setTransform(...transform)
    transformedContext.drawImage(canvas, 0, 0)
    transformedContext.resetTransform()
    return transformed.toDataURL('image/png')
  }

  private drawPaintedMask(context: CanvasRenderingContext2D, scope: MaskScope, negated = false) {
    const painted = document.createElement('canvas')
    painted.width = this.stageWidth
    painted.height = this.stageHeight
    const paintedContext = painted.getContext('2d')!
    paintedContext.drawImage(this.maskCanvasForScope(scope), 0, 0)
    const imageData = paintedContext.getImageData(0, 0, this.stageWidth, this.stageHeight)
    for (let index = 0; index < imageData.data.length; index += 4) {
      const alpha = imageData.data[index + 3]
      const channel = negated ? 0 : 255
      imageData.data[index] = channel
      imageData.data[index + 1] = channel
      imageData.data[index + 2] = channel
      imageData.data[index + 3] = alpha
    }
    paintedContext.putImageData(imageData, 0, 0)
    context.drawImage(painted, 0, 0)
  }

  private drawRegionMasks(context: CanvasRenderingContext2D, target: RegionTarget, layerId?: string) {
    for (const region of this.regions.filter((item) => item.target === target && (target === 'scene' || item.layerId === layerId))) {
      if (target === 'layer' && layerId && this.hasLayerRegionMaskPixels(layerId, region)) continue
      const rect = this.normalizedRegionRect(region.rect)
      context.save()
      context.globalCompositeOperation = 'source-over'
      context.filter = region.outerBlur ? `blur(${region.outerBlur}px)` : 'none'
      const channel = region.negate ? 0 : 255
      context.fillStyle = `rgba(${channel},${channel},${channel},${region.maskStrength})`
      this.fillRegionShape(context, region, rect)
      if (region.innerBlur) {
        context.filter = `blur(${region.innerBlur}px)`
        context.fillStyle = region.negate ? '#000' : '#fff'
        this.fillRegionShape(context, region, rect)
      }
      context.restore()
    }
  }

  private fillRegionShape(context: CanvasRenderingContext2D, region: RegionItem, rect: SelectionRect) {
    const points = region.rect.points
    if (points?.length) {
      context.beginPath()
      context.moveTo(points[0].x, points[0].y)
      for (const point of points.slice(1)) context.lineTo(point.x, point.y)
      context.closePath()
      context.fill()
      return
    }
    context.fillRect(rect.x, rect.y, rect.width, rect.height)
  }

  private normalizedRegionRect(rect: SelectionRect) {
    const x = Math.max(0, Math.min(this.stageWidth, rect.x))
    const y = Math.max(0, Math.min(this.stageHeight, rect.y))
    return {
      x,
      y,
      width: Math.max(1, Math.min(this.stageWidth - x, rect.width)),
      height: Math.max(1, Math.min(this.stageHeight - y, rect.height)),
      inverted: rect.inverted,
    }
  }

  private applyOutput = () => {
    if (!this.outputImage || !this.fabricCanvas) return
    this.pushHistory()
    FabricImage.fromURL(this.outputImage).then((image) => {
      this.fitImage(image)
      image.set({ name: 'Applied output' })
      this.fabricCanvas?.add(image)
      this.fabricCanvas?.setActiveObject(image)
      this.fabricCanvas?.requestRenderAll()
      this.clearMask()
      this.syncLayers()
    })
  }

  private downloadOutput = () => {
    const url = this.outputImage || this.exportCanvasImage(true)
    if (!url) return
    const link = document.createElement('a')
    link.href = url
    link.download = 'rtdiffusion-frame.png'
    link.click()
  }

  private createZip(files: { name: string; bytes: Uint8Array }[]) {
    const chunks: Uint8Array[] = []
    const centralDirectory: Uint8Array[] = []
    let offset = 0
    for (const file of files) {
      const name = new TextEncoder().encode(file.name)
      const crc = this.crc32(file.bytes)
      const local = new Uint8Array(30 + name.length)
      const localView = new DataView(local.buffer)
      localView.setUint32(0, 0x04034b50, true)
      localView.setUint16(4, 20, true)
      localView.setUint16(8, 0, true)
      localView.setUint32(14, crc, true)
      localView.setUint32(18, file.bytes.length, true)
      localView.setUint32(22, file.bytes.length, true)
      localView.setUint16(26, name.length, true)
      local.set(name, 30)
      chunks.push(local, file.bytes)

      const central = new Uint8Array(46 + name.length)
      const centralView = new DataView(central.buffer)
      centralView.setUint32(0, 0x02014b50, true)
      centralView.setUint16(4, 20, true)
      centralView.setUint16(6, 20, true)
      centralView.setUint32(16, crc, true)
      centralView.setUint32(20, file.bytes.length, true)
      centralView.setUint32(24, file.bytes.length, true)
      centralView.setUint16(28, name.length, true)
      centralView.setUint32(42, offset, true)
      central.set(name, 46)
      centralDirectory.push(central)
      offset += local.length + file.bytes.length
    }
    const centralOffset = offset
    for (const chunk of centralDirectory) {
      chunks.push(chunk)
      offset += chunk.length
    }
    const end = new Uint8Array(22)
    const endView = new DataView(end.buffer)
    endView.setUint32(0, 0x06054b50, true)
    endView.setUint16(8, files.length, true)
    endView.setUint16(10, files.length, true)
    endView.setUint32(12, offset - centralOffset, true)
    endView.setUint32(16, centralOffset, true)
    chunks.push(end)
    return new Blob(chunks as BlobPart[], { type: 'application/zip' })
  }

  private readZipEntries(bytes: Uint8Array) {
    const entries = new Map<string, Uint8Array>()
    const decoder = new TextDecoder()
    let offset = 0
    while (offset + 30 <= bytes.length) {
      const view = new DataView(bytes.buffer, bytes.byteOffset + offset, bytes.byteLength - offset)
      const signature = view.getUint32(0, true)
      if (signature !== 0x04034b50) break
      const flags = view.getUint16(6, true)
      const method = view.getUint16(8, true)
      if (flags & 0x08) throw new Error('ZIP archives with data descriptors are not supported')
      if (method !== 0) throw new Error('Only stored ZIP entries are supported')
      const compressedSize = view.getUint32(18, true)
      const nameLength = view.getUint16(26, true)
      const extraLength = view.getUint16(28, true)
      const nameStart = offset + 30
      const dataStart = nameStart + nameLength + extraLength
      const dataEnd = dataStart + compressedSize
      if (dataEnd > bytes.length) throw new Error('Truncated ZIP entry')
      const name = decoder.decode(bytes.slice(nameStart, nameStart + nameLength))
      entries.set(name, bytes.slice(dataStart, dataEnd))
      offset = dataEnd
    }
    return entries
  }

  private mediaTypeForPath(path: string) {
    const lower = path.toLowerCase()
    if (lower.endsWith('.png')) return 'image/png'
    if (lower.endsWith('.jpg') || lower.endsWith('.jpeg')) return 'image/jpeg'
    if (lower.endsWith('.webp')) return 'image/webp'
    if (lower.endsWith('.gif')) return 'image/gif'
    if (lower.endsWith('.mp4')) return 'video/mp4'
    if (lower.endsWith('.webm')) return 'video/webm'
    return 'application/octet-stream'
  }

  private crc32(bytes: Uint8Array) {
    let crc = 0xffffffff
    for (const byte of bytes) {
      crc ^= byte
      for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1))
    }
    return (crc ^ 0xffffffff) >>> 0
  }

  static styles = css`
    :host {
      display: block;
      height: 100svh;
      color: #050507;
      background: #dfe0f4;
      overflow: hidden;
    }

    .shell {
      height: 100svh;
      padding: 4px;
      display: grid;
      grid-template-rows: 48px minmax(0, 1fr) minmax(36px, auto);
      gap: 4px;
      overflow: hidden;
      box-sizing: border-box;
    }

    .topbar,
    .workspace {
      max-width: none;
      min-width: 0;
    }

    .topbar {
      display: grid;
      grid-template-columns: max-content minmax(0, 1fr);
      align-items: center;
      gap: 8px;
      min-height: 0;
      padding: 4px;
      border: 2px solid #24245d;
      background: #f0f0ff;
    }

    h1,
    h2 {
      margin: 0;
      letter-spacing: 0;
    }

    h1 {
      font-size: 32px;
      line-height: 1;
    }

    h2 {
      font-size: 14px;
      color: #050507;
    }

    .quick-actions,
    .tool-options {
      display: grid;
      grid-auto-flow: column;
      grid-auto-columns: max-content;
      gap: 6px;
      align-items: center;
      min-width: 0;
    }

    .quick-actions strong {
      padding: 0 6px;
      font-size: 13px;
      color: #111;
    }

    .tool-options {
      overflow-x: auto;
    }

    .workspace {
      display: grid;
      grid-template-columns: var(--left-panel-width, 220px) minmax(320px, 1fr) var(--right-panel-width, 320px);
      grid-template-rows: minmax(0, 1fr);
      gap: 4px;
      align-items: stretch;
      min-height: 0;
      overflow: hidden;
    }

    .panel-resizer {
      position: absolute;
      top: 0;
      bottom: 0;
      z-index: 10;
      width: 6px;
      min-width: 6px;
      min-height: 0;
      height: 100%;
      padding: 0;
      border: 0;
      border-radius: 0;
      background: #24245d;
      cursor: col-resize;
      opacity: 0.45;
    }

    .left-resizer {
      right: -4px;
    }

    .right-resizer {
      left: -4px;
    }

    .panel-resizer:hover,
    .panel-resizer:active {
      opacity: 1;
      background: #e9b44c;
    }

    .panel {
      position: relative;
      border: 2px solid #24245d;
      border-radius: 0;
      background: #aaaef3;
      min-width: 0;
      min-height: 0;
      box-sizing: border-box;
    }

    .tools-panel,
    .inspector,
    .stack {
      display: grid;
      gap: 10px;
    }

    .tools-panel,
    .inspector {
      align-content: start;
      height: 100%;
      min-height: 0;
      overflow: auto;
      overscroll-behavior: contain;
    }

    .tools-panel {
      grid-template-rows: auto auto auto minmax(180px, 1fr);
      overflow-x: hidden;
      overflow-y: auto;
    }

    .tools-panel > .button-grid {
      padding: 0 8px;
      box-sizing: border-box;
    }

    .inspector {
      grid-template-rows: minmax(0, 1fr);
      overflow: hidden;
    }

    .inspector-fold {
      display: grid;
      gap: 8px;
      min-height: 0;
      overflow: hidden;
      border-top: 1px solid rgba(36, 36, 93, 0.55);
      padding-top: 6px;
    }

    .inspector-fold summary {
      cursor: pointer;
      font-weight: 700;
      font-size: 13px;
      list-style-position: inside;
    }

    .inspector-fold:not([open]) {
      gap: 0;
    }

    .dock-section {
      display: grid;
      gap: 6px;
      min-height: 0;
      padding: 6px;
      min-width: 0;
      max-width: 100%;
      overflow: hidden;
      box-sizing: border-box;
    }

    .layers-dock {
      align-self: stretch;
      grid-template-rows: auto auto minmax(0, 1fr);
      gap: 5px;
      height: 100%;
      padding-left: 0;
      padding-right: 0;
      overflow: hidden;
    }

    .layers-dock > h2 {
      padding-left: 8px;
      padding-right: 8px;
    }

    .scene-entity-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 4px;
      padding: 0 6px;
      box-sizing: border-box;
    }

    .scene-entity {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 1px;
      align-items: center;
      min-height: 30px;
      margin: 0;
      padding: 4px 6px;
      color: #eeeeee;
      border: 1px solid #3a3f45;
      border-radius: 0;
      background: #202833;
      text-align: left;
      overflow: hidden;
    }

    .scene-entity span,
    .scene-entity small {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .scene-entity.selected {
      border-color: #2f9bff;
      background: #1f63a8;
    }

    .scene-entity small {
      color: #cbd5e3;
      font-size: 11px;
    }

    .field {
      display: grid;
      gap: 5px;
      color: #111;
      font-size: 13px;
    }

    .field span {
      color: #111;
    }

    textarea,
    input,
    select {
      width: min(100%, 280px);
      box-sizing: border-box;
      color: #050507;
      border: 1px solid #30304f;
      border-radius: 2px;
      background: #f8f8ff;
      padding: 8px;
      outline: none;
    }

    input[type='number'] {
      appearance: textfield;
      font-variant-numeric: tabular-nums;
      text-align: right;
    }

    input[type='number']::-webkit-outer-spin-button,
    input[type='number']::-webkit-inner-spin-button {
      appearance: none;
      margin: 0;
    }

    textarea {
      width: 100%;
      max-width: none;
      min-height: 72px;
      resize: vertical;
    }

    .inspector textarea {
      min-height: 64px;
      resize: none;
    }

    .compact,
    .inline,
    .dimensions {
      grid-template-columns: 78px minmax(0, 1fr);
      align-items: center;
    }

    .inline {
      display: grid;
    }

    .dimensions,
    .tool-row,
    .button-grid {
      display: grid;
      gap: 8px;
    }

    .button-grid {
      grid-template-columns: 1fr 1fr;
    }

    .stream-status-line {
      grid-column: 1 / -1;
      font-size: 11px;
      color: var(--fg-2, #aaa);
      text-align: center;
      padding: 2px 0 0;
      min-height: 16px;
    }

    .tool-row {
      grid-template-columns: repeat(auto-fill, 24px);
      gap: 4px;
      align-items: center;
      justify-content: start;
      min-width: 0;
    }

    .dest-switch {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 4px;
      min-width: 0;
    }

    .dest-switch button {
      min-width: 0;
      min-height: 28px;
      padding: 4px 6px;
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .dest-switch button.active {
      color: #ffffff;
      border-color: #2f9bff;
      background: #1f63a8;
    }

    .action-row {
      display: grid;
      grid-template-columns: repeat(auto-fill, 24px);
      gap: 4px;
      align-items: center;
      padding: 0 6px;
      box-sizing: border-box;
      min-width: 0;
      max-width: 100%;
      overflow: hidden;
    }

    .tool-icon {
      display: grid;
      place-items: center;
      width: 24px;
      min-width: 24px;
      height: 24px;
      min-height: 24px;
      padding: 0;
      border-radius: 3px;
      font-size: 14px;
      line-height: 1;
      font-weight: 700;
    }

    .number-field {
      color: #d8d4cb;
      font-size: 13px;
    }

    .number-field.embedded {
      grid-column: 1 / -1;
    }

    .number-control {
      display: grid;
      grid-template-columns: 26px minmax(0, 1fr) minmax(72px, max-content) 26px;
      align-items: center;
      width: min(100%, 240px);
      min-height: 30px;
      border: 1px solid #555a60;
      border-radius: 999px;
      overflow: hidden;
      background: #222;
      box-shadow: inset 0 0 0 1px #2f3338;
      cursor: ew-resize;
      user-select: none;
    }

    .inspector .number-control,
    .layer .number-control {
      grid-template-columns: 20px minmax(42px, 1fr) minmax(48px, max-content) 20px;
    }

    .inspector .number-control input,
    .layer .number-control input {
      min-width: 44px;
      font-size: 14px;
    }

    .inspector .scrub-name,
    .layer .scrub-name {
      font-size: 13px;
    }

    .inspector .scrub-arrow,
    .layer .scrub-arrow {
      font-size: 14px;
    }

    .number-control input,
    .number-control button,
    .scrub-name {
      border: 0;
      border-radius: 0;
      min-height: 28px;
    }

    .number-control input {
      width: 100%;
      min-width: 72px;
      padding: 2px 10px 2px 6px;
      background: transparent;
      color: #eeeeee;
      font-size: 22px;
      line-height: 1;
      cursor: ew-resize;
    }

    .scrub-name {
      display: flex;
      align-items: center;
      min-width: 0;
      padding: 0 4px;
      color: #9fa1a5;
      font-size: 20px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .scrub-arrow {
      min-height: 0;
      padding: 0;
      color: #eeeeee;
      background: transparent;
      font-size: 22px;
      line-height: 1;
      cursor: pointer;
    }

    .number-control:hover {
      border-color: #7a7f86;
      background: #272727;
    }

    .scrub-arrow:hover {
      color: #e9b44c;
      background: transparent;
    }

    button {
      box-sizing: border-box;
      min-height: 36px;
      color: #f3f1ec;
      background: #242a31;
      border: 1px solid #3b424c;
      border-radius: 6px;
      cursor: pointer;
    }

    button:hover,
    button.active {
      border-color: #e9b44c;
      background: #30312c;
    }

    .stream {
      width: 100%;
      background: #2f6f73;
      border-color: #5fb7aa;
      font-weight: 700;
    }

    .stream:hover,
    .stream.active {
      background: #24676d;
      border-color: #83d0bf;
    }

    .color-panel {
      display: grid;
      grid-template-columns: 52px minmax(0, 1fr);
      gap: 6px;
      align-items: start;
      min-width: 0;
      max-width: 100%;
      overflow: hidden;
    }

    .color-stack {
      position: relative;
      min-height: 54px;
    }

    .color-chip {
      position: absolute;
      width: 32px;
      min-height: 32px;
      padding: 0;
      background:
        linear-gradient(var(--chip), var(--chip)),
        linear-gradient(45deg, #d8dce2 25%, transparent 25%, transparent 75%, #d8dce2 75%),
        linear-gradient(45deg, #d8dce2 25%, #ffffff 25%, #ffffff 75%, #d8dce2 75%);
      background-position: 0 0, 0 0, 5px 5px;
      background-size: auto, 10px 10px, 10px 10px;
      border: 2px solid #f3f1ec;
      border-radius: 0;
      box-shadow: 0 0 0 1px #050507;
    }

    .color-chip.primary {
      top: 0;
      left: 0;
      z-index: 2;
    }

    .color-chip.secondary {
      right: 0;
      bottom: 0;
      z-index: 1;
    }

    .color-chip.active {
      background:
        linear-gradient(var(--chip), var(--chip)),
        linear-gradient(45deg, #d8dce2 25%, transparent 25%, transparent 75%, #d8dce2 75%),
        linear-gradient(45deg, #d8dce2 25%, #ffffff 25%, #ffffff 75%, #d8dce2 75%);
      background-position: 0 0, 0 0, 5px 5px;
      background-size: auto, 10px 10px, 10px 10px;
      outline: 2px solid #e9b44c;
      outline-offset: 2px;
      z-index: 3;
    }

    .swap-colors {
      position: absolute;
      right: 0;
      top: 0;
      width: 18px;
      min-height: 18px;
      padding: 0;
      border-radius: 3px;
      font-size: 12px;
      line-height: 1;
    }

    .color-editor {
      display: grid;
      gap: 4px;
      min-width: 0;
      overflow: hidden;
    }

    .color-native {
      display: grid;
      grid-template-columns: 28px minmax(0, 1fr);
      gap: 4px;
      align-items: center;
      min-width: 0;
    }

    .color-native input[type='color'] {
      width: 28px;
      height: 24px;
      padding: 0;
      border: 1px solid #3b424c;
      background: #1b1f25;
    }

    .hex-input {
      width: 100%;
      min-width: 0;
      box-sizing: border-box;
      height: 24px;
      padding: 3px 5px;
      font-size: 12px;
      text-transform: uppercase;
    }

    .channel-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 4px;
    }

    .channel-input {
      display: grid;
      grid-template-columns: 10px minmax(0, 1fr);
      gap: 2px;
      align-items: center;
      font-size: 10px;
      color: #1f2329;
      font-weight: 700;
    }

    .channel-input input {
      width: 100%;
      min-width: 0;
      height: 22px;
      padding: 1px 2px;
      font-size: 10px;
    }

    .hsl-sliders {
      display: grid;
      gap: 3px;
    }

    .color-slider {
      display: grid;
      grid-template-columns: 14px minmax(0, 1fr) 28px;
      gap: 4px;
      align-items: center;
      font-size: 10px;
      color: #1f2329;
      font-weight: 700;
    }

    .color-slider input {
      width: 100%;
      min-width: 0;
    }

    .color-slider output {
      font-variant-numeric: tabular-nums;
      text-align: right;
    }

    .palette {
      display: grid;
      grid-column: 1 / -1;
      grid-template-columns: repeat(8, minmax(0, 1fr));
      gap: 3px;
      align-items: center;
      min-width: 0;
    }

    .swatch {
      min-height: 22px;
      padding: 0;
      background:
        linear-gradient(var(--swatch), var(--swatch)),
        linear-gradient(45deg, #d8dce2 25%, transparent 25%, transparent 75%, #d8dce2 75%),
        linear-gradient(45deg, #d8dce2 25%, #ffffff 25%, #ffffff 75%, #d8dce2 75%);
      background-position: 0 0, 0 0, 5px 5px;
      background-size: auto, 10px 10px, 10px 10px;
      border-color: #4a505b;
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.18);
    }

    .quick-palette {
      grid-template-rows: repeat(2, 22px);
      overflow: hidden;
    }

    .palette-more {
      display: grid;
      place-items: center;
      color: #050507;
      background: #f8f8ff;
      font-size: 17px;
      font-weight: 700;
    }

    .swatch.active {
      outline: 2px solid #f3f1ec;
      outline-offset: 1px;
    }

    .planner {
      display: grid;
      grid-column: 1 / -1;
      gap: 5px;
      padding-top: 2px;
      border-top: 1px solid rgba(36, 36, 93, 0.32);
    }

    .planner-mode {
      grid-template-columns: 42px minmax(0, 1fr);
      margin: 0;
    }

    .planner-row {
      display: grid;
      grid-template-columns: repeat(7, minmax(0, 1fr));
      gap: 4px;
    }

    .color-popup {
      position: fixed;
      z-index: 30;
      width: min(420px, calc(100vw - 16px));
      max-height: min(620px, calc(100vh - 16px));
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      border: 2px solid #24245d;
      background: #aaaef3;
      box-shadow: 0 18px 42px rgba(0, 0, 0, 0.32);
      overflow: hidden;
    }

    .color-popup-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 24px;
      align-items: center;
      gap: 6px;
      min-height: 30px;
      padding: 4px 6px;
      color: #050507;
      background: #f0f0ff;
      border-bottom: 1px solid #24245d;
      cursor: move;
      user-select: none;
    }

    .color-popup-head button {
      width: 24px;
      min-width: 24px;
      height: 24px;
      min-height: 24px;
      padding: 0;
      border-radius: 3px;
    }

    .color-popup-body {
      display: grid;
      gap: 8px;
      min-height: 0;
      padding: 8px;
      overflow: auto;
    }

    .popup-color-editor {
      display: grid;
      grid-template-columns: 54px minmax(0, 1fr);
      gap: 6px;
      align-items: start;
    }

    .color-preview-large {
      width: 54px;
      height: 54px;
      border: 2px solid #f3f1ec;
      box-shadow: 0 0 0 1px #050507;
      background:
        linear-gradient(var(--chip), var(--chip)),
        linear-gradient(45deg, #d8dce2 25%, transparent 25%, transparent 75%, #d8dce2 75%),
        linear-gradient(45deg, #d8dce2 25%, #ffffff 25%, #ffffff 75%, #d8dce2 75%);
      background-position: 0 0, 0 0, 6px 6px;
      background-size: auto, 12px 12px, 12px 12px;
    }

    .popup-rgba,
    .hsl-sliders {
      grid-column: 1 / -1;
    }

    .full-palette {
      grid-template-columns: repeat(12, minmax(0, 1fr));
    }

    .popup-planner .swatch,
    .full-palette .swatch {
      min-height: 24px;
    }

    .studio {
      display: grid;
      grid-template-rows: minmax(0, 1fr);
      gap: 4px;
      min-width: 0;
      min-height: 0;
      height: 100%;
      overflow: hidden;
    }

    .document-bar {
      display: grid;
      grid-template-columns: minmax(180px, 280px) 96px 96px auto auto;
      gap: 8px;
      align-items: end;
      min-width: 0;
    }

    .error {
      color: #4b0000;
      border: 1px solid #9e3f4d;
      background: #ffd4d8;
      overflow-wrap: anywhere;
    }

    .viewer {
      display: grid;
      gap: 4px;
      min-height: 0;
      min-width: 0;
      overflow: hidden;
      overscroll-behavior: contain;
      border: 2px solid #24245d;
      border-radius: 0;
      background: #dfe0f4;
    }

    .workspace.portrait .viewer {
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      grid-template-rows: minmax(0, 1fr);
      align-items: stretch;
    }

    .workspace.landscape .viewer {
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      grid-template-rows: minmax(0, 1fr);
      align-items: stretch;
    }

    .work-pane {
      position: relative;
      display: block;
      min-width: 0;
      min-height: 0;
      overflow: hidden;
      border: 1px solid #24245d;
      background: #c9cbef;
    }

    .viewport-overlay {
      position: absolute;
      z-index: 12;
      left: 0px;
      right: 0px;
      display: grid;
      grid-template-columns: max-content minmax(0, 1fr) max-content;
      align-items: center;
      gap: 8px;
      min-width: 0;
      min-height: 24px;
      padding: 2px 7px;
      border: 1px solid rgba(20, 20, 46, 0.4);
      background: rgba(25, 25, 32, 0.58);
      color: #f7f7fb;
      font-size: 12px;
      opacity: 0;
      transform: translateY(-2px);
      pointer-events: none;
      transition: opacity 140ms ease, transform 140ms ease;
      backdrop-filter: blur(4px);
    }

    .viewport-overlay.visible {
      opacity: 1;
      transform: translateY(0);
    }

    .pane-head {
      top: 0px;
    }

    .pane-foot {
      bottom: 0px;
      transform: translateY(2px);
    }

    .pane-foot.visible {
      transform: translateY(0);
    }

    .viewport-overlay span {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .pane-action-overlay {
      grid-template-columns: max-content;
      left: auto;
      right: 6px;
    }

    .viewport-overlay button {
      min-height: 24px;
      padding: 2px 10px;
      justify-self: end;
      pointer-events: auto;
      border-color: rgba(255, 255, 255, 0.16);
      background: rgba(20, 20, 28, 0.82);
    }

    .viewport-toggles {
      display: flex;
      justify-content: end;
      gap: 4px;
      min-width: 0;
    }

    .viewport-toggles button {
      min-height: 22px;
      padding: 2px 7px;
      color: #d7d8e8;
    }

    .viewport-toggles button.active {
      border-color: #6fb8ff;
      color: #ffffff;
      background: #245f94;
    }

    .work-surface {
      display: grid;
      place-items: center;
      width: 100%;
      height: 100%;
      overflow: auto;
      background: #c9cbef;
      container-type: size;
      cursor: default;
      overscroll-behavior: contain;
    }

    .work-surface:active {
      cursor: grabbing;
    }

    .stage,
    .preview.native {
      position: relative;
      box-sizing: border-box;
      width: calc(
        min(calc(100cqw - 6px), calc((100cqh - 6px) * var(--stage-width) / var(--stage-height))) * var(--zoom)
      );
      aspect-ratio: var(--stage-width) / var(--stage-height);
      border: 0;
      outline: 3px solid #1b1b4f;
      border-radius: 0;
      box-shadow: none;
      overflow: hidden;
    }

    .preview.native.sampling img {
      filter: saturate(1.02) contrast(1.02);
    }

    .stage {
      cursor: crosshair;
      justify-self: center;
      background-color: #b8bdc4;
      background-image:
        linear-gradient(45deg, #8f969f 25%, transparent 25%),
        linear-gradient(-45deg, #8f969f 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #8f969f 75%),
        linear-gradient(-45deg, transparent 75%, #8f969f 75%);
      background-position: 0 0, 0 10px, 10px -10px, -10px 0;
      background-size: 20px 20px;
    }

    #edit-canvas,
    #mask-canvas,
    .canvas-container,
    .lower-canvas,
    .upper-canvas {
      position: absolute !important;
      inset: 0 !important;
      width: 100% !important;
      height: 100% !important;
      box-sizing: border-box !important;
    }

    .canvas-container {
      z-index: 1;
    }

    .stage.paint-hidden .canvas-container {
      opacity: 0;
    }

    .mask,
    .mask-preview {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      opacity: 0.74;
      touch-action: none;
    }

    .mask-preview {
      z-index: 3;
    }

    .mask {
      z-index: 2;
    }

    #editor-overlay {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      z-index: 4;
      pointer-events: none;
    }

    .mask-preview:not(.visible),
    .mask:not(.active) {
      opacity: 0;
    }

    .mask.active {
      z-index: 6;
      pointer-events: auto;
      cursor: crosshair;
    }

    .selection-box {
      position: absolute;
      z-index: 5;
      pointer-events: none;
      border: 3px solid #00e5ff;
      outline: 2px solid #050507;
      outline-offset: 2px;
      box-shadow: 0 0 0 1px #ffffff, inset 0 0 0 2px rgba(5, 5, 7, 0.72), 0 0 18px rgba(0, 229, 255, 0.62);
      background: rgba(0, 229, 255, 0.1);
    }

    .selection-box.inverted {
      border-color: #e9b44c;
      background: rgba(233, 180, 76, 0.14);
    }

    .selection-path {
      position: absolute;
      inset: 0;
      z-index: 5;
      pointer-events: none;
      overflow: visible;
    }

    .selection-path polygon {
      fill: rgba(0, 229, 255, 0.12);
      stroke: #00e5ff;
      stroke-width: 0.55;
      stroke-dasharray: 1.4 0.7;
      paint-order: stroke;
      filter: drop-shadow(0 0 1px #000000) drop-shadow(0 0 2px rgba(0, 229, 255, 0.75));
    }

    .transform-panel {
      display: grid;
      gap: 8px;
      min-width: 0;
      padding: 7px;
      border: 1px solid rgba(36, 36, 93, 0.62);
      background: rgba(240, 240, 255, 0.58);
    }

    .transform-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) max-content;
      gap: 6px;
      align-items: center;
      min-width: 0;
      color: #111;
    }

    .transform-head span {
      color: #404756;
      font-size: 11px;
      white-space: nowrap;
    }

    .transform-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 5px;
      min-width: 0;
    }

    .transform-panel .number-control {
      width: 100%;
      max-width: none;
    }
    .cursor {
      position: absolute;
      z-index: 7;
      pointer-events: none;
      display: none;
      transform: translate(-50%, -50%);
      border: 2px solid #ffffff;
      border-radius: 50%;
      box-shadow: 0 0 0 1px #000000, inset 0 0 0 1px #000000;
    }

    .cursor.visible {
      display: block;
    }

    .layer-box {
      display: grid;
      grid-template-rows: minmax(0, 1fr) auto;
      min-height: 0;
      height: 100%;
      border: 1px solid #3a3f45;
      border-radius: 0;
      overflow: hidden;
      background: #232323;
    }

    .layer-list,
    .lora-list,
    .variation-list {
      display: grid;
      gap: 6px;
      max-height: min(260px, 34svh);
      min-height: 0;
      overflow: auto;
      overscroll-behavior: contain;
    }

    .inspector > .stack {
      min-height: 0;
      overflow: hidden;
    }

    .compact-sampler {
      gap: 6px;
    }

    .compact-sampler textarea {
      min-height: 44px;
      max-height: 56px;
    }

    .compact-sampler .field {
      gap: 3px;
    }

    .layer-list {
      align-content: start;
      gap: 0;
      max-height: none;
      height: 100%;
      background: #232323;
    }

    .generator-tab {
      height: 100%;
      grid-template-rows: minmax(185px, 0.58fr) auto minmax(180px, 1fr);
      gap: 8px;
      overflow: hidden;
    }

    .alive-tab {
      align-content: start;
      gap: 8px;
      overflow-y: auto;
      overflow-x: hidden;
      overscroll-behavior: contain;
    }

    .motion-cache-list {
      display: grid;
      gap: 6px;
      min-height: 120px;
    }

    .motion-clip {
      display: grid;
      grid-template-columns: minmax(0, 1fr) max-content;
      gap: 6px;
      align-items: center;
      padding: 6px;
      border: 1px solid #30343b;
      background: #20242a;
      color: #f3f1ec;
    }

    .motion-clip.active {
      border-color: #5fb7aa;
      background: #243235;
    }

    .motion-clip div {
      display: grid;
      gap: 2px;
      min-width: 0;
    }

    .motion-clip span {
      color: #a9b0ba;
      font-size: 12px;
      overflow-wrap: anywhere;
    }

    .generator-controls {
      display: grid;
      align-content: start;
      gap: 8px;
      min-height: 0;
      overflow-y: auto;
      overflow-x: hidden;
      overscroll-behavior: contain;
    }

    .generator-controls textarea {
      min-height: 44px;
      max-height: 54px;
    }

    .generator-controls .field {
      gap: 3px;
    }

    .generator-controls .inline {
      grid-template-columns: 90px minmax(0, 1fr);
    }

    .generator-actions {
      display: grid;
      gap: 6px;
      min-width: 0;
      overflow: hidden;
    }

    .generator-actions > button {
      width: 100%;
      justify-self: stretch;
      min-height: 34px;
    }

    .generator-action-row {
      display: grid;
      grid-template-columns: minmax(0, 0.82fr) minmax(0, 1.18fr);
      gap: 6px;
      min-width: 0;
    }

    .generator-action-row > button {
      min-width: 0;
      min-height: 30px;
      padding-inline: 6px;
    }

    .generator-controls select,
    .generator-controls input,
    .generator-controls .number-control,
    .generator-actions .number-control {
      min-width: 0;
      max-width: 100%;
      box-sizing: border-box;
    }

    .generator-actions .resolution-grid {
      min-width: 0;
    }

    .task-item {
      display: grid;
      gap: 4px;
      padding: 7px;
      border: 1px solid #384454;
      border-radius: 6px;
      background: #18202b;
      color: #e6edf7;
      font-size: 12px;
    }

    .task-item.error-task {
      border-color: #9e3f4d;
      background: #2a171b;
    }

    .task-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) max-content;
      gap: 8px;
      align-items: center;
    }

    .task-row span,
    .task-item p {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .task-detail {
      color: #aeb7c6;
    }

    .task-item p {
      margin: 0;
      color: #cbd5e3;
    }

    .task-bar {
      height: 7px;
      overflow: hidden;
      border-radius: 999px;
      background: #0d1218;
    }

    .task-bar span {
      display: block;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, #3f77ff, #64c6b8);
      transition: width 120ms ease;
    }

    .motion-frame-strip {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 4px;
      min-width: 0;
    }

    .motion-frame-strip img {
      width: 100%;
      aspect-ratio: 1;
      object-fit: cover;
      border-radius: 4px;
      border: 1px solid #2c3644;
      background: #0d1218;
    }

    .variation-dock {
      display: grid;
      min-height: 0;
      overflow: hidden;
    }

    .layer-actions {
      display: grid;
      grid-template-columns: repeat(auto-fill, 24px);
      gap: 2px;
      padding: 4px;
      justify-content: start;
      border-top: 1px solid #3a3f45;
      background: #1d1d1d;
    }

    .layer-actions button {
      width: 24px;
      min-width: 24px;
      min-height: 24px;
      padding: 0;
      color: #8fc7ff;
      border: 0;
      border-radius: 3px;
      background: transparent;
      font-size: 11px;
    }

    .layer-actions button:hover {
      background: #303943;
    }

    .variation-list {
      grid-template-columns: repeat(auto-fill, minmax(92px, 1fr));
      align-content: start;
      max-height: 320px;
      border: 1px solid #24245d;
      background: rgba(20, 20, 28, 0.12);
    }

    .variation {
      position: relative;
      display: grid;
      grid-template-rows: minmax(82px, 108px) auto;
      gap: 4px;
      min-height: 124px;
      padding: 5px;
      text-align: left;
      border: 2px solid #171b23;
      border-radius: 6px;
      background: #161b24;
      overflow: hidden;
    }

    .variation.selected {
      border-color: #3f77ff;
      background: #1c2943;
    }

    .variation img {
      width: 100%;
      height: 100%;
      min-height: 82px;
      object-fit: contain;
      border-radius: 4px;
      background-color: #b8bdc4;
      background-image:
        linear-gradient(45deg, #8f969f 25%, transparent 25%),
        linear-gradient(-45deg, #8f969f 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #8f969f 75%),
        linear-gradient(-45deg, transparent 75%, #8f969f 75%);
      background-position: 0 0, 0 8px, 8px -8px, -8px 0;
      background-size: 16px 16px;
    }

    .variation-label {
      position: absolute;
      top: 6px;
      left: 6px;
      min-width: 18px;
      padding: 1px 5px;
      border-radius: 999px;
      color: #f7f7fb;
      background: rgba(20, 20, 28, 0.72);
      font-size: 11px;
      text-align: center;
    }

    .variation-seed {
      color: #d6dae6;
      font-size: 11px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .variation-empty {
      display: grid;
      place-items: center;
      min-height: 120px;
      border: 1px dashed #24245d;
    }

    .lora {
      display: grid;
      gap: 6px;
      align-items: center;
      color: #d8d4cb;
      border: 1px solid #2f343c;
      border-radius: 6px;
      background: #111419;
      padding: 7px;
      font-size: 13px;
    }

    .inspector .lora {
      min-height: 28px;
      padding: 4px 6px;
      font-size: 12px;
    }

    .layer {
      display: grid;
      grid-template-columns: 54px minmax(0, 1fr) 22px;
      grid-template-rows: 30px 30px;
      gap: 0 8px;
      align-items: center;
      min-height: 64px;
      padding: 4px;
      color: #eeeeee;
      border: 1px solid transparent;
      background: transparent;
      cursor: pointer;
    }

    .layer.selected,
    .lora.checked {
      border-color: #2f9bff;
      background: #1f63a8;
    }

    .layer-thumb {
      grid-row: 1 / 3;
      width: 50px;
      height: 32px;
      object-fit: contain;
      border: 1px solid #4b4f54;
      background-color: #b8bdc4;
      background-image:
        linear-gradient(45deg, #8f969f 25%, transparent 25%),
        linear-gradient(-45deg, #8f969f 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #8f969f 75%),
        linear-gradient(-45deg, transparent 75%, #8f969f 75%);
      background-position: 0 0, 0 6px, 6px -6px, -6px 0;
      background-size: 12px 12px;
    }

    .layer-name {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .layer-visible {
      justify-self: end;
      width: 16px;
      height: 16px;
      padding: 0;
      accent-color: #2f9bff;
    }

    .layer .number-field {
      grid-column: 2 / 4;
      align-self: center;
    }

    .layer .number-control {
      min-height: 22px;
      grid-template-columns: 20px minmax(0, 1fr) minmax(54px, max-content) 20px;
      border-color: #59616a;
      background: #1b1f24;
    }

    .layer .number-control input,
    .layer .number-control button,
    .layer .scrub-name {
      min-height: 20px;
      font-size: 13px;
    }

    .layer-panel {
      display: grid;
      gap: 10px;
      min-height: 0;
    }

    .tabs {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 4px;
    }

    .tabs button {
      min-height: 30px;
    }

    .layer-tab {
      display: grid;
      gap: 10px;
      min-height: 0;
    }

    .layer-empty {
      padding: 10px;
      border: 1px dashed #3a3f45;
      border-radius: 6px;
    }

    .lora {
      grid-template-columns: auto 1fr;
    }

    .lora span {
      overflow-wrap: anywhere;
    }

    .preview {
      display: grid;
      place-items: center;
      border: 1px solid #30343b;
      border-radius: 6px;
      background-color: #b8bdc4;
      background-image:
        linear-gradient(45deg, #8f969f 25%, transparent 25%),
        linear-gradient(-45deg, #8f969f 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #8f969f 75%),
        linear-gradient(-45deg, transparent 75%, #8f969f 75%);
      background-position: 0 0, 0 10px, 10px -10px, -10px 0;
      background-size: 20px 20px;
    }

    .preview img,
    .preview video {
      display: block;
      width: 100%;
      height: 100%;
      object-fit: contain;
      position: relative;
      z-index: 1;
    }

    .preview .stream-pending,
    .preview .stream-preview-fallback {
      grid-area: 1 / 1;
    }

    .stream-output-video {
      display: block;
      grid-area: 1 / 1;
      width: 100%;
      height: 100%;
      object-fit: contain;
    }

    .stream-output-canvas {
      display: none;
    }

    .preview .stream-pending {
      opacity: 0;
    }

    .preview .stream-preview-fallback {
      z-index: 2;
    }

    .output-pane {
      min-width: 0;
      min-height: 0;
      position: relative;
    }

    /* ── Debug stream overlay ─────────────────────────────── */
    .debug-overlay {
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      z-index: 20;
      background: rgba(10, 10, 18, 0.92);
      border-top: 1px solid rgba(255,255,255,0.12);
      backdrop-filter: blur(4px);
      font-size: 11px;
      color: #b0b8cc;
      display: flex;
      flex-direction: column;
      max-height: 38vh;
    }

    .debug-overlay-empty {
      flex-direction: row;
      align-items: center;
      gap: 10px;
      padding: 6px 10px;
      justify-content: space-between;
    }

    .debug-overlay-toolbar {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 4px 8px;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      flex-shrink: 0;
    }

    .debug-overlay-toolbar button {
      font-size: 11px;
      padding: 2px 8px;
      min-height: 20px;
      background: rgba(40,40,56,0.9);
      border: 1px solid rgba(255,255,255,0.14);
      color: #c0c8e0;
      cursor: pointer;
      border-radius: 3px;
    }

    .debug-overlay-toolbar button:hover { background: rgba(60,60,80,0.9); }

    .debug-overlay-toolbar span { flex: 1; }
    .debug-channel-label { font-weight: 600; color: #7ec8ff; }

    .debug-mosaic-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
      padding: 6px;
      overflow-y: auto;
      overscroll-behavior: contain;
    }

    .debug-mosaic-cell {
      position: relative;
      flex: 0 0 auto;
      cursor: pointer;
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 3px;
      overflow: hidden;
      transition: border-color 0.15s;
    }

    .debug-mosaic-cell:hover { border-color: #7ec8ff; }

    .debug-mosaic-cell img {
      display: block;
      height: 90px;
      width: auto;
      max-width: 180px;
      object-fit: contain;
      background: #0a0a12;
    }

    .debug-cell-label {
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      background: rgba(0,0,0,0.72);
      padding: 2px 4px;
      font-size: 10px;
      color: #b8c4e0;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .debug-overlay-single {
      flex: 1;
    }

    .debug-preview-full {
      display: block;
      width: 100%;
      max-height: calc(38vh - 32px);
      object-fit: contain;
      background: #0a0a12;
    }

    .debug-preview-text {
      padding: 8px;
      overflow-y: auto;
      max-height: calc(38vh - 32px);
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
      align-content: flex-start;
    }

    .debug-text-cell {
      width: 100%;
      cursor: default;
    }

    .debug-tag-content {
      padding: 4px;
      display: flex;
      flex-wrap: wrap;
      gap: 3px;
      min-height: 60px;
      align-content: flex-start;
    }

    .debug-tag-chip {
      display: inline-block;
      background: rgba(60, 80, 120, 0.6);
      border: 1px solid rgba(100, 140, 200, 0.3);
      border-radius: 3px;
      padding: 1px 5px;
      font-size: 10px;
      color: #b8d0f0;
      white-space: nowrap;
    }

    .debug-tag-empty {
      font-size: 10px;
      color: #606880;
      font-style: italic;
    }

    .context-panel {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 8px;
      min-height: 0;
      overflow: hidden;
    }

    .context-body {
      min-height: 0;
      overflow-y: auto;
      overflow-x: hidden;
      overscroll-behavior: contain;
      padding: 6px;
      box-sizing: border-box;
    }

    .context-body textarea {
      min-height: 48px;
      max-height: 58px;
    }

    .context-tabs {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(0, 1fr));
      gap: 2px;
      padding: 6px 6px 0;
    }

    .context-tabs button {
      min-height: 24px;
      padding: 2px 4px;
      color: #050507;
      background: #e8e8ff;
      border-color: #24245d;
      border-radius: 0;
      font-size: 12px;
    }

    .context-tabs button.active {
      background: #3f77ff;
      color: #050507;
    }

    .check-field {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 6px;
      align-items: center;
      font-size: 13px;
    }

    .resolution-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }

    .source-tab {
      gap: 8px;
    }

    .source-preview {
      display: grid;
      place-items: center;
      min-height: 120px;
      border: 1px solid #24245d;
      background-color: #b8bdc4;
      background-image:
        linear-gradient(45deg, #8f969f 25%, transparent 25%),
        linear-gradient(-45deg, #8f969f 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #8f969f 75%),
        linear-gradient(-45deg, transparent 75%, #8f969f 75%);
      background-position: 0 0, 0 8px, 8px -8px, -8px 0;
      background-size: 16px 16px;
      overflow: hidden;
    }

    .source-preview img {
      max-width: 100%;
      max-height: 180px;
      object-fit: contain;
    }

    .source-select {
      min-height: 112px;
      font-size: 12px;
    }

    .sampling-controls,
    .scene-tab,
    .record-tab {
      min-width: 0;
    }

    .sampling-controls {
      display: grid;
      gap: 8px;
    }

    .advanced-sampling {
      display: grid;
      gap: 6px;
      padding: 6px 0;
      border-top: 1px solid rgba(36, 36, 93, 0.35);
      border-bottom: 1px solid rgba(36, 36, 93, 0.35);
    }

    .sampling-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 6px;
    }

    .popup-picker {
      position: relative;
      width: 100%;
      min-width: 0;
      color: #111;
    }

    .picker-trigger {
      display: grid;
      grid-template-columns: minmax(0, 1fr) max-content;
      gap: 10px;
      align-items: center;
      width: 100%;
      max-width: 100%;
      min-height: 32px;
      padding: 5px 8px;
      box-sizing: border-box;
      border: 1px solid #30304f;
      border-radius: 3px;
      color: #111;
      background: #f8f8ff;
      cursor: pointer;
      text-align: left;
    }

    .picker-trigger span,
    .picker-trigger strong {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .picker-trigger strong {
      font-size: 12px;
      font-weight: 700;
      text-align: left;
    }

    .picker-trigger span {
      color: #4f5965;
      font-size: 11px;
    }

    .picker-dialog {
      position: fixed;
      inset: var(--picker-top) auto auto var(--picker-left);
      width: var(--picker-width);
      max-width: calc(100vw - 16px);
      max-height: var(--picker-max-height);
      margin: 0;
      padding: 0;
      border: 2px solid #24245d;
      border-radius: 0;
      background: #f0f0ff;
      color: #111;
      box-shadow: 0 18px 48px rgba(0, 0, 0, 0.44);
    }

    .picker-dialog::backdrop {
      background: transparent;
    }

    .picker-dialog-panel {
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      gap: 8px;
      max-height: var(--picker-max-height);
      padding: 10px;
      box-sizing: border-box;
    }

    .picker-dialog-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) max-content;
      gap: 8px;
      align-items: center;
      min-height: 30px;
    }

    .picker-dialog-head strong {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .picker-dialog-head button {
      width: 28px;
      min-width: 28px;
      min-height: 28px;
      padding: 0;
    }

    .picker-list {
      display: grid;
      gap: 3px;
      overflow: auto;
      overscroll-behavior: contain;
    }

    .dialog-picker-list {
      max-height: calc(var(--picker-max-height) - 66px);
      padding-right: 2px;
    }

    .picker-option {
      display: grid;
      grid-template-columns: minmax(0, 1fr) max-content;
      gap: 8px;
      align-items: center;
      min-height: 28px;
      padding: 4px 7px;
      color: #111;
      background: #ffffff;
      border-color: #c7cbe8;
      text-align: left;
    }

    .picker-option span,
    .picker-option small {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .picker-option small {
      color: #4d5560;
      font-size: 11px;
    }

    .picker-option.selected {
      color: #f3f1ec;
      background: #1f63a8;
      border-color: #2f9bff;
    }

    .picker-dialog .search {
      width: 100%;
      max-width: none;
      margin-bottom: 6px;
    }

    .lora-picker {
      display: grid;
      gap: 6px;
      color: #111;
    }

    .record-button {
      display: grid;
      grid-template-columns: max-content minmax(0, 1fr);
      gap: 8px;
      align-items: center;
      width: 100%;
      min-height: 38px;
      padding: 7px 9px;
    }

    .record-button.recording {
      border-color: #f04f65;
      background: #4b1720;
    }

    .record-dot {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: #f04f65;
      box-shadow: 0 0 0 2px rgba(240, 79, 101, 0.24);
    }

    .mask-tab,
    .regions-tab {
      display: grid;
      gap: 8px;
    }

    .region-list {
      display: grid;
      gap: 8px;
    }

    .region-card {
      display: grid;
      gap: 7px;
      padding: 8px;
      border: 1px solid #d3d7dc;
      border-radius: 4px;
      background: #f5f6f8;
    }

    .region-head {
      display: grid;
      grid-template-columns: 28px minmax(0, 1fr) max-content;
      gap: 6px;
      align-items: center;
    }

    .region-head input {
      min-width: 0;
    }

    .region-color {
      width: 28px;
      min-width: 28px;
      height: 28px;
      min-height: 28px;
      padding: 2px;
    }

    .region-color-button {
      background: var(--region-color);
      border: 1px solid #8b94a3;
      cursor: pointer;
    }

    .interval-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 6px;
    }

    .region-meta {
      font-size: 11px;
      color: #5b6570;
    }

    .schedule-timeline {
      display: grid;
      gap: 5px;
      min-width: 0;
      padding: 7px;
      border: 1px solid #30304f;
      background: #f8f8ff;
    }

    .timeline-head,
    .timeline-row {
      display: grid;
      grid-template-columns: minmax(72px, 0.34fr) minmax(0, 1fr);
      gap: 8px;
      align-items: center;
      min-width: 0;
    }

    .timeline-head {
      grid-template-columns: minmax(72px, 0.34fr) 1fr max-content max-content;
      color: #4d5560;
      font-size: 11px;
    }

    .timeline-row > span {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 11px;
    }

    .timeline-row.muted {
      opacity: 0.38;
    }

    .timeline-track {
      position: relative;
      height: 18px;
      overflow: hidden;
      border: 1px solid #c4c8dd;
      background:
        linear-gradient(90deg, transparent calc(50% - 1px), #ccd0e5 calc(50% - 1px), #ccd0e5 calc(50% + 1px), transparent calc(50% + 1px)),
        #eef0fb;
    }

    .timeline-track i {
      position: absolute;
      top: 3px;
      bottom: 3px;
      border-radius: 2px;
      box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.22);
    }

    .timeline-empty {
      min-height: 24px;
      display: grid;
      place-items: center;
      color: #5b6570;
      font-size: 11px;
    }

    .status-bar {
      display: grid;
      position: relative;
      grid-template-columns: max-content max-content max-content max-content max-content minmax(72px, 1fr) max-content minmax(220px, 360px);
      gap: 0;
      align-items: center;
      min-width: 0;
      padding: 0;
      border: 2px solid #24245d;
      background: #f0f0ff;
      color: #111;
      overflow: visible;
      font-size: 13px;
    }

    .status-bar > span {
      min-width: 0;
      min-height: 26px;
      padding: 5px 10px;
      border-right: 1px solid #c7c8eb;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .footer-task-host {
      position: relative;
      display: grid;
      gap: 3px;
      min-width: 0;
      min-height: 26px;
      padding: 3px 10px;
      border-left: 1px solid #c7c8eb;
    }

    .footer-error-host {
      position: relative;
      min-height: 26px;
      border-left: 1px solid #c7c8eb;
      border-right: 1px solid #c7c8eb;
    }

    .footer-error-button {
      display: grid;
      place-items: center;
      width: 100%;
      min-height: 26px;
      padding: 0 10px;
      border: 0;
      color: #555b67;
      background: transparent;
      font-size: 12px;
      white-space: nowrap;
    }

    .footer-error-button.active {
      color: #b01222;
      background: #ffe2e5;
      font-weight: 700;
    }

    .error-popover {
      position: absolute;
      right: 0;
      bottom: calc(100% + 8px);
      z-index: 42;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 8px;
      width: min(520px, calc(100vw - 28px));
      max-height: min(340px, 56svh);
      padding: 8px;
      border: 2px solid #6f1f2b;
      background: #180e12;
      box-shadow: 0 12px 36px rgba(0, 0, 0, 0.42);
    }

    .error-popover-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) max-content;
      gap: 8px;
      align-items: center;
      color: #ffecef;
    }

    .error-popover-head button {
      min-height: 24px;
      padding: 0 8px;
    }

    .error-list {
      display: grid;
      gap: 6px;
      min-height: 0;
      overflow: auto;
    }

    .error-item {
      display: grid;
      gap: 3px;
      width: 100%;
      min-height: 0;
      padding: 7px;
      border: 1px solid #7f2a35;
      color: #ffe8eb;
      background: #2a1419;
      text-align: left;
    }

    .error-item span {
      color: #ffb8c0;
      font-size: 11px;
    }

    .error-item p {
      margin: 0;
      overflow-wrap: anywhere;
      line-height: 1.35;
    }

    .footer-task-button {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, max-content));
      gap: 8px;
      align-items: center;
      justify-content: space-between;
      min-height: 22px;
      min-width: 0;
      padding: 0;
      border: 0;
      color: #111;
      background: transparent;
      font-size: 12px;
      text-align: left;
    }

    .footer-task-button span {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .footer-task-button:hover {
      color: #050507;
      background: transparent;
    }

    .footer-task-bar {
      height: 5px;
      overflow: hidden;
      border-radius: 999px;
      background: #c9cbef;
    }

    .footer-task-bar span {
      display: block;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, #3f77ff, #64c6b8);
      transition: width 120ms ease;
    }

    .task-popover {
      position: absolute;
      right: 0;
      bottom: calc(100% + 8px);
      z-index: 40;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 8px;
      width: min(420px, calc(100vw - 28px));
      max-height: min(360px, 58svh);
      padding: 8px;
      border: 2px solid #24245d;
      background: #10141b;
      box-shadow: 0 12px 36px rgba(0, 0, 0, 0.42);
    }

    .task-popover-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) max-content;
      gap: 8px;
      align-items: center;
      color: #e6edf7;
    }

    .task-popover-head button {
      min-height: 24px;
      padding: 0 8px;
    }

    .task-popover-list {
      display: grid;
      gap: 6px;
      min-height: 0;
      overflow: auto;
    }

    .task-empty {
      min-height: 80px;
      display: grid;
      place-items: center;
      color: #aeb7c6;
    }

    .preview.native {
      max-height: none;
      min-height: 0;
    }

    .empty {
      color: #8c9498;
      font-size: 14px;
    }

    @media (max-width: 1180px) {
      :host {
        overflow: auto;
      }

      .shell {
        height: auto;
        min-height: 100svh;
        overflow: visible;
      }

      .topbar {
        grid-template-columns: 1fr;
      }

      .metrics {
        justify-content: start;
        grid-template-columns: repeat(2, minmax(72px, max-content));
      }

      .workspace {
        grid-template-columns: minmax(150px, var(--left-panel-width, 220px)) minmax(260px, 1fr) minmax(220px, var(--right-panel-width, 320px));
        overflow: auto hidden;
      }

      .workspace.portrait .viewer,
      .workspace.landscape .viewer {
        grid-template-columns: 1fr;
        grid-template-rows: auto auto;
      }

      .work-pane {
        min-height: 70svh;
      }
    }

    @supports (grid-template-rows: subgrid) {
      .workspace {
        grid-template-rows: minmax(0, 1fr);
      }

      .tools-panel,
      .studio,
      .inspector {
        grid-row: 1;
      }
    }
  `
}
