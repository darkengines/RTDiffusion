import type { ColorPlannerMode, SizePreset, ToolMode } from '../types'
import { ARCHIVE_ROOT_URL, CANVAS_CUSTOM_PROPERTIES } from '../project-state'

// ── Canvas / stage ──────────────────────────────────────────────────────────
export { ARCHIVE_ROOT_URL as ARCHIVE_ROOT, CANVAS_CUSTOM_PROPERTIES }

export const OPTIONS_KEY = 'rtdiffusion.options.v1'
export const OPTIONS_VERSION = 4
export const PROJECT_STATE_KEY = 'rtdiffusion.projectState.v1'
export const PROJECT_RESOURCE_PREFIX = 'rtdiffusion.resource.v1.'
export const PROJECT_RESOURCE_DB = 'rtdiffusion.resources.v1'
export const PROJECT_RESOURCE_STORE = 'resources'
export const PROJECT_VERSION = 1
export const HISTORY_LIMIT = 60
export const SCENE_PRESETS_KEY = 'rtdiffusion.scenePresets.v1'
export const SAMPLER_PRESETS_KEY = 'rtdiffusion.samplerPresets.v1'

// ── Size presets ────────────────────────────────────────────────────────────
export const SIZE_PRESETS: SizePreset[] = [
  { label: '512 x 512', width: 512, height: 512 },
  { label: '768 x 768', width: 768, height: 768 },
  { label: '832 x 1216', width: 832, height: 1216 },
  { label: '1216 x 832', width: 1216, height: 832 },
  { label: '704 x 1024', width: 704, height: 1024 },
  { label: '1024 x 704', width: 1024, height: 704 },
  { label: '576 x 832', width: 576, height: 832 },
  { label: '832 x 576', width: 832, height: 576 },
]

// ── Samplers & schedulers ───────────────────────────────────────────────────
export const SAMPLERS = [
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

export const SCHEDULERS = [
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

// ── Colour palettes ─────────────────────────────────────────────────────────
export const PALETTE = [
  '#000000', '#1f2329', '#3b424c', '#6f7782', '#a9b0ba', '#d8dce2', '#ffffff', '#fff4d7',
  '#6d1f2c', '#b83246', '#f04f65', '#ff8a9a', '#ffd1d8', '#7a2f16', '#c75524', '#f28b38',
  '#ffbf69', '#ffe0ad', '#735200', '#b98300', '#e9b44c', '#ffe066', '#fff3a8', '#4d6219',
  '#7aa12b', '#a8d84f', '#d8f59a', '#123d30', '#1f7a5f', '#35b58f', '#79e0c0', '#c9ffee',
  '#123d44', '#1e6f7a', '#35b7c8', '#83e6f0', '#d2fbff', '#173b70', '#2f6dd1', '#5f8cff',
  '#9ab5ff', '#d8e4ff', '#2f246f', '#5d46c7', '#8a6cff', '#bfaeff', '#ebe4ff', '#551f6d',
  '#9638c7', '#c05cff', '#dea9ff', '#f4ddff', '#6b1f4b', '#bd3f84', '#f266b2', '#ffa6d6',
  '#ffd8ed', '#3c2820', '#704936', '#a66b4b', '#d4976b', '#f1c9a6', '#24413a', '#456d63',
]

export const QUICK_PALETTE = [
  '#000000', '#1f2329', '#3b424c', '#6f7782', '#a9b0ba', '#d8dce2', '#ffffff', '#fff4d7',
  '#6d1f2c', '#b83246', '#f04f65', '#ff8a9a', '#f28b38', '#e9b44c', '#35b58f', '#5f8cff',
]

// ── Tool definitions ────────────────────────────────────────────────────────
export const TOOL_DEFS: Record<ToolMode, { icon: string; label: string; shortcut: string }> = {
  select: { icon: '↖️', label: 'Move Selected Pixels', shortcut: 'Ctrl+1' },
  moveSelection: { icon: '✥', label: 'Move Selection', shortcut: 'Ctrl+2' },
  region: { icon: '▣', label: 'Rectangle Select', shortcut: 'Ctrl+3' },
  lassoSelect: { icon: '〰', label: 'Lasso Select', shortcut: 'Ctrl+4' },
  ellipseSelect: { icon: '◌', label: 'Ellipse Select', shortcut: 'Ctrl+5' },
  magicWand: { icon: '✦', label: 'Magic Wand', shortcut: 'Ctrl+6' },
  brush: { icon: '🖌️', label: 'Paintbrush', shortcut: 'Ctrl+7' },
  eraser: { icon: '🧹', label: 'Eraser', shortcut: 'Ctrl+8' },
  text: { icon: 'T', label: 'Text', shortcut: 'Ctrl+9' },
  line: { icon: '📐', label: 'Line / Curve', shortcut: 'Ctrl+0' },
  rect: { icon: '⬜', label: 'Rectangle Shape', shortcut: 'Ctrl+Alt+1' },
  ellipse: { icon: '◯', label: 'Ellipse Shape', shortcut: 'Ctrl+Alt+2' },
  fill: { icon: '🪣', label: 'Paint Bucket', shortcut: 'Ctrl+Alt+3' },
  eyedropper: { icon: '💧', label: 'Color Picker', shortcut: 'Ctrl+Alt+4' },
}

export const TOOL_SHORTCUTS: Record<string, ToolMode> = {
  '1': 'select',
  '2': 'moveSelection',
  '3': 'region',
  '4': 'lassoSelect',
  '5': 'ellipseSelect',
  '6': 'magicWand',
  '7': 'brush',
  '8': 'eraser',
  '9': 'text',
  '0': 'line',
}

export const COLOR_PLANNER_MODES: { label: string; value: ColorPlannerMode }[] = [
  { label: 'Complement', value: 'complementary' },
  { label: 'Analogous', value: 'analogous' },
  { label: 'Triad', value: 'triad' },
  { label: 'Tetrad', value: 'tetrad' },
  { label: 'Split', value: 'split' },
  { label: 'Mono', value: 'monochrome' },
]
