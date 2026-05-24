import { map } from 'nanostores'
import type { MotionTaskProgress, VideoFpsMode, VideoLoopMode } from '../types'

export interface MotionState {
  prompt: string
  arrivalPrompt: string
  negativePrompt: string
  model: string
  fps: number
  duration: number
  intensity: number
  fastVideoModelPath: string
  device: string
  activeFrame: string
  activeVideo: string
  activeIsStream: boolean
  streamHasSegments: boolean
  isGenerating: boolean
  chunkIndex: number
  task: MotionTaskProgress | undefined
  sanaVideoModel: string
  sanaVideoNumFrames: number
  sanaVideoGuidance: number
  sanaVideoSteps: number
  sanaVideoTask: Record<string, unknown> | null
  sanaVideoTaskId: string
  videoUrl: string
  videoLoopMode: VideoLoopMode
  videoStart: number
  videoEnd: number
  videoFpsMode: VideoFpsMode
  videoFps: number
  videoDuration: number
}

export const $motion = map<MotionState>({
  prompt: 'adult woman holding a lollipop, calm gaze at the viewer, subtle natural motion, same identity, same lighting',
  arrivalPrompt: '',
  negativePrompt: 'identity change, outfit change, camera cut, large motion, deformation, flicker',
  model: 'causal-forcing-1step',
  fps: 16,
  duration: 2,
  intensity: 1,
  fastVideoModelPath: '',
  device: '',
  activeFrame: '',
  activeVideo: '',
  activeIsStream: false,
  streamHasSegments: false,
  isGenerating: false,
  chunkIndex: 0,
  task: undefined,
  sanaVideoModel: '480p',
  sanaVideoNumFrames: 81,
  sanaVideoGuidance: 6.0,
  sanaVideoSteps: 20,
  sanaVideoTask: null,
  sanaVideoTaskId: '',
  videoUrl: '',
  videoLoopMode: 'loop',
  videoStart: 0,
  videoEnd: 0,
  videoFpsMode: 'fps',
  videoFps: 30,
  videoDuration: 0,
})
