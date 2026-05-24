import { map } from 'nanostores'
import type { ScenePanelTab, StreamMotionMode, StreamRuntimePreset } from '../types'

export type StreamStatus = 'offline' | 'connecting' | 'streaming' | 'error' | 'backend unavailable' | string

export interface StreamState {
  scenePanelTab: ScenePanelTab
  isStreaming: boolean
  realtimeVideoActive: boolean
  status: StreamStatus
  fps: number
  displayFps: number
  latency: number
  backendMode: string
  outputImage: string
  outputImageNonce: number
  realtimeDevice: string
  sessionDirectory: string
  runtimePreset: StreamRuntimePreset
  timestepIndices: string
  frameBufferSize: number
  cfgType: string
  similarityThreshold: number
  maxSkipFrames: number
  motionMode: StreamMotionMode
  motionIntensity: number
  motionSpeed: number
  tritonCompile: boolean
  sanaSteps: number
  debugStreamsEnabled: boolean
  debugChannelNames: string[]
  debugMosaicMode: boolean
  debugPreviewChannel: string
}

export const $stream = map<StreamState>({
  scenePanelTab: 'sdxl',
  isStreaming: false,
  realtimeVideoActive: false,
  status: 'offline',
  fps: 0,
  displayFps: 0,
  latency: 0,
  backendMode: 'mock',
  outputImage: '',
  outputImageNonce: 0,
  realtimeDevice: '',
  sessionDirectory: '',
  runtimePreset: 'diffusers',
  timestepIndices: '16,32',
  frameBufferSize: 1,
  cfgType: 'self',
  similarityThreshold: 0.98,
  maxSkipFrames: 0,
  motionMode: 'none',
  motionIntensity: 0.35,
  motionSpeed: 1,
  tritonCompile: false,
  sanaSteps: 2,
  debugStreamsEnabled: false,
  debugChannelNames: [],
  debugMosaicMode: true,
  debugPreviewChannel: '',
})
