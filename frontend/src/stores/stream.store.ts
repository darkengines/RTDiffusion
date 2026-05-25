import { map } from 'nanostores'
import type { ScenePanelTab, StreamMotionMode, StreamOutputTransport, StreamRuntimePreset, StreamTimingMap, StreamVaeMode } from '../types'

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
  motionMode: StreamMotionMode
  motionIntensity: number
  motionSpeed: number
  vaeMode: StreamVaeMode
  outputTransport: StreamOutputTransport
  outputTimings: StreamTimingMap
  outputTimingsExpanded: boolean
  tritonCompile: boolean
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
  motionMode: 'none',
  motionIntensity: 0.35,
  motionSpeed: 1,
  vaeMode: 'auto',
  outputTransport: 'video',
  outputTimings: {},
  outputTimingsExpanded: false,
  tritonCompile: false,
  debugStreamsEnabled: false,
  debugChannelNames: [],
  debugMosaicMode: true,
  debugPreviewChannel: '',
})
