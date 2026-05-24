import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { $stream } from './stores/stream.store'
import { $scene } from './stores/scene.store'

class FakeSocket {
  static OPEN = 1
  readyState = FakeSocket.OPEN
  bufferedAmount = 0
  sent: Record<string, unknown>[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  binaryType = ''

  send(data: string | ArrayBuffer | Blob) {
    if (typeof data === 'string') {
      this.sent.push(JSON.parse(data) as Record<string, unknown>)
    }
  }

  close() {
    this.readyState = 3
    this.onclose?.()
  }
}

const sockets: FakeSocket[] = []

vi.mock('./services/api', () => ({
  openInpaintSocket: () => {
    const socket = new FakeSocket()
    sockets.push(socket)
    return socket as unknown as WebSocket
  },
  openRenderSessionSocket: () => {
    const socket = new FakeSocket()
    sockets.push(socket)
    return socket as unknown as WebSocket
  },
  rtcDeletePeer: vi.fn(),
  rtcOffer: vi.fn(),
  rtcStreamUrl: vi.fn(),
}))

vi.mock('./services/layer.service', () => ({ upsertTask: vi.fn() }))
vi.mock('./stores/ui.store', () => ({ reportError: vi.fn() }))
vi.mock('./services/mask-blob.service', () => ({
  clearMaskBlobCache: vi.fn(),
  setMaskBlobUploader: vi.fn(),
}))

async function flushTimers() {
  await new Promise((resolve) => window.setTimeout(resolve, 0))
}

async function wait(ms: number) {
  await new Promise((resolve) => window.setTimeout(resolve, ms))
}

async function waitForSent(socket: FakeSocket, count: number) {
  for (let attempt = 0; attempt < 20; attempt++) {
    if (socket.sent.length >= count) return
    await wait(10)
  }
}

const dataSvg = (fill: string) => `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"><rect width="1" height="1" fill="${fill}"/></svg>`)}`
const IMAGE_RESOURCE = dataSvg('red')
const MASK_RESOURCE = dataSvg('green')
const LAYER_RESOURCE = dataSvg('blue')

describe('startInpaintStream live staging behavior', () => {
  beforeEach(() => {
    vi.stubGlobal('WebSocket', FakeSocket)
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn((blob: Blob) => `blob:${blob.type}:${blob.size}`) })
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() })
    sockets.length = 0
    $stream.set({
      ...$stream.get(),
      scenePanelTab: 'sdxl',
      isStreaming: false,
      status: 'offline',
      outputImage: '',
      fps: 0,
      latency: 0,
    })
    $scene.setKey('seedMode', 'fixed')
    $scene.setKey('seedRotationMode', 'off')
    $scene.setKey('seedRotationIntervalMs', 1000)
  })

  afterEach(async () => {
    const { stopStream } = await import('./services/stream.service')
    stopStream()
    vi.unstubAllGlobals()
  })

  it('drops delayed responses and applies only the latest input response', async () => {
    const service = await import('./services/stream.service')
    let image = 'input-a'
    service.wireFullFrameExporter(async () => JSON.stringify({
      image,
      mask: 'mask',
      width: 512,
      height: 512,
    }))

    service.startInpaintStream()
    const socket = sockets[0]
    socket.onopen?.()
    await waitForSent(socket, 1)

    expect(socket.sent).toHaveLength(1)
    expect(socket.sent[0].client_frame_id).toBe(1)
    expect(socket.sent[0].client_input_id).toBe(1)

    image = 'input-b'
    service.requestInpaintFrameUpdate()

    await wait(150)

    expect(socket.sent).toHaveLength(2)
    expect(socket.sent[1].client_frame_id).toBe(2)
    expect(socket.sent[1].client_input_id).toBe(2)
    expect(socket.sent[1].image).toBe('input-b')

    socket.onmessage?.({
      data: JSON.stringify({
        client_frame_id: 1,
        client_input_id: 1,
        image: 'old-output',
        fps: 1,
        latency_ms: 10,
        mode: 'sdxl',
      }),
    })
    await flushTimers()

    expect($stream.get().outputImage).toBe('')

    socket.onmessage?.({
      data: JSON.stringify({
        client_frame_id: 2,
        client_input_id: 2,
        image: 'new-output',
        fps: 1,
        latency_ms: 10,
        mode: 'sdxl',
      }),
    })

    expect($stream.get().outputImage).toBe('new-output')
    await wait(50)
    expect(socket.sent).toHaveLength(2)
  })

  it('does not render another SDXL/Z-Image frame until input changes', async () => {
    const service = await import('./services/stream.service')
    let image = 'input-a'
    service.wireFullFrameExporter(async () => JSON.stringify({
      image,
      mask: 'mask',
      width: 512,
      height: 512,
    }))

    service.startInpaintStream()
    const socket = sockets[0]
    socket.onopen?.()
    await waitForSent(socket, 1)

    socket.onmessage?.({
      data: JSON.stringify({
        client_frame_id: 1,
        client_input_id: 1,
        image: 'first-output',
        fps: 1,
        latency_ms: 10,
        mode: 'sdxl',
      }),
    })
    expect($stream.get().outputImage).toBe('first-output')
    await wait(50)
    expect(socket.sent).toHaveLength(1)

    image = 'input-b'
    service.requestInpaintFrameUpdate()
    await wait(150)
    const latestSent = socket.sent.at(-1)!
    expect(latestSent.image).toBe('input-b')
    expect(latestSent.client_input_id).toBe(2)
    socket.onmessage?.({
      data: JSON.stringify({
        client_frame_id: latestSent.client_frame_id,
        client_input_id: 2,
        image: 'second-output',
        fps: 1,
        latency_ms: 10,
        mode: 'sdxl',
      }),
    })

    expect($stream.get().outputImage).toBe('second-output')
  })

  it('drops a response when a newer edit is dirty but not yet sent', async () => {
    const service = await import('./services/stream.service')
    service.wireFullFrameExporter(async () => JSON.stringify({
      image: 'input-a',
      mask: 'mask',
      width: 512,
      height: 512,
    }))

    service.startInpaintStream()
    const socket = sockets[0]
    socket.onopen?.()
    await waitForSent(socket, 1)

    expect(socket.sent).toHaveLength(1)
    service.requestInpaintFrameUpdate()

    socket.onmessage?.({
      data: JSON.stringify({
        client_frame_id: 1,
        client_input_id: 1,
        image: 'visible-output',
        fps: 1,
        latency_ms: 10,
        mode: 'sdxl',
      }),
    })

    expect($stream.get().outputImage).toBe('')
  })

  it('keeps only the newest staged user input displayable after several inflight edits', async () => {
    const service = await import('./services/stream.service')
    let image = 'input-a'
    service.wireFullFrameExporter(async () => JSON.stringify({
      image,
      mask: 'mask',
      width: 512,
      height: 512,
    }))

    service.startInpaintStream()
    const socket = sockets[0]
    socket.onopen?.()
    await waitForSent(socket, 1)

    image = 'input-b'
    service.requestInpaintFrameUpdate()
    await wait(150)
    image = 'input-c'
    service.requestInpaintFrameUpdate()
    await wait(150)

    expect(socket.sent.map((packet) => packet.client_input_id)).toEqual([1, 2, 3])
    expect(socket.sent[2].image).toBe('input-c')

    for (const packet of socket.sent.slice(0, 2)) {
      socket.onmessage?.({
        data: JSON.stringify({
          client_frame_id: packet.client_frame_id,
          client_input_id: packet.client_input_id,
          image: `old-output-${packet.client_input_id}`,
          fps: 1,
          latency_ms: 10,
          mode: 'sdxl',
        }),
      })
    }
    expect($stream.get().outputImage).toBe('')

    socket.onmessage?.({
      data: JSON.stringify({
        client_frame_id: socket.sent[2].client_frame_id,
        client_input_id: 3,
        image: 'latest-output',
        fps: 1,
        latency_ms: 10,
        mode: 'sdxl',
      }),
    })
    expect($stream.get().outputImage).toBe('latest-output')
  })

  it('populates image resource debug data from inpaint websocket payloads', async () => {
    const service = await import('./services/stream.service')
    service.wireFullFrameExporter(async () => JSON.stringify({
      image: IMAGE_RESOURCE,
      mask: MASK_RESOURCE,
      width: 512,
      height: 512,
      layer_conditions: [{
        layer_id: 'layer a',
        region_id: 'mask y',
        image: LAYER_RESOURCE,
      }],
    }))

    service.startInpaintStream()
    const socket = sockets[0]
    socket.onopen?.()
    await waitForSent(socket, 1)
    await wait(10)

    const resources = service.getSceneResourceDebugData()
    const resourceLabels = [...resources.keys()].sort()
    expect(resourceLabels).toEqual(['layer_a/RGBA'])

    service.wireFullFrameExporter(async () => JSON.stringify({
      image: IMAGE_RESOURCE,
      mask: MASK_RESOURCE,
      width: 512,
      height: 512,
      layer_conditions: [],
    }))
    service.requestInpaintFrameUpdate()
    await wait(150)
    await wait(10)

    const nextLabels = [...service.getSceneResourceDebugData().keys()].sort()
    expect(nextLabels).toHaveLength(0)
  })

  it('routes unified realtime updates to the inpaint websocket when WebRTC is inactive', async () => {
    const service = await import('./services/stream.service')
    let prompt = 'initial'
    service.wireFullFrameExporter(async () => ({
      image: IMAGE_RESOURCE,
      mask: MASK_RESOURCE,
      prompt,
      width: 512,
      height: 512,
      layer_conditions: [{
        layer_id: 'layer-a',
        region_id: 'inherited:layer-a',
        prompt,
        image: LAYER_RESOURCE,
        prompt_mask: MASK_RESOURCE,
      }],
    }))

    service.startInpaintStream()
    const socket = sockets[0]
    socket.onopen?.()
    await waitForSent(socket, 1)

    prompt = 'monster'
    service.requestRealtimeFrameUpdate({ refreshLayerConditions: true })
    await wait(150)
    await waitForSent(socket, 2)

    expect(socket.sent[1].prompt).toBe('monster')
    const condition = (socket.sent[1].layer_conditions as Record<string, unknown>[])[0]
    expect(condition.prompt).toBe('monster')
  })
})

describe('render session WebSocket server-driven input readiness', () => {
  beforeEach(() => {
    vi.stubGlobal('WebSocket', FakeSocket)
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn((blob: Blob) => `blob:${blob.type}:${blob.size}`) })
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() })
    sockets.length = 0
    $stream.set({
      ...$stream.get(),
      scenePanelTab: 'sdxl',
      isStreaming: false,
      status: 'offline',
      outputImage: '',
      fps: 0,
      latency: 0,
    })
    $scene.setKey('seedMode', 'fixed')
    $scene.setKey('seedRotationMode', 'off')
    $scene.setKey('seedRotationIntervalMs', 1000)
  })

  afterEach(async () => {
    const { stopStream } = await import('./services/stream.service')
    stopStream()
    vi.unstubAllGlobals()
  })

  it('captures and sends only when the backend emits input_ready', async () => {
    const service = await import('./services/stream.service')
    let prompt = 'input-a'
    let exportCount = 0
    service.wireFullFrameExporter(async () => {
      exportCount++
      return JSON.stringify({
        image: 'static-image',
        mask: 'static-mask',
        prompt,
        width: 512,
        height: 512,
      })
    })

    await service.startWebRtc()
    const socket = sockets[0]
    socket.onopen?.()
    socket.onmessage?.({ data: JSON.stringify({ type: 'hello', session_id: 'session-a' }) })
    await flushTimers()

    expect(exportCount).toBe(0)
    expect(socket.sent).toHaveLength(0)

    prompt = 'input-b'
    service.requestWebRtcSettingsUpdate()
    service.requestWebRtcSettingsUpdate()
    await flushTimers()

    expect(exportCount).toBe(0)
    expect(socket.sent).toHaveLength(0)

    socket.onmessage?.({ data: JSON.stringify({ type: 'input_ready', reason: 'session_started' }) })
    await waitForSent(socket, 1)

    expect(exportCount).toBe(1)
    expect(socket.sent).toHaveLength(1)
    expect(socket.sent[0].type).toBe('scene')
    expect(((socket.sent[0].scene as Record<string, unknown>).settings as Record<string, unknown>).prompt).toBe('input-b')

    prompt = 'input-c'
    service.requestWebRtcSettingsUpdate()
    socket.onmessage?.({ data: JSON.stringify({ type: 'frame', data: 'data:image/jpeg;base64,abc' }) })
    await flushTimers()

    expect(exportCount).toBe(1)
    expect(socket.sent).toHaveLength(1)

    socket.onmessage?.({ data: JSON.stringify({ type: 'input_ready', reason: 'rendered' }) })
    await waitForSent(socket, 2)

    expect(exportCount).toBe(2)
    expect(socket.sent).toHaveLength(2)
    expect(((socket.sent[1].scene as Record<string, unknown>).settings as Record<string, unknown>).prompt).toBe('input-c')
  })

  it('sends layer prompt metadata patches without resending resources', async () => {
    const service = await import('./services/stream.service')
    service.wireFullFrameExporter(async () => ({
      image: IMAGE_RESOURCE,
      mask: MASK_RESOURCE,
      prompt: 'scene prompt',
      width: 512,
      height: 512,
      layer_conditions: [{
        layer_id: 'layer-a',
        region_id: 'inherited:layer-a',
        prompt: 'scene prompt',
        negative_prompt: '',
        image: LAYER_RESOURCE,
        prompt_mask: MASK_RESOURCE,
      }],
    }))
    service.wireSceneSettingsExporter(async () => ({ prompt: 'scene prompt', width: 512, height: 512 }))
    service.wireLayerConditionsMetadataExporter(async () => [{
      layer_id: 'layer-a',
      region_id: 'inherited:layer-a',
      prompt: 'monster',
      negative_prompt: '',
      name: 'monster',
      cfg: 1,
      denoise: 1,
    }])

    await service.startWebRtc()
    const socket = sockets[0]
    socket.onopen?.()
    socket.onmessage?.({ data: JSON.stringify({ type: 'hello', session_id: 'session-a' }) })
    socket.onmessage?.({ data: JSON.stringify({ type: 'input_ready', reason: 'session_started' }) })
    await waitForSent(socket, 1)

    const firstScene = socket.sent[0].scene as Record<string, unknown>
    const firstCondition = (firstScene.layer_conditions as Record<string, unknown>[])[0]
    expect(firstCondition.image_ref).toBeTruthy()
    expect(firstCondition.prompt_mask_ref_name).toBeTruthy()

    service.requestWebRtcLayerConditionsUpdate()
    socket.onmessage?.({ data: JSON.stringify({ type: 'input_ready', reason: 'rendered' }) })
    await waitForSent(socket, 2)

    expect(socket.sent[1].type).toBe('scene_patch')
    expect(socket.sent[1].events).toEqual(expect.arrayContaining([
      expect.objectContaining({ type: 'property.set', path: ['settings', 'prompt'], value: 'scene prompt' }),
      expect.objectContaining({ type: 'layer_conditions.replace', layer_conditions: expect.any(Array) }),
    ]))
    const patch = socket.sent[1].patch as Record<string, unknown>
    expect((patch.settings as Record<string, unknown>).input_revision).toBeUndefined()
    const condition = (patch.layer_conditions as Record<string, unknown>[])[0]
    expect(condition.prompt).toBe('monster')
    expect(condition.image_ref).toBe(firstCondition.image_ref)
    expect(condition.prompt_mask_ref_name).toBe(firstCondition.prompt_mask_ref_name)
    expect(condition.image).toBeUndefined()
    expect(condition.prompt_mask).toBeUndefined()
  })

  it('sends canonical layer conditions when prompt metadata has no existing refs', async () => {
    const service = await import('./services/stream.service')
    let includeLayer = false
    let exportCount = 0
    service.wireFullFrameExporter(async () => {
      exportCount++
      return {
        image: IMAGE_RESOURCE,
        mask: MASK_RESOURCE,
        prompt: 'scene prompt',
        width: 512,
        height: 512,
        layer_conditions: includeLayer ? [{
          layer_id: 'layer-a',
          region_id: 'inherited:layer-a',
          prompt: 'monster',
          negative_prompt: '',
          image: LAYER_RESOURCE,
          prompt_mask: MASK_RESOURCE,
        }] : [],
      }
    })
    service.wireSceneSettingsExporter(async () => ({ prompt: 'scene prompt', width: 512, height: 512 }))
    service.wireLayerConditionsMetadataExporter(async () => [{
      layer_id: 'layer-a',
      region_id: 'inherited:layer-a',
      prompt: 'monster',
      negative_prompt: '',
      name: 'monster',
      cfg: 1,
      denoise: 1,
    }])

    await service.startWebRtc()
    const socket = sockets[0]
    socket.onopen?.()
    socket.onmessage?.({ data: JSON.stringify({ type: 'hello', session_id: 'session-a' }) })
    socket.onmessage?.({ data: JSON.stringify({ type: 'input_ready', reason: 'session_started' }) })
    await waitForSent(socket, 1)

    includeLayer = true
    service.requestWebRtcLayerConditionsUpdate()
    socket.onmessage?.({ data: JSON.stringify({ type: 'input_ready', reason: 'rendered' }) })
    await waitForSent(socket, 2)

    expect(exportCount).toBe(2)
    expect(socket.sent[1].type).toBe('scene_patch')
    expect(socket.sent[1].events).toEqual(expect.arrayContaining([
      expect.objectContaining({ type: 'signal.update', signal: expect.objectContaining({ type: 'rgba', resource_ref: expect.any(String) }) }),
      expect.objectContaining({ type: 'signal.update', signal: expect.objectContaining({ type: 'prompt_mask', resource_ref: expect.any(String) }) }),
      expect.objectContaining({ type: 'layer_conditions.replace', layer_conditions: expect.any(Array) }),
    ]))
    const patch = socket.sent[1].patch as Record<string, unknown>
    expect((patch.settings as Record<string, unknown>).input_revision).toBeUndefined()
    const condition = (patch.layer_conditions as Record<string, unknown>[])[0]
    expect(condition.prompt).toBe('monster')
    expect(condition.image_ref).toBeTruthy()
    expect(condition.prompt_mask_ref_name).toBeTruthy()
    expect(condition.image).toBeUndefined()
    expect(condition.prompt_mask).toBeUndefined()
  })

  it('preserves distinct per-channel layer mask resources in WebRTC scenes', async () => {
    const service = await import('./services/stream.service')
    service.wireFullFrameExporter(async () => ({
      image: IMAGE_RESOURCE,
      mask: MASK_RESOURCE,
      prompt: 'scene prompt',
      width: 512,
      height: 512,
      layer_conditions: [{
        layer_id: 'layer-a',
        region_id: 'region-a',
        prompt: 'monster',
        negative_prompt: '',
        image: LAYER_RESOURCE,
        color_mask: dataSvg('cyan'),
        prompt_mask: dataSvg('magenta'),
        cfg_mask: dataSvg('yellow'),
        denoise_mask: dataSvg('black'),
      }],
    }))

    await service.startWebRtc()
    const socket = sockets[0]
    socket.onopen?.()
    socket.onmessage?.({ data: JSON.stringify({ type: 'hello', session_id: 'session-a' }) })
    socket.onmessage?.({ data: JSON.stringify({ type: 'input_ready', reason: 'session_started' }) })
    await waitForSent(socket, 1)

    const scene = socket.sent[0].scene as Record<string, unknown>
    const condition = (scene.layer_conditions as Record<string, unknown>[])[0]
    const signals = scene.signals as Record<string, unknown>[]
    const refs = [
      condition.color_mask_ref_name,
      condition.prompt_mask_ref_name,
      condition.cfg_mask_ref_name,
      condition.denoise_mask_ref_name,
    ]
    expect(refs.every(Boolean)).toBe(true)
    expect(new Set(refs).size).toBe(4)
    expect(condition.color_mask).toBeUndefined()
    expect(condition.prompt_mask).toBeUndefined()
    expect(condition.cfg_mask).toBeUndefined()
    expect(condition.denoise_mask).toBeUndefined()
    expect(signals).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: 'input.image', type: 'rgba', role: 'input', resource_ref: expect.any(String) }),
      expect.objectContaining({ id: 'input.mask', type: 'denoise_mask', role: 'input_mask', resource_ref: expect.any(String) }),
      expect.objectContaining({ id: 'layer.layer-a.region-a.color_mask', type: 'rgba_mask', channel: 'color', layer_id: 'layer-a', region_id: 'region-a', resource_ref: condition.color_mask_ref_name }),
      expect.objectContaining({ id: 'layer.layer-a.region-a.prompt_mask', type: 'prompt_mask', channel: 'prompt', layer_id: 'layer-a', region_id: 'region-a', resource_ref: condition.prompt_mask_ref_name }),
      expect.objectContaining({ id: 'layer.layer-a.region-a.cfg_mask', type: 'cfg_mask', channel: 'cfg', layer_id: 'layer-a', region_id: 'region-a', resource_ref: condition.cfg_mask_ref_name }),
      expect.objectContaining({ id: 'layer.layer-a.region-a.denoise_mask', type: 'denoise_mask', channel: 'denoise', layer_id: 'layer-a', region_id: 'region-a', resource_ref: condition.denoise_mask_ref_name }),
    ]))
    const debugResources = service.getSceneResourceDebugData()
    const debugNames = [...debugResources.keys()]
    expect(debugNames).toEqual(expect.arrayContaining([
      'layer-a/RGBA',
      'layer-a/RGBAPrompt',
      'layer-a/Prompt1',
      'layer-a/CFG',
      'layer-a/Denoise',
    ]))
  })

  it('sends inherited renderer prompt changes through layer condition metadata', async () => {
    const service = await import('./services/stream.service')
    let prompt = 'room'
    service.wireFullFrameExporter(async () => ({
      image: IMAGE_RESOURCE,
      mask: MASK_RESOURCE,
      prompt,
      width: 512,
      height: 512,
      layer_conditions: [{
        layer_id: 'layer-a',
        region_id: 'inherited:layer-a',
        prompt,
        negative_prompt: '',
        image: LAYER_RESOURCE,
        prompt_mask: MASK_RESOURCE,
      }],
    }))
    service.wireSceneSettingsExporter(async () => ({ prompt, width: 512, height: 512 }))
    service.wireLayerConditionsMetadataExporter(async () => [{
      layer_id: 'layer-a',
      region_id: 'inherited:layer-a',
      prompt,
      negative_prompt: '',
      name: 'inherited',
      cfg: 1,
      denoise: 1,
    }])

    await service.startWebRtc()
    const socket = sockets[0]
    socket.onopen?.()
    socket.onmessage?.({ data: JSON.stringify({ type: 'hello', session_id: 'session-a' }) })
    socket.onmessage?.({ data: JSON.stringify({ type: 'input_ready', reason: 'session_started' }) })
    await waitForSent(socket, 1)

    prompt = 'monster'
    service.requestRealtimeFrameUpdate({ refreshLayerConditions: true })
    socket.onmessage?.({ data: JSON.stringify({ type: 'input_ready', reason: 'rendered' }) })
    await waitForSent(socket, 2)

    const patch = socket.sent[1].patch as Record<string, unknown>
    expect(socket.sent[1].events).toEqual(expect.arrayContaining([
      expect.objectContaining({ type: 'property.set', path: ['settings', 'prompt'], value: 'monster' }),
      expect.objectContaining({ type: 'layer_conditions.replace', layer_conditions: expect.any(Array) }),
    ]))
    expect((patch.settings as Record<string, unknown>).prompt).toBe('monster')
    const condition = (patch.layer_conditions as Record<string, unknown>[])[0]
    expect(condition.prompt).toBe('monster')
    expect(condition.image_ref).toBeTruthy()
    expect(condition.prompt_mask_ref_name).toBeTruthy()
  })

  it('sends frame seed rotations even when the visual scene is unchanged', async () => {
    const service = await import('./services/stream.service')
    $scene.setKey('seedMode', 'random')
    $scene.setKey('seedRotationMode', 'frame')
    service.wireFullFrameExporter(async () => JSON.stringify({
      image: 'static-image',
      mask: 'static-mask',
      prompt: 'same prompt',
      seed_mode: 'fixed',
      width: 512,
      height: 512,
    }))

    await service.startWebRtc()
    const socket = sockets[0]
    socket.onopen?.()
    socket.onmessage?.({ data: JSON.stringify({ type: 'hello', session_id: 'session-a' }) })

    socket.onmessage?.({ data: JSON.stringify({ type: 'input_ready', reason: 'session_started' }) })
    await waitForSent(socket, 1)
    const firstScene = socket.sent[0].scene as Record<string, unknown>
    const firstSettings = firstScene.settings as Record<string, unknown>
    expect(firstSettings.seed_mode).toBe('random')
    expect(firstSettings.seed_rotation_serial).toBe(1)

    socket.onmessage?.({ data: JSON.stringify({ type: 'input_ready', reason: 'rendered' }) })
    await waitForSent(socket, 2)
    const secondScene = socket.sent[1].scene as Record<string, unknown>
    const secondSettings = secondScene.settings as Record<string, unknown>
    expect(secondSettings.seed_mode).toBe('random')
    expect(secondSettings.seed_rotation_serial).toBe(2)
    expect(secondScene.id).not.toBe(firstScene.id)
  })

  it('keeps emitting frame rotations with fixed seed mode', async () => {
    const service = await import('./services/stream.service')
    $scene.setKey('seedMode', 'fixed')
    $scene.setKey('seedRotationMode', 'frame')
    $scene.setKey('seed', '1234')
    service.wireFullFrameExporter(async () => JSON.stringify({
      image: 'static-image',
      mask: 'static-mask',
      prompt: 'same prompt',
      seed: 1234,
      seed_mode: 'fixed',
      width: 512,
      height: 512,
    }))

    await service.startWebRtc()
    const socket = sockets[0]
    socket.onopen?.()
    socket.onmessage?.({ data: JSON.stringify({ type: 'hello', session_id: 'session-a' }) })

    socket.onmessage?.({ data: JSON.stringify({ type: 'input_ready', reason: 'session_started' }) })
    await waitForSent(socket, 1)
    const firstScene = socket.sent[0].scene as Record<string, unknown>
    const firstSettings = firstScene.settings as Record<string, unknown>
    expect(firstSettings.seed).toBe(1234)
    expect(firstSettings.seed_mode).toBe('fixed')
    expect(firstSettings.seed_rotation_serial).toBe(1)

    socket.onmessage?.({ data: JSON.stringify({ type: 'input_ready', reason: 'rendered' }) })
    await waitForSent(socket, 2)
    const secondScene = socket.sent[1].scene as Record<string, unknown>
    const secondSettings = secondScene.settings as Record<string, unknown>
    expect(secondSettings.seed).toBe(1234)
    expect(secondSettings.seed_mode).toBe('fixed')
    expect(secondSettings.seed_rotation_serial).toBe(2)
    expect(secondScene.id).not.toBe(firstScene.id)
  })
})
