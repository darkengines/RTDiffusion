import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  clearMaskBlobCache,
  serializeChannelRefs,
  setMaskBlobUploader,
  uploadMaskBlob,
} from './mask-blob.service'

function blob(text: string) {
  return new Blob([text], { type: 'image/png' })
}

afterEach(() => {
  vi.unstubAllGlobals()
  clearMaskBlobCache()
})

describe('mask blob uploads', () => {
  it('uploads identical blob content only once per session', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ id: 'server-id' }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(uploadMaskBlob('pc-1', blob('same'))).resolves.toBe('server-id')
    await expect(uploadMaskBlob('pc-1', blob('same'))).resolves.toBe('server-id')

    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('separates the local cache by rtc session id', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ id: 'pc1-id' }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ id: 'pc2-id' }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(uploadMaskBlob('pc-1', blob('same'))).resolves.toBe('pc1-id')
    await expect(uploadMaskBlob('pc-2', blob('same'))).resolves.toBe('pc2-id')

    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('serializes channel blobs to backend ref field names', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ id: 'denoise-id' }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ id: 'prompt-id' }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(serializeChannelRefs('pc-1', {
      denoise: blob('denoise'),
      prompt: blob('prompt'),
    })).resolves.toEqual({
      denoise_mask_ref: 'denoise-id',
      prompt_mask_ref: 'prompt-id',
    })
  })

  it('passes coherent resource names to the uploader', async () => {
    const uploader = vi.fn(async () => ({ id: 'cfg-id' }))
    setMaskBlobUploader(uploader)

    await expect(serializeChannelRefs('pc-1', {
      cfg: blob('cfg'),
    }, {
      cfg: 'Layer A/Mask 1/cfg',
    })).resolves.toEqual({
      cfg_mask_ref: 'cfg-id',
    })

    expect(uploader).toHaveBeenCalledWith('pc-1', expect.any(Blob), 'Layer A/Mask 1/cfg')
  })
})
