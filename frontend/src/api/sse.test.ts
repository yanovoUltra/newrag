// SSE 解析与流式消费单测（纯逻辑，无 DOM 依赖）
import { describe, expect, it, vi } from 'vitest'

import { parseSseBlock, postSse } from './sse'

function sseStream(...chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder()
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(enc.encode(c))
      controller.close()
    },
  })
}

describe('parseSseBlock', () => {
  it('解析 event + JSON data', () => {
    const p = parseSseBlock('event: meta\ndata: {"top_k": 8}')
    expect(p).toEqual({ event: 'meta', data: { top_k: 8 } })
  })

  it('缺省 event 为 message', () => {
    expect(parseSseBlock('data: hello')?.event).toBe('message')
  })

  it('忽略注释行', () => {
    const p = parseSseBlock(': keepalive\nevent: done\ndata: {}')
    expect(p?.event).toBe('done')
  })

  it('多行 data 拼接，且剥离 data: 后的单个空格', () => {
    const p = parseSseBlock('data: line1\ndata: line2')
    expect(p?.data).toBe('line1\nline2')
  })

  it('data 非法 JSON 时原样返回字符串', () => {
    const p = parseSseBlock('event: token\ndata: 你好')
    expect(p).toEqual({ event: 'token', data: '你好' })
  })

  it('无 data 返回 null', () => {
    expect(parseSseBlock('event: keepalive')).toBeNull()
  })
})

describe('postSse', () => {
  it('完整消费流并按块触发事件', async () => {
    const events: Array<[string, unknown]> = []
    const onEvent = (e: string, d: unknown) => events.push([e, d])
    const onError = vi.fn()

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      body: sseStream(
        'event: meta\ndata: {"top_k": 8}\n\n',
        'event: token\ndata: 浦发银行\n\n',
        'event: done\ndata: {"usage": {}}\n\n',
      ),
    }))

    await postSse({
      url: '/api/v1/chat',
      body: { question: 'q' },
      signal: new AbortController().signal,
      onEvent,
      onError,
    })
    expect(onError).not.toHaveBeenCalled()
    expect(events.map((e) => e[0])).toEqual(['meta', 'token', 'done'])
    expect(events[1][1]).toBe('浦发银行')
  })

  it('跨 chunk 边界的事件也能拼接解析（缓冲）', async () => {
    const events: Array<[string, unknown]> = []
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      body: sseStream(
        'event: meta\ndata: {"top',
        '_k": 8}\n\nevent: token\ndata: x',
        '\n\n',
      ),
    }))
    await postSse({
      url: '/api/v1/chat', body: {}, signal: new AbortController().signal,
      onEvent: (e, d) => events.push([e, d]), onError: vi.fn(),
    })
    expect(events[0]).toEqual(['meta', { top_k: 8 }])
    expect(events[1]).toEqual(['token', 'x'])
  })

  it('非 2xx 时读取 detail 并触发 onError', async () => {
    const onError = vi.fn()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({ detail: '问题不能为空' }),
    }))
    await postSse({
      url: '/api/v1/chat', body: {}, signal: new AbortController().signal,
      onEvent: vi.fn(), onError,
    })
    expect(onError).toHaveBeenCalledWith(new Error('问题不能为空'))
  })

  it('fetch 抛错时触发 onError（AbortError 除外）', async () => {
    const onError = vi.fn()
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('network down')))
    await postSse({
      url: '/api/v1/chat', body: {}, signal: new AbortController().signal,
      onEvent: vi.fn(), onError,
    })
    expect(onError).toHaveBeenCalled()

    // AbortError 不应触发
    onError.mockClear()
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(Object.assign(new Error('abort'), { name: 'AbortError' })))
    await postSse({
      url: '/api/v1/chat', body: {}, signal: new AbortController().signal,
      onEvent: vi.fn(), onError,
    })
    expect(onError).not.toHaveBeenCalled()
  })

  it('无响应体时提示浏览器不支持', async () => {
    const onError = vi.fn()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, body: null }))
    await postSse({
      url: '/api/v1/chat', body: {}, signal: new AbortController().signal,
      onEvent: vi.fn(), onError,
    })
    expect(onError).toHaveBeenCalled()
  })
})
