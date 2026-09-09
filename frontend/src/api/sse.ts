// ---- SSE over POST：EventSource 不支持 POST，使用 fetch + ReadableStream 解析 text/event-stream ----
import { authorizationHeaders } from '@/auth'

export interface SseOptions {
  url: string
  body: Record<string, unknown>
  signal: AbortSignal
  onEvent: (event: string, data: unknown) => void
  onError: (err: Error) => void
}

export async function postSse({ url, body, signal, onEvent, onError }: SseOptions): Promise<void> {
  let res: Response
  try {
    const authHeaders = await authorizationHeaders()
    res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
        ...authHeaders,
      },
      body: JSON.stringify(body),
      signal,
    })
  } catch (e) {
    if ((e as Error).name !== 'AbortError') onError(e as Error)
    return
  }
  if (!res.ok) {
    // 非流式错误（如 400）：尽力读取 detail
    let detail = `HTTP ${res.status}`
    try {
      const j = await res.json()
      if (typeof j?.detail === 'string') detail = j.detail
    } catch {
      /* ignore */
    }
    onError(new Error(detail))
    return
  }
  if (!res.body) {
    onError(new Error('当前浏览器不支持流式响应'))
    return
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buf = ''
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')
      let idx: number
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const raw = buf.slice(0, idx)
        buf = buf.slice(idx + 2)
        const parsed = parseSseBlock(raw)
        if (parsed) onEvent(parsed.event, parsed.data)
      }
    }
  } catch (e) {
    if ((e as Error).name !== 'AbortError') onError(e as Error)
  }
}

interface ParsedSse {
  event: string
  data: unknown
}

/** 解析单个 SSE 块（event/data 行）；data 尽力 JSON.parse */
export function parseSseBlock(block: string): ParsedSse | null {
  let event = 'message'
  let data = ''
  for (const line of block.split('\n')) {
    if (line.startsWith(':')) continue // 注释行
    if (line.startsWith('event:')) {
      event = line.slice(6).trim()
    } else if (line.startsWith('data:')) {
      const payload = line.slice(5)
      data += (data ? '\n' : '') + (payload.startsWith(' ') ? payload.slice(1) : payload)
    }
  }
  if (!data) return null
  try {
    return { event, data: JSON.parse(data) }
  } catch {
    return { event, data }
  }
}
