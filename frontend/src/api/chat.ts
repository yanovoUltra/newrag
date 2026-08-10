// ---- 流式问答 API（SSE over POST）----
import { postSse } from './sse'
import type { ChatEvent } from '@/types'

export interface StreamChatParams {
  question: string
  orgId: string
  userVisibility: string
  sessionId?: string
  signal: AbortSignal
  onEvent: (evt: ChatEvent) => void
  onError: (err: Error) => void
}

export function streamChat(p: StreamChatParams): Promise<void> {
  return postSse({
    url: '/api/v1/chat',
    body: {
      question: p.question,
      org_id: p.orgId,
      user_visibility: p.userVisibility,
      session_id: p.sessionId ?? '',
    },
    signal: p.signal,
    onEvent: (event, data) => p.onEvent({ event, data } as ChatEvent),
    onError: p.onError,
  })
}
