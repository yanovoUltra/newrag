// ---- HTTP 客户端：统一错误处理与 JSON 解析 ----
import { ElMessage } from 'element-plus'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

export async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    throw new ApiError(res.status, await extractDetail(res))
  }
  return res.json() as Promise<T>
}

/** 从 FastAPI 错误体中提取 detail 文案 */
export async function extractDetail(res: Response): Promise<string> {
  try {
    const body = await res.json()
    if (typeof body?.detail === 'string') return body.detail
    if (Array.isArray(body?.detail))
      return body.detail.map((d: { msg?: string }) => d?.msg ?? '').filter(Boolean).join('; ')
    if (body?.message) return String(body.message)
  } catch {
    /* 非 JSON 响应 */
  }
  return `HTTP ${res.status}`
}

export function notifyError(err: unknown): void {
  const msg = err instanceof Error ? err.message : String(err)
  ElMessage.error(msg)
}
