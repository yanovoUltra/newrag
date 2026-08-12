// ---- 文档 & 任务 API ----
import { request, extractDetail } from './client'
import type { DocumentItem, DocumentPage, TaskItem } from '@/types'

export interface UploadResult {
  task_id: string
  doc_id: string
  status: 'pending' | 'duplicate'
}

export interface UploadForm {
  file: File
  org_id: string
  visibility: string
  fiscal_year?: number | null
  fiscal_quarter?: number | null
}

export function listDocuments(orgId?: string, limit = 50, offset = 0): Promise<DocumentItem[]> {
  const q = new URLSearchParams()
  if (orgId) q.set('org_id', orgId)
  q.set('limit', String(limit))
  q.set('offset', String(offset))
  return request(`/api/v1/documents?${q.toString()}`)
}

export interface DocumentPageParams {
  orgId?: string
  filename?: string
  status?: string
  limit?: number
  offset?: number
  signal?: AbortSignal
}

export function listDocumentPage(params: DocumentPageParams = {}): Promise<DocumentPage> {
  const q = new URLSearchParams()
  if (params.orgId) q.set('org_id', params.orgId)
  if (params.filename) q.set('filename', params.filename)
  if (params.status) q.set('status', params.status)
  q.set('limit', String(params.limit ?? 20))
  q.set('offset', String(params.offset ?? 0))
  return request(`/api/v1/documents/page?${q.toString()}`, { signal: params.signal })
}

export function getDocument(id: string): Promise<DocumentItem> {
  return request(`/api/v1/documents/${id}`)
}

export async function uploadDocument(form: UploadForm): Promise<UploadResult> {
  const fd = new FormData()
  fd.append('file', form.file)
  fd.append('org_id', form.org_id)
  fd.append('visibility', form.visibility)
  if (form.fiscal_year) fd.append('fiscal_year', String(form.fiscal_year))
  if (form.fiscal_quarter) fd.append('fiscal_quarter', String(form.fiscal_quarter))
  const res = await fetch('/api/v1/documents', { method: 'POST', body: fd })
  if (!res.ok) {
    throw new Error(await extractDetail(res))
  }
  return res.json()
}

export function deleteDocument(id: string): Promise<{ doc_id: string; deleted: boolean }> {
  return request(`/api/v1/documents/${id}`, { method: 'DELETE' })
}

export function getTask(id: string): Promise<TaskItem> {
  return request(`/api/v1/tasks/${id}`)
}

/** 轮询 ingest 任务直到 success/failed；onUpdate 每次拿到最新状态，resolve 时返回最终任务 */
export function pollTask(
  id: string,
  onUpdate: (t: TaskItem) => void,
  signal: AbortSignal,
  intervalMs = 1200,
): Promise<TaskItem> {
  return new Promise<TaskItem>((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException('aborted', 'AbortError'))
      return
    }
    const onAbort = () => reject(new DOMException('aborted', 'AbortError'))
    signal.addEventListener('abort', onAbort, { once: true })
    const tick = async () => {
      try {
        const t = await getTask(id)
        onUpdate(t)
        if (t.status === 'success' || t.status === 'failed') {
          signal.removeEventListener('abort', onAbort)
          resolve(t)
        } else {
          setTimeout(tick, intervalMs)
        }
      } catch (e) {
        signal.removeEventListener('abort', onAbort)
        reject(e)
      }
    }
    tick()
  })
}
