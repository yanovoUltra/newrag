// ---- 全局类型定义（与后端 /api/v1 契约对齐）----

// ---- 文档 ----
export interface DocumentItem {
  id: string
  filename: string
  file_type: string
  status: string
  org_id: string
  visibility: string
  fiscal_year: number | null
  fiscal_quarter: number | null
  chunk_count: number
  size_bytes: number
  error: string | null
  created_at: string
  updated_at: string
}

export interface DocumentPage {
  items: DocumentItem[]
  total: number
  limit: number
  offset: number
}

// ---- 任务（ingest 进度轮询）----
export interface TaskItem {
  id: string
  type: string
  doc_id: string
  status: 'queued' | 'running' | 'success' | 'failed'
  stage: string
  progress: number
  message: string
  created_at: string
  updated_at: string
}

// ---- SSE 问答事件 ----
export interface ChatMetaData {
  org_id: string
  top_k: number
  intent: string
  complexity: string
  needs_hyde: boolean
  year: number | null
  year_fell_back: boolean
  session_id?: string
  question?: string
}

export interface CitationData {
  doc_id: string
  doc_name: string
  section_path: string | null
  page: number | null
  chunk_type: string | null
  score: number
}

export interface WarningData {
  message: string
}

export interface TokenData {
  delta: string
}

export interface GroundingData {
  checked: number
  missing: string[]
  citations: number
}

// 字段抽取：结构化指标精确取值（metric 类问题命中独立字段索引）
export interface FieldData {
  metric: string
  metric_label: string
  year: number
  value: number
  unit: string | null
  raw: string
  doc_id: string
  doc_name: string
  page: number
  section_path: string
}

export interface FieldEventData {
  fields: FieldData[]
}

export interface ErrorData {
  message: string
}

export interface DoneData {
  usage: Record<string, unknown>
}

export type ChatEvent =
  | { event: 'meta'; data: ChatMetaData }
  | { event: 'citation'; data: CitationData }
  | { event: 'warning'; data: WarningData }
  | { event: 'token'; data: TokenData }
  | { event: 'grounding'; data: GroundingData }
  | { event: 'field'; data: FieldEventData }
  | { event: 'error'; data: ErrorData }
  | { event: 'done'; data: DoneData }

// ---- 前端消息结构 ----
export interface MessageTurn {
  id: string
  role: 'user' | 'assistant'
  question?: string
  content: string
  citations: CitationData[]
  fields?: FieldData[]
  meta?: ChatMetaData
  warning: string | null
  grounding: GroundingData | null
  error: string | null
  streaming: boolean
  createdAt: number
}
