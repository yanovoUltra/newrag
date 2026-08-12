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

// ---- 财报竞品对标 Agent ----
export interface BenchmarkMetricOption {
  key: string
  label: string
  kind: 'amount' | 'percent' | 'per_share'
  display_unit: string
  direction: 'higher' | 'lower' | 'neutral'
  available_records: number
}

export interface BenchmarkCatalog {
  companies: string[]
  years: number[]
  metrics: BenchmarkMetricOption[]
}

export interface BenchmarkEvidence {
  doc_id: string
  doc_name: string
  page: number
  section_path: string
  chunk_id: string
  source: string
  raw: string
}

export interface BenchmarkObservation {
  company: string
  year: number
  metric: string
  status: 'available' | 'missing'
  value: number | null
  normalized_value: number | null
  display_value: string
  unit: string | null
  comparable: boolean
  conflict: boolean
  change_value: number | null
  change_unit: string | null
  evidence: BenchmarkEvidence | null
  warnings: string[]
}

export interface BenchmarkMetricResult {
  key: string
  label: string
  kind: string
  display_unit: string
  direction: string
  observations: BenchmarkObservation[]
  insights: string[]
}

export interface BenchmarkStage {
  key: string
  label: string
  status: 'complete' | 'warning'
  detail: string
}

export interface BenchmarkAnalysis {
  id: string
  name: string
  generated_at: string
  org_id: string
  visibility: string
  companies: string[]
  years: number[]
  metrics: BenchmarkMetricResult[]
  summary: {
    requested_cells: number
    available_cells: number
    comparable_cells: number
    evidenced_cells: number
    coverage: number
    comparable_coverage: number
    evidence_coverage: number
    conflicts: number
    confidence: 'high' | 'medium' | 'low'
  }
  insights: string[]
  warnings: string[]
  stages: BenchmarkStage[]
}

export interface BenchmarkRunSummary {
  id: string
  name: string
  companies: string[]
  years: number[]
  metric_count: number
  coverage: number
  confidence: string
  created_at: string
}
