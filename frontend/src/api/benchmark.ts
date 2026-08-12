import { request } from './client'
import type { BenchmarkAnalysis, BenchmarkCatalog, BenchmarkRunSummary } from '@/types'

export interface BenchmarkRequest {
  org_id: string
  user_visibility: 'public' | 'internal' | 'restricted'
  companies: string[]
  years: number[]
  metrics: string[]
  name?: string
}

function scopeQuery(orgId: string, visibility: string): URLSearchParams {
  return new URLSearchParams({ org_id: orgId, user_visibility: visibility })
}

export function getBenchmarkCatalog(orgId: string, visibility: string): Promise<BenchmarkCatalog> {
  return request(`/api/v1/benchmark/catalog?${scopeQuery(orgId, visibility)}`)
}

export function analyzeBenchmark(payload: BenchmarkRequest): Promise<BenchmarkAnalysis> {
  return request('/api/v1/benchmark/analyze', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function listBenchmarkRuns(
  orgId: string,
  visibility: string,
): Promise<BenchmarkRunSummary[]> {
  return request(`/api/v1/benchmark/runs?${scopeQuery(orgId, visibility)}`)
}

export function getBenchmarkRun(
  id: string,
  orgId: string,
  visibility: string,
): Promise<BenchmarkAnalysis> {
  return request(`/api/v1/benchmark/runs/${id}?${scopeQuery(orgId, visibility)}`)
}
