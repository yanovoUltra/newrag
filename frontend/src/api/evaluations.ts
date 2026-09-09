import { request } from './client'
import type { EvaluationOverview } from '@/types'

export function getEvaluationOverview(orgId: string): Promise<EvaluationOverview> {
  const query = new URLSearchParams({ org_id: orgId })
  return request(`/api/v1/evaluations/overview?${query}`)
}
