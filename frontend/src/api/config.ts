import { request } from './client'

export interface PublicConfig {
  max_upload_mb: number
  accepted_extensions: string[]
  fiscal_year_min: number
  current_year: number
  runtime_mode: 'demo' | 'real'
  runtime_profile:
    | 'local-real'
    | 'local-parser-benchmark'
    | 'legacy-demo'
    | 'legacy-real'
  model_features: {
    embedding: boolean
    generation: boolean
    rerank: boolean
    ocr: boolean
  }
  model_services: Record<
    'embedding' | 'generation' | 'rerank' | 'ocr',
    {
      state: 'disabled' | 'misconfigured' | 'configured' | 'reachable' | 'healthy' | 'unhealthy' | 'unreachable'
      configured: boolean
      reachable: boolean | null
      healthy: boolean | null
    }
  >
  auth: {
    mode: 'disabled' | 'hmac' | 'oidc'
    issuer: string
    client_id: string
  }
}

export function getPublicConfig(): Promise<PublicConfig> {
  return request('/api/v1/config/public')
}
