import { request } from './client'

export interface PublicConfig {
  max_upload_mb: number
  accepted_extensions: string[]
  fiscal_year_min: number
  current_year: number
}

export function getPublicConfig(): Promise<PublicConfig> {
  return request('/api/v1/config/public')
}
