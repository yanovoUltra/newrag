import Keycloak from 'keycloak-js'
import type { PublicConfig } from '@/api/config'

export interface IdentityContext {
  orgId: string
  visibility: 'public' | 'internal' | 'restricted'
  roles: string[]
  username: string
}

let keycloak: Keycloak | null = null
let identity: IdentityContext | null = null

export async function initializeAuth(): Promise<PublicConfig> {
  const response = await fetch('/api/v1/config/public')
  if (!response.ok) throw new Error('无法读取公开运行时配置')
  const config = (await response.json()) as PublicConfig
  if (config.auth.mode !== 'oidc') return config

  const parsed = parseKeycloakIssuer(config.auth.issuer)
  keycloak = new Keycloak({
    url: parsed.url,
    realm: parsed.realm,
    clientId: config.auth.client_id,
  })
  const authenticated = await keycloak.init({
    onLoad: 'login-required',
    pkceMethod: 'S256',
    checkLoginIframe: false,
  })
  if (!authenticated || !keycloak.tokenParsed) throw new Error('登录失败')
  identity = identityFromToken(keycloak.tokenParsed as Record<string, unknown>)
  return config
}

export async function authorizationHeaders(): Promise<Record<string, string>> {
  if (!keycloak) return {}
  try {
    await keycloak.updateToken(30)
  } catch {
    await keycloak.login()
    return {}
  }
  return keycloak.token ? { Authorization: `Bearer ${keycloak.token}` } : {}
}

export function getIdentityContext(): IdentityContext | null {
  return identity
}

export async function logout(): Promise<void> {
  if (keycloak) await keycloak.logout({ redirectUri: window.location.origin })
}

function parseKeycloakIssuer(issuer: string): { url: string; realm: string } {
  const marker = '/realms/'
  const index = issuer.lastIndexOf(marker)
  if (index <= 0 || index + marker.length >= issuer.length) {
    throw new Error('OIDC issuer 不是合法的 Keycloak realm 地址')
  }
  return {
    url: issuer.slice(0, index),
    realm: issuer.slice(index + marker.length).replace(/\/$/, ''),
  }
}

function identityFromToken(token: Record<string, unknown>): IdentityContext {
  const realmAccess = (token.realm_access ?? {}) as { roles?: unknown[] }
  const visibility = String(token.visibility ?? 'public')
  if (!['public', 'internal', 'restricted'].includes(visibility)) {
    throw new Error('身份令牌中的可见级别非法')
  }
  const orgId = String(token.org_id ?? '').trim()
  if (!orgId) throw new Error('身份令牌缺少 org_id')
  return {
    orgId,
    visibility: visibility as IdentityContext['visibility'],
    roles: (realmAccess.roles ?? []).map(String),
    username: String(token.preferred_username ?? token.sub ?? ''),
  }
}
