import { afterEach, describe, expect, it, vi } from 'vitest'
import { analyzeBenchmark, getBenchmarkCatalog, getBenchmarkRun } from './benchmark'

afterEach(() => vi.unstubAllGlobals())

describe('benchmark api', () => {
  it('encodes catalog scope', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ companies: [], years: [], metrics: [] }), { status: 200 }),
      )
    vi.stubGlobal('fetch', fetchMock)
    await getBenchmarkCatalog('research team', 'internal')
    expect(fetchMock.mock.calls[0][0]).toBe(
      '/api/v1/benchmark/catalog?org_id=research+team&user_visibility=internal',
    )
  })

  it('posts an explicit analysis scope', async () => {
    const payload = {
      org_id: 'default',
      user_visibility: 'public' as const,
      companies: ['甲公司', '乙公司'],
      years: [2023, 2024],
      metrics: ['revenue'],
    }
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    await analyzeBenchmark(payload)
    expect(fetchMock.mock.calls[0][0]).toBe('/api/v1/benchmark/analyze')
    expect(fetchMock.mock.calls[0][1]).toMatchObject({
      method: 'POST',
      body: JSON.stringify(payload),
    })
  })

  it('scopes saved reports to organization and visibility', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    await getBenchmarkRun('run-1', 'default', 'restricted')
    expect(fetchMock.mock.calls[0][0]).toBe(
      '/api/v1/benchmark/runs/run-1?org_id=default&user_visibility=restricted',
    )
  })
})
