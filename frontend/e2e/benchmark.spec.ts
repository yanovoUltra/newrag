import { expect, test } from '@playwright/test'

const catalog = {
  companies: ['甲公司', '乙公司'],
  years: [2024, 2023],
  metrics: [
    {
      key: 'revenue',
      label: '营业收入',
      kind: 'amount',
      display_unit: '亿元',
      direction: 'higher',
      available_records: 4,
    },
  ],
}

const report = {
  id: 'agent-run-001',
  name: '甲公司 vs 乙公司 · 2023–2024 对标',
  generated_at: '2026-08-13T08:00:00Z',
  org_id: 'default',
  visibility: 'public',
  companies: ['甲公司', '乙公司'],
  years: [2023, 2024],
  metrics: [
    {
      key: 'revenue',
      label: '营业收入',
      kind: 'amount',
      display_unit: '亿元',
      direction: 'higher',
      insights: ['2024年，甲公司的营业收入为120.00 亿元，在本次可比公司中最高。'],
      observations: [
        ['甲公司', 2023, '100.00 亿元', null],
        ['甲公司', 2024, '120.00 亿元', 20],
        ['乙公司', 2023, '90.00 亿元', null],
        ['乙公司', 2024, '105.00 亿元', 16.6667],
      ].map(([company, year, display, change]) => ({
        company,
        year,
        metric: 'revenue',
        status: 'available',
        value: Number.parseFloat(String(display)),
        normalized_value: Number.parseFloat(String(display)),
        display_value: display,
        unit: '亿元',
        comparable: true,
        conflict: false,
        change_value: change,
        change_unit: '%',
        warnings: [],
        evidence: {
          doc_id: `${company}-doc`,
          doc_name: `${company}2024年报.pdf`,
          page: 18,
          section_path: '主要财务数据',
          chunk_id: `${company}-revenue-${year}`,
          source: 'table',
          raw: `${company} ${year} 营业收入 ${display}`,
        },
      })),
    },
  ],
  summary: {
    requested_cells: 4,
    available_cells: 4,
    comparable_cells: 4,
    evidenced_cells: 4,
    coverage: 1,
    comparable_coverage: 1,
    evidence_coverage: 1,
    conflicts: 0,
    confidence: 'high',
  },
  insights: ['2024年，甲公司的营业收入为120.00 亿元，在本次可比公司中最高。'],
  warnings: [],
  stages: ['范围规划', '字段取数', '口径归一', '确定性计算', '证据核验', '生成报告'].map(
    (label, index) => ({ key: `stage-${index}`, label, status: 'complete', detail: '已完成' }),
  ),
}

test.beforeEach(async ({ page }) => {
  await page.route('**/healthz', (route) =>
    route.fulfill({ json: { app: 'ok', qdrant: 'ok', redis: 'ok' } }),
  )
  await page.route('**/api/v1/benchmark/catalog**', (route) => route.fulfill({ json: catalog }))
  await page.route('**/api/v1/benchmark/runs**', (route) => route.fulfill({ json: [] }))
  await page.route('**/api/v1/benchmark/analyze', (route) => route.fulfill({ json: report }))
})

test('builds an auditable competitor benchmark and opens source evidence', async ({ page }) => {
  await page.goto('/benchmark')
  await expect(page.getByRole('heading', { name: '财报竞品对标' })).toBeVisible()
  const runButton = page.getByRole('button', { name: '生成可审计对标报告' })
  await expect(runButton).toBeEnabled()
  await runButton.click()

  await expect(page.getByText('高可信')).toBeVisible()
  await expect(page.getByText('120.00 亿元', { exact: true })).toBeVisible()
  await expect(page.getByText('确定性计算', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: '查看 甲公司 2024年 营业收入 的证据' }).click()
  await expect(page.getByRole('heading', { name: '原始财报证据' })).toBeVisible()
  await expect(page.getByText('甲公司2024年报.pdf')).toBeVisible()
})

test('keeps the benchmark setup usable at 375px', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 752 })
  await page.goto('/benchmark')
  await expect(page.getByRole('heading', { name: '定义对标任务' })).toBeVisible()
  const hasOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
  )
  expect(hasOverflow).toBe(false)
})
