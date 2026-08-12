import AxeBuilder from '@axe-core/playwright'
import { expect, test } from '@playwright/test'

const routes = ['/chat', '/documents', '/upload']
const viewports = [
  { width: 375, height: 752 },
  { width: 768, height: 900 },
  { width: 1440, height: 900 },
]

for (const viewport of viewports) {
  for (const route of routes) {
    test(`${route} at ${viewport.width}px has no axe violations`, async ({ page }) => {
      await page.setViewportSize(viewport)
      await page.goto(route)
      await page.waitForLoadState('networkidle')

      const results = await new AxeBuilder({ page }).analyze()

      const violations = results.violations.map((violation) => ({
        id: violation.id,
        impact: violation.impact,
        targets: violation.nodes.flatMap((node) => node.target),
      }))
      expect(violations).toEqual([])
    })
  }
}
