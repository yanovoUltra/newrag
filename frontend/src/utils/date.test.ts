import { describe, expect, it } from 'vitest'

import { getCurrentYear } from './date'

describe('getCurrentYear', () => {
  it('返回传入日期对应的本地年份', () => {
    expect(getCurrentYear(new Date(2032, 0, 1))).toBe(2032)
  })

  it('默认返回当前本地年份', () => {
    expect(getCurrentYear()).toBe(new Date().getFullYear())
  })
})
