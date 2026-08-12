export const MIN_FISCAL_YEAR = 1990

export function getCurrentYear(now: Date = new Date()): number {
  return now.getFullYear()
}
