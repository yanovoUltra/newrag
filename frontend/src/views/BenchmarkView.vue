<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import {
  ArrowRight,
  Clock,
  DataAnalysis,
  DocumentChecked,
  Download,
  Refresh,
  View,
  Warning,
} from '@element-plus/icons-vue'
import {
  analyzeBenchmark,
  getBenchmarkCatalog,
  getBenchmarkRun,
  listBenchmarkRuns,
} from '@/api/benchmark'
import { notifyError } from '@/api/client'
import type {
  BenchmarkAnalysis,
  BenchmarkCatalog,
  BenchmarkEvidence,
  BenchmarkMetricResult,
  BenchmarkObservation,
  BenchmarkRunSummary,
} from '@/types'

const route = useRoute()
const router = useRouter()
const orgId = ref(localStorage.getItem('newrag:chat:settings:org') ?? 'default')
const visibility = ref<'public' | 'internal' | 'restricted'>('public')
const catalog = ref<BenchmarkCatalog>({ companies: [], years: [], metrics: [] })
const selectedCompanies = ref<string[]>([])
const selectedYears = ref<number[]>([])
const selectedMetrics = ref<string[]>([])
const reportName = ref('')
const result = ref<BenchmarkAnalysis | null>(null)
const runs = ref<BenchmarkRunSummary[]>([])
const loadingCatalog = ref(false)
const running = ref(false)
const loadingRun = ref(false)
const evidenceOpen = ref(false)
const activeEvidence = ref<BenchmarkEvidence | null>(null)

const canRun = computed(
  () =>
    selectedCompanies.value.length >= 2 &&
    selectedYears.value.length > 0 &&
    selectedMetrics.value.length > 0 &&
    !running.value,
)

const confidenceLabel = computed(() => {
  const labels = { high: '高可信', medium: '需复核', low: '数据不足' }
  return result.value ? labels[result.value.summary.confidence] : ''
})

function percent(value: number): string {
  return `${Math.round(value * 100)}%`
}

function changeText(item: BenchmarkObservation): string {
  if (item.change_value === null) return '基期'
  const sign = item.change_value > 0 ? '+' : ''
  return `${sign}${item.change_value.toFixed(2)}${item.change_unit ?? ''}`
}

function changeClass(item: BenchmarkObservation): string {
  if (item.change_value === null || item.change_value === 0) return 'neutral'
  return item.change_value > 0 ? 'positive' : 'negative'
}

function metricObservation(
  metric: BenchmarkMetricResult,
  company: string,
  year: number,
): BenchmarkObservation | undefined {
  return metric.observations.find((item) => item.company === company && item.year === year)
}

function openEvidence(item?: BenchmarkObservation): void {
  if (!item?.evidence) return
  activeEvidence.value = item.evidence
  evidenceOpen.value = true
}

async function loadCatalog(resetSelection = false): Promise<void> {
  loadingCatalog.value = true
  try {
    const data = await getBenchmarkCatalog(orgId.value.trim() || 'default', visibility.value)
    catalog.value = data
    localStorage.setItem('newrag:chat:settings:org', orgId.value.trim() || 'default')
    const validCompanies = selectedCompanies.value.filter((item) => data.companies.includes(item))
    const validYears = selectedYears.value.filter((item) => data.years.includes(item))
    const metricKeys = new Set(data.metrics.map((item) => item.key))
    const validMetrics = selectedMetrics.value.filter((item) => metricKeys.has(item))
    if (resetSelection || validCompanies.length < 2)
      selectedCompanies.value = data.companies.slice(0, 3)
    else selectedCompanies.value = validCompanies
    if (resetSelection || !validYears.length) selectedYears.value = data.years.slice(0, 3).sort()
    else selectedYears.value = validYears.sort()
    if (resetSelection || !validMetrics.length)
      selectedMetrics.value = data.metrics.slice(0, 5).map((item) => item.key)
    else selectedMetrics.value = validMetrics
    runs.value = await listBenchmarkRuns(orgId.value.trim() || 'default', visibility.value)
  } catch (error) {
    notifyError(error)
  } finally {
    loadingCatalog.value = false
  }
}

async function runAnalysis(): Promise<void> {
  if (!canRun.value) {
    ElMessage.warning('请选择至少 2 家公司、1 个财年和 1 项指标')
    return
  }
  running.value = true
  try {
    result.value = await analyzeBenchmark({
      org_id: orgId.value.trim() || 'default',
      user_visibility: visibility.value,
      companies: selectedCompanies.value,
      years: selectedYears.value,
      metrics: selectedMetrics.value,
      name: reportName.value.trim(),
    })
    void router.replace({ query: { run: result.value.id } })
    runs.value = await listBenchmarkRuns(orgId.value.trim() || 'default', visibility.value)
    ElMessage.success('对标分析已生成并保存')
  } catch (error) {
    notifyError(error)
  } finally {
    running.value = false
  }
}

async function openRun(runId: string): Promise<void> {
  loadingRun.value = true
  try {
    result.value = await getBenchmarkRun(runId, orgId.value.trim() || 'default', visibility.value)
    selectedCompanies.value = [...result.value.companies]
    selectedYears.value = [...result.value.years]
    selectedMetrics.value = result.value.metrics.map((item) => item.key)
    reportName.value = result.value.name
    void router.replace({ query: { run: runId } })
  } catch (error) {
    notifyError(error)
  } finally {
    loadingRun.value = false
  }
}

function escapeHtml(value: unknown): string {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;')
}

function exportHtml(): void {
  if (!result.value) return
  const report = result.value
  const sections = report.metrics
    .map((metric) => {
      const rows = metric.observations
        .map(
          (item) =>
            `<tr><td>${escapeHtml(item.company)}</td><td>${item.year}</td><td>${escapeHtml(item.display_value)}</td><td>${escapeHtml(changeText(item))}</td><td>${escapeHtml(item.evidence?.doc_name ?? '无')}</td><td>${item.evidence?.page ?? '—'}</td></tr>`,
        )
        .join('')
      return `<section><h2>${escapeHtml(metric.label)}</h2><table><thead><tr><th>公司</th><th>财年</th><th>数值</th><th>变化</th><th>证据</th><th>页码</th></tr></thead><tbody>${rows}</tbody></table>${metric.insights.map((item) => `<p>${escapeHtml(item)}</p>`).join('')}</section>`
    })
    .join('')
  const html = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>${escapeHtml(report.name)}</title><style>body{font:15px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#172033}h1,h2{color:#0f5132}table{width:100%;border-collapse:collapse;margin:12px 0 28px}th,td{border:1px solid #cbd5e1;padding:8px;text-align:left}th{background:#eef7f1}.meta{color:#64748b}.warning{color:#92400e}</style></head><body><h1>${escapeHtml(report.name)}</h1><p class="meta">生成时间：${escapeHtml(new Date(report.generated_at).toLocaleString('zh-CN'))} · 数据覆盖：${percent(report.summary.coverage)} · 证据覆盖：${percent(report.summary.evidence_coverage)}</p>${report.warnings.map((item) => `<p class="warning">${escapeHtml(item)}</p>`).join('')}${sections}<p class="meta">本报告由 NewRAG 基于已入库财报结构化字段确定性生成；不构成投资建议。</p></body></html>`
  const url = URL.createObjectURL(new Blob([html], { type: 'text/html;charset=utf-8' }))
  const link = document.createElement('a')
  link.href = url
  link.download = `${report.name.replace(/[\\/:*?"<>|]/g, '_')}.html`
  link.click()
  URL.revokeObjectURL(url)
}

onMounted(async () => {
  await loadCatalog()
  const runId = typeof route.query.run === 'string' ? route.query.run : ''
  if (runId) await openRun(runId)
})
</script>

<template>
  <div class="benchmark-view" v-loading="loadingRun">
    <header class="page-header">
      <div>
        <div class="eyebrow">FINANCIAL BENCHMARK AGENT</div>
        <h1>财报竞品对标</h1>
        <p>统一口径取数、确定性计算、逐项证据核验；缺失值不推断。</p>
      </div>
      <el-button v-if="result" :icon="Download" @click="exportHtml">导出审计报告</el-button>
    </header>

    <div class="workspace">
      <aside class="control-panel card" aria-label="分析范围">
        <div class="panel-heading">
          <div>
            <span class="step-kicker">分析范围</span>
            <h2>定义对标任务</h2>
          </div>
          <el-button
            text
            :icon="Refresh"
            :loading="loadingCatalog"
            aria-label="刷新可用数据"
            @click="loadCatalog(true)"
          />
        </div>

        <label class="field-label" for="benchmark-org">机构 ID</label>
        <div class="scope-row">
          <el-input id="benchmark-org" v-model="orgId" aria-label="机构 ID" />
          <el-select v-model="visibility" aria-label="可见性" @change="loadCatalog(true)">
            <el-option label="公开" value="public" />
            <el-option label="内部" value="internal" />
            <el-option label="受限" value="restricted" />
          </el-select>
          <el-button :loading="loadingCatalog" @click="loadCatalog(true)">载入</el-button>
        </div>

        <label class="field-label">对标公司 <span>2–5 家</span></label>
        <el-select
          v-model="selectedCompanies"
          multiple
          collapse-tags
          :max-collapse-tags="2"
          :multiple-limit="5"
          placeholder="选择公司"
          aria-label="选择对标公司"
        >
          <el-option v-for="item in catalog.companies" :key="item" :label="item" :value="item" />
        </el-select>

        <label class="field-label">财年 <span>最多 5 年</span></label>
        <el-select
          v-model="selectedYears"
          multiple
          collapse-tags
          :multiple-limit="5"
          placeholder="选择财年"
          aria-label="选择财年"
        >
          <el-option
            v-for="item in catalog.years"
            :key="item"
            :label="`${item} 财年`"
            :value="item"
          />
        </el-select>

        <label class="field-label">核心指标 <span>最多 12 项</span></label>
        <el-select
          v-model="selectedMetrics"
          multiple
          collapse-tags
          :max-collapse-tags="2"
          :multiple-limit="12"
          placeholder="选择指标"
          aria-label="选择对标指标"
        >
          <el-option
            v-for="item in catalog.metrics"
            :key="item.key"
            :label="`${item.label} · ${item.available_records}`"
            :value="item.key"
          />
        </el-select>

        <label class="field-label" for="benchmark-name">报告名称 <span>可选</span></label>
        <el-input
          id="benchmark-name"
          v-model="reportName"
          placeholder="例如：银行同业 2022–2025 对标"
        />

        <el-button
          class="run-button"
          type="primary"
          :icon="DataAnalysis"
          :loading="running"
          :disabled="!canRun"
          @click="runAnalysis"
        >
          生成可审计对标报告
        </el-button>
        <p v-if="catalog.companies.length < 2" class="empty-hint">
          当前机构不足 2 家公司。请先上传并完成至少两家公司财报的字段抽取。
        </p>

        <section v-if="runs.length" class="history">
          <div class="history-title">
            <el-icon><Clock /></el-icon>最近分析
          </div>
          <button
            v-for="run in runs.slice(0, 5)"
            :key="run.id"
            type="button"
            @click="openRun(run.id)"
          >
            <span>{{ run.name }}</span>
            <small>{{ percent(run.coverage) }} 覆盖</small>
          </button>
        </section>
      </aside>

      <div class="report-panel">
        <section v-if="!result" class="empty-report card">
          <div class="empty-mark"><DocumentChecked /></div>
          <h2>从财报证据生成对标结论</h2>
          <p>选择公司、财年和指标后，Agent 将运行六阶段可审计工作流。</p>
          <ol>
            <li>限定公司与报告范围</li>
            <li>批量读取结构化财务字段</li>
            <li>统一金额和比率展示口径</li>
            <li>由代码计算同比变化</li>
            <li>核验每个单元格的原始证据</li>
            <li>只基于可比数据生成结论</li>
          </ol>
        </section>

        <template v-else>
          <section class="report-hero card">
            <div class="report-title-row">
              <div>
                <span class="step-kicker"
                  >ANALYSIS / {{ result.id.slice(0, 8).toUpperCase() }}</span
                >
                <h2>{{ result.name }}</h2>
                <p>{{ result.companies.join(' · ') }} / {{ result.years.join('–') }}</p>
              </div>
              <el-tag
                :type="result.summary.confidence === 'high' ? 'success' : 'warning'"
                effect="dark"
              >
                {{ confidenceLabel }}
              </el-tag>
            </div>

            <div class="score-strip">
              <div>
                <strong>{{ percent(result.summary.coverage) }}</strong>
                <span>数据覆盖</span>
              </div>
              <div>
                <strong>{{ percent(result.summary.comparable_coverage) }}</strong>
                <span>已有值可比</span>
              </div>
              <div>
                <strong>{{ percent(result.summary.evidence_coverage) }}</strong>
                <span>证据覆盖</span>
              </div>
              <div>
                <strong>{{ result.summary.conflicts }}</strong>
                <span>冲突单元格</span>
              </div>
            </div>

            <div class="stage-rail" aria-label="Agent 工作流状态">
              <div v-for="(stage, index) in result.stages" :key="stage.key" class="stage-item">
                <span class="stage-index" :class="stage.status">{{ index + 1 }}</span>
                <div>
                  <strong>{{ stage.label }}</strong
                  ><small>{{ stage.detail }}</small>
                </div>
                <el-icon v-if="index < result.stages.length - 1"><ArrowRight /></el-icon>
              </div>
            </div>
          </section>

          <el-alert
            v-for="warning in result.warnings"
            :key="warning"
            class="report-warning"
            :title="warning"
            type="warning"
            show-icon
            :closable="false"
          />

          <section v-if="result.insights.length" class="insight-card card">
            <div class="section-heading">
              <div>
                <span class="step-kicker">EVIDENCE-BASED</span>
                <h2>关键发现</h2>
              </div>
              <span>仅使用可比较且有来源的数据</span>
            </div>
            <ul>
              <li v-for="item in result.insights" :key="item">{{ item }}</li>
            </ul>
          </section>

          <section v-for="metric in result.metrics" :key="metric.key" class="metric-card card">
            <div class="metric-heading">
              <div>
                <span class="metric-code">{{ metric.key }}</span>
                <h2>{{ metric.label }}</h2>
              </div>
              <span>统一展示：{{ metric.display_unit }}</span>
            </div>

            <div class="metric-grid" :style="{ '--company-count': result.companies.length }">
              <article v-for="company in result.companies" :key="company" class="company-column">
                <h3>{{ company }}</h3>
                <div v-for="year in result.years" :key="year" class="value-row">
                  <span class="value-year">{{ year }}</span>
                  <template
                    v-for="item in [metricObservation(metric, company, year)]"
                    :key="
                      item ? `${item.metric}-${item.company}-${item.year}` : `${company}-${year}`
                    "
                  >
                    <template v-if="item">
                      <div class="value-main">
                        <strong :class="{ unavailable: item.status === 'missing' }">{{
                          item.display_value
                        }}</strong>
                        <small :class="changeClass(item)">{{
                          item.status === 'available' ? changeText(item) : '无数据'
                        }}</small>
                      </div>
                      <button
                        v-if="item.evidence"
                        type="button"
                        class="evidence-button"
                        :aria-label="`查看 ${company} ${year}年 ${metric.label} 的证据`"
                        @click="openEvidence(item)"
                      >
                        <el-icon><View /></el-icon><span>证据</span>
                      </button>
                      <el-tooltip v-if="item.warnings.length" :content="item.warnings.join('；')">
                        <el-icon class="cell-warning" aria-label="该数据需要复核"
                          ><Warning
                        /></el-icon>
                      </el-tooltip>
                    </template>
                  </template>
                </div>
              </article>
            </div>
            <ul v-if="metric.insights.length" class="metric-insights">
              <li v-for="item in metric.insights" :key="item">{{ item }}</li>
            </ul>
          </section>

          <p class="disclaimer">数据来自已入库财报及字段索引，结果用于研究辅助，不构成投资建议。</p>
        </template>
      </div>
    </div>

    <el-drawer v-model="evidenceOpen" title="原始财报证据" size="min(480px, 92vw)">
      <article v-if="activeEvidence" class="evidence-detail">
        <div class="evidence-badge"><DocumentChecked /> 已定位来源</div>
        <dl>
          <div>
            <dt>文件</dt>
            <dd>{{ activeEvidence.doc_name }}</dd>
          </div>
          <div>
            <dt>页码</dt>
            <dd>{{ activeEvidence.page || '未标注' }}</dd>
          </div>
          <div>
            <dt>章节</dt>
            <dd>{{ activeEvidence.section_path || '未标注' }}</dd>
          </div>
          <div>
            <dt>来源类型</dt>
            <dd>{{ activeEvidence.source === 'table' ? '财务表格' : '正文' }}</dd>
          </div>
          <div>
            <dt>文档 ID</dt>
            <dd class="mono">{{ activeEvidence.doc_id }}</dd>
          </div>
          <div>
            <dt>知识块 ID</dt>
            <dd class="mono">{{ activeEvidence.chunk_id || '未标注' }}</dd>
          </div>
        </dl>
        <h3>抽取原文</h3>
        <blockquote>{{ activeEvidence.raw || '字段记录未保存原文' }}</blockquote>
      </article>
    </el-drawer>
  </div>
</template>

<style scoped>
.benchmark-view {
  height: 100%;
  overflow: auto;
}
.page-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 20px;
  padding: 22px 28px 18px;
  border-bottom: 1px solid var(--color-border);
  background: linear-gradient(110deg, rgba(34, 197, 94, 0.07), transparent 45%);
}
.eyebrow,
.step-kicker,
.metric-code {
  color: var(--color-accent);
  font: 600 11px/1.2 var(--font-mono);
  letter-spacing: 0.12em;
}
.page-header h1 {
  margin: 4px 0 2px;
  font-size: 24px;
  line-height: 1.2;
}
.page-header p,
.report-title-row p {
  margin: 0;
  color: var(--color-fg-muted);
  font-size: 13px;
}
.workspace {
  display: grid;
  grid-template-columns: 340px minmax(0, 1fr);
  gap: 18px;
  max-width: 1500px;
  margin: 0 auto;
  padding: 18px 24px 42px;
}
.control-panel {
  position: sticky;
  top: 18px;
  align-self: start;
  padding: 18px;
}
.panel-heading,
.section-heading,
.metric-heading,
.report-title-row {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
}
.panel-heading h2,
.section-heading h2,
.metric-heading h2,
.report-title-row h2 {
  margin: 3px 0 0;
  font-size: 17px;
}
.field-label {
  display: flex;
  justify-content: space-between;
  margin: 16px 0 6px;
  color: var(--color-fg);
  font-size: 13px;
  font-weight: 600;
}
.field-label span {
  color: var(--color-fg-faint);
  font-weight: 400;
}
.scope-row {
  display: grid;
  grid-template-columns: 1fr 92px auto;
  gap: 6px;
}
.control-panel :deep(.el-select) {
  width: 100%;
}
.run-button {
  width: 100%;
  min-height: 44px;
  margin-top: 20px;
}
.empty-hint {
  color: var(--color-warn);
  font-size: 12px;
  line-height: 1.5;
}
.history {
  margin: 22px -4px 0;
  padding-top: 16px;
  border-top: 1px solid var(--color-border);
}
.history-title {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 0 4px 6px;
  color: var(--color-fg-muted);
  font-size: 12px;
  font-weight: 600;
}
.history button {
  width: 100%;
  min-height: 44px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  border: 0;
  border-radius: 7px;
  padding: 8px;
  background: transparent;
  color: var(--color-fg-muted);
  text-align: left;
  cursor: pointer;
  transition:
    background 160ms ease,
    color 160ms ease;
}
.history button:hover {
  background: var(--color-muted);
  color: var(--color-fg);
}
.history button span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.history small {
  flex-shrink: 0;
  color: var(--color-accent);
}
.report-panel {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.empty-report {
  min-height: 520px;
  display: grid;
  align-content: center;
  justify-items: center;
  padding: 40px;
  text-align: center;
}
.empty-mark {
  width: 64px;
  height: 64px;
  display: grid;
  place-items: center;
  border: 1px solid rgba(34, 197, 94, 0.5);
  border-radius: 18px;
  background: var(--color-accent-soft);
  color: var(--color-accent);
}
.empty-mark svg {
  width: 30px;
}
.empty-report h2 {
  margin: 20px 0 6px;
  font-size: 20px;
}
.empty-report p {
  margin: 0;
  color: var(--color-fg-muted);
}
.empty-report ol {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px 28px;
  margin: 24px 0 0;
  padding-left: 24px;
  text-align: left;
  color: var(--color-fg-muted);
  font-size: 13px;
}
.report-hero {
  padding: 20px;
  overflow: hidden;
}
.score-strip {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  margin: 20px 0;
  border: 1px solid var(--color-border);
  border-radius: 10px;
  background: #08101f;
}
.score-strip div {
  display: flex;
  flex-direction: column;
  padding: 14px 16px;
  border-right: 1px solid var(--color-border);
}
.score-strip div:last-child {
  border: 0;
}
.score-strip strong {
  font: 600 22px/1.2 var(--font-mono);
  color: var(--color-fg);
}
.score-strip span {
  margin-top: 4px;
  color: var(--color-fg-muted);
  font-size: 11px;
}
.stage-rail {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 4px;
}
.stage-item {
  min-width: 0;
  display: grid;
  grid-template-columns: 24px minmax(0, 1fr) auto;
  align-items: center;
  gap: 7px;
}
.stage-index {
  width: 24px;
  height: 24px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  background: var(--color-accent);
  color: #04130a;
  font: 700 11px var(--font-mono);
}
.stage-index.warning {
  background: var(--color-warn);
}
.stage-item strong,
.stage-item small {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.stage-item strong {
  font-size: 11px;
}
.stage-item small {
  color: var(--color-fg-faint);
  font-size: 9px;
}
.stage-item > .el-icon {
  color: var(--color-border);
}
.insight-card,
.metric-card {
  padding: 20px;
}
.section-heading > span,
.metric-heading > span {
  color: var(--color-fg-muted);
  font-size: 11px;
}
.insight-card ul {
  margin: 16px 0 0;
  padding: 0;
  display: grid;
  gap: 8px;
  list-style: none;
}
.insight-card li {
  position: relative;
  padding: 9px 12px 9px 30px;
  border-left: 2px solid var(--color-accent);
  background: rgba(34, 197, 94, 0.06);
  font-size: 13px;
}
.insight-card li::before {
  content: '↗';
  position: absolute;
  left: 11px;
  color: var(--color-accent);
  font-family: var(--font-mono);
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(var(--company-count), minmax(210px, 1fr));
  gap: 10px;
  margin-top: 16px;
}
.company-column {
  min-width: 0;
  border: 1px solid var(--color-border);
  border-radius: 9px;
  overflow: hidden;
  background: #08101f;
}
.company-column h3 {
  margin: 0;
  padding: 10px 12px;
  border-bottom: 1px solid var(--color-border);
  font-size: 13px;
}
.value-row {
  min-height: 72px;
  display: grid;
  grid-template-columns: 46px minmax(0, 1fr) 48px 18px;
  align-items: center;
  gap: 8px;
  padding: 9px 10px;
  border-bottom: 1px solid rgba(51, 65, 85, 0.65);
}
.value-row:last-child {
  border: 0;
}
.value-year {
  color: var(--color-fg-faint);
  font: 500 11px var(--font-mono);
}
.value-main {
  min-width: 0;
}
.value-main strong {
  display: block;
  overflow: hidden;
  color: var(--color-fg);
  font: 600 15px var(--font-mono);
  text-overflow: ellipsis;
  white-space: nowrap;
}
.value-main strong.unavailable {
  color: var(--color-fg-faint);
}
.value-main small {
  font: 500 10px var(--font-mono);
}
.positive {
  color: #4ade80;
}
.negative {
  color: #fb7185;
}
.neutral {
  color: var(--color-fg-faint);
}
.evidence-button {
  min-width: 44px;
  min-height: 44px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 1px;
  border: 0;
  border-radius: 6px;
  background: transparent;
  color: #38bdf8;
  font-size: 9px;
  cursor: pointer;
  transition: background 160ms ease;
}
.evidence-button:hover {
  background: rgba(56, 189, 248, 0.1);
}
.cell-warning {
  color: var(--color-warn);
}
.metric-insights {
  margin: 14px 0 0;
  padding-left: 20px;
  color: var(--color-fg-muted);
  font-size: 12px;
}
.disclaimer {
  margin: 2px 0;
  color: var(--color-fg-faint);
  font-size: 11px;
  text-align: center;
}
.evidence-detail {
  color: var(--color-fg);
}
.evidence-badge {
  display: flex;
  align-items: center;
  gap: 8px;
  width: max-content;
  padding: 7px 10px;
  border-radius: 999px;
  background: var(--color-accent-soft);
  color: var(--color-accent);
  font-size: 12px;
}
.evidence-badge svg {
  width: 16px;
}
.evidence-detail dl {
  margin: 20px 0;
}
.evidence-detail dl div {
  display: grid;
  grid-template-columns: 82px 1fr;
  gap: 12px;
  padding: 10px 0;
  border-bottom: 1px solid var(--color-border);
}
.evidence-detail dt {
  color: var(--color-fg-faint);
  font-size: 12px;
}
.evidence-detail dd {
  margin: 0;
  word-break: break-word;
}
.mono {
  font: 11px var(--font-mono);
}
.evidence-detail h3 {
  font-size: 13px;
}
.evidence-detail blockquote {
  margin: 0;
  padding: 14px;
  border-left: 3px solid var(--color-accent);
  background: var(--color-muted);
  line-height: 1.75;
}

@media (max-width: 1100px) {
  .workspace {
    grid-template-columns: 300px minmax(0, 1fr);
  }
  .stage-rail {
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }
  .metric-grid {
    grid-template-columns: 1fr;
  }
}
@media (max-width: 768px) {
  .page-header {
    align-items: flex-start;
    padding: 16px 14px;
  }
  .page-header p {
    max-width: 230px;
  }
  .workspace {
    display: block;
    padding: 12px 12px 30px;
  }
  .control-panel {
    position: static;
    margin-bottom: 12px;
  }
  .scope-row {
    grid-template-columns: 1fr 100px;
  }
  .scope-row > .el-button {
    grid-column: 1/-1;
    min-height: 44px;
  }
  .report-panel {
    gap: 10px;
  }
  .empty-report {
    min-height: 360px;
    padding: 24px 18px;
  }
  .empty-report ol {
    grid-template-columns: 1fr;
  }
  .score-strip {
    grid-template-columns: 1fr 1fr;
  }
  .score-strip div:nth-child(2) {
    border-right: 0;
  }
  .score-strip div:nth-child(-n + 2) {
    border-bottom: 1px solid var(--color-border);
  }
  .stage-rail {
    grid-template-columns: 1fr 1fr;
  }
  .stage-item > .el-icon {
    display: none;
  }
  .metric-card,
  .insight-card,
  .report-hero {
    padding: 15px;
  }
  .metric-grid {
    grid-template-columns: 1fr;
  }
  .value-row {
    grid-template-columns: 42px minmax(0, 1fr) 52px 18px;
  }
  .page-header .el-button span {
    display: none;
  }
}
</style>
