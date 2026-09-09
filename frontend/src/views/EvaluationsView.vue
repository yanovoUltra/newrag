<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { CircleCheck, Lock, Refresh, Warning } from '@element-plus/icons-vue'
import { getEvaluationOverview } from '@/api/evaluations'
import { notifyError } from '@/api/client'
import { getIdentityContext } from '@/auth'
import type { EvaluationMetricTree, EvaluationOverview, EvaluationRun } from '@/types'

const identity = getIdentityContext()
const orgId = identity?.orgId ?? localStorage.getItem('newrag:chat:settings:org') ?? 'default'
const overview = ref<EvaluationOverview | null>(null)
const loading = ref(false)

const metricLabels: Record<string, string> = {
  ndcg_at_8: 'NDCG@8',
  ndcg_at_10: 'NDCG@10',
  precision_at_1: 'P@1',
  mrr: 'MRR',
  recall_at_8: 'Recall@8',
  recall_at_20: 'Recall@20',
  all_hit_at_8: 'All-Hit@8',
  all_hit_at_20: 'All-Hit@20',
  candidate_group_recall_at_200: '候选组召回@200',
  oracle_all_hit_at_20: 'Oracle All-Hit@20',
  faithfulness: '忠实度',
  answer_relevancy: '回答相关性',
  context_precision: '上下文精确率',
  context_recall: '上下文召回率',
  citation_correctness: '引用正确率',
  p95_latency_ms: 'P95 延迟',
  peak_vram_gib: '峰值显存',
}

const contractRows = computed(() =>
  Object.entries(overview.value?.evaluation_contract.targets ?? {}).map(([key, target]) => ({
    key,
    label: metricLabels[key] ?? key,
    target,
  })),
)

const rerankerRuns = computed(() =>
  (overview.value?.published_runs ?? []).filter((run) => run.kind === 'frozen_candidate_rerank'),
)

const retrievalRuns = computed(() =>
  (overview.value?.published_runs ?? []).filter((run) => run.kind === 'end_to_end_retrieval'),
)

function flatMetrics(metrics: EvaluationMetricTree): Array<[string, number]> {
  return Object.entries(metrics)
    .filter((entry): entry is [string, number] => typeof entry[1] === 'number')
    .filter(([key]) => key !== 'peak_vram_gib')
}

function formatMetric(key: string, value: number): string {
  if (key === 'p95_latency_ms') return `${(value / 1000).toFixed(1)}s`
  if (key === 'peak_vram_gib') return `${value.toFixed(2)} GiB`
  return value.toFixed(4)
}

function metricPercent(key: string, value: number): number {
  if (key === 'p95_latency_ms' || key === 'peak_vram_gib') return 0
  return Math.max(0, Math.min(100, value * 100))
}

function runTone(run: EvaluationRun): string {
  if (run.status?.includes('rejected')) return 'danger'
  if (run.status?.includes('historical')) return 'warning'
  return 'success'
}

function kindLabel(kind: string): string {
  const labels: Record<string, string> = {
    end_to_end_retrieval: '端到端检索',
    frozen_candidate_rerank: '冻结候选池精排',
    stored_evaluation: '评估记录',
  }
  return labels[kind] ?? kind
}

async function load(): Promise<void> {
  loading.value = true
  try {
    overview.value = await getEvaluationOverview(orgId)
  } catch (error) {
    notifyError(error)
  } finally {
    loading.value = false
  }
}

onMounted(load)
</script>

<template>
  <div class="evaluation-view" v-loading="loading">
    <header class="page-header">
      <div>
        <div class="eyebrow">QUALITY CONTROL ROOM · EVIDENCE FIRST</div>
        <h1>模型与 RAG 评估</h1>
        <p>把候选召回、精排和生成质量分层展示；每个数字都携带口径、状态和来源哈希。</p>
      </div>
      <el-button :icon="Refresh" :loading="loading" aria-label="刷新评估数据" @click="load">
        刷新
      </el-button>
    </header>

    <section v-if="overview" class="truth-strip" :class="overview.runtime.mode">
      <div class="truth-mark" aria-hidden="true">{{ overview.runtime.mode === 'real' ? 'R' : 'D' }}</div>
      <div>
        <span class="section-kicker">当前实例运行真相</span>
        <strong>{{ overview.runtime.mode === 'real' ? '真实模型路径' : 'Mock 演示路径' }}</strong>
        <p>{{ overview.runtime.note }}</p>
      </div>
      <dl class="feature-list">
        <div v-for="(enabled, feature) in overview.runtime.features" :key="feature">
          <dt>{{ feature }}</dt>
          <dd :class="enabled ? 'enabled' : 'disabled'">{{ enabled ? '可用' : '关闭' }}</dd>
        </div>
      </dl>
    </section>

    <div v-if="overview" class="contract-grid">
      <section class="card contract-card">
        <div class="section-heading">
          <div>
            <span class="section-kicker">RELEASE CONTRACT</span>
            <h2>非指标题上线门槛</h2>
          </div>
          <el-icon class="success-icon" :size="22"><CircleCheck /></el-icon>
        </div>
        <div class="target-grid">
          <div v-for="row in contractRows" :key="row.key" class="target-item">
            <span>{{ row.label }}</span>
            <strong>≥ {{ row.target.toFixed(2) }}</strong>
          </div>
        </div>
        <p class="contract-note">
          指标题允许最大回归：{{
            (overview.evaluation_contract.indicator_regression_tolerance * 100).toFixed(1)
          }} 个百分点。
        </p>
      </section>

      <section class="card sealed-card">
        <div class="section-heading">
          <div>
            <span class="section-kicker">SEALED OUTER TEST</span>
            <h2>六公司封存盲测</h2>
          </div>
          <el-icon class="lock-icon" :size="22"><Lock /></el-icon>
        </div>
        <div class="sealed-count">
          <strong>{{ overview.sealed_blind_test.questions }}</strong>
          <span>道题 · {{ overview.sealed_blind_test.companies }} 家未见公司</span>
        </div>
        <div class="sealed-status"><span></span>未开启，标签保持封存</div>
        <p>{{ overview.sealed_blind_test.note }}</p>
      </section>
    </div>

    <section v-if="overview" class="pipeline-section">
      <div class="section-title-row">
        <div>
          <span class="section-kicker">RETRIEVAL PIPELINE</span>
          <h2>端到端检索：基线与撤回实验</h2>
        </div>
        <span class="scope-chip">71 道非指标题 · 28,424 chunks</span>
      </div>
      <div class="run-grid">
        <article v-for="run in retrievalRuns" :key="run.id" class="run-card" :class="runTone(run)">
          <header>
            <div>
              <span class="run-kind">{{ kindLabel(run.kind) }}</span>
              <h3>{{ run.label }}</h3>
            </div>
            <span class="run-state">{{ run.status?.includes('rejected') ? '已撤回' : '参照基线' }}</span>
          </header>
          <div class="metric-stack">
            <div v-for="([key, value]) in flatMetrics(run.metrics)" :key="key" class="metric-row">
              <div class="metric-copy">
                <span>{{ metricLabels[key] ?? key }}</span>
                <strong>{{ formatMetric(key, value) }}</strong>
              </div>
              <div v-if="metricPercent(key, value)" class="metric-track" aria-hidden="true">
                <span :style="{ width: `${metricPercent(key, value)}%` }"></span>
              </div>
            </div>
          </div>
          <p class="caveat"><el-icon><Warning /></el-icon>{{ run.caveat }}</p>
          <details v-if="run.source">
            <summary>查看可复核来源</summary>
            <code>{{ run.source.report }}</code>
            <code>SHA-256 {{ run.source.report_sha256 }}</code>
          </details>
        </article>
      </div>
    </section>

    <section v-if="overview" class="pipeline-section">
      <div class="section-title-row">
        <div>
          <span class="section-kicker">RERANKER LAB</span>
          <h2>本地 8B：同一冻结候选池比较</h2>
        </div>
        <span class="scope-chip caution">不是端到端生产结果</span>
      </div>
      <div class="comparison-table-wrap">
        <table class="comparison-table">
          <caption class="sr-only">Qwen3 Reranker Base 与 Adapter v1 比较</caption>
          <thead>
            <tr>
              <th scope="col">模型</th>
              <th scope="col">NDCG@8</th>
              <th scope="col">P@1</th>
              <th scope="col">MRR</th>
              <th scope="col">Recall@8</th>
              <th scope="col">All-Hit@8</th>
              <th scope="col">结论</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="run in rerankerRuns" :key="run.id">
              <th scope="row">
                {{ run.label }}
                <small>{{ run.sample_count }} 题</small>
              </th>
              <td>{{ formatMetric('ndcg_at_8', Number(run.metrics.ndcg_at_8)) }}</td>
              <td>{{ formatMetric('precision_at_1', Number(run.metrics.precision_at_1)) }}</td>
              <td>{{ formatMetric('mrr', Number(run.metrics.mrr)) }}</td>
              <td>{{ formatMetric('recall_at_8', Number(run.metrics.recall_at_8)) }}</td>
              <td>{{ formatMetric('all_hit_at_8', Number(run.metrics.all_hit_at_8)) }}</td>
              <td>{{ run.status?.includes('historical') ? '历史候选，未生产化' : 'Base 对照' }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p class="footnote">外部 rerank API 尚未完成同池公平比较，因此页面不填推测数字。</p>
    </section>

    <div v-if="overview" class="lower-grid">
      <section class="card">
        <div class="section-heading">
          <div>
            <span class="section-kicker">OFFICIAL EVALUATOR</span>
            <h2>Ragas 接入状态</h2>
          </div>
          <span class="version-chip">v{{ overview.official_ragas.version ?? '未安装' }}</span>
        </div>
        <p>官方包结果与项目历史自实现指标分开标记，不把相似公式冒充官方 Ragas。</p>
        <div v-if="overview.recent_stored_runs.length" class="stored-runs">
          <article v-for="run in overview.recent_stored_runs.slice(0, 5)" :key="run.id">
            <strong>{{ run.label }}</strong>
            <span>{{ run.created_at ? new Date(run.created_at).toLocaleString('zh-CN') : '' }}</span>
          </article>
        </div>
        <div v-else class="empty-run">尚无本租户的官方评估记录。</div>
      </section>

      <section class="card">
        <span class="section-kicker">PENDING, NOT HIDDEN</span>
        <h2>未完成的公平比较</h2>
        <ul class="pending-list">
          <li v-for="item in overview.pending_comparisons" :key="item.label">
            <strong>{{ item.label }}</strong>
            <span>{{ item.note }}</span>
          </li>
        </ul>
      </section>
    </div>
  </div>
</template>

<style scoped>
.evaluation-view {
  height: 100%;
  overflow-y: auto;
  padding: 32px clamp(20px, 4vw, 56px) 64px;
  color: var(--color-fg);
}
.page-header,
.section-heading,
.section-title-row,
.run-card header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 20px;
}
.page-header { margin-bottom: 24px; }
.page-header h1 { margin: 4px 0 8px; font-size: clamp(28px, 4vw, 42px); letter-spacing: -1.2px; }
.page-header p,
.truth-strip p,
.card p { margin: 0; color: var(--color-fg-muted); line-height: 1.65; }
.eyebrow,
.section-kicker,
.run-kind {
  color: var(--color-accent);
  font-size: 11px;
  font-weight: 750;
  letter-spacing: 1.6px;
}
.truth-strip {
  display: grid;
  grid-template-columns: auto minmax(240px, 1fr) minmax(300px, 1fr);
  gap: 20px;
  align-items: center;
  padding: 20px;
  border: 1px solid rgba(245, 158, 11, 0.38);
  border-radius: 14px;
  background: linear-gradient(110deg, rgba(245, 158, 11, 0.12), rgba(15, 23, 42, 0.35));
}
.truth-strip.real { border-color: rgba(34, 197, 94, 0.4); background: linear-gradient(110deg, rgba(34, 197, 94, 0.13), rgba(15, 23, 42, 0.35)); }
.truth-mark {
  display: grid;
  place-items: center;
  width: 48px;
  height: 48px;
  border-radius: 12px;
  background: #f59e0b;
  color: #111827;
  font: 800 22px/1 ui-monospace, monospace;
}
.truth-strip.real .truth-mark { background: var(--color-accent); }
.truth-strip strong { display: block; margin: 3px 0; font-size: 18px; }
.feature-list { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 0; }
.feature-list div { padding: 10px; border-radius: 8px; background: rgba(15, 23, 42, 0.55); }
.feature-list dt { color: var(--color-fg-faint); font-size: 11px; text-transform: uppercase; }
.feature-list dd { margin: 3px 0 0; font-size: 13px; font-weight: 700; }
.feature-list dd.enabled { color: #4ade80; }
.feature-list dd.disabled { color: #fbbf24; }
.contract-grid,
.lower-grid { display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 18px; margin-top: 18px; }
.card,
.run-card {
  border: 1px solid var(--color-border);
  border-radius: 14px;
  background: var(--color-panel);
  box-shadow: 0 16px 40px rgba(0, 0, 0, 0.08);
}
.card { padding: 22px; }
h2 { margin: 4px 0 0; font-size: 20px; }
.target-grid { display: grid; grid-template-columns: repeat(5, minmax(90px, 1fr)); gap: 8px; margin: 20px 0 14px; }
.target-item { padding: 12px; border: 1px solid var(--color-border); border-radius: 9px; background: var(--color-bg); }
.target-item span { display: block; color: var(--color-fg-muted); font-size: 12px; }
.target-item strong { display: block; margin-top: 6px; color: var(--color-accent); font-size: 17px; }
.success-icon { color: var(--color-accent); }
.lock-icon { color: #fbbf24; }
.contract-note { font-size: 12px; }
.sealed-count { margin: 18px 0 12px; }
.sealed-count strong { margin-right: 8px; font: 800 34px/1 ui-monospace, monospace; }
.sealed-count span { color: var(--color-fg-muted); }
.sealed-status { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; color: #fbbf24; font-size: 13px; }
.sealed-status span { width: 8px; height: 8px; border-radius: 50%; background: #f59e0b; box-shadow: 0 0 0 5px rgba(245, 158, 11, 0.12); }
.pipeline-section { margin-top: 34px; }
.section-title-row { align-items: flex-end; margin-bottom: 14px; }
.scope-chip,
.version-chip,
.run-state { padding: 6px 9px; border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 999px; color: var(--color-accent); background: rgba(34, 197, 94, 0.08); font-size: 11px; white-space: nowrap; }
.scope-chip.caution { border-color: rgba(245, 158, 11, 0.35); color: #fbbf24; background: rgba(245, 158, 11, 0.08); }
.run-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
.run-card { position: relative; overflow: hidden; padding: 22px; }
.run-card::before { content: ''; position: absolute; inset: 0 auto 0 0; width: 3px; background: var(--color-accent); }
.run-card.danger::before { background: #ef4444; }
.run-card.warning::before { background: #f59e0b; }
.run-card h3 { margin: 4px 0 0; font-size: 19px; }
.run-card.danger .run-state { border-color: rgba(239, 68, 68, 0.35); color: #f87171; background: rgba(239, 68, 68, 0.08); }
.metric-stack { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px 22px; margin-top: 20px; }
.metric-copy { display: flex; justify-content: space-between; gap: 12px; color: var(--color-fg-muted); font-size: 12px; }
.metric-copy strong { color: var(--color-fg); font: 700 13px/1 ui-monospace, monospace; }
.metric-track { height: 3px; margin-top: 6px; overflow: hidden; border-radius: 999px; background: var(--color-muted); }
.metric-track span { display: block; height: 100%; background: var(--color-accent); }
.danger .metric-track span { background: #ef4444; }
.caveat { display: flex; gap: 7px; margin-top: 18px !important; padding-top: 14px; border-top: 1px solid var(--color-border); font-size: 12px; }
details { margin-top: 12px; color: var(--color-fg-muted); font-size: 12px; }
summary { cursor: pointer; min-height: 32px; }
details code { display: block; margin-top: 5px; overflow-wrap: anywhere; color: var(--color-fg-faint); font-size: 10px; }
.comparison-table-wrap { overflow-x: auto; border: 1px solid var(--color-border); border-radius: 14px; }
.comparison-table { width: 100%; min-width: 850px; border-collapse: collapse; background: var(--color-panel); }
.comparison-table th,
.comparison-table td { padding: 15px 16px; border-bottom: 1px solid var(--color-border); text-align: left; font-size: 13px; }
.comparison-table thead th { color: var(--color-fg-faint); background: var(--color-bg); font-size: 11px; letter-spacing: 0.5px; }
.comparison-table tbody tr:last-child th,
.comparison-table tbody tr:last-child td { border-bottom: 0; }
.comparison-table tbody tr:last-child { background: rgba(245, 158, 11, 0.05); }
.comparison-table small { display: block; margin-top: 4px; color: var(--color-fg-faint); font-weight: 400; }
.footnote { margin: 9px 0 0; color: var(--color-fg-faint); font-size: 12px; }
.stored-runs { display: grid; gap: 8px; margin-top: 14px; }
.stored-runs article { display: flex; justify-content: space-between; gap: 12px; padding: 10px; border-radius: 8px; background: var(--color-bg); font-size: 12px; }
.stored-runs span,
.empty-run { color: var(--color-fg-faint); }
.empty-run { margin-top: 14px; padding: 20px; border: 1px dashed var(--color-border); border-radius: 8px; text-align: center; }
.pending-list { display: grid; gap: 12px; margin: 18px 0 0; padding: 0; list-style: none; }
.pending-list li { padding-left: 14px; border-left: 2px solid #f59e0b; }
.pending-list strong,
.pending-list span { display: block; }
.pending-list span { margin-top: 4px; color: var(--color-fg-muted); font-size: 12px; line-height: 1.5; }
.sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; }

@media (max-width: 1050px) {
  .truth-strip { grid-template-columns: auto 1fr; }
  .feature-list { grid-column: 1 / -1; }
  .target-grid { grid-template-columns: repeat(3, 1fr); }
}
@media (max-width: 768px) {
  .evaluation-view { padding: 20px 14px 40px; }
  .page-header,
  .section-title-row { align-items: stretch; flex-direction: column; }
  .page-header .el-button { min-height: 44px; }
  .truth-strip { grid-template-columns: auto 1fr; padding: 16px; }
  .feature-list { grid-template-columns: repeat(2, 1fr); }
  .contract-grid,
  .lower-grid,
  .run-grid { grid-template-columns: 1fr; }
  .target-grid { grid-template-columns: repeat(2, 1fr); }
  .metric-stack { grid-template-columns: 1fr; }
  .scope-chip { align-self: flex-start; }
}
</style>
