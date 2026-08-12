<script setup lang="ts">
import { ref } from 'vue'
import {
  Document,
  CircleCheckFilled,
  CircleCloseFilled,
  Aim,
  Pointer,
} from '@element-plus/icons-vue'
import MarkdownText from './MarkdownText.vue'
import type { CitationData, MessageTurn } from '@/types'

defineProps<{ message: MessageTurn; citeIndex: number }>()

const emit = defineEmits<{ cite: [citation: CitationData] }>()

const hoverIndex = ref<number | null>(null)

const intentLabels: Record<string, string> = {
  factual: '事实查询',
  summary: '摘要归纳',
  abstract: '抽象推理',
  multi_hop: '多跳推理',
  comparison: '对比分析',
  other: '通用',
}

function intentLabel(v: string | undefined): string {
  if (!v) return ''
  return intentLabels[v] ?? v
}
</script>

<template>
  <div class="msg-row" :class="message.role">
    <!-- 头像 -->
    <div class="avatar" :class="message.role">
      <el-icon v-if="message.role === 'assistant'" :size="16"><Aim /></el-icon>
      <el-icon v-else :size="16"><Pointer /></el-icon>
    </div>

    <div class="msg-body msg-in">
      <!-- 用户消息 -->
      <template v-if="message.role === 'user'">
        <div class="bubble user">{{ message.question }}</div>
      </template>

      <!-- 助手消息 -->
      <template v-else>
        <div class="bubble assistant">
          <!-- meta 徽标 -->
          <div v-if="message.meta" class="meta-chips">
            <span class="chip">
              意图 <b>{{ intentLabel(message.meta.intent) }}</b>
            </span>
            <span class="chip">
              复杂度 <b>{{ message.meta.complexity }}</b>
            </span>
            <span class="chip"
              >知识块 <b>{{ message.meta.top_k }}</b></span
            >
            <span v-if="message.meta.year" class="chip">
              年份过滤 <b>{{ message.meta.year }}</b>
            </span>
            <el-tooltip
              v-if="message.meta.year_fell_back"
              content="检索年份下无结果，已自动回退为全量检索"
              placement="top"
            >
              <span class="chip warn">年份回退</span>
            </el-tooltip>
            <span v-if="message.meta.needs_hyde" class="chip">HyDE 增强</span>
          </div>

          <!-- 低置信警告 -->
          <el-alert
            v-if="message.warning"
            class="warning-alert"
            :title="message.warning"
            type="warning"
            :closable="false"
            show-icon
          />

          <!-- 字段抽取命中（结构化指标精确取值） -->
          <div v-if="message.fields?.length" class="field-hits">
            <span class="field-hits-label">字段命中</span>
            <div
              v-for="(f, fi) in message.fields"
              :key="`${f.doc_id}-${f.metric}-${f.year}-${fi}`"
              class="field-hit"
            >
              <span class="field-name">{{ f.metric_label }}</span>
              <span class="field-year">{{ f.year }}年</span>
              <span class="field-value">{{ f.raw }}</span>
              <span class="field-src">{{ f.doc_name }}-第{{ f.page }}页</span>
            </div>
          </div>

          <!-- 内容 -->
          <MarkdownText v-if="message.content" :text="message.content" />
          <div v-else-if="message.streaming" class="typing-dots" aria-label="生成中">
            <span></span><span></span><span></span>
          </div>

          <!-- 流式光标 -->
          <span
            v-if="message.streaming && message.content"
            class="stream-caret"
            aria-hidden="true"
          ></span>

          <!-- 生成失败 -->
          <el-alert
            v-if="message.error"
            class="warning-alert"
            :title="message.error"
            type="error"
            :closable="false"
            show-icon
          />

          <!-- grounding 校验 -->
          <div v-if="message.grounding && !message.streaming" class="grounding">
            <div class="grounding-item ok">
              <el-icon :size="13"><CircleCheckFilled /></el-icon>
              <span>数字校验 {{ message.grounding.checked }} 处全部有据可查</span>
            </div>
            <div v-if="message.grounding.missing.length" class="grounding-item miss">
              <el-icon :size="13"><CircleCloseFilled /></el-icon>
              <span>
                以下数字未在知识块中定位到，请核对：
                <b>{{ message.grounding.missing.join('、') }}</b>
              </span>
            </div>
            <div v-if="message.grounding.citations > 0" class="grounding-item cite">
              <el-icon :size="13"><Document /></el-icon>
              <span>正文含 {{ message.grounding.citations }} 处来源引用标记</span>
            </div>
          </div>

          <!-- 引用 -->
          <div v-if="message.citations.length" class="citations">
            <span class="citations-label">引用来源</span>
            <button
              v-for="(c, i) in message.citations"
              :key="`${c.doc_id}-${i}`"
              type="button"
              class="cite-chip"
              :class="{ hover: hoverIndex === i }"
              @mouseenter="hoverIndex = i"
              @mouseleave="hoverIndex = null"
              @click="emit('cite', c)"
            >
              <el-icon :size="12"><Document /></el-icon>
              <span>[{{ citeIndex + i + 1 }}]</span>
              <span class="cite-name">{{ c.doc_name }}</span>
              <span v-if="c.chunk_type === 'table'" class="cite-type">表格</span>
              <span class="cite-score">{{ c.score.toFixed(3) }}</span>
            </button>
          </div>
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.msg-row {
  display: flex;
  gap: 12px;
  padding: 18px 0;
}
.msg-row.user {
  flex-direction: row-reverse;
}

.avatar {
  width: 34px;
  height: 34px;
  border-radius: 50%;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-top: 2px;
}
.avatar.assistant {
  background: var(--color-accent-soft);
  color: var(--color-accent);
  border: 1px solid rgba(34, 197, 94, 0.4);
}
.avatar.user {
  background: #1e3a8a;
  color: #93c5fd;
  border: 1px solid #1d4ed8;
}

.msg-body {
  max-width: 82%;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.msg-row.user .msg-body {
  align-items: flex-end;
}

.bubble {
  padding: 12px 16px;
  border-radius: 12px;
  font-size: 15px;
}
.bubble.user {
  background: #162a52;
  color: #e2e8f0;
  border: 1px solid #1d4ed8;
  border-top-right-radius: 3px;
  white-space: pre-wrap;
  word-break: break-word;
}
.bubble.assistant {
  background: var(--color-panel);
  border: 1px solid var(--color-border);
  border-top-left-radius: 3px;
}

/* meta 徽标 */
.meta-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 6px;
}
.chip {
  font-size: 11.5px;
  color: var(--color-fg-muted);
  background: var(--color-muted);
  border: 1px solid var(--color-border);
  border-radius: 999px;
  padding: 2px 10px;
}
.chip b {
  color: var(--color-fg);
  font-weight: 600;
}
.chip.warn {
  color: var(--color-warn);
  border-color: rgba(245, 158, 11, 0.4);
  cursor: help;
}

.warning-alert {
  margin: 6px 0;
}

/* 字段抽取命中 */
.field-hits {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 6px;
  padding: 10px 12px;
  background: rgba(34, 197, 94, 0.06);
  border: 1px solid rgba(34, 197, 94, 0.25);
  border-radius: 8px;
}
.field-hits-label {
  font-size: 11.5px;
  color: var(--color-accent);
  font-weight: 600;
}
.field-hit {
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 8px;
  font-size: 13px;
}
.field-name {
  color: var(--color-fg);
  font-weight: 600;
}
.field-year {
  color: var(--color-fg-muted);
  font-size: 12px;
}
.field-value {
  color: var(--color-accent);
  font-family: var(--font-mono);
  font-weight: 600;
}
.field-src {
  color: var(--color-fg-faint);
  font-size: 11.5px;
}

/* grounding */
.grounding {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px dashed var(--color-border);
}
.grounding-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12.5px;
  color: var(--color-fg-muted);
}
.grounding-item.ok {
  color: var(--color-accent);
}
.grounding-item.miss {
  color: var(--color-warn);
}
.grounding-item.cite {
  color: #38bdf8;
}

/* citations */
.citations {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  margin-top: 10px;
}
.citations-label {
  font-size: 12px;
  color: var(--color-fg-faint);
  margin-right: 2px;
}
.cite-chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 12px;
  color: #93c5fd;
  background: rgba(37, 99, 235, 0.14);
  border: 1px solid rgba(59, 130, 246, 0.35);
  border-radius: 6px;
  padding: 3px 9px;
  min-height: 44px;
  font-family: inherit;
  cursor: pointer;
  transition:
    background 150ms ease,
    border-color 150ms ease,
    transform 150ms ease;
  user-select: none;
}
.cite-chip:focus-visible {
  border-color: var(--color-accent);
}
.cite-chip.hover {
  background: rgba(37, 99, 235, 0.28);
  border-color: #3b82f6;
  transform: translateY(-1px);
}
.cite-name {
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.cite-type {
  color: var(--color-accent);
  font-size: 11px;
  border: 1px solid rgba(34, 197, 94, 0.4);
  border-radius: 4px;
  padding: 0 4px;
}
.cite-score {
  color: var(--color-fg-faint);
  font-family: var(--font-mono);
}

@media (max-width: 768px) {
  .msg-body {
    max-width: 100%;
  }
  .cite-name {
    max-width: 120px;
  }
}
</style>
