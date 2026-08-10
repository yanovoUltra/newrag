<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import {
  Plus,
  Promotion,
  VideoPause,
  Refresh,
  ChatLineRound,
  Setting,
} from '@element-plus/icons-vue'
import ChatMessage from '@/components/ChatMessage.vue'
import { streamChat } from '@/api/chat'
import type { CitationData, MessageTurn } from '@/types'

/* ---------------- 会话状态 ---------------- */

const STORAGE_KEY = 'newrag:chat:v1'
const SETTINGS_KEY = 'newrag:chat:settings'

interface PersistedChat {
  sessionId: string
  messages: MessageTurn[]
}

const orgId = ref(localStorage.getItem(SETTINGS_KEY + ':org') ?? 'default')
const visibility = ref(localStorage.getItem(SETTINGS_KEY + ':vis') ?? 'public')
watch(orgId, (v) => localStorage.setItem(SETTINGS_KEY + ':org', v))
watch(visibility, (v) => localStorage.setItem(SETTINGS_KEY + ':vis', v))

const sessionId = ref('')
const messages = ref<MessageTurn[]>([])
const streaming = ref(false)
const input = ref('')
const abortCtrl = ref<AbortController | null>(null)
const listEl = ref<HTMLElement | null>(null)
const inputEl = ref<HTMLTextAreaElement | null>(null)
const stickToBottom = ref(true)
const citeDrawer = ref(false)
const activeCitation = ref<CitationData | null>(null)
const showSettings = ref(false)

const currentAssistant = computed(() =>
  messages.value[messages.value.length - 1]?.role === 'assistant'
    ? messages.value[messages.value.length - 1]
    : null,
)

/* ---------------- 持久化 ---------------- */

function persist() {
  const data: PersistedChat = { sessionId: sessionId.value, messages: messages.value }
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data))
}

function restore() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return
    const data = JSON.parse(raw) as PersistedChat
    if (data?.sessionId && Array.isArray(data.messages)) {
      sessionId.value = data.sessionId
      messages.value = data.messages
    }
  } catch {
    localStorage.removeItem(STORAGE_KEY)
  }
}

function newChat() {
  if (streaming.value) abort()
  sessionId.value = ''
  messages.value = []
  persist()
  localStorage.removeItem(STORAGE_KEY)
  input.value = ''
  nextTick(() => inputEl.value?.focus())
}

/* ---------------- 滚动 ---------------- */

function onScroll() {
  const el = listEl.value
  if (!el) return
  stickToBottom.value = el.scrollHeight - el.scrollTop - el.clientHeight < 96
}

async function scrollToBottom(smooth = false) {
  const el = listEl.value
  if (!el || !stickToBottom.value) return
  await nextTick()
  el.scrollTo({ top: el.scrollHeight, behavior: smooth ? 'smooth' : 'auto' })
}

watch(
  () => messages.value.map((m) => m.content + (m.citations.length ? 'c' : '')).join('\u0001'),
  () => scrollToBottom(true),
)

/* ---------------- 问答流 ---------------- */

function abort() {
  abortCtrl.value?.abort()
  abortCtrl.value = null
}

async function send(question?: string) {
  const q = (question ?? input.value).trim()
  if (!q || streaming.value) return
  input.value = ''
  stickToBottom.value = true

  // 首条消息时生成会话 id（多轮上下文）
  if (!sessionId.value) sessionId.value = crypto.randomUUID()

  messages.value.push({
    id: crypto.randomUUID(),
    role: 'user',
    question: q,
    content: '',
    citations: [],
    warning: null,
    grounding: null,
    error: null,
    streaming: false,
    createdAt: Date.now(),
  })
  messages.value.push({
    id: crypto.randomUUID(),
    role: 'assistant',
    content: '',
    citations: [],
    warning: null,
    grounding: null,
    error: null,
    streaming: true,
    createdAt: Date.now(),
  })
  persist()
  scrollToBottom()

  const ctrl = new AbortController()
  abortCtrl.value = ctrl
  streaming.value = true

  await streamChat({
    question: q,
    orgId: orgId.value,
    userVisibility: visibility.value,
    sessionId: sessionId.value,
    signal: ctrl.signal,
    onEvent: (evt) => {
      const cur = currentAssistant.value
      if (!cur) return
      switch (evt.event) {
        case 'meta':
          cur.meta = evt.data
          if (evt.data.session_id) sessionId.value = evt.data.session_id
          break
        case 'citation':
          cur.citations.push(evt.data)
          break
        case 'warning':
          cur.warning = evt.data.message
          break
        case 'token':
          cur.content += evt.data.delta
          break
        case 'grounding':
          cur.grounding = evt.data
          break
        case 'field':
          cur.fields = evt.data.fields
          break
        case 'error':
          cur.error = evt.data.message
          break
        case 'done':
          cur.streaming = false
          streaming.value = false
          abortCtrl.value = null
          break
      }
      persist()
      scrollToBottom()
    },
    onError: (err) => {
      const cur = currentAssistant.value
      if (cur) {
        cur.error = err.message
        cur.streaming = false
      }
      streaming.value = false
      abortCtrl.value = null
      persist()
      ElMessage.error(err.message)
    },
  })
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault()
    send()
  }
}

/* ---------------- 引用抽屉 ---------------- */

function openCitation(c: CitationData) {
  activeCitation.value = c
  citeDrawer.value = true
}

/* ---------------- 示例问题 ---------------- */

const samples = [
  '浦发银行 2024 年营业收入是多少亿元？',
  '浦发银行 2024 年归属于母公司股东的净利润是多少？',
  'Apple FY2024 全年净销售额是多少？',
  '2024 年浦发银行不良贷款率是多少？',
]

/* ---------------- 生命周期 ---------------- */

onMounted(() => {
  restore()
  nextTick(() => inputEl.value?.focus())
})

onUnmounted(() => abort())
</script>

<template>
  <div class="chat-view">
    <!-- 顶部栏 -->
    <header class="chat-header">
      <div class="header-title">
        <h1>智能问答</h1>
        <span class="header-sub">基于已入库财报，支持多轮追问与来源引用</span>
      </div>
      <div class="header-actions">
        <el-button
          size="small"
          :icon="Setting"
          @click="showSettings = !showSettings"
          :type="showSettings ? 'primary' : 'default'"
        >
          检索设置
        </el-button>
        <el-button size="small" :icon="Plus" @click="newChat">新会话</el-button>
      </div>
    </header>

    <!-- 检索设置（org/visibility） -->
    <div v-if="showSettings" class="settings-bar">
      <div class="settings-field">
        <label>机构 ID</label>
        <el-input v-model="orgId" size="small" placeholder="default" style="width: 180px" />
      </div>
      <div class="settings-field">
        <label>可见性</label>
        <el-select v-model="visibility" size="small" style="width: 140px">
          <el-option label="公开 public" value="public" />
          <el-option label="内部 internal" value="internal" />
          <el-option label="受限 restricted" value="restricted" />
        </el-select>
      </div>
      <div class="settings-hint">权限过滤：仅检索「org={{ orgId }}」下可见性不高于所选档位的知识块</div>
    </div>

    <!-- 消息区 -->
    <div ref="listEl" class="chat-list" @scroll.passive="onScroll">
      <!-- 空状态 -->
      <div v-if="!messages.length" class="empty-state">
        <div class="empty-icon">
          <el-icon :size="34"><ChatLineRound /></el-icon>
        </div>
        <h2>开始分析你的财报</h2>
        <p class="muted">上传财报后，针对文档内容提问。回答带来源引用，点击引用可查看定位。</p>
        <div class="sample-grid">
          <button
            v-for="s in samples"
            :key="s"
            class="sample-card"
            :disabled="streaming"
            @click="send(s)"
          >
            {{ s }}
          </button>
        </div>
      </div>

      <!-- 消息列表 -->
      <template v-else>
        <ChatMessage
          v-for="m in messages"
          :key="m.id"
          :message="m"
          :cite-index="0"
          @cite="openCitation"
        />
      </template>
      <div class="list-spacer" />
    </div>

    <!-- 输入区 -->
    <footer class="chat-input-bar">
      <div class="input-box">
        <el-input
          ref="inputEl"
          v-model="input"
          type="textarea"
          :rows="2"
          :autosize="{ minRows: 1, maxRows: 6 }"
          placeholder="输入你的问题，Enter 发送 / Shift+Enter 换行"
          resize="none"
          :disabled="streaming"
          @keydown="onKeydown"
        />
        <div class="input-actions">
          <span v-if="sessionId" class="session-tag">
            会话 #{{ sessionId.slice(0, 8) }}
            <el-icon class="refresh-icon" :size="13" @click="newChat"><Refresh /></el-icon>
          </span>
          <el-button
            v-if="streaming"
            type="danger"
            :icon="VideoPause"
            @click="abort"
          >
            停止
          </el-button>
          <el-button
            v-else
            type="primary"
            :icon="Promotion"
            :disabled="!input.trim()"
            @click="send()"
          >
            发送
          </el-button>
        </div>
      </div>
    </footer>

    <!-- 引用详情抽屉 -->
    <el-drawer v-model="citeDrawer" title="引用定位" size="380px" append-to-body>
      <div v-if="activeCitation" class="cite-detail">
        <div class="cite-detail-field">
          <span class="label">文档</span>
          <span class="value">{{ activeCitation.doc_name }}</span>
        </div>
        <div class="cite-detail-field">
          <span class="label">章节</span>
          <span class="value">{{ activeCitation.section_path || '—' }}</span>
        </div>
        <div class="cite-detail-field">
          <span class="label">页码</span>
          <span class="value">{{ activeCitation.page ?? '—' }}</span>
        </div>
        <div class="cite-detail-field">
          <span class="label">块类型</span>
          <span class="value">
            {{ activeCitation.chunk_type }}
            <el-tag v-if="activeCitation.chunk_type === 'table'" size="small" type="success">
              表格
            </el-tag>
          </span>
        </div>
        <div class="cite-detail-field">
          <span class="label">相关度分</span>
          <span class="value mono">{{ activeCitation.score.toFixed(4) }}</span>
        </div>
        <div class="cite-detail-note">
          文档 ID：<code>{{ activeCitation.doc_id }}</code>
        </div>
      </div>
    </el-drawer>
  </div>
</template>

<style scoped>
.chat-view {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 24px;
  border-bottom: 1px solid var(--color-border);
  background: rgba(2, 6, 23, 0.85);
  backdrop-filter: blur(6px);
  flex-shrink: 0;
}
.header-title h1 {
  margin: 0;
  font-size: 17px;
  font-weight: 700;
}
.header-sub {
  font-size: 12.5px;
  color: var(--color-fg-muted);
}
.header-actions {
  display: flex;
  gap: 8px;
}

.settings-bar {
  display: flex;
  align-items: flex-end;
  gap: 16px;
  padding: 10px 24px;
  border-bottom: 1px solid var(--color-border);
  background: var(--color-panel);
  flex-shrink: 0;
  flex-wrap: wrap;
}
.settings-field {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.settings-field label {
  font-size: 12px;
  color: var(--color-fg-muted);
}
.settings-hint {
  font-size: 12px;
  color: var(--color-fg-faint);
  align-self: center;
  padding-bottom: 4px;
}

.chat-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px 24px 0;
  scroll-behavior: smooth;
}
.list-spacer {
  height: 24px;
}

/* 空状态 */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  text-align: center;
  padding: 40px 20px;
}
.empty-icon {
  width: 76px;
  height: 76px;
  border-radius: 20px;
  background: var(--color-accent-soft);
  border: 1px solid rgba(34, 197, 94, 0.3);
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--color-accent);
  margin-bottom: 18px;
}
.empty-state h2 {
  margin: 0 0 6px;
  font-size: 20px;
}
.empty-state p {
  margin: 0 0 26px;
  max-width: 460px;
}
.sample-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  max-width: 620px;
  width: 100%;
}
.sample-card {
  background: var(--color-panel);
  border: 1px solid var(--color-border);
  border-radius: 10px;
  padding: 12px 16px;
  color: var(--color-fg-muted);
  font-size: 13.5px;
  font-family: inherit;
  text-align: left;
  cursor: pointer;
  transition: border-color 160ms ease, color 160ms ease, transform 160ms ease;
}
.sample-card:hover:not(:disabled) {
  border-color: var(--color-accent);
  color: var(--color-fg);
  transform: translateY(-1px);
}
.sample-card:disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

/* 输入区 */
.chat-input-bar {
  flex-shrink: 0;
  padding: 12px 24px 16px;
  background: var(--color-bg);
}
.input-box {
  max-width: 860px;
  margin: 0 auto;
  background: var(--color-panel);
  border: 1px solid var(--color-border);
  border-radius: 12px;
  padding: 10px 12px 8px;
  transition: border-color 160ms ease, box-shadow 160ms ease;
}
.input-box:focus-within {
  border-color: var(--color-accent);
  box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.15);
}
.input-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 8px;
}
.session-tag {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: var(--color-fg-faint);
  font-family: var(--font-mono);
}
.refresh-icon {
  cursor: pointer;
  color: var(--color-fg-muted);
  transition: color 150ms ease;
}
.refresh-icon:hover {
  color: var(--color-accent);
}

/* 引用抽屉 */
.cite-detail {
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.cite-detail-field {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.cite-detail-field .label {
  font-size: 12px;
  color: var(--color-fg-faint);
}
.cite-detail-field .value {
  font-size: 14px;
  color: var(--color-fg);
  word-break: break-all;
}
.cite-detail-field .value.mono {
  font-family: var(--font-mono);
  color: var(--color-accent);
}
.cite-detail-note {
  font-size: 12px;
  color: var(--color-fg-faint);
  margin-top: 8px;
}
.cite-detail-note code {
  font-family: var(--font-mono);
  background: var(--color-muted);
  padding: 1px 6px;
  border-radius: 4px;
}

@media (max-width: 768px) {
  .chat-header {
    padding: 12px 14px;
  }
  .header-sub {
    display: none;
  }
  .chat-list {
    padding: 8px 12px 0;
  }
  .chat-input-bar {
    padding: 10px 12px 12px;
  }
  .sample-grid {
    grid-template-columns: 1fr;
  }
}
</style>
