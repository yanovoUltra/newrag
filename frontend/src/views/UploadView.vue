<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import {
  UploadFilled,
  Document,
  CircleCheck,
  Warning,
  Delete,
  ArrowRight,
} from '@element-plus/icons-vue'
import { pollTask, uploadDocument } from '@/api/documents'
import { getPublicConfig } from '@/api/config'
import type { TaskItem } from '@/types'
import { getCurrentYear, MIN_FISCAL_YEAR } from '@/utils/date'

const router = useRouter()

const file = ref<File | null>(null)
const orgId = ref(localStorage.getItem('newrag:chat:settings:org') ?? 'default')
const visibility = ref('public')
const fiscalYear = ref<number | null>(null)
const fiscalQuarter = ref<number | null>(null)
const currentYear = ref(getCurrentYear())
const fiscalYearMin = ref(MIN_FISCAL_YEAR)
const maxUploadMb = ref(50)
const acceptedExtensions = ref(['.pdf', '.docx', '.xlsx', '.png', '.jpg', '.jpeg'])

const uploading = ref(false)
const task = ref<TaskItem | null>(null)
const taskAbort = ref<AbortController | null>(null)
const duplicateDocId = ref<string | null>(null)

const accept = computed(() => acceptedExtensions.value.join(','))
const acceptedLabel = computed(() =>
  acceptedExtensions.value.map((ext) => ext.replace('.', '').toUpperCase()).join(' / '),
)

const stageLabels: Record<string, string> = {
  queued: '排队等待',
  parse: '解析文档（版面 / OCR）',
  chunk: '构建章节树与切分',
  embed: '向量化（稠密 + 稀疏）',
  index: '写入向量库',
}

function onFilePick(e: Event) {
  const input = e.target as HTMLInputElement
  if (input.files?.length) file.value = input.files[0]
  input.value = ''
}

function onDrop(e: DragEvent) {
  const f = e.dataTransfer?.files?.[0]
  if (f) file.value = f
}

function clearFile() {
  file.value = null
}

const fileMeta = computed(() => {
  if (!file.value) return null
  const f = file.value
  let size: string
  if (f.size < 1024) size = `${f.size} B`
  else if (f.size < 1024 * 1024) size = `${(f.size / 1024).toFixed(1)} KB`
  else size = `${(f.size / 1024 / 1024).toFixed(1)} MB`
  return { name: f.name, size }
})

async function submit() {
  if (!file.value) {
    ElMessage.warning('请先选择文件')
    return
  }
  if (file.value.size > maxUploadMb.value * 1024 * 1024) {
    ElMessage.warning(`文件不能超过 ${maxUploadMb.value}MB`)
    return
  }
  if (
    fiscalYear.value !== null &&
    (fiscalYear.value < fiscalYearMin.value || fiscalYear.value > currentYear.value)
  ) {
    ElMessage.warning(`财年须在 ${fiscalYearMin.value}–${currentYear.value} 之间`)
    return
  }
  uploading.value = true
  duplicateDocId.value = null
  task.value = null
  try {
    const res = await uploadDocument({
      file: file.value,
      org_id: orgId.value,
      visibility: visibility.value,
      fiscal_year: fiscalYear.value,
      fiscal_quarter: fiscalQuarter.value,
    })
    if (res.status === 'duplicate') {
      duplicateDocId.value = res.doc_id
      ElMessage.info('该文件已入库（sha256 幂等），无需重复上传')
      return
    }
    const ctrl = new AbortController()
    taskAbort.value = ctrl
    const finished = await pollTask(
      res.task_id,
      (t) => (task.value = t),
      ctrl.signal,
    )
    if (finished.status === 'success') {
      ElMessage.success('文档解析入库完成')
    }
  } catch (e) {
    if ((e as Error).name === 'AbortError') return
    ElMessage.error(e instanceof Error ? e.message : String(e))
  } finally {
    uploading.value = false
    taskAbort.value = null
  }
}

function goDocuments() {
  router.push('/documents')
}

onMounted(async () => {
  try {
    const config = await getPublicConfig()
    maxUploadMb.value = config.max_upload_mb
    acceptedExtensions.value = config.accepted_extensions
    fiscalYearMin.value = config.fiscal_year_min
    currentYear.value = config.current_year
  } catch {
    // 保留与服务端默认一致的 50MB 与本地当前年。
  }
})
</script>

<template>
  <div class="upload-view">
    <header class="upload-header">
      <div class="header-title">
        <h1>上传文档</h1>
        <span class="header-sub">支持 PDF / DOCX / XLSX / 图片，解析后自动切分并向量化入库</span>
      </div>
    </header>

    <div class="upload-body">
      <!-- 拖拽区 -->
      <div
        class="dropzone"
        :class="{ active: file, dragging: uploading }"
        @dragover.prevent
        @drop.prevent="onDrop"
        @click="!uploading && ($refs.fileInput as any)?.click()"
        role="button"
        tabindex="0"
        @keydown.enter="!uploading && ($refs.fileInput as any)?.click()"
      >
        <input
          ref="fileInput"
          type="file"
          :accept="accept"
          hidden
          @change="onFilePick"
        />
        <template v-if="fileMeta">
          <el-icon :size="34" class="dz-icon"><Document /></el-icon>
          <div class="dz-name">{{ fileMeta.name }}</div>
          <div class="dz-meta muted">{{ fileMeta.size }}</div>
          <el-button
            link
            type="danger"
            :icon="Delete"
            :disabled="uploading"
            @click.stop="clearFile"
          >
            移除文件
          </el-button>
        </template>
        <template v-else>
          <el-icon :size="40" class="dz-icon"><UploadFilled /></el-icon>
          <div class="dz-main">拖拽文件到此处，或 <b>点击选择</b></div>
          <div class="dz-meta muted">{{ acceptedLabel }}，单文件 ≤ {{ maxUploadMb }}MB</div>
        </template>
      </div>

      <!-- 元数据表单 -->
      <div class="form-grid card">
        <div class="form-field">
          <label>机构 ID <span class="req">*</span></label>
          <el-input v-model="orgId" placeholder="default" :disabled="uploading" />
          <span class="field-hint">检索按此维度做权限隔离</span>
        </div>
        <div class="form-field">
          <label>可见性 <span class="req">*</span></label>
          <el-select v-model="visibility" :disabled="uploading">
            <el-option label="公开 public" value="public" />
            <el-option label="内部 internal" value="internal" />
            <el-option label="受限 restricted" value="restricted" />
          </el-select>
          <span class="field-hint">越高的档位仅更高权限用户可见</span>
        </div>
        <div class="form-field">
          <label>财年（可选）</label>
          <el-input-number
            v-model="fiscalYear"
            :min="fiscalYearMin"
            :max="currentYear"
            :disabled="uploading"
            placeholder="如 2024"
            controls-position="right"
            style="width: 100%"
          />
          <span class="field-hint">
            可填写 {{ fiscalYearMin }}–{{ currentYear }}，用于问题中年份的精确过滤
          </span>
        </div>
        <div class="form-field">
          <label>财季（可选）</label>
          <el-select v-model="fiscalQuarter" :disabled="uploading" clearable placeholder="全年">
            <el-option v-for="q in [1, 2, 3, 4]" :key="q" :label="`Q${q}`" :value="q" />
          </el-select>
          <span class="field-hint">仅财年报告需要填写</span>
        </div>
      </div>

      <!-- 重复提示 -->
      <el-alert
        v-if="duplicateDocId"
        title="文件已存在（内容相同，sha256 幂等）"
        type="info"
        :closable="false"
        show-icon
        class="dup-alert"
      >
        <template #default>
          doc_id：<code>{{ duplicateDocId }}</code>，已可直接提问
        </template>
      </el-alert>

      <!-- 任务进度 -->
      <div v-if="task" class="task-panel card">
        <div class="task-head">
          <span class="task-stage">{{ stageLabels[task.stage] ?? task.stage }}</span>
          <span v-if="task.status === 'failed'" class="task-status fail">
            <el-icon><Warning /></el-icon> 失败
          </span>
          <span v-else-if="task.status === 'success'" class="task-status ok">
            <el-icon><CircleCheck /></el-icon> 已完成
          </span>
          <span v-else class="task-status run">处理中</span>
        </div>
        <el-progress
          :percentage="task.progress"
          :status="task.status === 'failed' ? 'exception' : task.status === 'success' ? 'success' : undefined"
          :stroke-width="10"
        />
        <div v-if="task.message" class="task-msg muted">{{ task.message }}</div>
        <div v-if="task.status === 'failed'" class="task-err">{{ task.message }}</div>
        <div v-if="task.status === 'success'" class="task-actions">
          <el-button type="primary" :icon="ArrowRight" @click="goDocuments">
            去文档库查看
          </el-button>
        </div>
      </div>

      <!-- 提交 -->
      <div class="submit-bar">
        <el-button
          type="primary"
          size="large"
          :icon="UploadFilled"
          :loading="uploading"
          :disabled="!file"
          @click="submit"
        >
          {{ uploading ? '解析入库中…' : '开始上传并解析' }}
        </el-button>
        <span class="submit-hint muted">上传后在后台解析，可在上方查看实时进度</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.upload-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow-y: auto;
}

.upload-header {
  padding: 14px 24px;
  border-bottom: 1px solid var(--color-border);
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

.upload-body {
  max-width: 760px;
  margin: 0 auto;
  padding: 24px;
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 18px;
}

/* 拖拽区 */
.dropzone {
  border: 1.5px dashed var(--color-border);
  border-radius: 14px;
  background: var(--color-panel);
  padding: 42px 24px;
  text-align: center;
  cursor: pointer;
  transition: border-color 160ms ease, background 160ms ease;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}
.dropzone:hover {
  border-color: var(--color-accent);
}
.dropzone.active {
  border-color: var(--color-accent);
  background: var(--color-accent-soft);
}
.dz-icon {
  color: var(--color-accent);
}
.dz-main {
  font-size: 15px;
  color: var(--color-fg);
}
.dz-main b {
  color: var(--color-accent);
}
.dz-name {
  font-size: 16px;
  font-weight: 600;
  word-break: break-all;
}
.dz-meta {
  font-size: 12.5px;
}

/* 表单 */
.form-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
  padding: 20px;
}
.form-field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.form-field label {
  font-size: 13px;
  color: var(--color-fg-muted);
}
.req {
  color: var(--color-danger);
}
.field-hint {
  font-size: 11.5px;
  color: var(--color-fg-faint);
}

.dup-alert {
  margin-top: 4px;
}
.dup-alert code {
  font-family: var(--font-mono);
  background: var(--color-muted);
  padding: 1px 6px;
  border-radius: 4px;
}

/* 任务面板 */
.task-panel {
  padding: 18px 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.task-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.task-stage {
  font-size: 14px;
  font-weight: 600;
}
.task-status {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 12.5px;
}
.task-status.ok {
  color: var(--color-accent);
}
.task-status.fail {
  color: var(--color-danger);
}
.task-status.run {
  color: var(--color-fg-muted);
}
.task-msg {
  font-size: 12.5px;
}
.task-err {
  font-size: 12.5px;
  color: var(--color-danger);
  background: rgba(239, 68, 68, 0.1);
  border-radius: 6px;
  padding: 8px 10px;
}
.task-actions {
  display: flex;
  justify-content: flex-end;
}

.submit-bar {
  display: flex;
  align-items: center;
  gap: 12px;
}
.submit-hint {
  font-size: 12.5px;
}

@media (max-width: 768px) {
  .upload-header {
    padding: 12px 14px;
  }
  .upload-body {
    padding: 16px 14px;
  }
  .form-grid {
    grid-template-columns: 1fr;
  }
}
</style>
