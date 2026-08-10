<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  Refresh,
  Plus,
  Delete,
  Document,
  Search,
} from '@element-plus/icons-vue'
import { deleteDocument, listDocuments } from '@/api/documents'
import { notifyError } from '@/api/client'
import type { DocumentItem } from '@/types'

const router = useRouter()
const docs = ref<DocumentItem[]>([])
const loading = ref(false)
const keyword = ref('')

const filtered = computed(() => {
  const kw = keyword.value.trim().toLowerCase()
  if (!kw) return docs.value
  return docs.value.filter(
    (d) =>
      d.filename.toLowerCase().includes(kw) ||
      d.id.toLowerCase().includes(kw) ||
      d.org_id.toLowerCase().includes(kw),
  )
})

async function load() {
  loading.value = true
  try {
    docs.value = await listDocuments()
  } catch (e) {
    notifyError(e)
  } finally {
    loading.value = false
  }
}

async function onDelete(d: DocumentItem) {
  try {
    await ElMessageBox.confirm(
      `删除后将移除「${d.filename}」的全部向量与登记信息，确定删除？`,
      '删除文档',
      { type: 'warning', confirmButtonText: '删除', cancelButtonText: '取消' },
    )
  } catch {
    return // 用户取消
  }
  try {
    await deleteDocument(d.id)
    ElMessage.success('已删除')
    load()
  } catch (e) {
    notifyError(e)
  }
}

const statusMeta: Record<string, { type: string; label: string }> = {
  pending: { type: 'info', label: '排队中' },
  parsing: { type: 'warning', label: '解析中' },
  chunking: { type: 'warning', label: '切分中' },
  embedding: { type: 'warning', label: '向量化' },
  indexed: { type: 'success', label: '已入库' },
  failed: { type: 'danger', label: '失败' },
}

const BUSY_STATUS = new Set(['pending', 'parsing', 'chunking', 'embedding'])

function statusInfo(d: DocumentItem): { type: string; label: string } {
  return statusMeta[d.status] ?? { type: 'info', label: d.status }
}

const visibilityLabels: Record<string, string> = {
  public: '公开',
  internal: '内部',
  restricted: '受限',
}

function fmtDate(iso: string): string {
  if (!iso) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString('zh-CN', { hour12: false })
}

function fmtSizeBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

onMounted(load)
</script>

<template>
  <div class="doc-view">
    <header class="doc-header">
      <div class="header-title">
        <h1>文档库</h1>
        <span class="header-sub">共 {{ docs.length }} 份财报，检索与问答均基于已入库知识块</span>
      </div>
      <div class="header-actions">
        <el-input
          v-model="keyword"
          placeholder="搜索文件名 / ID"
          clearable
          :prefix-icon="Search"
          style="width: 220px"
        />
        <el-button :icon="Refresh" :loading="loading" @click="load">刷新</el-button>
        <el-button type="primary" :icon="Plus" @click="router.push('/upload')">
          上传文档
        </el-button>
      </div>
    </header>

    <div class="doc-body">
      <el-table
        :data="filtered"
        v-loading="loading"
        empty-text="暂无文档，点击右上角上传"
        stripe
      >
        <el-table-column label="文件名" min-width="260">
          <template #default="{ row }">
            <div class="doc-name">
              <el-icon :size="16" class="doc-icon"><Document /></el-icon>
              <div class="doc-name-text">
                <span class="doc-filename">{{ row.filename }}</span>
                <span class="doc-meta">{{ row.file_type?.toUpperCase() }} · {{ fmtSizeBytes(row.size_bytes ?? 0) }}</span>
              </div>
            </div>
          </template>
        </el-table-column>

        <el-table-column label="状态" width="100">
          <template #default="{ row }">
            <el-tooltip :content="row.error || statusInfo(row).label" placement="top" :disabled="!row.error">
              <el-tag :type="(statusInfo(row).type as any)" size="small" effect="dark">
                {{ statusInfo(row).label }}
              </el-tag>
            </el-tooltip>
          </template>
        </el-table-column>

        <el-table-column label="可见性" width="90">
          <template #default="{ row }">
            <el-tag size="small" effect="plain">
              {{ visibilityLabels[row.visibility] ?? row.visibility }}
            </el-tag>
          </template>
        </el-table-column>

        <el-table-column label="财年" width="80" align="center">
          <template #default="{ row }">
            <span v-if="row.fiscal_year">{{ row.fiscal_year }}<template v-if="row.fiscal_quarter">Q{{ row.fiscal_quarter }}</template></span>
            <span v-else class="muted">—</span>
          </template>
        </el-table-column>

        <el-table-column label="知识块" width="90" align="center" prop="chunk_count" />

        <el-table-column label="机构" width="110" prop="org_id" show-overflow-tooltip />

        <el-table-column label="上传时间" width="150">
          <template #default="{ row }">
            <span class="muted">{{ fmtDate(row.created_at) }}</span>
          </template>
        </el-table-column>

        <el-table-column label="操作" width="100" align="center" fixed="right">
          <template #default="{ row }">
            <el-tooltip content="删除文档及其向量" placement="top">
              <el-button
                link
                type="danger"
                :icon="Delete"
                :disabled="BUSY_STATUS.has(row.status)"
                @click="onDelete(row)"
              />
            </el-tooltip>
          </template>
        </el-table-column>
      </el-table>
    </div>
  </div>
</template>

<style scoped>
.doc-view {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.doc-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 24px;
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
  flex-wrap: wrap;
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
  align-items: center;
}

.doc-body {
  flex: 1;
  overflow: auto;
  padding: 16px 24px;
}

.doc-name {
  display: flex;
  align-items: center;
  gap: 10px;
}
.doc-icon {
  color: var(--color-accent);
  flex-shrink: 0;
}
.doc-name-text {
  display: flex;
  flex-direction: column;
  min-width: 0;
}
.doc-filename {
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.doc-meta {
  font-size: 11.5px;
  color: var(--color-fg-faint);
}

@media (max-width: 768px) {
  .doc-header {
    padding: 12px 14px;
  }
  .doc-body {
    padding: 12px;
  }
}
</style>
