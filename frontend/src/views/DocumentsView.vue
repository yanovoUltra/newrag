<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  Refresh,
  Plus,
  Delete,
  Document,
  Search,
} from '@element-plus/icons-vue'
import { deleteDocument, listDocumentPage } from '@/api/documents'
import { notifyError } from '@/api/client'
import type { DocumentItem } from '@/types'

const router = useRouter()
const docs = ref<DocumentItem[]>([])
const loading = ref(false)
const keyword = ref('')
const orgFilter = ref('')
const statusFilter = ref('')
const total = ref(0)
const page = ref(1)
const pageSize = 20
let loadAbort: AbortController | null = null
let searchTimer: number | undefined

async function load() {
  loadAbort?.abort()
  const ctrl = new AbortController()
  loadAbort = ctrl
  loading.value = true
  try {
    const result = await listDocumentPage({
      filename: keyword.value.trim() || undefined,
      orgId: orgFilter.value.trim() || undefined,
      status: statusFilter.value || undefined,
      limit: pageSize,
      offset: (page.value - 1) * pageSize,
      signal: ctrl.signal,
    })
    if (ctrl.signal.aborted) return
    docs.value = result.items
    total.value = result.total
  } catch (e) {
    if ((e as Error).name === 'AbortError') return
    notifyError(e)
  } finally {
    if (loadAbort === ctrl) loading.value = false
  }
}

function scheduleSearch() {
  page.value = 1
  window.clearTimeout(searchTimer)
  searchTimer = window.setTimeout(load, 250)
}

function changePage(next: number) {
  page.value = next
  load()
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
    if (docs.value.length === 1 && page.value > 1) page.value -= 1
    await load()
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

watch([keyword, orgFilter, statusFilter], scheduleSearch)
onMounted(load)
onUnmounted(() => {
  loadAbort?.abort()
  window.clearTimeout(searchTimer)
})
</script>

<template>
  <div class="doc-view">
    <header class="doc-header">
      <div class="header-title">
        <h1>文档库</h1>
        <span class="header-sub">共 {{ total }} 份财报，检索与问答均基于已入库知识块</span>
      </div>
      <div class="header-actions">
        <el-input
          v-model="keyword"
          placeholder="搜索文件名"
          aria-label="按文件名搜索"
          clearable
          :prefix-icon="Search"
          style="width: 220px"
        />
        <el-input
          v-model="orgFilter"
          placeholder="机构 ID"
          aria-label="按机构 ID 筛选"
          clearable
          style="width: 130px"
        />
        <el-select
          v-model="statusFilter"
          placeholder="全部状态"
          aria-label="按文档状态筛选"
          clearable
          style="width: 130px"
        >
          <el-option label="已入库" value="indexed" />
          <el-option label="处理中" value="pending" />
          <el-option label="失败" value="failed" />
        </el-select>
        <el-button :icon="Refresh" :loading="loading" @click="load">刷新</el-button>
        <el-button type="primary" :icon="Plus" @click="router.push('/upload')">
          上传文档
        </el-button>
      </div>
    </header>

    <div class="doc-body">
      <el-table
        class="desktop-table"
        :data="docs"
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
                class="table-delete"
                link
                type="danger"
                :icon="Delete"
                :aria-label="`删除文档 ${row.filename}`"
                :disabled="BUSY_STATUS.has(row.status)"
                @click="onDelete(row)"
              />
            </el-tooltip>
          </template>
        </el-table-column>
      </el-table>

      <div v-loading="loading" class="mobile-cards" aria-live="polite">
        <article v-for="doc in docs" :key="doc.id" class="doc-card">
          <div class="card-heading">
            <div class="doc-name">
              <el-icon :size="18" class="doc-icon"><Document /></el-icon>
              <div class="doc-name-text">
                <span class="doc-filename">{{ doc.filename }}</span>
                <span class="doc-meta">{{ doc.file_type?.toUpperCase() }} · {{ fmtSizeBytes(doc.size_bytes ?? 0) }}</span>
              </div>
            </div>
            <el-button
              class="card-delete"
              type="danger"
              text
              :icon="Delete"
              :aria-label="`删除文档 ${doc.filename}`"
              :disabled="BUSY_STATUS.has(doc.status)"
              @click="onDelete(doc)"
            />
          </div>
          <dl class="card-grid">
            <div><dt>状态</dt><dd>{{ statusInfo(doc).label }}</dd></div>
            <div><dt>机构</dt><dd>{{ doc.org_id }}</dd></div>
            <div><dt>财年</dt><dd>{{ doc.fiscal_year ?? '—' }}</dd></div>
            <div><dt>知识块</dt><dd>{{ doc.chunk_count }}</dd></div>
          </dl>
          <div class="card-date">上传于 {{ fmtDate(doc.created_at) }}</div>
        </article>
        <div v-if="!loading && docs.length === 0" class="mobile-empty">暂无符合条件的文档</div>
      </div>

      <el-pagination
        v-if="total > pageSize"
        class="pagination"
        background
        layout="prev, pager, next"
        :current-page="page"
        :page-size="pageSize"
        :total="total"
        @current-change="changePage"
      />
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

.mobile-cards {
  display: none;
}
.table-delete {
  width: 44px;
  height: 44px;
}
.pagination {
  justify-content: flex-end;
  margin-top: 16px;
}

@media (max-width: 768px) {
  .doc-header {
    padding: 12px 14px;
    align-items: stretch;
  }
  .header-actions {
    display: grid;
    grid-template-columns: 1fr 1fr;
  }
  .header-actions :deep(.el-input),
  .header-actions :deep(.el-select) {
    width: 100% !important;
  }
  .header-actions :deep(.el-input__wrapper),
  .header-actions :deep(.el-select__wrapper),
  .header-actions :deep(.el-button) {
    min-height: 44px;
  }
  .doc-body {
    padding: 12px;
  }
  .desktop-table {
    display: none;
  }
  .mobile-cards {
    display: grid;
    gap: 10px;
  }
  .doc-card {
    padding: 14px;
    border: 1px solid var(--color-border);
    border-radius: 12px;
    background: var(--color-panel);
  }
  .card-heading {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 8px;
  }
  .card-delete {
    width: 44px;
    height: 44px;
    flex-shrink: 0;
  }
  .card-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin: 14px 0 10px;
  }
  .card-grid div {
    min-width: 0;
  }
  .card-grid dt {
    font-size: 11px;
    color: var(--color-fg-faint);
  }
  .card-grid dd {
    margin: 2px 0 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .card-date,
  .mobile-empty {
    font-size: 12px;
    color: var(--color-fg-muted);
  }
  .mobile-empty {
    padding: 40px 0;
    text-align: center;
  }
  .pagination {
    justify-content: center;
  }
}
</style>
