<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import {
  ChatDotRound,
  DataAnalysis,
  Histogram,
  TrendCharts,
  UploadFilled,
  CircleCheckFilled,
  WarningFilled,
} from '@element-plus/icons-vue'
import { getPublicConfig } from '@/api/config'

const route = useRoute()
const health = ref<'ok' | 'down' | 'checking'>('checking')
const runtimeMode = ref<'demo' | 'real' | 'unknown'>('unknown')
let timer: number | undefined

async function checkHealth() {
  const ctrl = new AbortController()
  const timeout = window.setTimeout(() => ctrl.abort(), 3_000)
  try {
    const res = await fetch('/healthz', { signal: ctrl.signal })
    const body = await res.json()
    health.value = body.app === 'ok' && body.qdrant === 'ok' && body.redis === 'ok' ? 'ok' : 'down'
  } catch {
    health.value = 'down'
  } finally {
    window.clearTimeout(timeout)
  }
}

async function loadRuntimeMode() {
  try {
    runtimeMode.value = (await getPublicConfig()).runtime_mode
  } catch {
    runtimeMode.value = 'unknown'
  }
}

onMounted(() => {
  checkHealth()
  loadRuntimeMode()
  timer = window.setInterval(checkHealth, 30_000)
})
onUnmounted(() => window.clearInterval(timer))

const navItems = [
  { path: '/chat', label: '智能问答', icon: ChatDotRound },
  { path: '/benchmark', label: '竞品对标', icon: TrendCharts },
  { path: '/evaluations', label: '模型评估', icon: Histogram },
  { path: '/documents', label: '文档库', icon: DataAnalysis },
  { path: '/upload', label: '上传文档', icon: UploadFilled },
]
</script>

<template>
  <a class="skip-link" href="#main-content">跳到主内容</a>
  <div class="app-shell">
    <!-- 左侧导航 -->
    <aside class="sidebar">
      <router-link to="/chat" class="brand">
        <div class="brand-logo">
          <span class="brand-logo-bar"></span>
        </div>
        <div class="brand-text">
          <div class="brand-title">NewRAG</div>
          <div class="brand-sub">多模态财报深度分析</div>
        </div>
      </router-link>

      <nav class="nav">
        <router-link
          v-for="item in navItems"
          :key="item.path"
          :to="item.path"
          class="nav-item"
          :class="{ active: route.path.startsWith(item.path) }"
        >
          <el-icon :size="18"><component :is="item.icon" /></el-icon>
          <span>{{ item.label }}</span>
        </router-link>
      </nav>

      <div class="sidebar-footer">
        <div class="health" :class="health" role="status" aria-live="polite">
          <el-icon v-if="health === 'ok'" :size="14"><CircleCheckFilled /></el-icon>
          <el-icon v-else :size="14"><WarningFilled /></el-icon>
          <span v-if="health === 'ok'">服务正常</span>
          <span v-else-if="health === 'down'">后端离线</span>
          <span v-else>检测中…</span>
        </div>
      </div>
    </aside>

    <!-- 主内容 -->
    <main id="main-content" class="main" tabindex="-1">
      <div v-if="runtimeMode === 'demo'" class="demo-banner" role="status">
        演示模式：公网环境使用 Mock 模型，仅展示系统流程；模型评测数据来自本地真实实验。
      </div>
      <div class="route-content">
        <router-view v-slot="{ Component }">
          <transition name="fade-slide" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </div>
    </main>
  </div>
</template>

<style scoped>
.skip-link {
  position: fixed;
  top: 8px;
  left: 8px;
  z-index: 1000;
  padding: 8px 12px;
  border-radius: 6px;
  background: var(--color-accent);
  color: white;
  transform: translateY(-150%);
}
.skip-link:focus {
  transform: translateY(0);
}
.app-shell {
  display: flex;
  height: 100%;
}

.sidebar {
  display: flex;
  flex-direction: column;
  width: 236px;
  flex-shrink: 0;
  border-right: 1px solid var(--color-border);
  background: var(--color-panel);
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 20px 20px 16px;
  text-decoration: none;
  color: var(--color-fg);
}
.brand:hover {
  color: var(--color-fg);
}

.brand-logo {
  width: 36px;
  height: 36px;
  border-radius: 10px;
  background: linear-gradient(135deg, #16a34a, #22c55e);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 12px rgba(34, 197, 94, 0.35);
}
.brand-logo-bar {
  width: 16px;
  height: 16px;
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.22);
  border: 2px solid rgba(255, 255, 255, 0.85);
}

.brand-title {
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 0.3px;
}
.brand-sub {
  font-size: 12px;
  color: var(--color-fg-muted);
  margin-top: 1px;
}

.nav {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 8px 12px;
  flex: 1;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 8px;
  color: var(--color-fg-muted);
  text-decoration: none;
  font-size: 14px;
  transition:
    background 160ms ease,
    color 160ms ease;
  cursor: pointer;
}
.nav-item:hover {
  background: var(--color-muted);
  color: var(--color-fg);
}
.nav-item.active {
  background: var(--color-accent-soft);
  color: var(--color-accent);
  font-weight: 600;
}

.sidebar-footer {
  padding: 12px 20px;
  border-top: 1px solid var(--color-border);
}

.health {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
}
.health.ok {
  color: var(--color-accent);
}
.health.down {
  color: var(--color-danger);
}
.health.checking {
  color: var(--color-fg-faint);
}

.main {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-width: 0;
  overflow: hidden;
  background: var(--color-bg);
}

.demo-banner {
  flex: none;
  padding: 8px 16px;
  border-bottom: 1px solid rgba(245, 158, 11, 0.35);
  background: rgba(245, 158, 11, 0.1);
  color: #fbbf24;
  font-size: 13px;
  line-height: 20px;
  text-align: center;
}

.route-content {
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

/* 窄屏：顶部精简品牌区 + 底部三项导航 */
@media (max-width: 768px) {
  .app-shell {
    flex-direction: column;
  }
  .sidebar {
    width: 100%;
    height: auto;
    flex-direction: row;
    align-items: center;
    border-right: none;
    border-bottom: 1px solid var(--color-border);
    padding: 8px 12px;
  }
  .brand {
    padding: 0;
    margin-right: auto;
  }
  .brand-sub {
    display: none;
  }
  .nav {
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 20;
    flex-direction: row;
    gap: 0;
    padding: 4px max(8px, env(safe-area-inset-right)) max(4px, env(safe-area-inset-bottom))
      max(8px, env(safe-area-inset-left));
    border-top: 1px solid var(--color-border);
    background: var(--color-panel);
  }
  .nav-item {
    flex: 1;
    min-height: 52px;
    flex-direction: column;
    justify-content: center;
    gap: 2px;
    padding: 4px 8px;
    white-space: nowrap;
    font-size: 12px;
  }
  .sidebar-footer {
    display: none;
  }
  .main {
    padding-bottom: calc(60px + env(safe-area-inset-bottom));
    box-sizing: border-box;
  }
}
</style>
