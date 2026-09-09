import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/chat' },
    {
      path: '/chat',
      name: 'chat',
      component: () => import('@/views/ChatView.vue'),
      meta: { title: '智能问答' },
    },
    {
      path: '/benchmark',
      name: 'benchmark',
      component: () => import('@/views/BenchmarkView.vue'),
      meta: { title: '竞品对标' },
    },
    {
      path: '/evaluations',
      name: 'evaluations',
      component: () => import('@/views/EvaluationsView.vue'),
      meta: { title: '模型评估' },
    },
    {
      path: '/documents',
      name: 'documents',
      component: () => import('@/views/DocumentsView.vue'),
      meta: { title: '文档库' },
    },
    {
      path: '/upload',
      name: 'upload',
      component: () => import('@/views/UploadView.vue'),
      meta: { title: '上传文档' },
    },
    { path: '/:pathMatch(.*)*', redirect: '/chat' },
  ],
})

router.afterEach((to) => {
  const title = to.meta.title as string | undefined
  document.title = title ? `${title} · NewRAG` : 'NewRAG · 多模态财报深度分析'
})

export default router
