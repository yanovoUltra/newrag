<script setup lang="ts">
import { computed } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'

marked.setOptions({ gfm: true, breaks: true })

const props = defineProps<{ text: string }>()

const html = computed(() => {
  const raw = marked.parse(props.text ?? '', { async: false }) as string
  return DOMPurify.sanitize(raw, {
    USE_PROFILES: { html: true },
  })
})
</script>

<template>
  <div class="markdown-body" v-html="html"></div>
</template>
