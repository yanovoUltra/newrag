import pluginVue from 'eslint-plugin-vue'
import { defineConfigWithVueTs, vueTsConfigs } from '@vue/eslint-config-typescript'

export default defineConfigWithVueTs(
  { ignores: ['dist/**', 'node_modules/**', 'test-results/**'] },
  pluginVue.configs['flat/essential'],
  vueTsConfigs.recommended,
  {
    files: ['src/**/*.{ts,vue}', 'e2e/**/*.ts'],
    rules: {
      'vue/multi-word-component-names': 'off',
    },
  },
)
