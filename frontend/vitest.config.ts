import { defineConfig } from 'vitest/config'

export default defineConfig({
  envDir: '..',
  test: {
    environment: 'jsdom',
    globals: true,
  },
})
