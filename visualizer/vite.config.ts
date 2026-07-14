import { defineConfig } from 'vite';
import { imagetools } from 'vite-imagetools';
import { createHtmlPlugin } from 'vite-plugin-html';

const gitHash = process.env.NANOTRACE_GIT_HASH ?? 'dev';

export default defineConfig({
  base: '/nanotrace/',
  define: {
    __GIT_HASH__: JSON.stringify(gitHash),
  },
  server: {
    port: 4173,
  },
  plugins: [
    imagetools(),
    createHtmlPlugin({
      minify: {
        collapseWhitespace: true,
        removeComments: true,
        removeRedundantAttributes: true,
        removeScriptTypeAttributes: true,
        removeStyleLinkTypeAttributes: true,
        useShortDoctype: true,
        minifyCSS: true,
        minifyJS: true,
      },
    }),
  ],
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    emptyOutDir: true,
    minify: 'terser',
    cssMinify: true,
    terserOptions: {
      compress: {
        drop_console: false,
        drop_debugger: true,
        pure_funcs: [],
      },
      mangle: true,
      format: {
        comments: false,
      },
    },
  },
  resolve: {
    extensions: ['.ts', '.js'],
  },
  publicDir: 'public',
});
