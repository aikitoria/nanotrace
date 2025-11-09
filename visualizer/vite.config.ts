import { defineConfig } from 'vite';
import { imagetools } from 'vite-imagetools';
import { createHtmlPlugin } from 'vite-plugin-html';
import { execSync } from 'child_process';

// Get git commit hash (first 4 chars) for version string
let gitHash = 'dev';
try {
  gitHash = execSync('git rev-parse --short=4 HEAD').toString().trim();
} catch (e) {
  console.warn('Could not get git commit hash, using "dev"');
}

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
