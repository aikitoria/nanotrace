/**
 * TypeScript type declarations for Vite build system.
 *
 * Enables type-safe imports of image assets processed by vite-imagetools.
 * The plugin converts PNG images to WebP format with specified quality during build.
 */

/// <reference types="vite/client" />

/**
 * Module declaration for PNG imports with WebP conversion.
 * Usage: import avatarUrl from './avatar.png?format=webp&quality=85';
 * Returns the URL of the optimized WebP image in the build output.
 */
declare module '*.png?format=webp&quality=85' {
  const src: string;
  export default src;
}
