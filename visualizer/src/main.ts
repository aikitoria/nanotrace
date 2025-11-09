/**
 * Entry point for the Nanotrace visualizer application.
 *
 * This module initializes the WebGPU-based trace viewer and sets up the UI.
 * The avatar image is converted to WebP at build time by Vite for optimization.
 */

import { initApp } from './visualizer.js';
import avatarUrl from './assets/avatar.png?format=webp&quality=85';

// Set avatar image for the file selector UI
// Vite's imagetools plugin converts PNG to WebP with 85% quality during build
const avatarImg = document.querySelector('.avatar') as HTMLImageElement;
if (avatarImg) {
    avatarImg.src = avatarUrl;
}

// Initialize the visualizer with WebGPU setup
// Any initialization errors are displayed in the loading element
initApp().catch(err => {
    document.getElementById('loading')!.textContent = `Error: ${err.message}`;
    console.error(err);
});
