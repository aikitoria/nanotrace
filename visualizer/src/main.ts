/**
 * Entry point for the Nanotrace visualizer application.
 *
 * This module initializes the WebGPU-based trace viewer and sets up the UI.
 */

import { initApp } from './visualizer.js';

// Keep development sessions free of service-worker control. The production
// base also scopes installation to the GitHub Pages project directory.
if (import.meta.env.PROD && 'serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`, {
            scope: import.meta.env.BASE_URL,
            updateViaCache: 'none',
        }).catch(err => console.error('Service worker registration failed:', err));
    });
}

// Initialize the visualizer with WebGPU setup
// Any initialization errors are displayed in the loading element
document.fonts.load('12px "IBM Plex Sans"').then(() => initApp()).catch(err => {
    document.getElementById('loading')!.textContent = `Error: ${err.message}`;
    console.error(err);
});
