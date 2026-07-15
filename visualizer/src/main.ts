/**
 * Entry point for the Nanotrace visualizer application.
 *
 * This module initializes the WebGPU-based trace viewer and sets up the UI.
 */

import { initApp } from './visualizer.js';

// Initialize the visualizer with WebGPU setup
// Any initialization errors are displayed in the loading element
initApp().catch(err => {
    document.getElementById('loading')!.textContent = `Error: ${err.message}`;
    console.error(err);
});
