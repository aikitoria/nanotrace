/**
 * Main orchestrator for the Nanotrace visualizer.
 *
 * ZoneVisualizer coordinates all rendering subsystems (WebGPU, 2D canvas, timeline)
 * and manages the application lifecycle in two phases:
 * 1. WebGPU initialization (initWebGPU)
 * 2. Trace data loading and visualization setup (initVisualization)
 *
 * Architecture:
 * - WebGPU handles high-performance rendering of zones, blocks, and lanes
 * - Canvas 2D overlays text labels (blocks and zones)
 * - InteractionManager handles mouse input and hit detection
 * - Camera manages viewport transformations with independent X/Y zoom
 */

import { Camera } from './utils/camera.js';
import { parseTraceFile, buildHierarchy, formatString as formatStringHelper } from './utils/file-loader.js';
import { createGPUBuffers, createPipelines, GPUResources } from './renderers/gpu-renderer.js';
import { LabelRenderer } from './renderers/label-renderer.js';
import { TimelineRenderer } from './renderers/timeline-renderer.js';
import { InteractionManager } from './interaction-manager.js';
import {
    FormatDescriptor,
    Zone,
    Block,
    BlockLane,
    Lane
} from './utils/types.js';
import {
    BASE_TIME_RANGE,
    ZOOM_FACTOR,
    MIN_ZOOM_X,
    MAX_ZOOM_X,
    MIN_ZOOM_Y,
    MAX_ZOOM_Y,
    PAN_SPEED,
    SELECTION_EPSILON,
    SM_LABEL_WIDTH,
    TIMELINE_HEIGHT,
    INITIAL_ZOOM_PADDING,
    MIN_SELECTION_DISTANCE,
    LOADING_OVERLAY_DELAY,
    MAX_KERNEL_NAME_LENGTH,
    FPS_PADDING_WIDTH,
    MS_TO_NS,
    CLEAR_COLOR_R,
    CLEAR_COLOR_G,
    CLEAR_COLOR_B
} from './utils/constants.js';

// Git commit hash injected at build time by Vite
declare const __GIT_HASH__: string;

const VERSION = `0.1-${__GIT_HASH__}`;

/**
 * Main application class that coordinates all visualization subsystems.
 *
 * Manages the full application lifecycle from WebGPU setup through file loading,
 * event handling, and the render loop. Delegates specialized rendering tasks to
 * dedicated renderer classes while maintaining overall coordination.
 */
export class ZoneVisualizer {
    // Canvas elements for WebGPU and 2D text rendering
    private canvas: HTMLCanvasElement;
    private labelCanvas: HTMLCanvasElement;
    private labelCtx: CanvasRenderingContext2D;

    // UI element references
    private tooltip: HTMLElement;
    private loading: HTMLElement;
    private stats: HTMLElement;
    private laneLabelsContainer: HTMLElement;
    private timelineContainer: HTMLElement;
    private cursorLine: HTMLElement;
    private cursorTimestamp: HTMLElement;
    private selectionRegion: HTMLElement;
    private selectionLineStart: HTMLElement;
    private selectionLineEnd: HTMLElement;
    private selectionLabel: HTMLElement;
    private fileSelector: HTMLElement;
    private loadingOverlay: HTMLElement;
    private loadingText: HTMLElement;
    private fileInput: HTMLInputElement;

    // WebGPU resources (initialized in initWebGPU)
    private adapter: GPUAdapter | null = null;
    private device!: GPUDevice;
    private context!: GPUCanvasContext;
    private format!: GPUTextureFormat;
    private gpuResources?: GPUResources;

    // Trace data hierarchy (populated from file)
    private zones: Zone[] = [];
    private blocks: Block[] = [];
    private blockLanes: BlockLane[] = [];
    private lanes: Lane[] = [];
    private laneLabels: HTMLElement[] = [];
    private formatDescriptors: FormatDescriptor[] = [];

    // Trace metadata
    private TIME_RANGE: number = BASE_TIME_RANGE;
    private kernelName: string = '';
    private gridDimX: number = 0;
    private gridDimY: number = 0;
    private gridDimZ: number = 0;
    private clusterDimX: number = 0;
    private clusterDimY: number = 0;
    private clusterDimZ: number = 0;
    private worldHeight: number = 0;

    // Rendering subsystems (null until initVisualization)
    private camera: Camera | null = null;
    private labelRenderer: LabelRenderer | null = null;
    private timelineRenderer: TimelineRenderer | null = null;
    private interactionManager: InteractionManager | null = null;

    // Render loop state
    private animationFrameId: number | null = null;
    private eventListenersSetup: boolean = false;
    private isRendering: boolean = false;
    private lastTime: number = 0;
    private numZones: number = 0;

    // Preallocated buffers to avoid per-frame allocations (GC pressure reduction)
    private uniformData = new ArrayBuffer(112);
    private uniformFloatView = new Float32Array(this.uniformData);
    private uniformIntView = new Int32Array(this.uniformData);
    private backgroundUniformData = new ArrayBuffer(80);
    private backgroundFloatView = new Float32Array(this.backgroundUniformData);

    /**
     * Initializes the visualizer by acquiring DOM element references and setting up
     * file loading UI event handlers. WebGPU and visualization setup happen later
     * in initWebGPU() and initVisualization() respectively.
     */
    constructor() {
        this.canvas = document.getElementById('canvas') as HTMLCanvasElement;
        this.labelCanvas = document.getElementById('labelCanvas') as HTMLCanvasElement;
        const ctx = this.labelCanvas.getContext('2d');
        if (!ctx) {
            throw new Error('Failed to get 2D context for label canvas');
        }
        this.labelCtx = ctx;

        this.tooltip = this.getElement('tooltip');
        this.loading = this.getElement('loading');
        this.stats = this.getElement('stats');
        this.laneLabelsContainer = this.getElement('lane-labels');
        this.timelineContainer = this.getElement('timeline');
        this.cursorLine = this.getElement('cursor-line');
        this.cursorTimestamp = this.getElement('cursor-timestamp');
        this.selectionRegion = this.getElement('selection-region');
        this.selectionLineStart = this.getElement('selection-line-start');
        this.selectionLineEnd = this.getElement('selection-line-end');
        this.selectionLabel = this.getElement('selection-label');
        this.fileSelector = this.getElement('file-selector');
        this.loadingOverlay = this.getElement('loading-overlay');
        this.loadingText = this.loadingOverlay.querySelector('.loading-text') as HTMLElement;
        this.fileInput = document.getElementById('file-input') as HTMLInputElement;

        // Wire up file input handler for local .nanotrace files
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Setup sample trace loading menu with cancel button
        const loadSampleBtn = document.getElementById('load-sample-btn');
        const sampleMenu = document.getElementById('sample-menu');
        if (loadSampleBtn && sampleMenu) {
            loadSampleBtn.addEventListener('click', () => {
                this.fileSelector.classList.add('hidden');
                sampleMenu.classList.remove('hidden');
            });

            const sampleOptions = sampleMenu.querySelectorAll('.sample-option');
            sampleOptions.forEach(option => {
                option.addEventListener('click', (e) => {
                    const button = (e.currentTarget as HTMLElement);
                    const sampleName = button.getAttribute('data-sample');
                    if (sampleName) {
                        sampleMenu.classList.add('hidden');
                        this.loadSampleFile(sampleName);
                    }
                });
            });

            const cancelBtn = sampleMenu.querySelector('.sample-menu-cancel');
            if (cancelBtn) {
                cancelBtn.addEventListener('click', () => {
                    sampleMenu.classList.add('hidden');
                    this.fileSelector.classList.remove('hidden');
                });
            }
        }

        document.body.addEventListener('dragenter', (e) => {
            e.preventDefault();
            e.stopPropagation();
        });
        document.body.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
        });
        // Enable drag-and-drop for .nanotrace files anywhere on the page
        document.body.addEventListener('drop', (e) => this.handleFileDrop(e));
    }

    /**
     * Helper to get DOM element by ID with type safety and error handling.
     * Throws if element doesn't exist to catch configuration errors early.
     */
    private getElement(id: string): HTMLElement {
        const element = document.getElementById(id);
        if (!element) {
            throw new Error(`Element with id '${id}' not found`);
        }
        return element;
    }

    /**
     * Phase 1: Initialize WebGPU adapter, device, and canvas context.
     *
     * This must be called before any trace data is loaded. Requests maximum
     * buffer sizes to handle large traces (potentially millions of zones).
     * The preferred canvas format is auto-detected for optimal performance.
     */
    async initWebGPU(): Promise<void> {
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported');
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error('Failed to get GPU adapter');
        }
        this.adapter = adapter;

        this.device = await this.adapter.requestDevice({
            requiredLimits: {
                maxBufferSize: this.adapter.limits.maxBufferSize,
                maxStorageBufferBindingSize: this.adapter.limits.maxStorageBufferBindingSize,
            }
        });

        const context = this.canvas.getContext('webgpu');
        if (!context) {
            throw new Error('Failed to get WebGPU context');
        }
        this.context = context;

        this.format = navigator.gpu.getPreferredCanvasFormat();
        context.configure({
            device: this.device,
            format: this.format,
            alphaMode: 'opaque',
        });

        this.resizeCanvas();

        this.loading.textContent = 'Select a trace file to begin';
    }

    /**
     * Phase 2: Initialize visualization after trace data has been loaded.
     *
     * Creates all rendering subsystems (camera, renderers, interaction manager),
     * uploads data to GPU, sets up event listeners, and starts the render loop.
     * Calculates initial auto-zoom to fit the entire trace with padding.
     */
    async initVisualization(): Promise<void> {
        // Stop any existing render loop before reinitializing
        this.isRendering = false;

        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }

        this.loadingText.textContent = 'Initializing camera and renderers...';
        await new Promise(resolve => setTimeout(resolve, 0));

        this.resizeCanvas();

        // Create rendering subsystems with loaded trace data
        this.camera = new Camera(this.worldHeight, this.TIME_RANGE);
        this.labelRenderer = new LabelRenderer(this.labelCtx, this.canvas, this.camera);
        this.labelRenderer.updateBlocksCache(this.lanes);  // Cache flattened blocks to avoid per-frame allocations
        this.timelineRenderer = new TimelineRenderer(this.timelineContainer, this.canvas, this.camera);
        this.interactionManager = new InteractionManager(
            this.camera,
            this.canvas,
            this.tooltip,
            this.selectionRegion,
            this.selectionLineStart,
            this.selectionLineEnd,
            this.selectionLabel
        );

        // Calculate initial X-axis zoom to fit entire trace with 100px padding
        const rect = this.canvas.getBoundingClientRect();
        const aspect = rect.width / rect.height;
        const desiredZoomX = 2 * aspect * (rect.width - 100) / (this.TIME_RANGE * rect.width);
        this.camera.xZoomMultiplier = desiredZoomX / this.camera.zoom;

        // Upload trace data to GPU storage buffers and create render pipelines
        this.loadingText.textContent = 'Uploading to GPU...';
        await new Promise(resolve => setTimeout(resolve, 0));
        const buffers = createGPUBuffers(this.device, this.zones, this.blocks, this.blockLanes, this.lanes);
        this.gpuResources = createPipelines(
            this.device,
            this.format,
            buffers.positionBuffer,
            buffers.laneBuffer,
            buffers.blockLaneBuffer,
            buffers.blockBuffer
        );

        // Setup event listeners only once (they persist across reloads)
        this.setupEventListeners();

        this.loading.style.display = 'none';
        this.lastTime = performance.now();

        // Start the render loop
        this.isRendering = true;
        this.render();
    }

    /**
     * Handles file selection from <input type="file"> element.
     * Shows loading overlay, parses trace, and initializes visualization.
     * Clears the input value afterward to allow reloading the same file.
     */
    async handleFileSelect(event: Event): Promise<void> {
        const input = event.target as HTMLInputElement;
        const file = input.files?.[0];
        if (!file) return;

        this.fileSelector.classList.add('hidden');
        this.loadingOverlay.classList.remove('hidden');

        // Yield to browser to show loading overlay before blocking on file parse
        await new Promise(resolve => setTimeout(resolve, LOADING_OVERLAY_DELAY));

        try {
            await this.loadTraceFile(file);
            await this.initVisualization();
            this.loadingOverlay.classList.add('hidden');
        } catch (err) {
            console.error(err);
            alert(`Error loading trace: ${(err as Error).message}`);
            this.loadingOverlay.classList.add('hidden');
            this.fileSelector.classList.remove('hidden');
        } finally {
            // Clear input to allow reloading the same file
            this.fileInput.value = '';
        }
    }

    /**
     * Handles drag-and-drop file loading.
     * Validates .nanotrace extension before processing.
     */
    async handleFileDrop(event: DragEvent): Promise<void> {
        event.preventDefault();
        event.stopPropagation();

        const files = event.dataTransfer?.files;
        if (!files || files.length === 0) return;

        const file = files[0];
        if (!file.name.endsWith('.nanotrace')) {
            alert('Please drop a .nanotrace file');
            return;
        }

        this.fileSelector.classList.add('hidden');
        this.loadingOverlay.classList.remove('hidden');

        await new Promise(resolve => setTimeout(resolve, LOADING_OVERLAY_DELAY));

        try {
            await this.loadTraceFile(file);
            await this.initVisualization();
            this.loadingOverlay.classList.add('hidden');
        } catch (err) {
            console.error(err);
            alert(`Error loading trace: ${(err as Error).message}`);
            this.loadingOverlay.classList.add('hidden');
            this.fileSelector.classList.remove('hidden');
        }
    }

    /**
     * Loads bundled sample trace file from assets.
     * Sample 1: Minimal trace (1 block, 2 events)
     * Sample 2: Small random trace (~48K events, 16 SMs)
     * Sample 3: Large random trace (~10M events, 144 SMs)
     * Uses Vite's import.meta.url for proper bundled path resolution.
     */
    async loadSampleFile(sampleName: string): Promise<void> {
        // Map sample names to filenames (served from public directory)
        const sampleFiles: { [key: string]: string } = {
            'sample1': 'minimal.nanotrace',
            'sample2': 'random_small.nanotrace',
            'sample3': 'random.nanotrace',
        };

        const fileName = sampleFiles[sampleName];
        if (!fileName) {
            alert('This sample is not yet available. Only Samples 1-3 are currently included.');
            this.fileSelector.classList.remove('hidden');
            return;
        }

        this.loadingOverlay.classList.remove('hidden');
        this.loadingText.textContent = 'Downloading sample file...';

        await new Promise(resolve => setTimeout(resolve, 0));

        try {
            // Fetch sample file from public directory (with base path)
            const basePath = import.meta.env.BASE_URL || '/';
            const response = await fetch(`${basePath}${fileName}`);
            if (!response.ok) {
                throw new Error(`Failed to fetch ${fileName}: ${response.status} ${response.statusText}`);
            }
            const blob = await response.blob();
            const file = new File([blob], fileName, { type: 'application/octet-stream' });
            await this.loadTraceFile(file);
            await this.initVisualization();
            this.loadingOverlay.classList.add('hidden');
        } catch (err) {
            console.error(err);
            alert(`Error loading sample: ${(err as Error).message}`);
            this.loadingOverlay.classList.add('hidden');
            this.fileSelector.classList.remove('hidden');
        }
    }

    /**
     * Parses binary trace file and builds visualization hierarchy.
     *
     * This method:
     * 1. Parses the binary .nanotrace format
     * 2. Creates SM lane labels (one per streaming multiprocessor)
     * 3. Builds the hierarchical data structure (lanes → block lanes → blocks → zones)
     * 4. Stores data in instance variables for GPU upload and rendering
     */
    async loadTraceFile(file: File): Promise<void> {
        performance.mark('loadTraceFile:start');
        this.loadingText.textContent = 'Parsing trace file...';
        await new Promise(resolve => setTimeout(resolve, 0));
        const parsedData = await parseTraceFile(file, (message: string) => {
            this.loadingText.textContent = message;
        });
        // Truncate long kernel names for stats display
        this.kernelName = parsedData.kernelName.length > MAX_KERNEL_NAME_LENGTH
            ? parsedData.kernelName.substring(0, MAX_KERNEL_NAME_LENGTH) + '...'
            : parsedData.kernelName;
        this.gridDimX = parsedData.gridDimX;
        this.gridDimY = parsedData.gridDimY;
        this.gridDimZ = parsedData.gridDimZ;
        this.clusterDimX = parsedData.clusterDimX;
        this.clusterDimY = parsedData.clusterDimY;
        this.clusterDimZ = parsedData.clusterDimZ;
        this.formatDescriptors = parsedData.formatDescriptors;

        // Create SM lane labels dynamically based on trace data
        this.laneLabelsContainer.innerHTML = '';
        this.laneLabels = [];
        const maxSmId = parsedData.blockDescriptors.reduce((max, desc) => Math.max(max, desc.smId), -1);
        const numLanes = maxSmId + 1;
        for (let i = 0; i < numLanes; i++) {
            const label = document.createElement('div');
            label.className = 'lane-label glass mono';
            label.textContent = `SM ${i}`;
            this.laneLabelsContainer.appendChild(label);
            this.laneLabels.push(label);
        }

        this.loadingText.textContent = 'Building hierarchy...';
        await new Promise(resolve => setTimeout(resolve, 0));
        const hierarchy = buildHierarchy(
            parsedData.formatDescriptors,
            parsedData.blockDescriptors,
            parsedData.tracks
        );

        // Store hierarchy data for GPU upload and rendering
        this.zones = hierarchy.zones;
        this.blocks = hierarchy.blocks;
        this.blockLanes = hierarchy.blockLanes;
        this.lanes = hierarchy.lanes;
        this.worldHeight = hierarchy.worldHeight;
        this.TIME_RANGE = hierarchy.timeRange;
        this.numZones = this.zones.length;

        performance.mark('loadTraceFile:end');
        performance.measure('Load Trace File (Total)', 'loadTraceFile:start', 'loadTraceFile:end');
    }

    /**
     * Wrapper around formatStringHelper to use instance's format descriptors.
     * Replaces placeholders like {0}, {1} with parameter values.
     */
    formatString(formatDescId: number, params: number[]): string {
        return formatStringHelper(this.formatDescriptors, formatDescId, params);
    }

    /**
     * Resizes both canvases to match window size with device pixel ratio.
     * Called on window resize and during initialization.
     */
    resizeCanvas(): void {
        this.canvas.width = window.innerWidth * devicePixelRatio;
        this.canvas.height = window.innerHeight * devicePixelRatio;
        this.labelCanvas.width = window.innerWidth * devicePixelRatio;
        this.labelCanvas.height = window.innerHeight * devicePixelRatio;
    }

    /**
     * Sets up all mouse and keyboard event listeners.
     *
     * Only runs once (guarded by eventListenersSetup flag) since listeners
     * persist across file reloads. Handles:
     * - Pan (right-click drag)
     * - Zoom (scroll, shift+scroll, ctrl+scroll)
     * - Selection (left-click drag, double-click snap)
     * - Reset view (R key)
     * - Hover tooltips
     */
    setupEventListeners(): void {
        if (this.eventListenersSetup) return;
        this.eventListenersSetup = true;

        // Prevent right-click context menu on canvas
        this.canvas.addEventListener('contextmenu', (e) => {
            e.preventDefault();
        });

        window.addEventListener('resize', () => this.resizeCanvas());

        window.addEventListener('keydown', (e) => {
            if (e.key === 'r' || e.key === 'R') {
                if (!this.camera) return;
                this.camera.x = -this.camera.timeRange / 2;
                this.camera.y = -this.worldHeight + 0.5;
                this.camera.zoom = 2.0;

                const rect = this.canvas.getBoundingClientRect();
                const aspect = rect.width / rect.height;
                const desiredZoomX = 2 * aspect * (rect.width - INITIAL_ZOOM_PADDING) / (this.TIME_RANGE * rect.width);
                this.camera.xZoomMultiplier = desiredZoomX / this.camera.zoom;
            }
        });

        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            if (!this.camera) return;

            const rect = this.canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            const ndcX = (mouseX / rect.width) * 2 - 1;
            const ndcY = -((mouseY / rect.height) * 2 - 1);
            const aspect = rect.width / rect.height;

            if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
                const panSpeed = PAN_SPEED;
                this.camera.x -= e.deltaX * panSpeed / this.camera.zoomX * aspect;
                return;
            }

            const zoomFactor = ZOOM_FACTOR;
            const isCtrlPressed = e.ctrlKey || e.metaKey;
            const isShiftPressed = e.shiftKey;

            if (isCtrlPressed) {
                const oldZoomX = this.camera.zoomX;
                const oldZoomY = this.camera.zoomY;

                if (e.deltaY < 0) {
                    this.camera.zoom *= zoomFactor;
                } else {
                    this.camera.zoom /= zoomFactor;
                }
                this.camera.zoom = Math.max(MIN_ZOOM_Y, Math.min(MAX_ZOOM_Y, this.camera.zoom));

                const newZoomX = this.camera.zoomX;
                const newZoomY = this.camera.zoomY;
                this.camera.x += ndcX * aspect * (1/newZoomX - 1/oldZoomX);
                this.camera.y += ndcY * (1/newZoomY - 1/oldZoomY);
            } else if (isShiftPressed) {
                const oldZoomY = this.camera.zoomY;
                const oldZoom = this.camera.zoom;

                if (e.deltaY < 0) {
                    this.camera.zoom *= zoomFactor;
                } else {
                    this.camera.zoom /= zoomFactor;
                }
                this.camera.zoom = Math.max(MIN_ZOOM_Y, Math.min(MAX_ZOOM_Y, this.camera.zoom));

                this.camera.xZoomMultiplier *= (oldZoom / this.camera.zoom);

                const newZoomY = this.camera.zoomY;
                this.camera.y += ndcY * (1/newZoomY - 1/oldZoomY);
            } else {
                const oldZoomX = this.camera.zoomX;

                if (e.deltaY < 0) {
                    this.camera.xZoomMultiplier *= zoomFactor;
                } else {
                    this.camera.xZoomMultiplier /= zoomFactor;
                }
                this.camera.xZoomMultiplier = Math.max(MIN_ZOOM_X, Math.min(MAX_ZOOM_X, this.camera.xZoomMultiplier));

                const newZoomX = this.camera.zoomX;
                this.camera.x += ndcX * aspect * (1/newZoomX - 1/oldZoomX);
            }
        }, { passive: false });

        this.canvas.addEventListener('dblclick', (e) => {
            if (e.button === 0 && this.interactionManager) {
                const result = this.interactionManager.findZoneAtPosition(e.clientX, e.clientY, this.lanes, this.blocks);

                if (result.zone) {
                    const epsilon = SELECTION_EPSILON;
                    this.interactionManager.startSelection(Math.max(0, result.zone.startX - epsilon));
                    this.interactionManager.updateSelectionEnd(Math.min(this.TIME_RANGE, result.zone.endX + epsilon));
                    this.interactionManager.endSelection();
                    this.interactionManager.updateSelection();
                } else if (result.block) {
                    const epsilon = SELECTION_EPSILON;
                    this.interactionManager.startSelection(Math.max(0, result.block.startX - epsilon));
                    this.interactionManager.updateSelectionEnd(Math.min(this.TIME_RANGE, result.block.endX + epsilon));
                    this.interactionManager.endSelection();
                    this.interactionManager.updateSelection();
                }
            }
        });

        this.canvas.addEventListener('mousedown', (e) => {
            if (!this.camera || !this.interactionManager) return;

            if (e.button === 2) {
                this.camera.isDragging = true;
                this.camera.lastX = e.clientX;
                this.camera.lastY = e.clientY;
            } else if (e.button === 0) {
                const worldPos = this.camera.screenToWorld(e.clientX, e.clientY, this.canvas);
                const clampedX = Math.max(0, Math.min(this.TIME_RANGE, worldPos.x));
                this.interactionManager.startSelection(clampedX);
                this.interactionManager.hideSelectionUI();
            }
        });

        window.addEventListener('mouseup', (_e) => {
            if (!this.camera || !this.interactionManager) return;
            this.camera.isDragging = false;

            if (this.interactionManager.isCurrentlySelecting()) {
                const bounds = this.interactionManager.getSelectionBounds();
                const worldDistance = Math.abs(bounds.end - bounds.start);
                const rect = this.canvas.getBoundingClientRect();
                const aspect = rect.width / rect.height;
                const minWorldDistance = (MIN_SELECTION_DISTANCE / rect.width) * 2 / this.camera.zoomX * aspect;

                if (worldDistance > minWorldDistance) {
                    this.interactionManager.endSelection();
                } else {
                    this.interactionManager.clearSelection();
                }
            }
        });

        window.addEventListener('mousemove', (e) => {
            if (!this.camera || !this.interactionManager) return;

            if (this.camera.isDragging) {
                const dx = e.clientX - this.camera.lastX;
                const dy = e.clientY - this.camera.lastY;
                const rect = this.canvas.getBoundingClientRect();
                const aspect = rect.width / rect.height;
                this.camera.x += (dx / rect.width) * 2 / this.camera.zoomX * aspect;
                this.camera.y -= (dy / rect.height) * 2 / this.camera.zoomY;
                this.camera.lastX = e.clientX;
                this.camera.lastY = e.clientY;
            }

            if (this.interactionManager.isCurrentlySelecting()) {
                const worldPos = this.camera.screenToWorld(e.clientX, e.clientY, this.canvas);
                this.interactionManager.updateSelectionEnd(Math.max(0, Math.min(this.TIME_RANGE, worldPos.x)));
                this.interactionManager.updateSelection();
            }

            if (e.clientX >= 0 && e.clientX <= window.innerWidth &&
                e.clientY >= 0 && e.clientY <= window.innerHeight) {
                this.cursorLine.style.left = `${e.clientX}px`;
                this.cursorLine.style.display = 'block';

                const worldPos = this.camera.screenToWorld(e.clientX, e.clientY, this.canvas);
                const time = worldPos.x;

                const timeInNs = time * MS_TO_NS;
                const formattedTime = Math.round(timeInNs).toLocaleString() + ' ns';

                this.cursorTimestamp.textContent = formattedTime;
                this.cursorTimestamp.style.left = `${e.clientX + 4}px`;
                this.cursorTimestamp.style.display = 'block';
            } else {
                this.cursorLine.style.display = 'none';
                this.cursorTimestamp.style.display = 'none';
            }

            this.interactionManager.updateHover(e.clientX, e.clientY, this.lanes, this.blocks, this.formatDescriptors, this.formatString.bind(this));
        });

        document.body.addEventListener('mouseleave', () => {
            this.cursorLine.style.display = 'none';
            this.cursorTimestamp.style.display = 'none';
        });
    }

    /**
     * Delegates zone label rendering to LabelRenderer.
     * Convenience wrapper that passes instance data to the renderer.
     */
    renderZoneLabels(): void {
        if (!this.camera || !this.labelRenderer) return;
        this.labelRenderer.renderZoneLabels(this.lanes, this.blockLanes, this.formatDescriptors, this.formatString.bind(this));
    }

    /**
     * Updates SM lane label positions based on camera viewport.
     *
     * Positions labels on the left edge of the viewport (50px wide) and vertically
     * aligns them with their corresponding lanes. Labels are clamped to stay below
     * the timeline bar (30px) and are hidden when lanes scroll off-screen.
     */
    updateLaneLabels(): void {
        if (!this.camera) return;

        const rect = this.canvas.getBoundingClientRect();
        const aspect = rect.width / rect.height;

        const worldX = 0;
        const ndcX = (worldX + this.camera.x) * this.camera.zoomX / aspect;
        const laneStartScreenX = (ndcX + 1) * rect.width / 2;

        const labelWidth = SM_LABEL_WIDTH;
        const labelX = Math.max(0, laneStartScreenX - labelWidth);

        const timelineHeight = TIMELINE_HEIGHT;

        for (let i = 0; i < this.lanes.length; i++) {
            const lane = this.lanes[i];

            if (lane.blockLanes.length === 0) {
                this.laneLabels[i].style.display = 'none';
                continue;
            }

            const laneTopY = lane.y + lane.height;
            const laneBottomY = lane.y;

            const worldTopY = laneTopY + this.camera.y;
            const worldBottomY = laneBottomY + this.camera.y;
            const ndcTopY = worldTopY * this.camera.zoomY;
            const ndcBottomY = worldBottomY * this.camera.zoomY;
            const screenTopY = rect.height / 2 - (ndcTopY * rect.height / 2);
            const screenBottomY = rect.height / 2 - (ndcBottomY * rect.height / 2);

            const laneVisible = screenBottomY >= -20 && screenTopY <= rect.height + 20;

            if (laneVisible) {
                const clampedTopY = Math.max(timelineHeight, screenTopY);
                const clampedHeight = Math.max(0, screenBottomY - clampedTopY);

                this.laneLabels[i].style.display = 'flex';
                this.laneLabels[i].style.top = `${clampedTopY}px`;
                this.laneLabels[i].style.left = `${labelX}px`;
                this.laneLabels[i].style.height = `${clampedHeight}px`;
            } else {
                this.laneLabels[i].style.display = 'none';
            }
        }
    }


    /**
     * Main render loop (runs at ~60 FPS via requestAnimationFrame).
     *
     * Rendering pipeline:
     * 1. Update stats display (FPS, trace info, memory usage)
     * 2. Update UI overlays (lane labels, timeline, zone labels, selection)
     * 3. Build uniform data (view-projection matrix, hover/selection state)
     * 4. Execute WebGPU render passes (background → lanes → block lanes → blocks → zones)
     * 5. Submit command buffer to GPU
     * 6. Schedule next frame
     */
    render(): void {
        if (!this.isRendering) {
            return;
        }

        if (!this.camera || !this.gpuResources) return;

        // Calculate FPS for stats display
        const now = performance.now();
        const deltaTime = now - this.lastTime;
        const fps = deltaTime > 0 ? Math.round(1000 / deltaTime) : 60;

        // Update stats overlay with trace info and performance metrics
        const durationNs = Math.round(this.TIME_RANGE * MS_TO_NS);
        const formattedDuration = durationNs.toLocaleString();
        const fpsStr = String(fps).padStart(FPS_PADDING_WIDTH, ' ');

        // Only update stats if they don't exist yet (avoid recreating links every frame)
        if (!this.stats.querySelector('.stats-links')) {
            this.stats.innerHTML = `<span class="stats-dynamic"></span><br><span class="stats-links"><a href="https://github.com/aikitoria/nanotrace" target="_blank">nanotrace</a> ${VERSION} by <a href="https://github.com/aikitoria" target="_blank">aikitoria</a></span>`;
        }

        // Update only the dynamic content
        const dynamicStats = this.stats.querySelector('.stats-dynamic');
        if (dynamicStats) {
            dynamicStats.innerHTML = `${this.kernelName}<br>Duration: ${formattedDuration} ns<br>Grid: (${this.gridDimX}, ${this.gridDimY}, ${this.gridDimZ}) | Cluster: (${this.clusterDimX}, ${this.clusterDimY}, ${this.clusterDimZ})<br>SMs: ${this.lanes.length.toLocaleString()} | Blocks: ${this.blocks.length.toLocaleString()} | Zones: ${this.numZones.toLocaleString()}<br>Zoom: ${this.camera.zoomX.toFixed(2)} × ${this.camera.zoomY.toFixed(2)} | FPS: ${fpsStr}`;
        }
        this.lastTime = now;

        // Update 2D overlays (labels, timeline, selection UI)
        this.updateLaneLabels();
        this.timelineRenderer!.updateTimeline(this.TIME_RANGE);
        this.renderZoneLabels();

        if (this.interactionManager && (this.interactionManager.hasActiveSelection() || this.interactionManager.isCurrentlySelecting())) {
            this.interactionManager.updateSelection();
        }

        // Prepare uniform data for shaders (112 bytes total)
        const aspect = this.canvas.width / this.canvas.height;
        const viewProjMatrix = this.camera.getViewProjectionMatrix(aspect);

        // Reuse preallocated uniform buffers to avoid GC pressure
        const floatView = this.uniformFloatView;
        const intView = this.uniformIntView;

        // Uniform layout: mat4x4 viewProj, int hoveredId, float zoomX, float zoomY,
        // float selectionStart, float selectionEnd, int hasSelection, int hoveredBlockId,
        // float camera_x_high, float camera_x_low, float camera_y, float scale_x, float scale_y
        floatView.set(viewProjMatrix, 0);  // Offset 0-15: 4x4 matrix
        intView[16] = this.interactionManager!.getHoveredZoneId();
        floatView[17] = this.camera.zoomX;
        floatView[18] = this.camera.zoomY;

        const selectionActive = this.interactionManager!.hasActiveSelection() || this.interactionManager!.isCurrentlySelecting();
        const selectionBounds = this.interactionManager!.getSelectionBounds();
        floatView[19] = selectionActive ? selectionBounds.start : 0.0;
        floatView[20] = selectionActive ? selectionBounds.end : 0.0;
        intView[21] = selectionActive ? 1 : 0;
        intView[22] = this.interactionManager!.getHoveredBlockId();

        // Double-single camera position for high-precision zone rendering
        const [camera_x_high, camera_x_low] = this.camera.getCameraXDoubleSingle();
        floatView[23] = camera_x_high;
        floatView[24] = camera_x_low;
        floatView[25] = this.camera.y;

        // Scale factors for manual transformation in shader
        const scale_x = this.camera.zoomX / aspect;
        const scale_y = this.camera.zoomY;
        floatView[26] = scale_x;
        floatView[27] = scale_y;

        this.device.queue.writeBuffer(this.gpuResources.uniformBuffer, 0, this.uniformData);

        const topLane = this.lanes[0];
        const backgroundHeight = topLane.y + topLane.height;

        // Reuse preallocated background uniform buffer to avoid GC pressure
        this.backgroundFloatView.set(viewProjMatrix, 0);
        this.backgroundFloatView[16] = this.TIME_RANGE;
        this.backgroundFloatView[17] = backgroundHeight;
        this.device.queue.writeBuffer(this.gpuResources.backgroundUniformBuffer, 0, this.backgroundUniformData);

        const commandEncoder = this.device.createCommandEncoder();
        const textureView = this.context.getCurrentTexture().createView();

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: { r: CLEAR_COLOR_R, g: CLEAR_COLOR_G, b: CLEAR_COLOR_B, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }]
        });

        renderPass.setPipeline(this.gpuResources.passes.background.pipeline);
        renderPass.setBindGroup(0, this.gpuResources.passes.background.bindGroup);
        renderPass.draw(6, 1, 0, 0);

        renderPass.setPipeline(this.gpuResources.passes.lane.pipeline);
        renderPass.setBindGroup(0, this.gpuResources.passes.lane.bindGroup);
        renderPass.draw(6, this.lanes.length, 0, 0);

        renderPass.setPipeline(this.gpuResources.passes.blockLane.pipeline);
        renderPass.setBindGroup(0, this.gpuResources.passes.blockLane.bindGroup);
        renderPass.draw(6, this.blockLanes.length, 0, 0);

        renderPass.setPipeline(this.gpuResources.passes.blockBg.pipeline);
        renderPass.setBindGroup(0, this.gpuResources.passes.blockBg.bindGroup);
        renderPass.draw(6, this.blocks.length, 0, 0);

        renderPass.setPipeline(this.gpuResources.passes.block.pipeline);
        renderPass.setBindGroup(0, this.gpuResources.passes.block.bindGroup);
        renderPass.draw(6, this.blocks.length, 0, 0);

        renderPass.setPipeline(this.gpuResources.passes.zone.pipeline);
        renderPass.setBindGroup(0, this.gpuResources.passes.zone.bindGroup);
        renderPass.draw(6, this.numZones, 0, 0);
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);

        this.animationFrameId = requestAnimationFrame(() => this.render());
    }
}

/**
 * Application entry point called from main.ts.
 * Creates the visualizer and performs initial WebGPU setup.
 * File loading happens later via UI interaction.
 */
export async function initApp(): Promise<void> {
    const visualizer = new ZoneVisualizer();
    await visualizer.initWebGPU();
}
