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
import { parseTraceFile, projectTraceData, buildHierarchy, formatString as formatStringHelper, formatTrackString as formatTrackStringHelper, formatBlockString as formatBlockStringHelper, formatTooltipString as formatTooltipStringHelper, formatTrackTooltipString as formatTrackTooltipStringHelper, formatBlockTooltipString as formatBlockTooltipStringHelper, TrackExpansionMode, type ParsedTraceData } from './utils/file-loader.js';
import { createGPUBuffers, createPipelines, GPUResources } from './renderers/gpu-renderer.js';
import { LabelRenderer } from './renderers/label-renderer.js';
import { TimelineRenderer } from './renderers/timeline-renderer.js';
import { InteractionManager } from './interaction-manager.js';
import {
    HierarchyData
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
    TRACK_LABEL_WIDTH,
    TIMELINE_HEIGHT,
    INITIAL_ZOOM_PADDING,
    INITIAL_BASE_ZOOM,
    MIN_SELECTION_DISTANCE,
    MIN_ZONE_LABEL_HEIGHT,
    MIN_ZONE_LABEL_WIDTH,
    LABEL_FONT_SIZE,
    MIN_LABEL_FONT_SIZE,
    MIN_LABEL_ZOOM_Y,
    SUBLANE_HEIGHT,
    LOADING_OVERLAY_DELAY,
    MAX_KERNEL_NAME_LENGTH,
    FPS_PADDING_WIDTH,
    MS_TO_NS,
    LANE_PADDING,
    BLOCK_LANE_PADDING
} from './utils/constants.js';

// Git commit hash injected at build time by Vite
declare const __GIT_HASH__: string;

const VERSION = `0.1-${__GIT_HASH__}`;

interface ExpandedKernelGroup {
    startNs: number;
    endNs: number;
    topTrackIndex: number;
    bottomTrackIndex: number;
}

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
    private trackFrame: HTMLElement;
    private kernelFramesContainer: HTMLElement;
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
    private closeBtn: HTMLElement;
    private helpBtn: HTMLElement;
    private expandAllBtn: HTMLButtonElement;

    // WebGPU resources (initialized in initWebGPU)
    private adapter: GPUAdapter | null = null;
    private device!: GPUDevice;
    private context!: GPUCanvasContext;
    private format!: GPUTextureFormat;
    private gpuResources?: GPUResources;
    private cachedPasses?: GPUResources['passes'];

    // Trace data hierarchy (populated from file, SoA structures)
    private hierarchy: HierarchyData | null = null;
    private fullTraceData: ParsedTraceData | null = null;
    private expandedEventIds = new Set<bigint>();
    private expandedTrackIds = new Set<bigint>();
    private materializedEventIds = new Set<bigint>();
    private eventZoneIndices = new Map<bigint, number[]>();
    private childZoneIndices = new Map<bigint, number[]>();
    private rowBaseYs = new Float32Array();
    private rowOffsets = new Float32Array();
    private rowVisible = new Uint8Array();
    private rowVisibleZoneCounts = new Uint32Array();
    private rowLayout = new Float32Array();
    private zoneVisibility = new Uint32Array();
    private treeParentRows = new Int32Array();
    private treeTrackIds: bigint[] = [];
    private rowTimeBounded = new Float32Array();
    private rowExpansionGroupIds: bigint[] = [];
    private rowExpansionModes: TrackExpansionMode[] = [];
    private expansionLabelRows = new Int32Array();
    private suppressedExpansionLabels = new Uint8Array();
    private expandedTreeTrackIds = new Set<bigint>();
    private treeDisclosures = new Map<bigint, HTMLElement>();
    private expandedKernelGroups: ExpandedKernelGroup[] = [];
    private kernelFrames: HTMLElement[] = [];
    private laneLabels: HTMLElement[] = [];

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
    private uniformData = new ArrayBuffer(128);
    private uniformFloatView = new Float32Array(this.uniformData);
    private uniformIntView = new Int32Array(this.uniformData);
    private backgroundUniformData = new ArrayBuffer(32);
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
        this.trackFrame = this.getElement('track-frame');
        this.kernelFramesContainer = this.getElement('kernel-frames');
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
        this.closeBtn = this.getElement('close-btn');
        this.helpBtn = this.getElement('help-btn');
        this.expandAllBtn = this.getElement(
            'expand-all-btn') as HTMLButtonElement;
        this.expandAllBtn.addEventListener('click', () => {
            void this.toggleAllDetails();
        });

        // Wire up file input handler for local .nanotrace files
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Wire up close button to return to file selector
        this.closeBtn.addEventListener('click', () => this.closeVisualization());

        // Wire up help button to show controls overlay
        const helpOverlay = this.getElement('help-overlay');
        this.helpBtn.addEventListener('click', () => {
            helpOverlay.classList.remove('hidden');
            // Hide cursor line, timestamp, and tooltip when help is open
            this.cursorLine.style.display = 'none';
            this.cursorTimestamp.style.display = 'none';
            this.tooltip.classList.remove('visible');
        });

        const helpCloseBtn = helpOverlay.querySelector('.help-close');
        if (helpCloseBtn) {
            helpCloseBtn.addEventListener('click', () => {
                helpOverlay.classList.add('hidden');
            });
        }

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
            alphaMode: 'premultiplied',
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

        // Clean up old timeline renderer if it exists
        if (this.timelineRenderer) {
            this.timelineRenderer.destroy();
            this.timelineRenderer = null;
        }

        this.loadingText.textContent = 'Initializing camera and renderers...';
        await new Promise(resolve => setTimeout(resolve, 0));

        this.resizeCanvas();

        if (!this.hierarchy) {
            throw new Error('Hierarchy data not loaded');
        }

        // Create rendering subsystems with loaded trace data
        this.camera = new Camera(this.worldHeight, this.TIME_RANGE);
        this.labelRenderer = new LabelRenderer(this.labelCtx, this.canvas, this.camera);
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
        this.interactionManager.updateLayout(
            this.rowOffsets, this.rowVisible, this.zoneVisibility);

        this.fitTraceToViewport();

        // Upload trace data to GPU storage buffers and create render pipelines
        this.loadingText.textContent = 'Uploading to GPU...';
        await new Promise(resolve => setTimeout(resolve, 0));
        this.destroyGpuResources();
        const buffers = createGPUBuffers(
            this.device,
            this.hierarchy.zones,
            this.hierarchy.blocks,
            this.hierarchy.blockLanes,
            this.hierarchy.lanes,
            this.rowBaseYs,
            this.rowTimeBounded,
            this.rowLayout,
            this.zoneVisibility
        );
        this.gpuResources = createPipelines(
            this.device,
            this.format,
            buffers.positionBuffer,
            buffers.laneBuffer,
            buffers.blockLaneBuffer,
            buffers.blockBuffer,
            buffers.rowLayoutBuffer,
            buffers.zoneVisibilityBuffer,
            this.cachedPasses
        );
        this.cachedPasses = this.gpuResources.passes;

        // Setup event listeners only once (they persist across reloads)
        this.setupEventListeners();

        this.loading.style.display = 'none';
        this.lastTime = performance.now();

        // Show UI elements
        document.body.classList.add('visualization-active');
        this.stats.style.display = 'block';

        // Show close and help buttons now that visualization is loaded
        this.closeBtn.style.display = 'block';
        this.helpBtn.style.display = 'block';
        this.expandAllBtn.style.display = 'block';

        // Start the render loop
        this.isRendering = true;
        this.render();
    }

    /** Releases buffers owned by the previous visible projection. */
    private destroyGpuResources(): void {
        if (!this.gpuResources) return;

        this.gpuResources.uniformBuffer.destroy();
        this.gpuResources.backgroundUniformBuffer.destroy();
        this.gpuResources.buffers.position.destroy();
        this.gpuResources.buffers.lane.destroy();
        this.gpuResources.buffers.blockLane.destroy();
        this.gpuResources.buffers.block.destroy();
        this.gpuResources.buffers.rowLayout.destroy();
        this.gpuResources.buffers.zoneVisibility.destroy();
        this.gpuResources = undefined;
    }

    /**
     * Closes the current visualization and returns to file selector.
     * Stops the render loop and cleans up resources before showing file selector.
     */
    closeVisualization(): void {
        // Stop render loop
        this.isRendering = false;

        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }

        // Clean up timeline renderer DOM elements
        if (this.timelineRenderer) {
            this.timelineRenderer.destroy();
            this.timelineRenderer = null;
        }

        // Clean up lane labels
        this.laneLabelsContainer.innerHTML = '';
        this.laneLabels = [];

        // Hide UI elements
        this.tooltip.classList.remove('visible');
        this.stats.style.display = 'none';
        this.cursorLine.style.display = 'none';
        this.cursorTimestamp.style.display = 'none';
        this.selectionRegion.style.display = 'none';
        this.selectionLineStart.style.display = 'none';
        this.selectionLineEnd.style.display = 'none';
        this.selectionLabel.style.display = 'none';

        // Clear label canvas
        const rect = this.labelCanvas.getBoundingClientRect();
        this.labelCtx.clearRect(0, 0, rect.width, rect.height);

        // Clear interaction manager
        this.interactionManager = null;

        // Clear other renderers
        this.labelRenderer = null;
        this.camera = null;

        // Hide close and help buttons and show file selector
        document.body.classList.remove('visualization-active');
        this.closeBtn.style.display = 'none';
        this.helpBtn.style.display = 'none';
        this.expandAllBtn.style.display = 'none';
        this.fileSelector.classList.remove('hidden');
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
            this.closeBtn.style.display = 'none';
            this.helpBtn.style.display = 'none';
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
            this.closeBtn.style.display = 'none';
            this.helpBtn.style.display = 'none';
            this.fileSelector.classList.remove('hidden');
        }
    }

    /** Loads a bundled nanotrace v4 sample. */
    async loadSampleFile(sampleName: string): Promise<void> {
        const sampleFiles: { [key: string]: string } = {
            'unified': 'unified_trace.nanotrace',
            'multistream-graph': 'multistream_graph.nanotrace',
            'cpu-hierarchy': 'cpu_hierarchy.nanotrace',
            'tma-static': 'tma_bandwidth_static_sm120a.nanotrace',
            'tma-atomic': 'tma_bandwidth_atomic_sm120a.nanotrace',
        };

        const fileName = sampleFiles[sampleName];
        if (!fileName) {
            alert('This sample is not available.');
            this.closeBtn.style.display = 'none';
            this.helpBtn.style.display = 'none';
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
            this.closeBtn.style.display = 'none';
            this.helpBtn.style.display = 'none';
            this.fileSelector.classList.remove('hidden');
        }
    }

    /**
     * Parses binary trace file and builds visualization hierarchy (SoA version).
     *
     * This method:
     * 1. Parses the binary .nanotrace format directly into SoA structures
     * 2. Creates SM lane labels (one per streaming multiprocessor)
     * 3. Builds the hierarchical data structure (lanes → block lanes → blocks → zones)
     * 4. Stores data in instance variables for GPU upload and rendering
     */
    async loadTraceFile(file: File): Promise<void> {
        performance.mark('loadTraceFile:start');
        this.loadingText.textContent = 'Parsing trace file...';
        await new Promise(resolve => setTimeout(resolve, 0));
        this.fullTraceData = await parseTraceFile(file, (message: string) => {
            this.loadingText.textContent = message;
        });
        this.expandedEventIds.clear();
        this.expandedTrackIds.clear();
        this.expandedTreeTrackIds.clear();
        this.materializedEventIds.clear();
        for (let zoneIndex = 0;
            zoneIndex < this.fullTraceData.zones.count; zoneIndex++) {
            if (this.fullTraceData.zones.hasChildren[zoneIndex] !== 0) {
                this.materializedEventIds.add(
                    this.fullTraceData.zones.eventIds[zoneIndex]);
            }
        }
        const parsedData = projectTraceData(
            this.fullTraceData, this.materializedEventIds,
            this.expandedTrackIds);
        await this.applyTraceProjection(parsedData);

        performance.mark('loadTraceFile:end');
        performance.measure('Load Trace File (Total)', 'loadTraceFile:start', 'loadTraceFile:end');
    }

    /** Rebuilds the visible track projection without reparsing the trace file. */
    private async applyTraceProjection(parsedData: ParsedTraceData): Promise<void> {
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

        // Create labels for generic CPU, GPU stream, and intra-kernel tracks.
        this.laneLabelsContainer.innerHTML = '';
        this.laneLabels = [];
        this.treeDisclosures.clear();
        this.treeParentRows = new Int32Array(parsedData.trackNames.length);
        this.treeParentRows.fill(-1);
        this.rowExpansionGroupIds = parsedData.trackExpansionGroupIds;
        this.rowExpansionModes = parsedData.trackExpansionModes;
        this.expansionLabelRows = new Int32Array(parsedData.trackNames.length);
        this.expansionLabelRows.fill(-1);
        this.suppressedExpansionLabels = new Uint8Array(
            parsedData.trackNames.length);
        this.treeTrackIds = new Array<bigint>(
            parsedData.trackNames.length).fill(0n);
        const rowByTrackId = new Map<bigint, number>();
        for (let i = 0; i < parsedData.trackHierarchies.length; i++) {
            const hierarchy = parsedData.trackHierarchies[i];
            const track = hierarchy[hierarchy.length - 1];
            if (track) {
                this.treeTrackIds[i] = track.id;
                rowByTrackId.set(track.id, i);
            }
        }

        const treeParentRows = new Set<number>();
        for (let i = 0; i < parsedData.trackHierarchies.length; i++) {
            const hierarchy = parsedData.trackHierarchies[i];
            const track = hierarchy[hierarchy.length - 1];
            if (!track) continue;
            const parentRow = rowByTrackId.get(track.parentId);
            if (parentRow === undefined) continue;
            const parentHierarchy = parsedData.trackHierarchies[parentRow];
            const parentTrack = parentHierarchy[parentHierarchy.length - 1];
            const cpuTreeEdge = track.kind === 1 && parentTrack?.kind === 1;
            const gpuStreamEdge = track.kind === 3 && parentTrack?.kind === 2;
            if (!cpuTreeEdge && !gpuStreamEdge) continue;
            this.treeParentRows[i] = parentRow;
            treeParentRows.add(parentRow);
        }
        for (let parentRow = 0;
            parentRow < parsedData.trackNames.length; parentRow++) {
            if (this.rowExpansionModes[parentRow]
                    !== TrackExpansionMode.Collapsed) {
                continue;
            }
            const groupId = this.rowExpansionGroupIds[parentRow];
            for (let childRow = parentRow + 1;
                childRow < parsedData.trackNames.length; childRow++) {
                if (this.rowExpansionGroupIds[childRow] !== groupId) continue;
                if (this.rowExpansionModes[childRow]
                        !== TrackExpansionMode.Expanded) {
                    continue;
                }
                this.expansionLabelRows[parentRow] = childRow;
                this.suppressedExpansionLabels[childRow] = 1;
                break;
            }
        }
        this.rowTimeBounded = new Float32Array(parsedData.trackNames.length);
        for (let i = 0; i < parsedData.trackNames.length; i++) {
            this.rowTimeBounded[i] = parsedData.trackDepths[i] > 0
                && this.treeParentRows[i] < 0 ? 1 : 0;
        }

        for (let i = 0; i < parsedData.trackNames.length; i++) {
            const label = document.createElement('div');
            label.className = 'lane-label';
            label.style.setProperty(
                '--track-depth', parsedData.trackDepths[i].toString());
            label.title = parsedData.trackTooltips[i];

            const physicalSmRow = parsedData.trackNames[i].startsWith('SM ');
            if (parsedData.trackDepths[i] > 0
                || this.treeParentRows[i] >= 0) {
                label.classList.add('lane-label-child');
            }
            if (this.treeParentRows[i] >= 0 || physicalSmRow) {
                label.classList.add('lane-label-tree-child');
            }

            const trackDisclosureKey = parsedData.trackDisclosureKeys[i];
            const treeTrackId = this.treeTrackIds[i];
            if (trackDisclosureKey) {
                label.classList.add('lane-label-expandable');
                label.tabIndex = 0;
                label.setAttribute('role', 'button');
                const disclosure = document.createElement('span');
                disclosure.className = 'lane-disclosure';
                disclosure.textContent = parsedData.trackExpanded[i]
                    ? '\u25be' : '\u25b8';
                label.appendChild(disclosure);
                const trackId = BigInt(
                    trackDisclosureKey.slice('track:'.length));
                this.treeDisclosures.set(trackId, disclosure);
                const toggleTrack = (): void => {
                    void this.toggleDisclosure(trackDisclosureKey);
                };
                label.addEventListener('click', toggleTrack);
                label.addEventListener('keydown', (event) => {
                    if (event.key !== 'Enter' && event.key !== ' ') return;
                    event.preventDefault();
                    toggleTrack();
                });
            } else if (treeParentRows.has(i)) {
                label.classList.add('lane-label-expandable');
                label.tabIndex = 0;
                label.setAttribute('role', 'button');
                const disclosure = document.createElement('span');
                disclosure.className = 'lane-disclosure';
                disclosure.textContent = this.expandedTreeTrackIds.has(
                    treeTrackId) ? '\u25be' : '\u25b8';
                label.appendChild(disclosure);
                this.treeDisclosures.set(treeTrackId, disclosure);
                const toggleTree = (): void => {
                    this.toggleTreeTrack(treeTrackId);
                };
                label.addEventListener('click', toggleTree);
                label.addEventListener('keydown', (event) => {
                    if (event.key !== 'Enter' && event.key !== ' ') return;
                    event.preventDefault();
                    toggleTree();
                });
            }

            const name = document.createElement('span');
            name.className = 'lane-name';
            name.textContent = parsedData.trackNames[i];
            label.appendChild(name);
            this.laneLabelsContainer.appendChild(label);
            this.laneLabels.push(label);
        }

        this.loadingText.textContent = 'Building hierarchy...';
        await new Promise(resolve => setTimeout(resolve, 0));
        this.hierarchy = buildHierarchy(
            parsedData.kernelName,
            [parsedData.gridDimX, parsedData.gridDimY, parsedData.gridDimZ],
            [parsedData.clusterDimX, parsedData.clusterDimY, parsedData.clusterDimZ],
            parsedData.formatDescriptors,
            parsedData.tracks,
            parsedData.zones,
            parsedData.blocks,
            parsedData.trackNames,
            parsedData.trackDepths
        );
        // Store metadata for rendering
        this.worldHeight = this.hierarchy.worldHeight;
        this.TIME_RANGE = this.hierarchy.totalDurationNs * 1e-6;  // Convert ns to ms
        this.numZones = this.hierarchy.zones.count;
        this.initializeEventLayout();
    }

    /** Builds stable event indices and the initial collapsed row layout. */
    private initializeEventLayout(): void {
        if (!this.hierarchy) return;

        const zones = this.hierarchy.zones;
        const lanes = this.hierarchy.lanes;
        this.eventZoneIndices.clear();
        this.childZoneIndices.clear();
        this.rowBaseYs = lanes.ys.slice();
        this.rowOffsets = new Float32Array(lanes.count);
        this.rowVisible = new Uint8Array(lanes.count);
        this.rowVisibleZoneCounts = new Uint32Array(lanes.count);
        this.rowLayout = new Float32Array(lanes.count * 4);
        this.zoneVisibility = new Uint32Array(zones.count);

        for (let zoneIndex = 0; zoneIndex < zones.count; zoneIndex++) {
            const parentEventId = zones.parentEventIds[zoneIndex];
            if (parentEventId === 0n) {
                this.zoneVisibility[zoneIndex] = 1;
                this.rowVisibleZoneCounts[zones.smIndices[zoneIndex]]++;
            } else {
                const children = this.childZoneIndices.get(parentEventId)
                    ?? [];
                children.push(zoneIndex);
                this.childZoneIndices.set(parentEventId, children);
            }
        }

        for (let zoneIndex = 0; zoneIndex < zones.count; zoneIndex++) {
            const eventId = zones.eventIds[zoneIndex];
            if (this.childZoneIndices.has(eventId)) {
                const parentCopies = this.eventZoneIndices.get(eventId) ?? [];
                parentCopies.push(zoneIndex);
                this.eventZoneIndices.set(eventId, parentCopies);
            }
            zones.expanded[zoneIndex] = this.expandedEventIds.has(eventId)
                ? 1 : 0;
        }

        for (let zoneIndex = 0; zoneIndex < zones.count; zoneIndex++) {
            if (zones.parentEventIds[zoneIndex] === 0n
                && this.expandedEventIds.has(zones.eventIds[zoneIndex])) {
                this.setChildVisibility(zones.eventIds[zoneIndex], true);
            }
        }

        this.recalculateRowLayout(false);
        this.rebuildExpandedKernelGroups();
    }

    private visibleEventZoneIndex(eventId: bigint): number | undefined {
        const parentCopies = this.eventZoneIndices.get(eventId);
        if (!parentCopies || parentCopies.length === 0) return undefined;

        for (const zoneIndex of parentCopies) {
            const rowIndex = this.hierarchy?.zones.smIndices[zoneIndex];
            if (rowIndex !== undefined && this.rowVisible[rowIndex] !== 0) {
                return zoneIndex;
            }
        }
        return parentCopies[0];
    }

    /** Updates a small event subtree and returns the changed zone indices. */
    private setChildVisibility(
        parentEventId: bigint, parentVisible: boolean,
        changedZoneIndices?: number[]
    ): void {
        if (!this.hierarchy) return;
        const children = this.childZoneIndices.get(parentEventId);
        if (!children) return;

        const childrenVisible = parentVisible
            && this.expandedEventIds.has(parentEventId);
        const zones = this.hierarchy.zones;
        for (const zoneIndex of children) {
            const previousVisible = this.zoneVisibility[zoneIndex] !== 0;
            if (previousVisible !== childrenVisible) {
                this.zoneVisibility[zoneIndex] = childrenVisible ? 1 : 0;
                const rowIndex = zones.smIndices[zoneIndex];
                if (childrenVisible) {
                    this.rowVisibleZoneCounts[rowIndex]++;
                } else {
                    this.rowVisibleZoneCounts[rowIndex]--;
                }
                changedZoneIndices?.push(zoneIndex);
            }

            this.setChildVisibility(
                zones.eventIds[zoneIndex], childrenVisible,
                changedZoneIndices);
        }
    }

    private treeRowVisible(rowIndex: number): boolean {
        const expansionMode = this.rowExpansionModes[rowIndex]
            ?? TrackExpansionMode.Always;
        const expansionGroupId = this.rowExpansionGroupIds[rowIndex] ?? 0n;
        if (expansionMode === TrackExpansionMode.Collapsed
            && this.expandedTrackIds.has(expansionGroupId)) {
            return false;
        }
        if (expansionMode === TrackExpansionMode.Expanded
            && !this.expandedTrackIds.has(expansionGroupId)) {
            return false;
        }

        let parentRow = this.treeParentRows[rowIndex] ?? -1;

        while (parentRow >= 0) {
            const parentExpansionMode = this.rowExpansionModes[parentRow]
                ?? TrackExpansionMode.Always;
            if (parentExpansionMode === TrackExpansionMode.Always
                && !this.expandedTreeTrackIds.has(
                    this.treeTrackIds[parentRow])) {
                return false;
            }
            parentRow = this.treeParentRows[parentRow] ?? -1;
        }

        return true;
    }

    private toggleTreeTrack(trackId: bigint): void {
        if (this.expandedTreeTrackIds.has(trackId)) {
            this.expandedTreeTrackIds.delete(trackId);
        } else {
            this.expandedTreeTrackIds.add(trackId);
        }

        const disclosure = this.treeDisclosures.get(trackId);
        if (disclosure) {
            disclosure.textContent = this.expandedTreeTrackIds.has(trackId)
                ? '\u25be' : '\u25b8';
        }
        this.tooltip.classList.remove('visible');
        this.interactionManager?.clearSelection();
        this.recalculateRowLayout(true);
    }

    /** Repositions stable rows and uploads only their small offset table. */
    private recalculateRowLayout(updateGpu: boolean): void {
        if (!this.hierarchy) return;

        const lanes = this.hierarchy.lanes;
        const visibleAbove = new Int32Array(lanes.count);
        let previousVisibleRow = -1;
        for (let rowIndex = 0; rowIndex < lanes.count; rowIndex++) {
            const visible = this.treeRowVisible(rowIndex)
                && (lanes.depths[rowIndex] === 0
                    || this.rowVisibleZoneCounts[rowIndex] !== 0);
            this.rowVisible[rowIndex] = visible ? 1 : 0;
            visibleAbove[rowIndex] = previousVisibleRow;
            if (visible) previousVisibleRow = rowIndex;
        }

        const oldWorldHeight = this.worldHeight;
        const rowPadding = LANE_PADDING + BLOCK_LANE_PADDING;
        let currentY = 0;
        for (let rowIndex = lanes.count - 1; rowIndex >= 0; rowIndex--) {
            const visible = this.rowVisible[rowIndex] !== 0;
            const effectiveY = currentY;
            this.rowOffsets[rowIndex] = effectiveY - this.rowBaseYs[rowIndex];
            this.rowLayout[rowIndex * 4] = this.rowOffsets[rowIndex];
            this.rowLayout[rowIndex * 4 + 1] = visible ? 1 : 0;
            lanes.ys[rowIndex] = effectiveY;

            if (!visible) continue;
            currentY += lanes.heights[rowIndex] + rowPadding;
            const aboveRow = visibleAbove[rowIndex];
            if (lanes.depths[rowIndex] === 0 && aboveRow >= 0
                && lanes.depths[aboveRow] > 0) {
                currentY += rowPadding * 2;
            }
        }
        this.worldHeight = currentY;

        if (updateGpu && this.gpuResources) {
            this.device.queue.writeBuffer(
                this.gpuResources.buffers.rowLayout, 0, this.rowLayout);
        }
        if (updateGpu && this.camera) {
            this.camera.y += oldWorldHeight - this.worldHeight;
        }
    }

    /** Uploads only visibility words belonging to the toggled event subtree. */
    private uploadZoneVisibility(changedZoneIndices: number[]): void {
        if (!this.gpuResources || changedZoneIndices.length === 0) return;
        changedZoneIndices.sort((first, second) => first - second);

        let rangeStart = changedZoneIndices[0];
        let rangeEnd = rangeStart + 1;
        for (let i = 1; i <= changedZoneIndices.length; i++) {
            const zoneIndex = changedZoneIndices[i];
            if (i < changedZoneIndices.length && zoneIndex === rangeEnd) {
                rangeEnd++;
                continue;
            }

            this.device.queue.writeBuffer(
                this.gpuResources.buffers.zoneVisibility,
                rangeStart * Uint32Array.BYTES_PER_ELEMENT,
                this.zoneVisibility,
                rangeStart,
                rangeEnd - rangeStart);
            if (i < changedZoneIndices.length) {
                rangeStart = zoneIndex;
                rangeEnd = zoneIndex + 1;
            }
        }
    }

    /** Rebuilds the tiny set of frames for currently expanded events. */
    private rebuildExpandedKernelGroups(): void {
        this.expandedKernelGroups = [];
        this.kernelFramesContainer.replaceChildren();
        this.kernelFrames = [];
        if (!this.hierarchy) return;

        const zones = this.hierarchy.zones;
        for (const eventId of this.expandedEventIds) {
            const parentZoneIndex = this.visibleEventZoneIndex(eventId);
            const children = this.childZoneIndices.get(eventId);
            if (parentZoneIndex === undefined || !children) continue;

            let topTrackIndex = zones.smIndices[parentZoneIndex];
            let bottomTrackIndex = topTrackIndex;
            let hasVisibleChild = false;
            for (const childZoneIndex of children) {
                if (this.zoneVisibility[childZoneIndex] === 0) continue;
                const rowIndex = zones.smIndices[childZoneIndex];
                topTrackIndex = Math.min(topTrackIndex, rowIndex);
                bottomTrackIndex = Math.max(bottomTrackIndex, rowIndex);
                hasVisibleChild = true;
            }
            if (!hasVisibleChild) continue;

            this.expandedKernelGroups.push({
                startNs: zones.startsX[parentZoneIndex],
                endNs: zones.endsX[parentZoneIndex],
                topTrackIndex,
                bottomTrackIndex
            });
        }

        for (let i = 0; i < this.expandedKernelGroups.length; i++) {
            const frame = document.createElement('div');
            frame.className = 'kernel-frame';
            this.kernelFramesContainer.appendChild(frame);
            this.kernelFrames.push(frame);
        }
    }

    private disclosureTargets(): {
        eventIds: Set<bigint>;
        trackIds: Set<bigint>;
    } {
        const eventIds = this.materializedEventIds;
        const trackIds = new Set<bigint>();
        const streamsByGpu = new Map<bigint, Set<bigint>>();
        if (!this.fullTraceData) return { eventIds, trackIds };

        for (const hierarchy of this.fullTraceData.trackHierarchies) {
            const gpuNode = hierarchy.find(node => node.kind === 2);
            const streamNode = hierarchy.find(node => node.kind === 3);
            if (gpuNode && streamNode) {
                const streams = streamsByGpu.get(gpuNode.id)
                    ?? new Set<bigint>();
                streams.add(streamNode.id);
                streamsByGpu.set(gpuNode.id, streams);
            }
            for (const node of hierarchy) {
                if (node.kind === 5 || node.kind === 6) {
                    trackIds.add(node.id);
                }
            }
        }
        for (const [gpuId, streamIds] of streamsByGpu) {
            if (streamIds.size > 1) trackIds.add(gpuId);
        }
        return { eventIds, trackIds };
    }

    private allDetailsExpanded(): boolean {
        const targets = this.disclosureTargets();
        return targets.eventIds.size !== 0
            && this.expandedEventIds.size === targets.eventIds.size
            && Array.from(targets.trackIds).every(
                id => this.expandedTrackIds.has(id));
    }

    private updateExpandAllButton(): void {
        this.expandAllBtn.textContent = this.allDetailsExpanded()
            ? 'Collapse all' : 'Expand all';
    }

    private async rebuildProjection(
        screenX?: number, screenY?: number
    ): Promise<void> {
        if (!this.fullTraceData) return;
        this.tooltip.classList.remove('visible');
        const previousCamera = this.camera ? {
            x: this.camera.x,
            y: this.camera.y,
            zoom: this.camera.zoom,
            xZoomMultiplier: this.camera.xZoomMultiplier,
            worldHeight: this.worldHeight
        } : null;

        const projection = projectTraceData(
            this.fullTraceData, this.materializedEventIds,
            this.expandedTrackIds);
        await this.applyTraceProjection(projection);
        await this.initVisualization();

        if (previousCamera && this.camera) {
            this.camera.x = previousCamera.x;
            this.camera.y = -this.worldHeight
                + previousCamera.y + previousCamera.worldHeight;
            this.camera.zoom = previousCamera.zoom;
            this.camera.xZoomMultiplier = previousCamera.xZoomMultiplier;
        }

        this.updateExpandAllButton();
        this.interactionManager?.clearSelection();
        if (this.interactionManager && this.hierarchy
            && screenX !== undefined && screenY !== undefined) {
            this.interactionManager.updateHover(
                screenX,
                screenY,
                this.hierarchy,
                this.formatTooltipString.bind(this),
                this.formatTrackTooltipString.bind(this),
                this.formatBlockTooltipString.bind(this)
            );
        }
    }

    /** Toggles an event- or track-owned disclosure group. */
    private async toggleDisclosure(
        disclosureKey: string, screenX?: number, screenY?: number
    ): Promise<void> {
        if (!this.fullTraceData || !disclosureKey) return;

        if (disclosureKey.startsWith('track:')) {
            const trackId = BigInt(disclosureKey.slice('track:'.length));
            if (this.expandedTrackIds.has(trackId)) {
                this.expandedTrackIds.delete(trackId);
            } else {
                this.expandedTrackIds.add(trackId);
            }
            const expanded = this.expandedTrackIds.has(trackId);
            const disclosure = this.treeDisclosures.get(trackId);
            if (disclosure) {
                disclosure.textContent = expanded ? '\u25be' : '\u25b8';
            }
            this.tooltip.classList.remove('visible');
            this.interactionManager?.clearSelection();
            this.recalculateRowLayout(true);
            this.rebuildExpandedKernelGroups();
            this.updateExpandAllButton();
            if (this.interactionManager && this.hierarchy
                && screenX !== undefined && screenY !== undefined) {
                this.interactionManager.updateHover(
                    screenX, screenY, this.hierarchy,
                    this.formatTooltipString.bind(this),
                    this.formatTrackTooltipString.bind(this),
                    this.formatBlockTooltipString.bind(this));
            }
            return;
        } else {
            const eventId = BigInt(disclosureKey.slice('event:'.length));
            const parentZoneIndex = this.visibleEventZoneIndex(eventId);
            if (this.expandedEventIds.has(eventId)) {
                this.expandedEventIds.delete(eventId);
            } else {
                this.expandedEventIds.add(eventId);
            }
            if (parentZoneIndex !== undefined && this.hierarchy) {
                this.hierarchy.zones.expanded[parentZoneIndex] =
                    this.expandedEventIds.has(eventId) ? 1 : 0;
            }

            const changedZoneIndices: number[] = [];
            const parentVisible = parentZoneIndex !== undefined
                && this.zoneVisibility[parentZoneIndex] !== 0;
            this.setChildVisibility(
                eventId, parentVisible, changedZoneIndices);
            this.uploadZoneVisibility(changedZoneIndices);
            this.recalculateRowLayout(true);
            this.rebuildExpandedKernelGroups();
            this.updateExpandAllButton();
            this.interactionManager?.clearSelection();
            if (this.interactionManager && this.hierarchy
                && screenX !== undefined && screenY !== undefined) {
                this.interactionManager.updateHover(
                    screenX, screenY, this.hierarchy,
                    this.formatTooltipString.bind(this),
                    this.formatTrackTooltipString.bind(this),
                    this.formatBlockTooltipString.bind(this));
            }
            return;
        }
    }

    private async toggleAllDetails(): Promise<void> {
        const targets = this.disclosureTargets();
        if (this.allDetailsExpanded()) {
            this.expandedEventIds.clear();
            this.expandedTrackIds.clear();
        } else {
            for (const eventId of targets.eventIds) {
                this.expandedEventIds.add(eventId);
            }
            for (const trackId of targets.trackIds) {
                this.expandedTrackIds.add(trackId);
            }
        }

        await this.rebuildProjection();
    }

    /**
     * Wrapper around formatStringHelper to use instance's format descriptors.
     * Replaces placeholders like {0}, {1} with parameter values.
     */
    formatString(formatDescId: number, params: number[]): string {
        if (!this.hierarchy) return '';
        return formatStringHelper(this.hierarchy.formatDescriptors, formatDescId, params);
    }

    /**
     * Wrapper around formatTrackStringHelper to use instance's format descriptors.
     * Replaces {lane} placeholder and numbered placeholders like {0}, {1} with values.
     */
    formatTrackString(formatDescId: number, laneId: number, params: number[]): string {
        if (!this.hierarchy) return '';
        return formatTrackStringHelper(this.hierarchy.formatDescriptors, formatDescId, laneId, params);
    }

    /**
     * Wrapper around formatBlockStringHelper to use instance's format descriptors and grid dimensions.
     * Replaces special placeholders like {blockX}, {blockY}, {blockZ}, {clusterX}, etc.
     */
    formatBlockString(formatDescId: number, blockId: number, clusterId: number): string {
        if (!this.hierarchy) return '';
        return formatBlockStringHelper(
            this.hierarchy.formatDescriptors,
            formatDescId,
            blockId,
            clusterId
        );
    }

    /**
     * Wrapper around formatTooltipStringHelper to use instance's format descriptors.
     * Uses tooltip string instead of label string for hover display.
     */
    formatTooltipString(formatDescId: number, params: number[]): string {
        if (!this.hierarchy) return '';
        return formatTooltipStringHelper(this.hierarchy.formatDescriptors, formatDescId, params);
    }

    /**
     * Wrapper around formatTrackTooltipStringHelper to use instance's format descriptors.
     * Uses tooltip string instead of label string for hover display.
     */
    formatTrackTooltipString(formatDescId: number, laneId: number, params: number[]): string {
        if (!this.hierarchy) return '';
        return formatTrackTooltipStringHelper(this.hierarchy.formatDescriptors, formatDescId, laneId, params);
    }

    /**
     * Wrapper around formatBlockTooltipStringHelper to use instance's format descriptors and grid dimensions.
     * Uses tooltip string instead of label string for hover display.
     */
    formatBlockTooltipString(formatDescId: number, blockId: number, clusterId: number): string {
        if (!this.hierarchy) return '';
        return formatBlockTooltipStringHelper(
            this.hierarchy.formatDescriptors,
            formatDescId,
            blockId,
            clusterId
        );
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
                this.fitTraceToViewport();
            }
        });

        window.addEventListener('wheel', (e) => {
            if (!this.camera) return;

            const helpOverlay = document.getElementById('help-overlay');
            if (helpOverlay && !helpOverlay.classList.contains('hidden')) return;

            e.preventDefault();

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
            if (e.button === 0 && this.interactionManager && this.hierarchy) {
                const result = this.interactionManager.findZoneAtPosition(e.clientX, e.clientY, this.hierarchy);

                if (result.zoneIdx !== -1) {
                    if (this.hierarchy.zones.hasChildren[result.zoneIdx] !== 0) {
                        return;
                    }

                    const epsilon = SELECTION_EPSILON;
                    // Convert zone times from nanoseconds to milliseconds
                    const zoneStartMs = this.hierarchy.zones.startsX[result.zoneIdx] * 1e-6;
                    const zoneEndMs = this.hierarchy.zones.endsX[result.zoneIdx] * 1e-6;
                    this.interactionManager.startSelection(Math.max(0, zoneStartMs - epsilon));
                    this.interactionManager.updateSelectionEnd(Math.min(this.TIME_RANGE, zoneEndMs + epsilon));
                    this.interactionManager.endSelection();
                    this.interactionManager.updateSelection();
                } else if (result.blockIdx !== -1) {
                    const epsilon = SELECTION_EPSILON;
                    // Convert block times from nanoseconds to milliseconds
                    const blockStartMs = this.hierarchy.blocks.startsX[result.blockIdx] * 1e-6;
                    const blockEndMs = this.hierarchy.blocks.endsX[result.blockIdx] * 1e-6;
                    this.interactionManager.startSelection(Math.max(0, blockStartMs - epsilon));
                    this.interactionManager.updateSelectionEnd(Math.min(this.TIME_RANGE, blockEndMs + epsilon));
                    this.interactionManager.endSelection();
                    this.interactionManager.updateSelection();
                }
            }
        });

        this.canvas.addEventListener('click', (e) => {
            if (e.button !== 0 || e.detail !== 1
                || !this.interactionManager || !this.hierarchy) return;

            const result = this.interactionManager.findZoneAtPosition(
                e.clientX, e.clientY, this.hierarchy);
            if (result.zoneIdx !== -1
                && this.hierarchy.zones.hasChildren[result.zoneIdx] !== 0) {
                e.preventDefault();
                void this.toggleDisclosure(
                    this.hierarchy.zones.disclosureKeys[result.zoneIdx],
                    e.clientX, e.clientY);
            }
        });

        this.canvas.addEventListener('mousedown', (e) => {
            if (!this.camera || !this.interactionManager) return;

            if (e.button === 2) {
                this.camera.isDragging = true;
                this.camera.lastX = e.clientX;
                this.camera.lastY = e.clientY;
            } else if (e.button === 0) {
                if (this.hierarchy) {
                    const result = this.interactionManager.findZoneAtPosition(
                        e.clientX, e.clientY, this.hierarchy);
                    if (result.zoneIdx !== -1
                        && this.hierarchy.zones.hasChildren[result.zoneIdx] !== 0) {
                        return;
                    }
                }

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

            // Check if help overlay is open
            const helpOverlay = document.getElementById('help-overlay');
            const isHelpOpen = helpOverlay && !helpOverlay.classList.contains('hidden');

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

            // Don't show cursor line/timestamp if help overlay is open
            if (!isHelpOpen && e.clientX >= 0 && e.clientX <= window.innerWidth &&
                e.clientY >= 0 && e.clientY <= window.innerHeight) {
                this.cursorLine.style.left = `${e.clientX}px`;
                this.cursorLine.style.display = 'block';

                const worldPos = this.camera.screenToWorld(e.clientX, e.clientY, this.canvas);
                const time = worldPos.x;

                const timeInNs = time * MS_TO_NS;
                const formattedTime = Math.round(timeInNs).toLocaleString()
                    + ' ns';

                this.cursorTimestamp.textContent = formattedTime;
                this.cursorTimestamp.style.left = `${e.clientX + 4}px`;
                this.cursorTimestamp.style.display = 'block';
            } else {
                this.cursorLine.style.display = 'none';
                this.cursorTimestamp.style.display = 'none';
            }

            // Don't update hover if help overlay is open
            if (!isHelpOpen && this.hierarchy) {
                this.interactionManager.updateHover(e.clientX, e.clientY, this.hierarchy, this.formatTooltipString.bind(this), this.formatTrackTooltipString.bind(this), this.formatBlockTooltipString.bind(this));
            }
        });

        document.body.addEventListener('mouseleave', () => {
            this.cursorLine.style.display = 'none';
            this.cursorTimestamp.style.display = 'none';
        });
    }

    /** Fits the trace beside the hierarchical track-label gutter. */
    private fitTraceToViewport(): void {
        if (!this.camera) return;

        const rect = this.canvas.getBoundingClientRect();
        const aspect = rect.width / rect.height;
        const leftPadding = Math.min(
            TRACK_LABEL_WIDTH + 16, rect.width * 0.4);
        const rightPadding = INITIAL_ZOOM_PADDING / 2;
        const ndcLeft = 2 * leftPadding / rect.width - 1;
        const ndcRight = 1 - 2 * rightPadding / rect.width;
        const desiredZoomX = (ndcRight - ndcLeft) * aspect / this.TIME_RANGE;

        this.camera.zoom = INITIAL_BASE_ZOOM;
        this.camera.xZoomMultiplier = desiredZoomX / this.camera.zoom;
        this.camera.x = ndcLeft * aspect / desiredZoomX;
        const topPadding = TIMELINE_HEIGHT + 14;
        const topNdc = 1 - 2 * topPadding / rect.height;
        this.camera.y = topNdc / this.camera.zoom - this.worldHeight;
    }

    /**
     * Delegates zone label rendering to LabelRenderer (SoA version).
     * Convenience wrapper that passes instance data to the renderer.
     */
    renderZoneLabels(): void {
        if (!this.camera || !this.labelRenderer || !this.hierarchy) return;
        this.labelRenderer.renderZoneLabels(
            this.hierarchy,
            this.formatString.bind(this),
            this.formatBlockString.bind(this),
            this.rowOffsets,
            this.rowVisible,
            this.zoneVisibility);
    }

    /**
     * Updates visible track-label positions based on camera viewport.
     *
     * Positions labels on the left edge of the viewport and vertically
     * aligns them with their corresponding lanes. Labels are clamped to stay below
     * the timeline bar (30px) and are hidden when lanes scroll off-screen.
     */
    updateLaneLabels(): void {
        if (!this.camera || !this.hierarchy) return;

        if (this.camera.zoomY < MIN_LABEL_ZOOM_Y) {
            for (const label of this.laneLabels) {
                label.style.display = 'none';
            }
            return;
        }

        const rect = this.canvas.getBoundingClientRect();
        const aspect = rect.width / rect.height;

        const ndcX = this.camera.x * this.camera.zoomX / aspect;
        const laneStartScreenX = (ndcX + 1) * rect.width / 2;

        const labelWidth = TRACK_LABEL_WIDTH;
        const labelX = Math.max(0, laneStartScreenX - labelWidth);

        const timelineHeight = TIMELINE_HEIGHT;
        const screenZoneHeight = SUBLANE_HEIGHT * this.camera.zoomY
            * (rect.height / 2);
        const labelFontSize = LABEL_FONT_SIZE * screenZoneHeight
            / MIN_ZONE_LABEL_HEIGHT;
        const labelScale = labelFontSize / LABEL_FONT_SIZE;

        const lanes = this.hierarchy.lanes;
        for (let i = 0; i < lanes.count; i++) {
            const groupId = this.rowExpansionGroupIds[i] ?? 0n;
            const expandedGroup = this.expandedTrackIds.has(groupId);
            const expansionLabelRow = this.expansionLabelRows[i] ?? -1;
            const overlayExpandedGroup = expandedGroup
                && expansionLabelRow >= 0;
            if ((this.rowVisible[i] === 0 && !overlayExpandedGroup)
                || (expandedGroup
                    && this.suppressedExpansionLabels[i] !== 0)
                || labelFontSize < MIN_LABEL_FONT_SIZE) {
                this.laneLabels[i].style.display = 'none';
                continue;
            }
            const displayRow = overlayExpandedGroup ? expansionLabelRow : i;
            // Check if lane has any block lanes
            const hasBlockLanes = lanes.blockLanesEndIndices[displayRow]
                > lanes.blockLanesStartIndices[displayRow];
            if (!hasBlockLanes) {
                this.laneLabels[i].style.display = 'none';
                continue;
            }

            const laneTopY = lanes.ys[displayRow]
                + lanes.heights[displayRow];
            const laneBottomY = lanes.ys[displayRow];

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
                let currentLabelX = labelX;
                let currentLabelWidth = labelWidth;

                const physicalSmRow = lanes.names[i]?.startsWith('SM ') ?? false;
                const treeChild = this.treeParentRows[i] >= 0;
                if (lanes.depths[i] > 0 && !physicalSmRow && !treeChild) {
                    const startMs = lanes.startsX[i] / MS_TO_NS;
                    const endMs = lanes.widths[i] / MS_TO_NS;
                    const startNdc = (startMs + this.camera.x)
                        * this.camera.zoomX / aspect;
                    const endNdc = (endMs + this.camera.x)
                        * this.camera.zoomX / aspect;
                    const startScreenX = (startNdc + 1) * rect.width / 2;
                    const endScreenX = (endNdc + 1) * rect.width / 2;
                    currentLabelX = Math.max(0, startScreenX);
                    currentLabelWidth = Math.max(
                        0, Math.min(rect.width, endScreenX) - currentLabelX);
                }

                const childLabelTooSmall = lanes.depths[i] > 0
                    && !physicalSmRow
                    && !treeChild
                    && (currentLabelWidth < MIN_ZONE_LABEL_WIDTH
                        || clampedHeight < MIN_LABEL_FONT_SIZE);
                if (clampedHeight === 0 || currentLabelWidth < 1
                    || childLabelTooSmall) {
                    this.laneLabels[i].style.display = 'none';
                    continue;
                }

                this.laneLabels[i].style.display = 'flex';
                this.laneLabels[i].style.fontSize = `${labelFontSize}px`;
                this.laneLabels[i].style.gap = `${5 * labelScale}px`;
                this.laneLabels[i].style.top = `${clampedTopY}px`;
                this.laneLabels[i].style.left = `${currentLabelX}px`;
                this.laneLabels[i].style.width = `${currentLabelWidth}px`;
                this.laneLabels[i].style.height = `${clampedHeight}px`;
            } else {
                this.laneLabels[i].style.display = 'none';
            }
        }
    }

    /** Positions one frame ten pixels outside the complete visible track group. */
    private updateTrackFrame(): void {
        if (!this.camera || !this.hierarchy
            || this.hierarchy.lanes.count === 0) return;

        const framePadding = 10;
        const rect = this.canvas.getBoundingClientRect();
        const aspect = rect.width / rect.height;
        const lanes = this.hierarchy.lanes;
        const topWorld = lanes.ys[0] + lanes.heights[0];
        const bottomWorld = lanes.ys[lanes.count - 1];

        const leftNdc = this.camera.x * this.camera.zoomX / aspect;
        const rightNdc = (this.TIME_RANGE + this.camera.x)
            * this.camera.zoomX / aspect;
        const topNdc = (topWorld + this.camera.y) * this.camera.zoomY;
        const bottomNdc = (bottomWorld + this.camera.y) * this.camera.zoomY;

        const left = (leftNdc + 1) * rect.width / 2 - framePadding;
        const right = (rightNdc + 1) * rect.width / 2 + framePadding;
        const top = rect.height / 2 - topNdc * rect.height / 2 - framePadding;
        const bottom = rect.height / 2
            - bottomNdc * rect.height / 2 + framePadding;

        this.positionClippedFrame(
            this.trackFrame, left, right, top, bottom, rect.width, 4);
    }

    /** Positions rounded frames around each expanded kernel's child rows. */
    private updateKernelFrames(): void {
        if (!this.camera || !this.hierarchy) return;

        const framePadding = 2;
        const rect = this.canvas.getBoundingClientRect();
        const aspect = rect.width / rect.height;
        const lanes = this.hierarchy.lanes;

        for (let i = 0; i < this.expandedKernelGroups.length; i++) {
            const group = this.expandedKernelGroups[i];
            const frame = this.kernelFrames[i];
            const topLane = lanes.ys[group.topTrackIndex]
                + lanes.heights[group.topTrackIndex];
            const bottomLane = lanes.ys[group.bottomTrackIndex];
            const startMs = group.startNs / MS_TO_NS;
            const endMs = group.endNs / MS_TO_NS;
            const leftNdc = (startMs + this.camera.x)
                * this.camera.zoomX / aspect;
            const rightNdc = (endMs + this.camera.x)
                * this.camera.zoomX / aspect;
            const topNdc = (topLane + this.camera.y) * this.camera.zoomY;
            const bottomNdc = (bottomLane + this.camera.y) * this.camera.zoomY;

            const left = (leftNdc + 1) * rect.width / 2 - framePadding;
            const right = (rightNdc + 1) * rect.width / 2 + framePadding;
            const top = rect.height / 2
                - topNdc * rect.height / 2 - framePadding;
            const bottom = rect.height / 2
                - bottomNdc * rect.height / 2 + framePadding;

            this.positionClippedFrame(
                frame, left, right, top, bottom, rect.width, 1);
        }
    }

    /** Keeps overlay frames within Chrome's reliable CSS layout range. */
    private positionClippedFrame(
        frame: HTMLElement,
        left: number,
        right: number,
        top: number,
        bottom: number,
        viewportWidth: number,
        borderRadius: number
    ): void {
        const clippedLeft = Math.max(0, left);
        const clippedRight = Math.min(viewportWidth, right);

        if (clippedRight <= clippedLeft || bottom <= top) {
            frame.style.display = 'none';
            return;
        }

        const leftClipped = left < 0;
        const rightClipped = right > viewportWidth;
        frame.style.display = 'block';
        frame.style.left = `${clippedLeft}px`;
        frame.style.top = `${top}px`;
        frame.style.width = `${clippedRight - clippedLeft}px`;
        frame.style.height = `${bottom - top}px`;
        frame.style.borderLeftWidth = leftClipped ? '0' : '1px';
        frame.style.borderRightWidth = rightClipped ? '0' : '1px';
        const radius = `${borderRadius}px`;
        frame.style.borderTopLeftRadius = leftClipped ? '0' : radius;
        frame.style.borderBottomLeftRadius = leftClipped ? '0' : radius;
        frame.style.borderTopRightRadius = rightClipped ? '0' : radius;
        frame.style.borderBottomRightRadius = rightClipped ? '0' : radius;
    }


    /**
     * Main render loop (runs at ~60 FPS via requestAnimationFrame).
     *
     * Rendering pipeline:
     * 1. Update stats display (FPS, trace info, memory usage)
     * 2. Update UI overlays (lane labels, timeline, zone labels, selection)
     * 3. Build uniform data (view-projection matrix, hover/selection state)
     * 4. Execute WebGPU render passes (lanes → block lanes → blocks → zones)
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
            this.stats.innerHTML = `<span class="stats-dynamic"></span><br><span class="stats-links"><a href="https://github.com/aikitoria/nanotrace" target="_blank">Nanotrace</a> ${VERSION}</span>`;
        }

        // Update only the dynamic content
        const dynamicStats = this.stats.querySelector('.stats-dynamic');
        if (dynamicStats && this.hierarchy) {
            dynamicStats.innerHTML = `${this.kernelName}<br>Duration: ${formattedDuration} ns<br>Grid: (${this.gridDimX}, ${this.gridDimY}, ${this.gridDimZ}) | Cluster: (${this.clusterDimX}, ${this.clusterDimY}, ${this.clusterDimZ})<br>Rows: ${this.hierarchy.lanes.count.toLocaleString()} | Groups: ${this.hierarchy.blocks.count.toLocaleString()} | Zones: ${this.numZones.toLocaleString()}<br>Zoom: ${this.camera.zoomX.toFixed(2)} × ${this.camera.zoomY.toFixed(2)} | FPS: ${fpsStr}`;
        }
        this.lastTime = now;

        // Update 2D overlays (labels, timeline, selection UI)
        this.updateTrackFrame();
        this.updateKernelFrames();
        this.updateLaneLabels();
        this.timelineRenderer!.updateTimeline(this.TIME_RANGE);
        this.renderZoneLabels();

        if (this.interactionManager && (this.interactionManager.hasActiveSelection() || this.interactionManager.isCurrentlySelecting())) {
            this.interactionManager.updateSelection();
        }

        // Prepare uniform data for shaders (128 bytes total)
        const aspect = this.canvas.width / this.canvas.height;
        const viewProjMatrix = this.camera.getViewProjectionMatrix(aspect);

        // Reuse preallocated uniform buffers to avoid GC pressure
        const floatView = this.uniformFloatView;
        const intView = this.uniformIntView;

        // Uniform layout: mat4x4 viewProj, int hoveredId, float zoomX, float zoomY,
        // float selectionStart, float selectionEnd, int hasSelection, int hoveredBlockId,
        // float camera_x_high, float camera_x_low, float camera_y, float scale_x,
        // float scale_y, float viewport_width
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
        floatView[28] = this.canvas.width;

        this.device.queue.writeBuffer(this.gpuResources.uniformBuffer, 0, this.uniformData);

        if (!this.hierarchy) return;

        const topLaneY = this.hierarchy.lanes.ys[0];
        const topLaneHeight = this.hierarchy.lanes.heights[0];
        const backgroundHeight = topLaneY + topLaneHeight;

        // Split TIME_RANGE into dual float for high precision
        const [timeRange_high, timeRange_low] = Camera.splitDouble(this.TIME_RANGE);

        // Reuse preallocated background uniform buffer to avoid GC pressure
        // Layout: camera_x_high, camera_x_low, camera_y, scale_x, scale_y, timeRange_high, timeRange_low, worldHeight
        this.backgroundFloatView[0] = camera_x_high;
        this.backgroundFloatView[1] = camera_x_low;
        this.backgroundFloatView[2] = this.camera.y;
        this.backgroundFloatView[3] = scale_x;
        this.backgroundFloatView[4] = scale_y;
        this.backgroundFloatView[5] = timeRange_high;
        this.backgroundFloatView[6] = timeRange_low;
        this.backgroundFloatView[7] = backgroundHeight;
        this.device.queue.writeBuffer(this.gpuResources.backgroundUniformBuffer, 0, this.backgroundUniformData);

        const commandEncoder = this.device.createCommandEncoder();
        const textureView = this.context.getCurrentTexture().createView();

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
                loadOp: 'clear',
                storeOp: 'store',
            }]
        });

        renderPass.setPipeline(this.gpuResources.passes.lane.pipeline);
        renderPass.setBindGroup(0, this.gpuResources.passes.lane.bindGroup);
        renderPass.draw(6, this.hierarchy.lanes.count, 0, 0);

        renderPass.setPipeline(this.gpuResources.passes.blockLane.pipeline);
        renderPass.setBindGroup(0, this.gpuResources.passes.blockLane.bindGroup);
        renderPass.draw(6, this.hierarchy.blockLanes.count, 0, 0);

        renderPass.setPipeline(this.gpuResources.passes.blockBg.pipeline);
        renderPass.setBindGroup(0, this.gpuResources.passes.blockBg.bindGroup);
        renderPass.draw(6, this.hierarchy.blocks.count, 0, 0);

        renderPass.setPipeline(this.gpuResources.passes.block.pipeline);
        renderPass.setBindGroup(0, this.gpuResources.passes.block.bindGroup);
        renderPass.draw(6, this.hierarchy.blocks.count, 0, 0);

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
