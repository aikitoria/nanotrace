import type { TraceOverlays } from './types.js';

// Preserve the recorded hue while lifting dark colors for the viewer's dark
// surfaces. Labels are opaque application data and never select a palette.
export function semanticRangeColor(source: number): number {
    const linear = [16, 8, 0].map(shift => {
        const channel = ((source >> shift) & 255) / 255;
        return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
    });
    const luminance = linear[0] * 0.2126 + linear[1] * 0.7152 + linear[2] * 0.0722;
    const lift = luminance < 0.4 ? (0.4 - luminance) / (1 - luminance) : 0;
    return linear.reduce((color, channel) => {
        const lifted = channel + (1 - channel) * lift;
        const srgb = lifted <= 0.0031308 ? lifted * 12.92 : 1.055 * lifted ** (1 / 2.4) - 0.055;
        return (color << 8) | Math.round(srgb * 255);
    }, 0);
}

/** Identify where to add space above a group, never inside a row. */
export function semanticGroupTopRows(
    rowCount: number, overlays: TraceOverlays, visible?: Uint8Array
): Uint8Array {
    const tops = new Uint8Array(rowCount);
    for (const rows of overlays.groupRows) {
        let first = rowCount;
        for (const row of rows) {
            if ((!visible || visible[row]) && row < first) first = row;
        }
        if (first < rowCount) tops[first] = 1;
    }
    return tops;
}
