import type { TraceOverlays } from './types.js';

// The two alternating banks have six distinct slot/stage colors. CPU expert
// envelopes share the corresponding GPU Expert color. Dense MLP work uses
// the same stage color; token-boundary ranges retain their recorded color.
const BANK_STAGE_COLORS = [0x4A90D9, 0xE69F00, 0xB07CD8, 0x42B9B0, 0xE573A0, 0xB2BD48];

export function semanticRangeColor(label: string, fallback: number): number {
    const match = /^Slot (\d+) (Attention|Expert|MoE|MLP|Merge) Layer /.exec(label);
    if (!match) return fallback;
    const slot = Number(match[1]);
    const stage = match[2] === 'Attention' ? 0 : match[2] === 'Merge' ? 2 : 1;
    return BANK_STAGE_COLORS[(slot % 2) * 3 + stage];
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
