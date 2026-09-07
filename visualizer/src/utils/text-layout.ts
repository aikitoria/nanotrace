/** Align text rasterization to physical pixels without rounding trace geometry. */
export function snapTextPixel(value: number, pixelRatio: number): number {
    return Math.round(value * pixelRatio) / pixelRatio;
}

/** Shorten known C++ namespaces in drawn labels, preserving trace metadata. */
export function displayLabel(label: string): string {
    return label.replace(/mina::(?:kernels|components)::/g, '');
}
