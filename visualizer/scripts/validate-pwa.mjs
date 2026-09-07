import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { Script } from 'node:vm';

const output = resolve('dist');
const deployment = new URL('https://aikitoria.github.io/nanotrace/');
const html = readFileSync(resolve(output, 'index.html'), 'utf8');
const manifest = JSON.parse(readFileSync(resolve(output, 'manifest.webmanifest'), 'utf8'));

function deployedFile(relative, base = deployment) {
    const url = new URL(relative, base);
    assert.equal(url.origin, deployment.origin);
    assert.ok(url.pathname.startsWith(deployment.pathname), `Outside app scope: ${url}`);
    const file = resolve(output, url.pathname.slice(deployment.pathname.length));
    assert.ok(existsSync(file), `Missing deployed asset: ${url}`);
    return file;
}

for (const name of ['id', 'start_url', 'scope']) {
    assert.equal(new URL(manifest[name], deployment).href, deployment.href, name);
}
assert.equal(manifest.display, 'standalone');
assert.ok(manifest.name && manifest.short_name);
assert.notEqual(manifest.prefer_related_applications, true);
assert.equal(manifest.theme_color, '#111113');
assert.match(html, /<link\b[^>]*rel="manifest"[^>]*href="\/nanotrace\/manifest.webmanifest"/);
assert.match(html, /<meta\b[^>]*name="theme-color"[^>]*content="#111113"/);

for (const match of html.matchAll(/(?:href|src)="([^"#]+)"/g)) {
    if (match[1].startsWith(deployment.pathname)) deployedFile(match[1]);
}

function checkPng(file, size) {
    const bytes = readFileSync(file);
    assert.equal(bytes.subarray(0, 8).toString('hex'), '89504e470d0a1a0a');
    assert.equal(bytes.readUInt32BE(16), size);
    assert.equal(bytes.readUInt32BE(20), size);
}

for (const icon of manifest.icons) deployedFile(icon.src);
for (const size of [192, 512]) {
    const icon = manifest.icons.find(icon => icon.sizes === `${size}x${size}`);
    assert.ok(icon, `Missing ${size}px install icon`);
    assert.equal(icon.type, 'image/png');
    checkPng(deployedFile(icon.src), size);
}
checkPng(deployedFile('apple-touch-icon.png'), 180);
new Script(readFileSync(deployedFile('sw.js'), 'utf8'));
console.log('PWA deployment validated: scoped manifest, HTML links, icons and service worker.');
