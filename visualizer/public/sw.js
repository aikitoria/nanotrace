// Match the online viewer's network behavior: no traces or application assets
// are cached, and deploying a new version never reloads an active analysis.
self.addEventListener('install', (event) => {
    event.waitUntil(self.skipWaiting());
});
self.addEventListener('activate', (event) => {
    event.waitUntil(self.clients.claim());
});
