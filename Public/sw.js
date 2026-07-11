// NeuraMesh service worker — app-shell cache so the installed PWA opens
// instantly (and can show "looking for your mesh" when the coordinator
// is down) instead of a browser error page.
//
// Registered ONLY in secure contexts (https / localhost) — see main.tsx.
// Served over plain http on the trusted LAN, browsers refuse service
// workers entirely, and the app works fine without one; this file is
// the upgrade path for anyone terminating TLS in front of the mesh.
//
// Strategy: network-first for everything (live data must be live),
// falling back to the cached shell for navigations when the mesh is
// unreachable. /api, /ws and /health are never cached.

const CACHE = 'neuramesh-shell-v1';

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) =>
      cache.addAll(['/', '/manifest.webmanifest', '/icon-192.png'])),
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))),
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  const isLiveData = url.pathname.startsWith('/api')
    || url.pathname === '/ws' || url.pathname === '/health';
  if (event.request.method !== 'GET' || isLiveData) return; // straight through

  event.respondWith(
    fetch(event.request)
      .then((response) => {
        const copy = response.clone();
        caches.open(CACHE).then((cache) => cache.put(event.request, copy));
        return response;
      })
      .catch(() =>
        caches.match(event.request).then(
          (cached) => cached ?? caches.match('/'))),
  );
});
