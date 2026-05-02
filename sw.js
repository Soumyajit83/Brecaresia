self.addEventListener('install', (e) => {
  console.log('Service Worker Installed');
});

self.addEventListener('fetch', (e) => {
  // Logic for offline support would go here
  e.respondWith(fetch(e.request));
});