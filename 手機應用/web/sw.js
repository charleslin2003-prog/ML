/**
 * Service Worker：讓應用離線可用。
 *
 * 快取策略刻意分成兩種：
 *   - 程式碼與介面：network-first，確保使用者拿到最新版
 *   - 模型檔案（數十 MB）：cache-first，下載過一次就不再重抓
 */
const VERSION = 'v1';
const SHELL = `shell-${VERSION}`;
const MODEL = `model-${VERSION}`;

const SHELL_FILES = [
  './index.html', './app.js', './inference.js', './tokenizer.js',
  './styles.css', './manifest.json',
  './vendor/ort.min.js', './vendor/ort-wasm-simd-threaded.wasm',
  './vendor/ort-wasm-simd-threaded.mjs',
];

// skipWaiting + clients.claim 讓 SW 一安裝完就接管現有頁面。
// 沒有這組設定的話，首次造訪時模型的 fetch 會繞過 SW 而不被快取，
// 使用者要到第二次開啟才真正具備離線能力（實測確認過這個行為）。
self.addEventListener('install', e => {
  e.waitUntil(caches.open(SHELL).then(c => c.addAll(SHELL_FILES)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(
        keys.filter(k => k !== SHELL && k !== MODEL).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET') return;

  // 模型與詞彙表：cache-first（檔案大且不常變，不該每次重抓）
  if (url.pathname.includes('/model/')) {
    e.respondWith(
      caches.open(MODEL).then(async cache => {
        const hit = await cache.match(e.request);
        if (hit) return hit;
        const resp = await fetch(e.request);
        if (resp.ok) cache.put(e.request, resp.clone());
        return resp;
      })
    );
    return;
  }

  // 其餘：network-first，離線時退回快取
  e.respondWith(
    fetch(e.request)
      .then(resp => {
        const copy = resp.clone();
        caches.open(SHELL).then(c => c.put(e.request, copy));
        return resp;
      })
      .catch(() => caches.match(e.request))
  );
});
