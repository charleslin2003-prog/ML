import { DefamationClassifier } from './inference.js';

const $ = id => document.getElementById(id);
const clf = new DefamationClassifier();

const EXAMPLES = [
  '你這個智障，廢物一個',
  '他把公司的錢偷偷匯到自己帳戶',
  '幹你娘機掰，滾出去',
  '這個醫生根本沒有執照，是密醫',
];

const COLOR = { '公然侮辱': 'insult', '誹謗類': 'defame', '誹謗': 'defame' };

function fmtBytes(n) {
  if (!n) return '';
  return n > 1024 ** 2 ? `${(n / 1024 ** 2).toFixed(1)} MB` : `${(n / 1024).toFixed(0)} KB`;
}

async function boot() {
  try {
    const meta = await clf.init((stage, loaded, total) => {
      $('loadText').textContent = total > 1
        ? `${stage}… ${fmtBytes(loaded)} / ${fmtBytes(total)}`
        : `${stage}…`;
      $('loadBar').style.width = total ? `${(loaded / total) * 100}%` : '100%';
    });

    $('loading').hidden = true;
    $('main').hidden = false;
    $('go').disabled = false;
    $('aboutModel').textContent =
      `${meta.model_name}（${(meta.num_parameters / 1e6).toFixed(1)}M 參數，` +
      `量化後 ${meta.size_int8_mb} MB）`;
    $('aboutAcc').textContent = meta.accuracy
      ? `accuracy ${(meta.accuracy * 100).toFixed(1)}%、macro-F1 ${meta.macro_f1.toFixed(3)}`
      : '見技術報告';

    EXAMPLES.forEach(ex => {
      const b = document.createElement('button');
      b.textContent = ex.length > 12 ? ex.slice(0, 12) + '…' : ex;
      b.onclick = () => { $('text').value = ex; $('count').textContent = ex.length; };
      $('examples').appendChild(b);
    });
  } catch (err) {
    $('loadText').textContent = '模型載入失敗';
    $('loadNote').innerHTML =
      `<span class="warn">${err.message}</span><br>` +
      '請確認 model/ 目錄下有 .int8.onnx 與 vocab.txt，且是透過 http(s) 開啟（不能用 file://）。';
  }
}

$('text').addEventListener('input', e => {
  $('count').textContent = e.target.value.length;
});

$('go').addEventListener('click', async () => {
  const text = $('text').value.trim();
  if (!text) return;
  $('go').disabled = true;
  $('go').textContent = '判讀中…';
  try {
    const r = await clf.classify(text);
    render(text, r);
  } catch (err) {
    $('result').hidden = false;
    $('result').innerHTML = `<span class="warn">判讀失敗：${err.message}</span>`;
  } finally {
    $('go').disabled = false;
    $('go').textContent = '判讀';
  }
});

function render(text, r) {
  const el = $('result');
  el.hidden = false;

  // 信心度低時必須明講，不能只顯示一個看起來很確定的答案
  const conf = r.top.prob;
  const confNote = conf < 0.6
    ? '<p class="warn" style="font-size:.82rem;margin:.4rem 0">⚠ 兩個類別的機率接近，這則判讀的參考價值很低。</p>'
    : '';

  const rows = r.all.map(x => `
    <div class="row">
      <div class="name">${x.label}</div>
      <div class="track">
        <div class="fill ${COLOR[x.label] || ''}" style="width:${(x.prob * 100).toFixed(1)}%">
          ${(x.prob * 100).toFixed(1)}%
        </div>
      </div>
    </div>`).join('');

  el.innerHTML = `
    <p class="result-conf" style="margin-bottom:2px">用語較接近</p>
    <p class="result-label">${r.top.label} 類案件</p>
    <p class="result-conf">相似度 ${(conf * 100).toFixed(1)}%（非定罪機率）</p>
    ${confNote}
    ${rows}
    <div class="note">
      <span class="badge">${r.nTokens} tokens</span>
      <span class="badge">${r.latencyMs} ms</span>
      <span class="badge">裝置端運算</span>
      <br>
      這是「用語與過往判決案例的相似度」，不是定罪機率，也不是法律推理。
      是否成罪須依個案脈絡由法院認定。
      ${conf < 0.6 ? '這次的信心度偏低，建議不要採信。' : ''}
    </div>`;
  el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => navigator.serviceWorker.register('./sw.js').catch(() => {}));
}

/**
 * 顯示離線狀態與安裝引導。
 * iOS 特別重要：WebKit 的 ITP 會在使用者 7 天未與該站互動後，連同 Service Worker
 * 註冊與 Cache 內容一起刪除 —— 24.5 MB 的模型要重下一次。官方明文豁免的只有
 * 「已加入主畫面」的 web app，所以必須主動引導 iOS 使用者安裝。
 */
async function showCacheStatus() {
  const el = document.createElement('div');
  el.className = 'note';
  const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);
  const standalone = window.matchMedia('(display-mode: standalone)').matches ||
                     window.navigator.standalone === true;

  let cached = false;
  try {
    const c = await caches.open('model-v1');
    cached = (await c.keys()).length > 0;
  } catch { /* 不支援 Cache API */ }

  const parts = [cached ? '✓ 模型已存在裝置，可離線使用' : '首次使用需連線下載模型'];
  if (navigator.storage?.persisted) {
    const persisted = await navigator.storage.persisted().catch(() => false);
    if (persisted) parts.push('已申請持久化儲存');
  }
  if (isIOS && !standalone) {
    parts.push('<span class="warn">iOS 提醒：請用分享選單「加入主畫面」。'
             + '否則 Safari 會在 7 天未使用後清除快取，需重新下載模型。</span>');
  }
  el.innerHTML = parts.join('<br>');
  document.getElementById('about').prepend(el);
}
showCacheStatus();

boot();
