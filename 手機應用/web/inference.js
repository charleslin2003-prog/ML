/**
 * 裝置端推論引擎：載入量化後的 ONNX 模型，在瀏覽器裡跑完整的分類流程。
 * 使用者輸入的文字從頭到尾不會離開這台裝置。
 */
import { BertTokenizer } from './tokenizer.js';

// 網頁放在 web/，模型放在同層的 model/，所以要往上一層找
const MODEL_DIR = '../model';

export class DefamationClassifier {
  constructor() {
    this.session = null;
    this.tokenizer = null;
    this.meta = null;
    this.ready = false;
  }

  /** onProgress(stage, loadedBytes, totalBytes) 讓 UI 顯示下載進度 */
  async init(onProgress = () => {}) {
    // 向瀏覽器申請持久化儲存，降低模型被清掉的機率。
    // 注意這對 iOS Safari 的 ITP「7 天未互動即清除」規則幫助有限 ——
    // 在 iOS 上真正有效的是讓使用者「加入主畫面」，那才會被明文豁免。
    if (navigator.storage?.persist) {
      try { await navigator.storage.persist(); } catch { /* 不支援就算了 */ }
    }
    onProgress('讀取模型設定', 0, 1);
    this.meta = await (await fetch(`${MODEL_DIR}/model_meta.json`)).json();

    onProgress('載入詞彙表', 0, 1);
    const vocabText = await (await fetch(`${MODEL_DIR}/vocab.txt`)).text();
    this.tokenizer = new BertTokenizer(vocabText, {
      doLowerCase: this.meta.do_lower_case,
      maxLen: this.meta.max_length,
    });

    // 模型檔案較大，用串流讀取才能回報進度；下載完由 Service Worker 快取，之後離線可用
    const url = `${MODEL_DIR}/${this.meta.model_name}.int8.onnx`;
    const resp = await fetch(url);
    const total = Number(resp.headers.get('content-length')) || 0;
    const reader = resp.body.getReader();
    const chunks = [];
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      onProgress('下載模型', loaded, total);
    }
    const buf = new Uint8Array(loaded);
    let off = 0;
    for (const c of chunks) { buf.set(c, off); off += c.length; }

    onProgress('初始化推論引擎', 0, 1);
    // 選 wasm 而非 webgpu 有具體理由：ONNX Runtime 的 WebGPU 運算子清單不含
    // MatMulInteger 與 DynamicQuantizeLinear，INT8 量化模型的矩陣乘法會退回 CPU，
    // 反而比直接用 wasm 慢。要用 WebGPU 就得改走 fp16 或 q4，那又會讓體積變大。
    // 不設 wasmPaths：ORT 預設會相對於 ort.min.js 自身的位置找 .wasm，
    // 而我們已經把兩者放在同一個 vendor/ 目錄下，設了反而會變成 vendor/vendor/。
    ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 2, 4);
    ort.env.wasm.simd = true;
    this.session = await ort.InferenceSession.create(buf.buffer, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });

    this.ready = true;
    onProgress('完成', 1, 1);
    return this.meta;
  }

  static softmax(arr) {
    const max = Math.max(...arr);
    const exp = arr.map(v => Math.exp(v - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(v => v / sum);
  }

  async classify(text) {
    if (!this.ready) throw new Error('模型尚未載入完成');
    const t0 = performance.now();
    const enc = this.tokenizer.encode(text);
    const L = this.meta.max_length;
    const big = a => BigInt64Array.from(a.map(BigInt));

    const feeds = {
      input_ids: new ort.Tensor('int64', big(enc.inputIds), [1, L]),
      attention_mask: new ort.Tensor('int64', big(enc.attentionMask), [1, L]),
      token_type_ids: new ort.Tensor('int64', big(enc.tokenTypeIds), [1, L]),
    };
    const out = await this.session.run(feeds);
    const logits = Array.from(out.logits.data).map(Number);
    const probs = DefamationClassifier.softmax(logits);

    const ranked = this.meta.labels
      .map((label, i) => ({ label, prob: probs[i] }))
      .sort((a, b) => b.prob - a.prob);

    return {
      top: ranked[0],
      all: ranked,
      nTokens: enc.nRealTokens,
      latencyMs: Math.round(performance.now() - t0),
    };
  }
}
