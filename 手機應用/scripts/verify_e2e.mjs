/**
 * 端到端驗證：用 JS 的 tokenizer + ONNX 推論，跟 Python 的結果比對。
 * tokenizer 一致不代表整條鏈一致 —— 張量型別、padding、attention mask
 * 任何一個環節寫錯都會讓結果偏掉，所以要整條驗證過。
 */
import { readFileSync, writeFileSync } from 'node:fs';
import ort from 'onnxruntime-node';
import { fileURLToPath } from 'node:url';
import { BertTokenizer } from '../web/tokenizer.js';

const base = new URL('../model/', import.meta.url);
const meta = JSON.parse(readFileSync(new URL('model_meta.json', base), 'utf8'));
const tk = new BertTokenizer(readFileSync(new URL('vocab.txt', base), 'utf8'),
  { doLowerCase: meta.do_lower_case, maxLen: meta.max_length });
// 專案路徑含中文，URL.pathname 會是百分號編碼，必須用 fileURLToPath 還原
const sess = await ort.InferenceSession.create(
  fileURLToPath(new URL(`${meta.model_name}.int8.onnx`, base)));

const texts = JSON.parse(readFileSync(new URL('./e2e_cases.json', import.meta.url), 'utf8'));
const softmax = a => { const m = Math.max(...a); const e = a.map(v => Math.exp(v - m));
  const s = e.reduce((x, y) => x + y, 0); return e.map(v => v / s); };

const out = [];
for (const t of texts) {
  const enc = tk.encode(t);
  const L = meta.max_length;
  const big = a => BigInt64Array.from(a.map(BigInt));
  const r = await sess.run({
    input_ids: new ort.Tensor('int64', big(enc.inputIds), [1, L]),
    attention_mask: new ort.Tensor('int64', big(enc.attentionMask), [1, L]),
    token_type_ids: new ort.Tensor('int64', big(enc.tokenTypeIds), [1, L]),
  });
  const logits = Array.from(r.logits.data).map(Number);
  const p = softmax(logits);
  out.push({ text: t, logits, probs: p,
             label: meta.labels[p.indexOf(Math.max(...p))] });
}
writeFileSync(new URL('./e2e_js_out.json', import.meta.url), JSON.stringify(out, null, 2), 'utf8');
console.log('JS 端完成，結果已寫出');
for (const o of out) console.log(`  ${o.label}  ${(Math.max(...o.probs)*100).toFixed(1)}%  ${o.text}`);
