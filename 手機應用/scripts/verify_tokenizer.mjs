/**
 * 驗證瀏覽器端 tokenizer 與 Python 版 HuggingFace tokenizer 的輸出是否完全一致。
 * 這一步非做不可：tokenizer 只要差一個 token，模型的輸入就錯了，
 * 但推論仍會正常跑完並給出一個看似合理的答案 —— 這種錯誤最難從結果上察覺。
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { BertTokenizer } from '../web/tokenizer.js';

const vocab = readFileSync(new URL('../model/vocab.txt', import.meta.url), 'utf8');
const cases = JSON.parse(readFileSync(new URL('./tokenizer_cases.json', import.meta.url), 'utf8'));
const tk = new BertTokenizer(vocab, { doLowerCase: true, maxLen: 256 });

const out = cases.map(c => ({
  text: c.text,
  js_ids: tk.encode(c.text).inputIds.slice(0, c.py_ids.length),
  py_ids: c.py_ids,
}));
writeFileSync(new URL('./tokenizer_js_out.json', import.meta.url),
  JSON.stringify(out, null, 2), 'utf8');

let bad = 0;
for (const r of out) {
  const same = JSON.stringify(r.js_ids) === JSON.stringify(r.py_ids);
  if (!same) {
    bad++;
    console.log(`不一致: ${r.text}`);
    console.log(`  JS: ${r.js_ids.join(',')}`);
    console.log(`  PY: ${r.py_ids.join(',')}`);
  }
}
console.log(`${cases.length} 個測試案例，${cases.length - bad} 個一致，${bad} 個不一致`);
process.exit(bad ? 1 : 0);
