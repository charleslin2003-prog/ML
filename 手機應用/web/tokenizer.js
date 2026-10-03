/**
 * 瀏覽器端的中文 BERT WordPiece tokenizer。
 *
 * 為什麼要自己寫：ONNX Runtime Web 只負責跑模型，不含 tokenizer。
 * 中文 BERT 的 vocab 是字級為主（21,128 個 token），但英數字母仍會被切成子詞，
 * 所以不能只做「逐字切開」，必須實作完整的 WordPiece 貪婪最長匹配。
 *
 * 跟 HuggingFace BertTokenizer 對齊的處理順序：
 *   清理控制字元 → NFD正規化去重音 → 轉小寫 → CJK字元前後加空白
 *   → 依空白與標點切分 → 每個片段做 WordPiece
 */
export class BertTokenizer {
  constructor(vocabText, { doLowerCase = true, maxLen = 256 } = {}) {
    this.vocab = new Map();
    vocabText.split('\n').forEach((tok, i) => {
      const t = tok.replace(/\r$/, '');
      if (t.length) this.vocab.set(t, i);
    });
    this.doLowerCase = doLowerCase;
    this.maxLen = maxLen;
    this.unk = '[UNK]';
    this.cls = '[CLS]';
    this.sep = '[SEP]';
    this.pad = '[PAD]';
  }

  /** 中日韓統一表意文字的 Unicode 區段 —— 這些字元要被當成獨立 token */
  static isCJK(cp) {
    return (cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0x3400 && cp <= 0x4dbf) ||
           (cp >= 0x20000 && cp <= 0x2a6df) || (cp >= 0x2a700 && cp <= 0x2b73f) ||
           (cp >= 0x2b740 && cp <= 0x2b81f) || (cp >= 0x2b820 && cp <= 0x2ceaf) ||
           (cp >= 0xf900 && cp <= 0xfaff) || (cp >= 0x2f800 && cp <= 0x2fa1f);
  }

  /**
   * 對齊 HuggingFace BasicTokenizer._is_punctuation：
   * 只有 ASCII 標點區段與 Unicode 類別 P 算標點。
   * 刻意不包含類別 S（符號，含 emoji）—— 原本誤加 \p{S} 會把 emoji 當標點切開，
   * 但 Python 版是把「emoji😡」當成一個詞丟給 WordPiece、整個變成 [UNK]，
   * 兩邊的 token 序列就會不一致。
   */
  static isPunct(ch) {
    const cp = ch.codePointAt(0);
    if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
        (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
    return /\p{P}/u.test(ch);
  }

  /** 基礎切分：清理 → 正規化 → CJK 加空白 → 依空白與標點切開 */
  basicTokenize(text) {
    let s = String(text).replace(/[\u0000�]/g, '')
                        .replace(/[\u0001-\u001f\u007f-\u009f]/g, ' ');
    if (this.doLowerCase) {
      s = s.normalize('NFD').replace(/\p{Mn}/gu, '').toLowerCase();
    }
    let spaced = '';
    for (const ch of s) {
      const cp = ch.codePointAt(0);
      spaced += BertTokenizer.isCJK(cp) ? ` ${ch} ` : ch;
    }
    const out = [];
    for (const chunk of spaced.split(/\s+/)) {
      if (!chunk) continue;
      let buf = '';
      for (const ch of chunk) {
        if (BertTokenizer.isPunct(ch)) {
          if (buf) { out.push(buf); buf = ''; }
          out.push(ch);
        } else buf += ch;
      }
      if (buf) out.push(buf);
    }
    return out;
  }

  /** WordPiece 貪婪最長匹配：找不到就整個詞標成 [UNK] */
  wordpiece(word) {
    if (word.length > 100) return [this.unk];
    const sub = [];
    let start = 0;
    while (start < word.length) {
      let end = word.length, found = null;
      while (start < end) {
        const piece = (start > 0 ? '##' : '') + word.slice(start, end);
        if (this.vocab.has(piece)) { found = piece; break; }
        end -= 1;
      }
      if (found === null) return [this.unk];
      sub.push(found);
      start = end;
    }
    return sub;
  }

  /** 回傳模型需要的三個張量，長度已補齊到 maxLen 或截斷 */
  encode(text) {
    const pieces = [];
    for (const w of this.basicTokenize(text)) pieces.push(...this.wordpiece(w));
    // 保留兩格給 [CLS] 與 [SEP]
    const body = pieces.slice(0, this.maxLen - 2);
    const tokens = [this.cls, ...body, this.sep];

    const ids = tokens.map(t => this.vocab.has(t) ? this.vocab.get(t) : this.vocab.get(this.unk));
    const mask = new Array(ids.length).fill(1);
    const padId = this.vocab.get(this.pad) ?? 0;
    while (ids.length < this.maxLen) { ids.push(padId); mask.push(0); }

    return {
      inputIds: ids,
      attentionMask: mask,
      tokenTypeIds: new Array(this.maxLen).fill(0),
      tokens,
      nRealTokens: tokens.length,
    };
  }
}
