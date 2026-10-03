# 妨害名譽風險判讀器 — 手機應用

把前面訓練好的模型做成**手機上可離線使用的 PWA**。使用者輸入一則網路言論，
應用判讀它較接近「公然侮辱」還是「誹謗」，**所有運算都在裝置上完成，文字不會上傳**。

## 快速開始

```bash
python 手機應用/serve.py
```

- 電腦：開 http://localhost:8000/web/
- 手機：與電腦連同一個 Wi-Fi，開 `http://<電腦區網IP>:8000/web/`（啟動時會印出來）

> Service Worker（離線快取、加到主畫面）只在 `localhost` 或 HTTPS 下才會註冊。
> 用區網 IP 測試時推論功能正常，但離線功能要部署到 HTTPS 網域才會生效。

## 這個資料夾裝了什麼

```
手機應用/
├── web/                     PWA 本體（部署時放上靜態網站空間即可）
│   ├── index.html           介面：免責聲明置頂、輸入框、結果呈現
│   ├── app.js               流程控制與結果渲染
│   ├── inference.js         ONNX Runtime Web 推論引擎
│   ├── tokenizer.js         純 JS 實作的中文 BERT WordPiece tokenizer
│   ├── sw.js                Service Worker：模型 cache-first、程式碼 network-first
│   ├── manifest.json        PWA 設定（可加到主畫面）
│   └── styles.css
├── model/
│   ├── defamation_electra_small.int8.onnx   24.5 MB，INT8 量化
│   ├── vocab.txt                            21,128 個 token
│   └── model_meta.json                      標籤、尺寸、驗證結果
├── scripts/                 訓練、匯出、驗證
└── notebook/                完整技術報告（功能說明/文獻回顧/方法/成果/結論）
```

## 為什麼選 electra-small

手機部署的瓶頸是**下載體積**，不是準確度。實測五個候選模型：

| 模型 | 參數量 | accuracy | INT8 體積 |
|---|---|---|---|
| chinese-legal-electra-base | 102.3M | 0.7543 | ~98 MB |
| chinese-roberta-wwm-ext | 102.3M | 0.7507 | ~98 MB |
| **chinese-electra-180g-small-ex** | **24.6M** | **0.7419** | **24.5 MB** |
| rbt6 | 59.7M | 0.7432 | ~57 MB |
| rbt3 | 38.5M | 0.7253 | ~37 MB |

electra-small 只少 1.2 個百分點，但**體積小 4 倍**。對一個要讓人在手機上下載的工具，
這筆交換很划算。

## 量化的代價是多少（實測，不是估計）

| 階段 | accuracy | macro-F1 | 對原模型 |
|---|---|---|---|
| PyTorch fp32 | 0.7419 | 0.7311 | — |
| ONNX fp32 | 0.7419 | 0.7311 | ±0.0000 |
| **ONNX INT8** | **0.7409** | **0.7314** | **−0.0010** |

INT8 量化讓體積從 94.2 MB 降到 24.5 MB（壓縮 3.84 倍），accuracy 只掉 **0.1 個百分點**，
與原模型的預測一致率 96.77%（測試集 3,065 筆）。

## 正確性怎麼驗證的

跨語言移植最容易出錯的是 tokenizer —— 切錯一個 token，模型輸入就錯了，
但推論仍會正常跑完並給出看似合理的答案，**從結果上完全看不出來**。所以分三層驗證：

1. **`verify_tokenizer.mjs`**：12 個邊界案例（中英混合、全形字、emoji、多重空白、標點），
   比對 JS 與 Python 的 token id 序列 → **12/12 完全一致**
2. **`verify_e2e.mjs`**：JS 的 tokenizer + ONNX 推論，對照 Python 的完整流程
   → **8/8 案例的 logits 差異為 0**
3. **`test_browser.py`**：Playwright 模擬 iPhone 尺寸實際開啟網頁，
   驗證模型載入、推論、介面渲染 → **零主控台錯誤，單次推論 81~132 ms**

> 驗證過程中真的抓到一個 bug：原本 `isPunct` 用了 `\p{S}`（符號類）會把 emoji
> 當成標點切開，但 HuggingFace 的 `_is_punctuation` 只認 ASCII 標點與 Unicode `P` 類。
> 這個差異只在含 emoji 的輸入上才會顯現，若沒做比對測試根本不會發現。

## 部署到 HTTPS（讓離線功能生效）

```bash
# 以 GitHub Pages 為例：把 手機應用/ 的內容推到 gh-pages 分支
# 網址會是 https://<帳號>.github.io/<repo>/web/
```

部署後首次開啟會下載 24.5 MB 模型，之後 Service Worker 會快取起來，
關掉網路也能用，也可以「加到主畫面」當成一般 App 開。

## 已知限制

- **準確率 74%**：每四則就有一則判斷是錯的，不能當法律意見用
- **只能在「假設已成案」的前提下分類**：訓練資料全來自已被起訴判決的案件，
  沒有「正常不會被告的留言」當負樣本，所以模型**無法判斷一句話會不會被告**。
  丟一句無害的話進去，它仍然會輸出一個罪名
- **任務本質是多標籤**：同一則貼文可以同時構成公然侮辱與誹謗（實測在高品質
  逐句標籤資料中佔 25%），但目前是單標籤輸出
- **iOS Safari 的 Cache Storage 配額較嚴格**，長期不用可能被系統清掉需重新下載
