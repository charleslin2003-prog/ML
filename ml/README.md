# 妨害名譽罪名分類 — 模型實驗

把 `dataset/defamation_statements_ml_ready.csv` 當成 5 類文本分類任務，比較 5 個
HuggingFace 上的中文預訓練模型：**通用中文模型下載量前 3 名** + **中文法律領域模型
下載量前 2 名**。

## 模型選擇（依 HuggingFace API 實際下載量）

### 通用中文分類模型 Top 3

| 資料夾 | 模型 | 下載量 | likes |
|---|---|---|---|
| `01_bert-base-chinese/` | `google-bert/bert-base-chinese` | 1,028,517 | 1,469 |
| `02_chinese-bert-wwm/` | `hfl/chinese-bert-wwm` | 202,016 | 97 |
| `03_chinese-roberta-wwm-ext/` | `hfl/chinese-roberta-wwm-ext` | 56,766 | 427 |

下載量更高的 `shibing624/text2vec-base-chinese`（945,769）與 `BAAI/bge-base-zh-v1.5`
（677,894）刻意排除 —— 它們是 sentence-embedding 模型（`BertModel`，任務是相似度檢索），
不是分類 backbone。

### 中文法律領域模型 Top 2

| 資料夾 | 模型 | 下載量 | likes |
|---|---|---|---|
| `04_chinese-legal-electra-base/` | `hfl/chinese-legal-electra-base-discriminator` | 416 | 3 |
| `05_lawformer/` | `thunlp/Lawformer` | 251 | 23 |

法律領域模型的下載量比通用模型低三個數量級 —— 中文法律 NLP 的預訓練生態本來就很小，
這兩個已經是 HF 上的天花板。

## 任務定義

5 類單標籤分類：`公然侮辱` / `加重誹謗` / `散布文字誹謗` / `誹謗` / `無罪`。
輸入是單句社群平台言論（`言論內容`），輸出是該言論構成的罪名。

## 資料清理（`prepare_data.py`）

原始 CSV 63,044 筆 → 可用 26,721 筆。每一步都是必要的，不是保守起見：

1. **只留 5 個目標標籤** — 原始 102 種標籤裡混了 `上訴駁回。`、`原告之訴駁回。`、
   `再審之聲請駁回。` 等程序性處分，那些根本不是罪名。60,256 筆。
2. **全域去重 `(言論內容, 罪名)`** — 爬蟲是 append-only 可續跑設計，同一句話會重複
   出現；20,730 筆是完全重複。→ 39,526 筆。
3. **丟棄跨標籤衝突的句子** — fallback 抽取路徑會把判決書裡**任何**引號內容當成言論，
   所以法律套語（`意圖散布於眾`、`散布於眾`、`最低限度的道德`）同時掛在多個罪名下。
   這些句子一律整批丟棄（2,736 個不同句子），而不是任選一個標籤 —— 選了就是在教模型
   記住雜訊。→ 33,545 筆。
4. **句長 < 6 字丟棄** — 剩下的碎片（`同仁`、`、`）沒有可判斷的語義。→ 26,721 筆。

### 切分必須以「判決」為 group

26,721 筆只來自 **6,990 份判決**（平均一份判決貢獻 3.8 句，最多 161 句），而同一份判決
爆開的所有句子共用同一個罪名標籤、內容還高度相似。若用一般的 row-level random split，
同一份判決的句子會橫跨 train/test，模型只要記住「這句像是第 1234 號判決」就能答對 ——
測試分數會虛高。

因此用 `GroupShuffleSplit(groups=原文網址)` 切 70/15/15，並在腳本最後斷言
train/test 判決重疊為 0。

| split | 筆數 | 判決數 | 公然侮辱 | 加重誹謗 | 散布文字誹謗 | 誹謗 | 無罪 |
|---|---|---|---|---|---|---|---|
| train | 18,875 | 4,893 | 6,385 | 3,352 | 1,870 | 1,594 | 5,674 |
| val | 3,994 | 1,048 | 1,361 | 722 | 517 | 316 | 1,078 |
| test | 3,852 | 1,049 | 1,268 | 812 | 423 | 292 | 1,057 |

## 執行

```
python ml/prepare_data.py                              # 產生 data/{train,val,test}.csv
python ml/01_bert-base-chinese/train.py                # 逐個訓練（五個資料夾各一支）
python ml/compare.py                                   # 彙總成 RESULTS.md
```

環境：Python 3.14 / torch 2.9.0+cu129 / transformers 5.17.0 / RTX 4060 8GB。
四個 base 模型各約 8 分鐘，Lawformer 因 batch 小 4 倍、序列長 2 倍而顯著更久。

## 產出

每個模型資料夾內：

- `train.py` — 該模型的超參數，實作共用於 `../common.py`
- `README.md` — 模型來源、下載量、選它的理由、超參數設定的原因
- `results.json` — 測試集 accuracy / macro-F1 / weighted-F1 / 各類別 F1 / 混淆矩陣
- `classification_report.txt` — sklearn 完整報告
- `checkpoints/` — 只保留 val macro-F1 最佳的 epoch

跨模型比較見 [`RESULTS.md`](RESULTS.md)。

## 評估指標為何看 macro-F1

類別不平衡（公然侮辱 1,268 筆 vs 誹謗 292 筆，4.3 倍差距）。accuracy 會被大類主導，
`metric_for_best_model` 與模型排序都用 **macro-F1**，讓稀少罪名與大類等權。

## 已知限制

- **標籤本身有雜訊**：`加重誹謗`／`散布文字誹謗`／`誹謗` 三者在刑法上是同一條罪的不同
  態樣（刑法 310 條），判決書用詞不一致，模型要分開這三類本身就接近不可能。若分數
  在這三類上互相混淆，那是標籤定義問題，不是模型問題。
- **`無罪` 不是罪名**：它是判決結果。同一句話可能因為證據不足而無罪，言論內容本身
  不一定帶有可學的訊號，這類的 F1 預期偏低。
- **只有 176 筆來自高品質的 `extract_incident_table()` 路徑**，其餘都是 regex fallback，
  沒有 日期／平台／帳號 等 metadata，也無法當成額外特徵。
