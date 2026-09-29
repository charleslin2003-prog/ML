# 05 · thunlp/Lawformer

| 項目 | 值 |
|---|---|
| HuggingFace | [thunlp/Lawformer](https://huggingface.co/thunlp/Lawformer) |
| HF 下載量 | 251 /月（**中文法律領域模型第二名**） |
| HF likes | 23（法律類最高） |
| 類型 | 法律領域中文 Longformer |
| 架構 | LongformerForSequenceClassification · 126M 參數 · vocab 21,128 |

## 為何選它

清華 THUNLP 出品，論文《Lawformer: A Pre-trained Language Model for Chinese Legal
Long Documents》。在中文裁判文書上預訓練的 **Longformer**，用滑動窗注意力（window 512）
+ 全域注意力取代 full attention，因此能吃到 4,096 token 的長文，是為「整份判決書」設計的。

HF 下載量法律類第二，likes 23（法律類第一），是中文法律 NLP 研究最常引用的模型。

## 這裡的架構錯配，是刻意保留的

本資料集的輸入是**單句社群言論**，中位數僅 8 個字、99 百分位 212 字 —— Lawformer 的
長文優勢完全用不上，還要付出 Longformer 的計算代價。選它純粹因為它是法律領域下載量
第二名，作為「領域對、但架構粒度不對」的對照組：若它輸給通用 BERT，說明對這個
句級任務而言，長文預訓練換不到好處。

## 超參數

| 參數 | 值 | 原因 |
|---|---|---|
| max_length | 512 | Longformer 的 attention window 即 512，低於此無意義 |
| batch_size | **8** | 512 序列 + 滑動窗注意力，8GB VRAM 放不下 32 |
| gradient_checkpointing | **開啟** | 以計算時間換記憶體 |
| epochs | 3 | |
| learning_rate | 2e-5 | |

因此本模型的訓練時間顯著長於其他四個（batch 小 4 倍、序列長 2 倍）。

## 執行

```
python ml/05_lawformer/train.py
```
