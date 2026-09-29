# 02 · hfl/chinese-bert-wwm

| 項目 | 值 |
|---|---|
| HuggingFace | [hfl/chinese-bert-wwm](https://huggingface.co/hfl/chinese-bert-wwm) |
| HF 下載量 | 202,016 /月（中文 encoder 第二名） |
| HF likes | 97 |
| 類型 | 通用中文 BERT（Whole Word Masking） |
| 架構 | BertForSequenceClassification · 102M 參數 · vocab 21,128 |

## 為何選它

哈工大訊飛聯合實驗室（HFL）的中文 BERT，把 Google 的隨機 subword masking 換成
**整詞遮蔽（Whole Word Masking）**：預訓練時遮的是完整詞彙而非單字，逼模型學到詞級語義
而不是靠上下文補字。這是中文 BERT 最廣為採用的改良，`hfl/*` 整個系列的起點。

對本任務的意義：罪名判斷仰賴「誰罵了誰什麼詞」的詞級語義（「智障」vs「智慧」差一字、
語義天差地遠），整詞遮蔽的預訓練目標理論上比字級隨機遮蔽更貼近這個需求。

## 超參數

與 01 完全相同（max_length 256 / batch 32 / 3 epochs / lr 2e-5），
差異只有預訓練權重，方便單獨歸因整詞遮蔽的貢獻。

## 執行

```
python ml/02_chinese-bert-wwm/train.py
```
