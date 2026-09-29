# 01 · google-bert/bert-base-chinese

| 項目 | 值 |
|---|---|
| HuggingFace | [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese) |
| HF 下載量 | 1,028,517 /月（中文 encoder 第一名） |
| HF likes | 1,469 |
| 類型 | 通用中文 BERT |
| 架構 | BertForSequenceClassification · 102M 參數 · vocab 21,128 |

## 為何選它

Google 官方釋出的中文 BERT，是所有中文文本分類任務事實上的基準線。HuggingFace 上
中文 encoder 模型下載量第一，任何新模型都會拿它當對照組，所以這裡也用它當 baseline。

字元級 tokenizer（單字一 token），對本資料集大量的短句髒話／諧音字（「87」、「塑膠」）
不會因為 subword 切分而失真。

## 超參數

| 參數 | 值 |
|---|---|
| max_length | 256 |
| batch_size | 32 |
| epochs | 3 |
| learning_rate | 2e-5（BERT fine-tune 慣例值） |
| fp16 | 是 |

## 執行

```
python ml/01_bert-base-chinese/train.py
```

產出：`results.json`、`classification_report.txt`、`checkpoints/`（只留最佳 epoch）。
