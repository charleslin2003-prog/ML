# 03 · hfl/chinese-roberta-wwm-ext

| 項目 | 值 |
|---|---|
| HuggingFace | [hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext) |
| HF 下載量 | 56,766 /月（中文 encoder 第三名） |
| HF likes | 427（`hfl/*` 系列中最高） |
| 類型 | 通用中文 RoBERTa（整詞遮蔽 + 擴充語料） |
| 架構 | BertForSequenceClassification · 102M 參數 · vocab 21,128 |

## 為何選它

在 02 的整詞遮蔽之上再加兩件事：改用 **RoBERTa 訓練配方**（去掉 NSP 任務、動態遮蔽、
更大 batch）、並用 **ext 擴充語料**（5.4B 詞，遠多於原版的中文維基）。

雖然名字叫 RoBERTa，權重實際上是 BERT 架構（載入後就是 `BertForSequenceClassification`），
差別純粹在預訓練配方與語料量。中文 NLP 論文最常拿來當 SOTA baseline 的就是這個，
likes 數（427）也是三個通用模型裡最高的 —— 下載量輸給前兩名，但被實際研究引用得最多。

## 超參數

與 01、02 完全相同，三個通用模型構成可控的對照實驗。

## 執行

```
python ml/03_chinese-roberta-wwm-ext/train.py
```
