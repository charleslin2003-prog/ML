# 04 · hfl/chinese-legal-electra-base-discriminator

| 項目 | 值 |
|---|---|
| HuggingFace | [hfl/chinese-legal-electra-base-discriminator](https://huggingface.co/hfl/chinese-legal-electra-base-discriminator) |
| HF 下載量 | 416 /月（**中文法律領域模型第一名**） |
| HF likes | 3 |
| 類型 | 法律領域中文 ELECTRA |
| 架構 | ElectraForSequenceClassification · 102M 參數 · vocab 21,128 |

## 為何選它

HuggingFace 上下載量最高的中文法律領域預訓練模型。ELECTRA 的預訓練目標是
**replaced token detection**（判斷每個 token 是否被生成器替換過）而非遮蔽字還原，
樣本效率比 MLM 高；HFL 用中文法律語料（判決書、法條）繼續預訓練出這個版本。

對本任務的期待：判決書文體、法律術語（「意圖散布於眾」、「足生損害於公眾」）已在
預訓練階段見過，理論上比通用模型更快抓到罪名的判準。但要注意領域語料主要是**判決書本文**，
而本資料集的輸入是**社群平台原始言論**（口語、諧音、髒話），兩者文體落差很大 ——
領域預訓練不見得有優勢，這正是要用實驗驗證的假設。

## 超參數

| 參數 | 值 |
|---|---|
| max_length | 256 |
| batch_size | 32 |
| epochs | 3 |
| learning_rate | **5e-5** |

learning_rate 比 BERT 系列高一個檔次 —— ELECTRA discriminator 的 fine-tune 慣例值，
用 2e-5 會收斂不足。

## 執行

```
python ml/04_chinese-legal-electra-base/train.py
```
