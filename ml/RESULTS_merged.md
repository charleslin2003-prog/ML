# 合併標籤（3 類）vs 原始 5 類任務 比較

把 `加重誹謗`／`散布文字誹謗`／`誹謗` 合併成單一個「誹謗類」後，測試同一組模型的 macro-F1 是否回升。詳見 [`README_合併標籤實驗.md`](README_合併標籤實驗.md)。

| 模型 | 5 類 macro-F1 | 3 類(合併) macro-F1 | 變化 | 5 類 acc | 3 類 acc |
|---|---|---|---|---|---|
| bert-base-chinese | 0.3716 | 0.5184 | +0.1468 | 0.4551 | 0.5315 |
| chinese-bert-wwm | 0.3699 | 0.5153 | +0.1454 | 0.4556 | 0.5325 |
| chinese-roberta-wwm-ext | 0.3828 | 0.5232 | +0.1404 | 0.4603 | 0.5396 |
| chinese-legal-electra-base | 0.3566 | 0.5228 | +0.1662 | 0.4408 | 0.5336 |
| Lawformer | 0.2343 | 0.4337 | +0.1994 | 0.3738 | 0.4659 |

## 合併任務各類別 F1

| 模型 | 公然侮辱 | 誹謗類 | 無罪 |
|---|---|---|---|
| bert-base-chinese | 0.5448 | 0.6158 | 0.3946 |
| chinese-bert-wwm | 0.5415 | 0.6274 | 0.3770 |
| chinese-roberta-wwm-ext | 0.5562 | 0.6252 | 0.3882 |
| chinese-legal-electra-base | 0.5424 | 0.6055 | 0.4206 |
| Lawformer | 0.4872 | 0.5545 | 0.2594 |

## 運行時間與資源使用（06~10，common.py 加計時後的實測數據）

| 模型 | 參數量 | 總耗時 | Trainer train_runtime | 峰值 GPU 記憶體(已配置) | 峰值 GPU 記憶體(已保留) |
|---|---|---|---|---|---|
| bert-base-chinese | 102M | 6.5 分鐘 | 6.3 分鐘 | 4.52 GB | 5.87 GB |
| chinese-bert-wwm | 102M | 6.5 分鐘 | 6.3 分鐘 | 4.52 GB | 5.87 GB |
| chinese-roberta-wwm-ext | 102M | 6.5 分鐘 | 6.3 分鐘 | 4.52 GB | 5.87 GB |
| chinese-legal-electra-base | 102M | 6.5 分鐘 | 6.3 分鐘 | 4.52 GB | 5.87 GB |
| Lawformer | 126M | 103.8 分鐘 | 102.1 分鐘 | 2.28 GB | 3.09 GB |

> 「總耗時」含 tokenizer/模型載入、tokenize、train、test 預測的完整流程；「train_runtime」只算 Trainer 內部的訓練迴圈，兩者差距大致就是資料前處理與評估花的時間。GPU 記憶體是 fp16 + `per_device_train_batch_size` 各自不同下的峰值，Lawformer 用 512 長度、batch 8 且開了 gradient checkpointing，其餘四個用 256 長度、batch 32，數字不能只看「誰的記憶體用得少」，要連同 batch size 一起看。
