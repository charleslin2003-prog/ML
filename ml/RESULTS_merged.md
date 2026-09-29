# 合併標籤（3 類）vs 原始 5 類任務 比較

把 `加重誹謗`／`散布文字誹謗`／`誹謗` 合併成單一個「誹謗類」後，測試同一組模型的 macro-F1 是否回升。詳見 [`README_合併標籤實驗.md`](README_合併標籤實驗.md)。

| 模型 | 5 類 macro-F1 | 3 類(合併) macro-F1 | 變化 | 5 類 acc | 3 類 acc |
|---|---|---|---|---|---|
| bert-base-chinese | 0.3587 | 0.5166 | +0.1579 | 0.4566 | 0.5287 |
| chinese-bert-wwm | 0.3625 | 0.5142 | +0.1517 | 0.4460 | 0.5345 |
| chinese-roberta-wwm-ext | 0.3711 | 0.5215 | +0.1503 | 0.4569 | 0.5433 |
| chinese-legal-electra-base | 0.3594 | 0.5266 | +0.1672 | 0.4315 | 0.5392 |
| Lawformer | 0.2348 | 0.4283 | +0.1936 | 0.3710 | 0.4618 |

## 合併任務各類別 F1

| 模型 | 公然侮辱 | 誹謗類 | 無罪 |
|---|---|---|---|
| bert-base-chinese | 0.5479 | 0.6048 | 0.3972 |
| chinese-bert-wwm | 0.5493 | 0.6277 | 0.3657 |
| chinese-roberta-wwm-ext | 0.5574 | 0.6369 | 0.3701 |
| chinese-legal-electra-base | 0.5455 | 0.6139 | 0.4205 |
| Lawformer | 0.4793 | 0.5543 | 0.2514 |

## 運行時間與資源使用（06~10，common.py 加計時後的實測數據）

| 模型 | 參數量 | 總耗時 | Trainer train_runtime | 峰值 GPU 記憶體(已配置) | 峰值 GPU 記憶體(已保留) |
|---|---|---|---|---|---|
| bert-base-chinese | 102M | 6.8 分鐘 | 6.6 分鐘 | 4.52 GB | 5.87 GB |
| chinese-bert-wwm | 102M | 7.1 分鐘 | 6.9 分鐘 | 4.52 GB | 5.87 GB |
| chinese-roberta-wwm-ext | 102M | 7.1 分鐘 | 6.9 分鐘 | 4.52 GB | 5.87 GB |
| chinese-legal-electra-base | 102M | 6.5 分鐘 | 6.3 分鐘 | 4.52 GB | 5.87 GB |
| Lawformer | 126M | 103.9 分鐘 | 102.2 分鐘 | 2.28 GB | 3.09 GB |

> 「總耗時」含 tokenizer/模型載入、tokenize、train、test 預測的完整流程；「train_runtime」只算 Trainer 內部的訓練迴圈，兩者差距大致就是資料前處理與評估花的時間。GPU 記憶體是 fp16 + `per_device_train_batch_size` 各自不同下的峰值，Lawformer 用 512 長度、batch 8 且開了 gradient checkpointing，其餘四個用 256 長度、batch 32，數字不能只看「誰的記憶體用得少」，要連同 batch size 一起看。
