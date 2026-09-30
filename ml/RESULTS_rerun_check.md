# 01~10 重跑結果一致性檢查

比較重跑前後每個模型的 macro-F1，確認訓練可重現。

| 模型 | 重跑前 macro-F1 | 重跑後 macro-F1 | 差異 |
|---|---|---|---|
| 01_bert-base-chinese | — | 0.3587 | — |
| 02_chinese-bert-wwm | — | 0.3625 | — |
| 03_chinese-roberta-wwm-ext | — | 0.3711 | — |
| 04_chinese-legal-electra-base | — | 0.3594 | — |
| 05_lawformer | — | 0.2348 | — |
| 06_bert-base-chinese_merged | — | 0.5166 | — |
| 07_chinese-bert-wwm_merged | — | 0.5142 | — |
| 08_chinese-roberta-wwm-ext_merged | — | 0.5215 | — |
| 09_chinese-legal-electra-base_merged | — | 0.5266 | — |
| 10_lawformer_merged | — | 0.4283 | — |
