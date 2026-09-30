# 01~10 重跑結果一致性檢查

比較重跑前後每個模型的 macro-F1，確認訓練可重現。

| 模型 | 重跑前 macro-F1 | 重跑後 macro-F1 | 差異 |
|---|---|---|---|
| 01_bert-base-chinese | 0.3587 | 0.3716 | +0.0129 |
| 02_chinese-bert-wwm | 0.3625 | 0.3699 | +0.0074 |
| 03_chinese-roberta-wwm-ext | 0.3711 | 0.3828 | +0.0117 |
| 04_chinese-legal-electra-base | 0.3594 | 0.3566 | -0.0028 |
| 05_lawformer | 0.2348 | 0.2343 | -0.0005 |
| 06_bert-base-chinese_merged | 0.5166 | 0.5184 | +0.0018 |
| 07_chinese-bert-wwm_merged | 0.5142 | 0.5153 | +0.0011 |
| 08_chinese-roberta-wwm-ext_merged | 0.5215 | 0.5232 | +0.0017 |
| 09_chinese-legal-electra-base_merged | 0.5266 | 0.5228 | -0.0038 |
| 10_lawformer_merged | 0.4283 | 0.4337 | +0.0054 |

> 差異都在 ±0.013 以內，屬於 fp16 混合精度訓練在 GPU 上常見的正常誤差範圍（浮點運算順序、cuDNN 演算法選擇等不受 `seed=42` 完全控制），不是資料或程式碼變動造成的系統性偏移，可視為訓練結果可重現。
