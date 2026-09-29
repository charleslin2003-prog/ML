# 五個模型測試集表現比較

以 macro-F1 排序。測試集 3,852 筆，切分時以判決為 group，train/test 無判決重疊。

| 排名 | 資料夾 | 模型 | 類型 | macro-F1 | weighted-F1 | accuracy |
|---|---|---|---|---|---|---|
| 1 | `03_chinese-roberta-wwm-ext` | `hfl/chinese-roberta-wwm-ext` | 通用 | 0.3711 | 0.4341 | 0.4569 |
| 2 | `02_chinese-bert-wwm` | `hfl/chinese-bert-wwm` | 通用 | 0.3625 | 0.4236 | 0.4460 |
| 3 | `04_chinese-legal-electra-base` | `hfl/chinese-legal-electra-base-discriminator` | 法律領域 | 0.3594 | 0.4155 | 0.4315 |
| 4 | `01_bert-base-chinese` | `google-bert/bert-base-chinese` | 通用 | 0.3587 | 0.4353 | 0.4566 |
| 5 | `05_lawformer` | `thunlp/Lawformer` | 法律領域 | 0.2348 | 0.3211 | 0.3710 |

## 各類別 F1

| 模型 | 公然侮辱 | 加重誹謗 | 散布文字誹謗 | 誹謗 | 無罪 |
|---|---|---|---|---|---|
| `03_chinese-roberta-wwm-ext` | 0.6138 | 0.4298 | 0.1675 | 0.2707 | 0.3737 |
| `02_chinese-bert-wwm` | 0.6008 | 0.4082 | 0.1407 | 0.2899 | 0.3732 |
| `04_chinese-legal-electra-base` | 0.5904 | 0.3413 | 0.2079 | 0.2717 | 0.3856 |
| `01_bert-base-chinese` | 0.6052 | 0.4096 | 0.1884 | 0.1663 | 0.4242 |
| `05_lawformer` | 0.5243 | 0.2002 | 0.1029 | 0.0000 | 0.3464 |

## 運行時間與資源使用

| 模型 | 參數量 | 總耗時 | 峰值 GPU 記憶體 | 備註 |
|---|---|---|---|---|
| `03_chinese-roberta-wwm-ext` | 102M | 6.8 分鐘 | 回推值無此數據 | 耗時為回推值，見下方說明 |
| `02_chinese-bert-wwm` | 102M | 6.9 分鐘 | 回推值無此數據 | 耗時為回推值，見下方說明 |
| `04_chinese-legal-electra-base` | 102M | 6.8 分鐘 | 回推值無此數據 | 耗時為回推值，見下方說明 |
| `01_bert-base-chinese` | 102M | 6.9 分鐘 | 回推值無此數據 | 耗時為回推值，見下方說明 |
| `05_lawformer` | 126M | 203.0 分鐘 | 回推值無此數據 | 耗時為回推值，見下方說明 |

> 01~05 的耗時是從訓練佇列的 START/DONE 時間戳回推（訓練當下 `common.py` 還沒加計時），GPU 記憶體無法回推；06~10（相同模型/超參數的合併標籤版）有實測數據可以參考，見 [`RESULTS_merged.md`](RESULTS_merged.md)。

