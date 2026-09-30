# 五個模型測試集表現比較

以 macro-F1 排序。測試集 3,852 筆，切分時以判決為 group，train/test 無判決重疊。

| 排名 | 資料夾 | 模型 | 類型 | macro-F1 | weighted-F1 | accuracy |
|---|---|---|---|---|---|---|
| 1 | `03_chinese-roberta-wwm-ext` | `hfl/chinese-roberta-wwm-ext` | 通用 | 0.3828 | 0.4423 | 0.4603 |
| 2 | `01_bert-base-chinese` | `google-bert/bert-base-chinese` | 通用 | 0.3716 | 0.4332 | 0.4551 |
| 3 | `02_chinese-bert-wwm` | `hfl/chinese-bert-wwm` | 通用 | 0.3699 | 0.4336 | 0.4556 |
| 4 | `04_chinese-legal-electra-base` | `hfl/chinese-legal-electra-base-discriminator` | 法律領域 | 0.3566 | 0.4242 | 0.4408 |
| 5 | `05_lawformer` | `thunlp/Lawformer` | 法律領域 | 0.2343 | 0.3245 | 0.3738 |

## 各類別 F1

| 模型 | 公然侮辱 | 加重誹謗 | 散布文字誹謗 | 誹謗 | 無罪 |
|---|---|---|---|---|---|
| `03_chinese-roberta-wwm-ext` | 0.6103 | 0.4424 | 0.2032 | 0.2757 | 0.3824 |
| `01_bert-base-chinese` | 0.6098 | 0.4264 | 0.1841 | 0.2649 | 0.3728 |
| `02_chinese-bert-wwm` | 0.6149 | 0.4203 | 0.1908 | 0.2490 | 0.3745 |
| `04_chinese-legal-electra-base` | 0.5785 | 0.3952 | 0.1828 | 0.2093 | 0.4172 |
| `05_lawformer` | 0.5244 | 0.2258 | 0.0695 | 0.0000 | 0.3521 |

## 運行時間與資源使用

| 模型 | 參數量 | 總耗時 | 峰值 GPU 記憶體 | 備註 |
|---|---|---|---|---|
| `03_chinese-roberta-wwm-ext` | 102M | 6.5 分鐘 | 4.50 GB | 實測 |
| `01_bert-base-chinese` | 102M | 6.7 分鐘 | 4.50 GB | 實測 |
| `02_chinese-bert-wwm` | 102M | 6.6 分鐘 | 4.50 GB | 實測 |
| `04_chinese-legal-electra-base` | 102M | 6.6 分鐘 | 4.50 GB | 實測 |
| `05_lawformer` | 126M | 103.6 分鐘 | 2.28 GB | 實測 |

