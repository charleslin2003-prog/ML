# 妨害名譽罪名分類 — 訓練資料蒐集

從司法院裁判書查詢系統（judgment.judicial.gov.tw）爬取「妨害名譽」相關刑事判決書，整理成可用於機器學習的逐句訓練資料：給定一則社群平台言論（貼文/留言），預測構成哪一種罪名（例如公然侮辱、加重誹謗）。

## 環境設置

- Python 3.14.5（建議透過 `uv` 安裝管理），相依套件安裝在 `.venv` 內。
- 主要套件：`playwright`、`pandas`。
- 首次使用需安裝 Playwright 的 Chromium：
  ```
  playwright install chromium
  ```

## 使用方式

### 1. 爬取判決書

```
python dataset/crawler_defamation.py
```

會呼叫 `crawl_exhaustive()`，用司法院進階查詢的「裁判案由」欄位搜尋「妨害名譽」刑事判決，並以裁判日期區間遞迴切分查詢條件，繞過網站單次查詢只顯示前 500 筆結果的限制，盡量抓完整個資料庫。

- 結果即時寫入 `dataset/defamation_judgments.csv`（每處理完一份判決就落盤一次，不怕中途中斷遺失資料）。
- 進度記錄在 `dataset/defamation_judgments.csv.progress.txt`，重新執行同一指令會自動跳過已抓過的日期區間。
- 這是一個會跑很久（視資料量可能數小時到數天）的背景工作，建議用分離的背景程序執行（例如 Windows 上用 PowerShell 的 `Start-Process`），而不是佔用前景終端機。

### 2. 整理成 ML 可用資料

```
python dataset/clean_dataset.py
```

讀取 `defamation_judgments.csv`（不會修改原始檔），依序做以下整理：

1. 去除頭尾空白。
2. 去重複列——逐則表格資料用「原文網址+編號」判斷，整份摘要備援資料用「原文網址」判斷（因為爬蟲是
   append 寫檔且沒有終止條件，重跑可能把同一份判決再寫一次）。
3. 拆成兩份輸出：「言論內容為實際文字（非空、非`未明確提取`佔位字串）且案由有值」的列進
   `defamation_statements_ml_ready.csv`；其餘進 `defamation_unresolved_judgments.csv`。
4. `社交平台` 大小寫統一成大寫（`ig`/`IG` → `IG`）——只是格式統一，不改變內容。
5. **回頭清洗 `涉犯罪名`**：沒有逐則表格的判決，這欄原本是整段主文原文（夾雜被告姓名、刑度）。這裡會
   套用跟爬蟲 `_extract_crime_label()` 完全一樣的關鍵字規則，把它收斂成乾淨罪名標籤（例如「王孋儀犯
   散布文字誹謗罪，處...」→「散布文字誹謗」），這樣即使是舊爬蟲版本抓下來、還沒套用這個正規化的既有
   資料，也能不用重爬就拿到跟新資料一致的乾淨標籤。抓不到已知關鍵字的（多半是「上訴駁回」「自訴不受
   理」這類根本不是罪名的程序性結果）會保留原文，不會用猜的填入不確定的罪名。

輸出：

- `dataset/defamation_statements_ml_ready.csv` — 有實際言論內容與案由的可用資料
- `dataset/defamation_unresolved_judgments.csv` — 沒有抓到可用言論、待日後補強解析的判決

## 檔案說明

| 檔案 | 說明 |
|---|---|
| `CLAUDE.md` | 給 Claude Code 的專案說明（環境設定、執行方式、架構、已知待辦事項）。面向 AI 協作者，內容跟這份 README 有重疊但不完全一樣。 |
| `dataset/crawler_defamation.py` | 爬蟲主程式。`crawl_judgments()` 是單次查詢（受限於網站 500 筆上限，適合快速測試）；`crawl_exhaustive()` 會遞迴切分裁判日期區間繞過這個上限，並支援中斷續爬，是正式抓資料時用的版本。`extract_incident_table()` 優先解析判決書內附的逐則言論表格；解析不到表格時，退回用 `extract_statements_and_crimes()` 以正則從判決全文摘要言論與罪名。`_extract_crime_label()` 是罪名關鍵字標準化規則，`clean_dataset.py` 會直接複用同一份函式維持標籤一致。 |
| `dataset/run_full_crawl.py` | 背景執行 `crawl_exhaustive()` 的進入點（`headless=True`），用於長時間背景爬取，不佔用前景終端機。 |
| `dataset/clean_dataset.py` | 讀取 `defamation_judgments.csv`（不修改原始檔），去重、標準化欄位，拆分成「ML 可用」與「待解析」兩份輸出。 |
| `dataset/defamation_judgments.csv` | 爬蟲的原始輸出（append 寫入，可能含重複列）。有逐則言論表格的判決會展開成多列（一則言論一列）；沒有表格的判決退回整份摘要（一份判決一列，欄位較少）。 |
| `dataset/defamation_judgments.csv.progress.txt` | `crawl_exhaustive()` 的斷點續爬紀錄，記錄哪些裁判日期區間已經抓完；重新執行同一指令會自動跳過已完成的區間。 |
| `dataset/defamation_statements_ml_ready.csv` | `clean_dataset.py` 的正式輸出：已去重、罪名標籤已標準化，且言論內容與案由都有值，可直接用於 ML 訓練。 |
| `dataset/defamation_unresolved_judgments.csv` | `clean_dataset.py` 的另一份輸出：沒抓到可用言論內容或案由的列，留待補強解析規則（或改用 LLM 輔助抽取）後再重新歸類。 |
| `dataset/crawl.log` | 執行爬蟲時的標準輸出紀錄（各判決的解析結果、略過的區間等），方便事後檢查爬取狀況。 |
| `dataset/crawl_err.log` | 執行爬蟲時的標準錯誤輸出紀錄。 |
| `dataset/__pycache__/` | Python 執行時自動產生的位元組碼快取，非原始碼，可忽略。 |

## 輸出欄位

| 欄位 | 說明 |
|---|---|
| 裁判字號 | 判決書標題（法院、年度、字別、案號） |
| 裁判日期 | 判決宣判日期 |
| 案由 | 裁判案由（如「妨害名譽」） |
| 編號 | 該判決書內附言論表格的項次（僅逐則表格資料才有） |
| 日期 / 時間 | 該則言論的實際發文/留言時間（僅逐則表格資料才有） |
| 社交平台 / 平台帳號 | 被告使用的社群平台與帳號（僅逐則表格資料才有） |
| 言論內容 | 被告的實際發文/留言文字 |
| 事實要旨 | 該則言論對應的犯罪事實摘要 |
| 涉犯罪名 | 標準化後的罪名標籤（如公然侮辱、加重誹謗、無罪） |
| 告證編號 | 檢察官起訴時引用的證據編號（僅逐則表格資料才有） |
| 原文網址 | 判決書原文連結 |

## 資料來源的兩種粒度

部分判決書（尤其涉及社群平台的案件）會把檢察官起訴書的逐則言論附表原封不動放進判決全文，這類判決可以直接解析出上述完整欄位、一則言論一列。其餘判決沒有這種表格，只能用正則從判決全文摘要，產生的資料列會缺少日期/時間/社交平台/平台帳號/告證編號，且粒度是整份判決一列而非逐則言論。

## 已知限制

- 罪名標籤標準化（`_extract_crime_label`）是關鍵字比對的簡化做法，遇到單一主文列出多名被告、多項罪名的判決，只會取第一個比對到的關鍵字。
- 目前資料仍以「無逐則表格」的粗粒度列佔多數，且罪名類別分布不一定平衡（公然侮辱案件量通常遠大於加重誹謗）。
- 司法院網站沒有正式的 API，頁面結構純粹靠觀察現況得出（例如查詢結果實際上在 `iframe-data` 子頁框內），未來網站改版可能需要重新調整。
