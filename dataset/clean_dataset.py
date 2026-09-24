"""整理爬蟲輸出的原始 CSV，切分成「可用於 ML 的逐句資料」與「尚待人工/規則重新解析的判決」。

不修改原始檔（defamation_judgments.csv）。因為爬蟲是 append 模式且沒有上限，
可能重複爬到同一份判決，所以這裡一律先做去重再篩選。

用法：
    python dataset/clean_dataset.py
"""
import io
import sys

import pandas as pd

from crawler_defamation import _extract_crime_label

# Windows 主控台預設編碼（cp950）印不出判決書裡的罕見字元（如帶圈數字），改用 UTF-8 並容錯，避免整理途中崩潰。
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RAW_PATH = 'defamation_judgments.csv'
ML_READY_PATH = 'defamation_statements_ml_ready.csv'
UNRESOLVED_PATH = 'defamation_unresolved_judgments.csv'

UNRESOLVED_PLACEHOLDER = '未明確提取'

STR_COLUMNS = [
    '裁判字號', '裁判日期', '案由', '日期', '時間',
    '社交平台', '平台帳號', '言論內容', '事實要旨', '涉犯罪名', '原文網址',
]


def _strip_strings(df):
  for col in STR_COLUMNS:
    df[col] = df[col].astype('string').str.strip()
  return df


def clean_dataset(raw_path=RAW_PATH):
  df = pd.read_csv(raw_path, encoding='utf-8-sig')
  raw_count = len(df)

  df = _strip_strings(df)

  # 爬蟲會無止盡地爬、且用 append 寫檔，重跑可能把同一份判決的同一則言論再寫一次。
  # 逐則列（編號非空）用「原文網址+編號」判斷是否重複；備援整份列（編號空）用「原文網址」判斷。
  has_id = df['編號'].notna()
  dedup_incident = df[has_id].drop_duplicates(subset=['原文網址', '編號'], keep='first')
  dedup_fallback = df[~has_id].drop_duplicates(subset=['原文網址'], keep='first')
  df = pd.concat([dedup_incident, dedup_fallback], ignore_index=True)
  after_dedup_count = len(df)

  # 只要言論內容不是佔位字串/空值、且案由有值，就算可用的 ML 資料
  # （不強制要求有逐則表格編號——沒有表格但正則有抓到引號原文的整份判決摘要列也收錄）。
  usable_mask = (
      df['言論內容'].notna()
      & (df['言論內容'] != UNRESOLVED_PLACEHOLDER)
      & (df['言論內容'].str.len() > 0)
      & df['案由'].notna()
      & (df['案由'].str.len() > 0)
  )

  ml_ready = df[usable_mask].copy()
  unresolved = df[~usable_mask].copy()

  # 編號、告證編號在可用資料中不應有空值，可以安全轉成整數方便後續處理。
  for col in ('編號', '告證編號'):
    ml_ready[col] = ml_ready[col].astype('Int64')

  # 社交平台大小寫不一致（ig/IG/fb/FB），統一成大寫，這是格式標準化，不是竄改內容。
  ml_ready['社交平台'] = ml_ready['社交平台'].str.upper()

  # 沒有逐則表格的判決，涉犯罪名欄位是整段主文原文（夾雜被告姓名、刑度等雜訊）。
  # 用爬蟲既有的 _extract_crime_label 規則收斂成乾淨標籤，跟新爬的資料用同一套邏輯，
  # 這樣既有 CSV 不用重爬也能拿到跟新資料一致的罪名標籤。抓不到已知關鍵字的就保留原文，
  # 不用猜測填入不確定的罪名。
  ml_ready['涉犯罪名'] = ml_ready['涉犯罪名'].apply(_extract_crime_label)

  # 不可用的判決列，維持原欄位但拿掉逐則專屬欄位（本來就是空的，避免造成「有欄位但無意義」的誤導）。
  unresolved = unresolved.drop(columns=['編號', '日期', '時間', '社交平台', '平台帳號', '告證編號'])

  ml_ready.to_csv(ML_READY_PATH, index=False, encoding='utf-8-sig')
  unresolved.to_csv(UNRESOLVED_PATH, index=False, encoding='utf-8-sig')

  print(f'原始資料：{raw_count} 列')
  print(f'去重後：{after_dedup_count} 列（移除 {raw_count - after_dedup_count} 筆重複）')
  print(f'可用於 ML 的逐句資料：{len(ml_ready)} 列 -> {ML_READY_PATH}')
  print(f'  來自 {ml_ready["裁判字號"].nunique()} 份判決')
  print(f'  罪名分布：\n{ml_ready["涉犯罪名"].value_counts().to_string()}')
  print(f'  社交平台分布：\n{ml_ready["社交平台"].value_counts().to_string()}')
  na_cols = ml_ready.isna().sum()
  na_cols = na_cols[na_cols > 0]
  if len(na_cols):
    print(f'  仍存在的空值（未填補，如實保留）：\n{na_cols.to_string()}')
  print(f'尚待重新解析的判決（整份判決退回摘要，無逐句資料）：{len(unresolved)} 列 -> {UNRESOLVED_PATH}')
  print(f'  來自 {unresolved["裁判字號"].nunique()} 份判決')

  return ml_ready, unresolved


if __name__ == '__main__':
  clean_dataset()
