"""判決層級任務：把預測粒度對齊到標籤粒度。

問題根源：95.8% 的資料來自 fallback 抽取，一份判決抓出的所有句子（最多 161 句）
全部共用同一個標籤 —— 那個標籤來自判決主文，描述的是「這份判決關於什麼罪」，
而不是「這句話構成什麼罪」。用句子層級訓練，等於在教模型學一個它拿不到的答案。

這裡改成：把同一份判決的所有句子合併成一個樣本，標籤用該判決的罪名。
這樣標籤就是**正確**的（不再是把一個標籤硬攤到 161 句上），雜訊直接消失。

代價：樣本數從 1.6 萬降到約 4 千，而且實務用途變成「判斷整個案件」而非「判斷單則留言」。
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data_v3 import normalize_text

ROOT = Path(__file__).resolve().parent
SOURCE_CSV = ROOT.parent.parent / "dataset" / "defamation_statements_ml_ready.csv"
MERGED_DIR = ROOT.parent / "data_merged"
MERGE_MAP = {"公然侮辱": "公然侮辱", "加重誹謗": "誹謗類",
             "散布文字誹謗": "誹謗類", "誹謗": "誹謗類"}
LABELS = ["公然侮辱", "誹謗類"]
LABEL2ID = {n: i for i, n in enumerate(LABELS)}
MAX_SENT = 30          # 一份判決最多取 30 句，避免少數判決的 161 句灌爆輸入


def main():
    out = ROOT / "data_judgment"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    judgments = {s: set(pd.read_csv(MERGED_DIR / f"{s}.csv", encoding="utf-8-sig")["原文網址"])
                 for s in ("train", "val", "test")}

    src = pd.read_csv(SOURCE_CSV, encoding="utf-8-sig", dtype=str)
    src = src[src["涉犯罪名"].isin(MERGE_MAP)].copy()
    src["text"] = src["言論內容"].map(normalize_text)
    src["涉犯罪名"] = src["涉犯罪名"].map(MERGE_MAP)
    src = src[src["text"].str.len() >= 2].drop_duplicates(["原文網址", "text"])

    # 一份判決內若同時出現兩個罪名（表格路徑才會發生），無法用單標籤表達，略過
    multi = src.groupby("原文網址")["涉犯罪名"].nunique()
    src = src[src["原文網址"].isin(multi[multi == 1].index)]

    agg = (src.groupby(["原文網址", "涉犯罪名"])["text"]
           .apply(lambda s: " ".join(s.head(MAX_SENT)))
           .reset_index())
    agg["label"] = agg["涉犯罪名"].map(LABEL2ID)
    agg["n_chars"] = agg["text"].str.len()

    frames = {}
    assigned = set().union(*judgments.values())
    for s in ("train", "val", "test"):
        if s == "train":
            m = agg["原文網址"].isin(judgments["train"]) | ~agg["原文網址"].isin(assigned)
        else:
            m = agg["原文網址"].isin(judgments[s])
        frames[s] = agg[m].reset_index(drop=True)

    leak = set(frames["train"]["原文網址"]) & (
        set(frames["val"]["原文網址"]) | set(frames["test"]["原文網址"]))
    assert not leak, f"判決洩漏：{len(leak)}"

    for s, df in frames.items():
        df[["text", "label", "涉犯罪名", "原文網址"]].to_csv(
            out / f"{s}.csv", index=False, encoding="utf-8-sig")
        print(f"{s:6s} {len(df):5d} 份判決  標籤分布 {df['涉犯罪名'].value_counts().to_dict()}  "
              f"文字長度中位數 {int(df['n_chars'].median())}")

    (out / "label_map.json").write_text(
        json.dumps({"label2id": LABEL2ID, "labels": LABELS}, ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
