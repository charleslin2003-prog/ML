"""往「增加資料量」方向的變體 —— 階段1 的證據顯示資料量比資料乾淨度更重要。

階段1 發現：過濾越兇分數越差（train 18,590→10,999 時 macro-F1 掉 0.02），
而且清洗過的 test set 分數並沒有比較高，代表那些「雜訊」列並不比其他資料難分類。
所以這裡反過來做：把原本被 6 字門檻濾掉、但其實是真實辱罵的短句撈回來
（例如「死賤女人」「你娘的機掰」「吃餿水的」都是 公然侮辱 的典型訊號）。

安全性：test set 直接沿用 data_merged 的原始 test（只做正規化、不增不減），
新撈回來的資料只會進 train/val，而且會嚴格排除任何出現在 val/test 的判決，
確保不會有判決跨切分洩漏。
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
             "散布文字誹謗": "誹謗類", "誹謗": "誹謗類", "無罪": "無罪"}
LABELS = ["公然侮辱", "誹謗類", "無罪"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}


def build(name: str, min_chars: int) -> dict:
    out = ROOT / f"data_{name}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # 既有切分的判決歸屬（以判決為單位，維持跟基準線完全相同的切分邊界）
    splits = {s: pd.read_csv(MERGED_DIR / f"{s}.csv", encoding="utf-8-sig")
              for s in ("train", "val", "test")}
    judgments = {s: set(df["原文網址"]) for s, df in splits.items()}

    # 候選池：用跟 prepare_data_merged.py 相同的邏輯，只把長度門檻放寬
    src = pd.read_csv(SOURCE_CSV, encoding="utf-8-sig", dtype=str)
    src = src[src["涉犯罪名"].isin(MERGE_MAP)].copy()
    src["text"] = src["言論內容"].map(normalize_text)
    src["涉犯罪名"] = src["涉犯罪名"].map(MERGE_MAP)
    src = src.drop_duplicates(["text", "涉犯罪名"])
    conflict = src.groupby("text")["涉犯罪名"].nunique()
    src = src[~src["text"].isin(conflict[conflict > 1].index)]
    src = src[src["text"].str.len() >= min_chars]
    src["label"] = src["涉犯罪名"].map(LABEL2ID)

    # test 完全沿用原本的，只做正規化 —— 這樣才能跟基準線直接比較
    test = splits["test"].copy()
    test["text"] = test["text"].map(normalize_text)

    # val 只取原本就屬於 val 的判決（不新增判決，避免把評估集變大變簡單）
    val = src[src["原文網址"].isin(judgments["val"])]

    # train 可以吃新資料：原本就屬於 train 的判決，加上從未進過任何切分的判決
    assigned = judgments["train"] | judgments["val"] | judgments["test"]
    train = src[src["原文網址"].isin(judgments["train"]) | ~src["原文網址"].isin(assigned)]

    # 保險：確認 train 不含任何 val/test 的判決
    leak = set(train["原文網址"]) & (judgments["val"] | judgments["test"])
    assert not leak, f"判決洩漏到 train：{len(leak)} 份"

    cols = ["text", "label", "涉犯罪名", "原文網址"]
    for s, df in [("train", train), ("val", val), ("test", test)]:
        df[cols].reset_index(drop=True).to_csv(
            out / f"{s}.csv", index=False, encoding="utf-8-sig")

    (out / "label_map.json").write_text(
        json.dumps({"label2id": LABEL2ID, "labels": LABELS}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return {"variant": name, "min_chars": min_chars, "train": len(train),
            "val": len(val), "test": len(test),
            "train_判決數": train["原文網址"].nunique()}


if __name__ == "__main__":
    rows = [build("v4_more_len4", 4), build("v5_more_len2", 2)]
    print(pd.DataFrame(rows).to_string(index=False))
    base = pd.read_csv(MERGED_DIR / "train.csv", encoding="utf-8-sig")
    print(f"\n基準線 train = {len(base)} 筆 / {base['原文網址'].nunique()} 份判決")
