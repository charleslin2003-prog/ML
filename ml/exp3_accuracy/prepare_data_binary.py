"""第四種任務設定：二分類（公然侮辱 vs 誹謗類），拿掉「無罪」。

為什麼拿掉無罪：它是「判決結果」而不是「言論性質」。同一句「你這個智障」可能因為
不具公然性、證據不足、可受公評、善意發表言論等理由而無罪 —— 這些資訊**完全不在
言論文字裡**，模型無從判斷。實測也印證了：無罪類的 recall 只有 35~47%，
而且它會搶走另外兩個真罪名類別 600~700 筆預測。

拿掉之後任務回到 CLAUDE.md 寫的專案原始目標：「預測這則言論構成哪個罪名」。

切分一律沿用 data_merged 既有的判決歸屬，確保沒有判決跨切分洩漏。
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data_v3 import normalize_text, noise_mask

ROOT = Path(__file__).resolve().parent
SOURCE_CSV = ROOT.parent.parent / "dataset" / "defamation_statements_ml_ready.csv"
MERGED_DIR = ROOT.parent / "data_merged"

MERGE_MAP = {"公然侮辱": "公然侮辱", "加重誹謗": "誹謗類",
             "散布文字誹謗": "誹謗類", "誹謗": "誹謗類"}   # 無罪 直接不收
LABELS = ["公然侮辱", "誹謗類"]
LABEL2ID = {n: i for i, n in enumerate(LABELS)}


def build(name: str, normalize: bool, denoise: bool, min_chars: int,
          from_source: bool, clean_test: bool) -> dict:
    out = ROOT / f"data_{name}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    splits = {s: pd.read_csv(MERGED_DIR / f"{s}.csv", encoding="utf-8-sig")
              for s in ("train", "val", "test")}
    judgments = {s: set(df["原文網址"]) for s, df in splits.items()}

    if from_source:
        # 從原始 CSV 重建，可以把被 6 字門檻濾掉的短句也撈回來
        src = pd.read_csv(SOURCE_CSV, encoding="utf-8-sig", dtype=str)
        src = src[src["涉犯罪名"].isin(MERGE_MAP)].copy()
        src["text"] = src["言論內容"].map(normalize_text)
        src["涉犯罪名"] = src["涉犯罪名"].map(MERGE_MAP)
        src = src.drop_duplicates(["text", "涉犯罪名"])
        conflict = src.groupby("text")["涉犯罪名"].nunique()
        src = src[~src["text"].isin(conflict[conflict > 1].index)]
        pool = src
    else:
        pool = pd.concat([df.assign(_s=s) for s, df in splits.items()], ignore_index=True)
        pool = pool[pool["涉犯罪名"].isin(LABELS)].copy()
        if normalize:
            pool["text"] = pool["text"].map(normalize_text)

    def shape(df, is_test):
        df = df.copy()
        apply_filters = (not is_test) or clean_test
        if apply_filters:
            if denoise:
                df = df[~noise_mask(df["text"].astype(str))]
            df = df[df["text"].astype(str).str.len() >= min_chars]
            df = df.drop_duplicates(["text", "涉犯罪名"])
        df["label"] = df["涉犯罪名"].map(LABEL2ID)
        return df[["text", "label", "涉犯罪名", "原文網址"]].reset_index(drop=True)

    assigned = judgments["train"] | judgments["val"] | judgments["test"]
    if from_source:
        train = pool[pool["原文網址"].isin(judgments["train"]) | ~pool["原文網址"].isin(assigned)]
        val = pool[pool["原文網址"].isin(judgments["val"])]
        # test 一律沿用既有切分（只做正規化），不吃新撈回來的資料 ——
        # 否則各變體的測試集大小不同就無法互相比較
        test = splits["test"][splits["test"]["涉犯罪名"].isin(LABELS)].copy()
        test["text"] = test["text"].map(normalize_text)
    else:
        train = pool[pool["_s"] == "train"]
        val = pool[pool["_s"] == "val"]
        test = pool[pool["_s"] == "test"]

    out_frames = {"train": shape(train, False), "val": shape(val, False),
                  "test": shape(test, True)}
    leak = set(out_frames["train"]["原文網址"]) & (
        set(out_frames["val"]["原文網址"]) | set(out_frames["test"]["原文網址"]))
    assert not leak, f"判決洩漏：{len(leak)} 份"

    for s, df in out_frames.items():
        df.to_csv(out / f"{s}.csv", index=False, encoding="utf-8-sig")
    (out / "label_map.json").write_text(
        json.dumps({"label2id": LABEL2ID, "labels": LABELS}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return {"variant": name, **{s: len(df) for s, df in out_frames.items()},
            "test_類別分布": out_frames["test"]["涉犯罪名"].value_counts().to_dict()}


VARIANTS = [
    # name,           normalize, denoise, min_chars, from_source, clean_test
    ("bin_v0",        False,     False,   6,         False,       False),
    ("bin_v1_norm",   True,      False,   6,         False,       False),
    ("bin_v2_clean",  True,      True,    10,        False,       False),
    ("bin_v3_source", True,      False,   4,         True,        False),
    ("bin_v4_cleantest", True,   True,    10,        False,       True),
]

if __name__ == "__main__":
    rows = [build(*v) for v in VARIANTS]
    print(pd.DataFrame(rows).to_string(index=False))
