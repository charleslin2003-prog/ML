"""驗證 ml/README.md「已知限制」提出的假設：加重誹謗/散布文字誹謗/誹謗三類在刑法上
同屬第 310 條的不同態樣，模型分不清楚可能是標籤定義太細，而非模型能力問題。

把這三類合併成一個「誹謗類」，其餘沿用 prepare_data.py 完全相同的清洗流程，
輸出到獨立的 data_merged/ 目錄 —— 不覆寫 prepare_data.py 或既有的 data/*.csv，
兩份資料可以並存比較。
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "dataset" / "defamation_statements_ml_ready.csv"
OUT_DIR = ROOT / "data_merged"

ORIGINAL_LABELS = ["公然侮辱", "加重誹謗", "散布文字誹謗", "誹謗", "無罪"]
MERGE_MAP = {
    "公然侮辱": "公然侮辱",
    "加重誹謗": "誹謗類",
    "散布文字誹謗": "誹謗類",
    "誹謗": "誹謗類",
    "無罪": "無罪",
}
LABELS = ["公然侮辱", "誹謗類", "無罪"]
MIN_CHARS = 6
SEED = 42


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(SOURCE, encoding="utf-8-sig", dtype=str)
    df = df[["言論內容", "涉犯罪名", "原文網址", "裁判字號", "裁判日期", "社交平台"]]
    df = df[df["涉犯罪名"].isin(ORIGINAL_LABELS)].copy()
    df["言論內容"] = df["言論內容"].str.strip()

    df = df.drop_duplicates(["言論內容", "涉犯罪名"])
    df["merged_label"] = df["涉犯罪名"].map(MERGE_MAP)

    # 原版 prepare_data.py 是在合併前的 5 個細分標籤上抓「一句話對到多個標籤」的矛盾資料。
    # 這裡改成在合併後的 3 個粗分標籤上抓矛盾 —— 一句話同時被貼過「加重誹謗」和「誹謗」
    # 在合併後已經不算矛盾了（都變成「誹謗類」），理應保留，這正是驗證假設的關鍵差異。
    label_count = df.groupby("言論內容")["merged_label"].nunique()
    df = df[~df["言論內容"].isin(label_count[label_count > 1].index)]

    df = df[df["言論內容"].str.len() >= MIN_CHARS]
    return df.reset_index(drop=True)


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    groups = df["原文網址"]
    train_idx, rest_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED).split(df, groups=groups)
    )
    rest = df.iloc[rest_idx].reset_index(drop=True)
    val_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED).split(
            rest, groups=rest["原文網址"]
        )
    )
    return {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": rest.iloc[val_idx].reset_index(drop=True),
        "test": rest.iloc[test_idx].reset_index(drop=True),
    }


def main() -> None:
    df = load_and_clean()
    label2id = {name: i for i, name in enumerate(LABELS)}
    df["label"] = df["merged_label"].map(label2id)
    df = df.rename(columns={"言論內容": "text", "涉犯罪名": "原始罪名", "merged_label": "涉犯罪名"})

    OUT_DIR.mkdir(exist_ok=True)
    splits = split(df)
    for name, part in splits.items():
        part.to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    (OUT_DIR / "label_map.json").write_text(
        json.dumps(
            {"label2id": label2id, "labels": LABELS, "merge_map": MERGE_MAP},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    print(f"總計 {len(df)} 筆，{df['原文網址'].nunique()} 份判決（3 類合併版）")
    for name, part in splits.items():
        print(f"\n[{name}] {len(part)} 筆 / {part['原文網址'].nunique()} 份判決")
        print(part["涉犯罪名"].value_counts().to_string())
    overlap = set(splits["train"]["原文網址"]) & set(splits["test"]["原文網址"])
    print(f"\ntrain/test 判決重疊: {len(overlap)}")


if __name__ == "__main__":
    main()
