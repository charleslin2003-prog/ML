"""從 defamation_statements_ml_ready.csv 產出 5 類分類任務的 train/val/test。"""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "dataset" / "defamation_statements_ml_ready.csv"
OUT_DIR = ROOT / "data"

LABELS = ["公然侮辱", "加重誹謗", "散布文字誹謗", "誹謗", "無罪"]
MIN_CHARS = 6
SEED = 42


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(SOURCE, encoding="utf-8-sig", dtype=str)
    df = df[["言論內容", "涉犯罪名", "原文網址", "裁判字號", "裁判日期", "社交平台"]]
    df = df[df["涉犯罪名"].isin(LABELS)].copy()
    df["言論內容"] = df["言論內容"].str.strip()

    df = df.drop_duplicates(["言論內容", "涉犯罪名"])

    # fallback 路徑會把判決裡任何引號內容都當成言論，法律套語（「意圖散布於眾」等）
    # 因此同時掛在多個罪名下 —— 出現跨標籤衝突的句子一律丟棄，而非任選一個標籤。
    label_count = df.groupby("言論內容")["涉犯罪名"].nunique()
    df = df[~df["言論內容"].isin(label_count[label_count > 1].index)]

    df = df[df["言論內容"].str.len() >= MIN_CHARS]
    return df.reset_index(drop=True)


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # 同一份判決的句子共用同一個標籤，若隨機切分會讓幾乎相同的句子橫跨
    # train/test 造成洩漏，所以以 原文網址 為 group 切分。
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
    df["label"] = df["涉犯罪名"].map(label2id)
    df = df.rename(columns={"言論內容": "text"})

    OUT_DIR.mkdir(exist_ok=True)
    splits = split(df)
    for name, part in splits.items():
        part.to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    (OUT_DIR / "label_map.json").write_text(
        json.dumps({"label2id": label2id, "labels": LABELS}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"總計 {len(df)} 筆，{df['原文網址'].nunique()} 份判決")
    for name, part in splits.items():
        print(f"\n[{name}] {len(part)} 筆 / {part['原文網址'].nunique()} 份判決")
        print(part["涉犯罪名"].value_counts().to_string())
    overlap = set(splits["train"]["原文網址"]) & set(splits["test"]["原文網址"])
    print(f"\ntrain/test 判決重疊: {len(overlap)}")


if __name__ == "__main__":
    main()
