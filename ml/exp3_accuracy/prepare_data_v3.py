"""第三種實驗：在不改動原始資料的前提下，產生多個「資料清洗變體」來測試哪種整理方式最能提高準確度。

刻意從 ml/data_merged/ 既有的 train/val/test 切分出發（唯讀），只做欄位內容的轉換與列的篩選，
不重新切分 —— 這樣跟 06~10 的基準線比較時，唯一的變數就是資料清洗方式本身，
不會混入「切分不同」這個干擾因素。

輸出到 ml/exp3_accuracy/data_<variant>/，完全不碰 ml/data/ 與 ml/data_merged/。
"""
import json
import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "data_merged"   # 唯讀來源
MIN_CHARS_DEFAULT = 6

# fallback 抽取路徑會把判決書裡任何引號內容當成言論，以下樣態都不是「被告發表的言論」
NOISE_PATTERNS = {
    "court_qa": r"（問[：:]|（答[：:]|\(問[：:]|^問[：:]",
    "court_doc": r"法院|判決|裁定|上訴人|檢察官|起訴書|偵查|本院|告訴狀|本件|聲請人",
    "legal_narrative": r"^伊|伊[說稱表認]|被告[^，。]{0,6}(?:表示|稱|說)|告訴人|證人",
    "legal_term": r"故意|過失|真實惡意|善意發表|誹謗罪|侮辱罪|構成要件|阻卻違法|不法所有",
    "pure_ascii": r"^[A-Za-z0-9\s\.\-_/＆&]+$",
    "org_name": r"公司|財團法人|委員會|辦公室|股份有限|協會|工會|學校|事務所",
}


def normalize_text(s: str) -> str:
    """把判決書排版造成的換行／連續空白壓掉，並統一常見的全形符號。

    判決書原文是分欄排版，抽取出來的言論內容有 51% 帶有換行＋縮排空白
    （例如 "信口開河、無知\n    亂講"），這些空白在中文裡沒有語意，卻會
    讓 tokenizer 切出大量無意義的 token，稀釋真正的語意訊號。
    """
    s = str(s)
    s = s.replace("　", " ")           # 全形空白
    s = re.sub(r"[\r\n]+", "", s)          # 換行直接去掉（中文不需要空白分詞）
    s = re.sub(r"\s{2,}", " ", s)          # 連續空白壓成一個
    s = s.replace("．．．", "…").replace("...", "…")
    s = re.sub(r"…{2,}", "…", s)
    return s.strip()


def noise_mask(series: pd.Series) -> pd.Series:
    """回傳布林遮罩：True 代表這一列看起來不是真的網路言論。"""
    mask = pd.Series(False, index=series.index)
    for pattern in NOISE_PATTERNS.values():
        mask |= series.str.contains(pattern, regex=True, na=False)
    return mask


def build_variant(name: str, normalize: bool, drop_noise: bool, min_chars: int,
                  dedup: bool, filter_test: bool) -> dict:
    """產生一個資料變體。

    filter_test=False 時，test 只做 normalize、不丟列 —— 這樣 macro-F1 才能
    跟原始基準線（06~10）直接比較，否則「丟掉難的測試樣本」會讓分數虛假變好。
    """
    out_dir = ROOT / f"data_{name}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    stats = {"variant": name}
    for split in ("train", "val", "test"):
        df = pd.read_csv(SOURCE / f"{split}.csv", encoding="utf-8-sig")
        before = len(df)

        if normalize:
            df["text"] = df["text"].map(normalize_text)

        apply_row_filters = filter_test or split != "test"
        if apply_row_filters:
            if drop_noise:
                df = df[~noise_mask(df["text"].astype(str))]
            df = df[df["text"].astype(str).str.len() >= min_chars]
            if dedup:
                df = df.drop_duplicates(["text", "涉犯罪名"])

        df = df.reset_index(drop=True)
        df.to_csv(out_dir / f"{split}.csv", index=False, encoding="utf-8-sig")
        stats[split] = f"{before}→{len(df)}"

    shutil.copy(SOURCE / "label_map.json", out_dir / "label_map.json")
    return stats


VARIANTS = [
    # name,               normalize, drop_noise, min_chars, dedup, filter_test
    ("v0_baseline",       False,     False,      MIN_CHARS_DEFAULT, False, False),
    ("v1_normalize",      True,      False,      MIN_CHARS_DEFAULT, False, False),
    ("v2_denoise",        True,      True,       MIN_CHARS_DEFAULT, False, False),
    ("v3_denoise_len10",  True,      True,       10,                True,  False),
    # 下面這個把 test 也一起清洗，用來看「實務部署時只分類真的言論」能到多準
    ("v3_cleantest",      True,      True,       10,                True,  True),
]


def main() -> None:
    rows = []
    for args in VARIANTS:
        rows.append(build_variant(*args))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    (ROOT / "variants_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
