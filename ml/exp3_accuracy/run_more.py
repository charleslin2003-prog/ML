"""階段3：測試「增加資料量」方向（階段1 的否定結果指向的反方向）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v3 import run

EXP_ROOT = Path(__file__).resolve().parent
BACKBONE = "hfl/chinese-roberta-wwm-ext"
CLEAN_TEST = EXP_ROOT / "data_v3_cleantest"

if __name__ == "__main__":
    run("E7_more_len4", EXP_ROOT / "data_v4_more_len4",
        model_id=BACKBONE, extra_test=CLEAN_TEST)
    run("E8_more_len2", EXP_ROOT / "data_v5_more_len2",
        model_id=BACKBONE, extra_test=CLEAN_TEST)
    # 把資料量方向跟技術手段疊起來看有沒有加成
    run("E9_more_weighted", EXP_ROOT / "data_v4_more_len4",
        model_id=BACKBONE, class_weight=True, extra_test=CLEAN_TEST)
