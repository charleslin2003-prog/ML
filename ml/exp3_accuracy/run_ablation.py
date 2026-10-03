"""第三種實驗的消融測試執行器。

分兩階段，確保每次只改一個變數：
  階段1（資料面）：固定模型與超參數，只換資料清洗變體，看哪種整理方式最有效
  階段2（技術面）：固定在階段1 勝出的資料上，逐一加上類別加權、更多 epoch、更大模型

所有實驗都在「同一份未過濾的 test set」上評分，才能跟原始基準線（06~10 的 0.5232）
直接比較；另外對清洗過的 test set 再評一次，代表實務部署情境。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v3 import run

EXP_ROOT = Path(__file__).resolve().parent
BEST_BACKBONE = "hfl/chinese-roberta-wwm-ext"   # 前兩輪實驗的勝出者
CLEAN_TEST = EXP_ROOT / "data_v3_cleantest"

STAGE1 = [
    ("E0_baseline",  "data_v0_baseline",      {}),
    ("E1_normalize", "data_v1_normalize",     {}),
    ("E2_denoise",   "data_v2_denoise",       {}),
    ("E3_denoise10", "data_v3_denoise_len10", {}),
]


def stage1():
    results = {}
    for name, data, kw in STAGE1:
        r = run(name, EXP_ROOT / data, model_id=BEST_BACKBONE,
                extra_test=CLEAN_TEST, **kw)
        results[name] = r
    return results


def stage2(best_data: str):
    """best_data 由階段1 的結果決定，在這份資料上疊加技術手段。"""
    results = {}
    results["E4_weighted"] = run(
        "E4_weighted", EXP_ROOT / best_data, model_id=BEST_BACKBONE,
        class_weight=True, extra_test=CLEAN_TEST)
    results["E5_weighted_6ep"] = run(
        "E5_weighted_6ep", EXP_ROOT / best_data, model_id=BEST_BACKBONE,
        class_weight=True, epochs=6, early_stopping=True, extra_test=CLEAN_TEST)
    results["E6_large"] = run(
        "E6_large", EXP_ROOT / best_data,
        model_id="hfl/chinese-roberta-wwm-ext-large",
        class_weight=True, batch_size=16, learning_rate=1e-5,
        extra_test=CLEAN_TEST)
    return results


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "stage1"
    if which == "stage1":
        stage1()
    else:
        best = sys.argv[2]
        stage2(best)
