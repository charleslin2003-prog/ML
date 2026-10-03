"""訓練手機部署用的候選模型 —— 重點是「準確度 vs 模型大小」的取捨。

base 模型（102M 參數）量化後仍約 100MB，手機下載負擔太大。
這裡測試幾個小模型，看犧牲多少準確度能換到多小的體積。
"""
import sys
from pathlib import Path

ML = Path(__file__).resolve().parents[2] / "ml"
sys.path.insert(0, str(ML / "exp3_accuracy"))
from train_v3 import run, EXP_ROOT

DATA = ML / "exp3_accuracy" / "data_bin_v3_source"   # 句子層級二分類，最佳資料變體

CANDIDATES = [
    ("M1_electra_small", "hfl/chinese-electra-180g-small-ex-discriminator", 5e-5),
    ("M2_rbt3",          "hfl/rbt3",                                        3e-5),
    ("M3_rbt6",          "hfl/rbt6",                                        3e-5),
]

if __name__ == "__main__":
    for name, model_id, lr in CANDIDATES:
        run(name, DATA, model_id=model_id, epochs=6, early_stopping=True,
            learning_rate=lr, batch_size=32)
