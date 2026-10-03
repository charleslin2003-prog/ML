"""二分類任務（公然侮辱 vs 誹謗類）的消融實驗。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v3 import run

E = Path(__file__).resolve().parent
BB = "hfl/chinese-roberta-wwm-ext"
CLEAN = E / "data_bin_v4_cleantest"

if __name__ == "__main__":
    run("B0_bin_raw",    E / "data_bin_v0",        model_id=BB, extra_test=CLEAN)
    run("B1_bin_norm",   E / "data_bin_v1_norm",   model_id=BB, extra_test=CLEAN)
    run("B2_bin_clean",  E / "data_bin_v2_clean",  model_id=BB, extra_test=CLEAN)
    run("B3_bin_source", E / "data_bin_v3_source", model_id=BB, extra_test=CLEAN)
