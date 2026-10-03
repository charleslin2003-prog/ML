"""衝 80%：在最佳二分類資料上，疊加類別加權＋多模型集成。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v3 import run

E = Path(__file__).resolve().parent
DATA = E / "data_bin_v3_source"      # 正規化＋增量資料，macro-F1 最佳
CLEAN = E / "data_bin_v4_cleantest"

if __name__ == "__main__":
    # 單模型：最佳資料＋加權＋更多 epoch
    run("B4_bin_best", DATA, model_id="hfl/chinese-roberta-wwm-ext",
        class_weight=True, epochs=6, early_stopping=True, extra_test=CLEAN)
    # 集成成員：換不同 backbone，之後用 softmax 平均投票
    run("B5_bin_bertwwm", DATA, model_id="hfl/chinese-bert-wwm",
        class_weight=True, epochs=6, early_stopping=True, extra_test=CLEAN)
    run("B6_bin_electra", DATA, model_id="hfl/chinese-legal-electra-base-discriminator",
        class_weight=True, epochs=6, early_stopping=True, learning_rate=5e-5,
        extra_test=CLEAN)
    run("B7_bin_macbert", DATA, model_id="hfl/chinese-macbert-base",
        class_weight=True, epochs=6, early_stopping=True, extra_test=CLEAN)
