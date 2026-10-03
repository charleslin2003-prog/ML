"""判決層級任務的實驗（含集成成員）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v3 import run

E = Path(__file__).resolve().parent
D = E / "data_judgment"

if __name__ == "__main__":
    run("J0_judgment", D, model_id="hfl/chinese-roberta-wwm-ext",
        max_length=512, batch_size=16, epochs=6, early_stopping=True, class_weight=True)
    run("J1_bertwwm", D, model_id="hfl/chinese-bert-wwm",
        max_length=512, batch_size=16, epochs=6, early_stopping=True, class_weight=True)
    run("J2_electra", D, model_id="hfl/chinese-legal-electra-base-discriminator",
        max_length=512, batch_size=16, epochs=6, early_stopping=True,
        learning_rate=5e-5, class_weight=True)
