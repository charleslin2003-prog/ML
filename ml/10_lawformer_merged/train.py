import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ML_ROOT, run

if __name__ == "__main__":
    run(
        model_id="thunlp/Lawformer",
        out_dir=Path(__file__).resolve().parent,
        data_dir=ML_ROOT / "data_merged",
        max_length=512,
        batch_size=8,
        epochs=3,
        learning_rate=2e-5,
        gradient_checkpointing=True,
    )
