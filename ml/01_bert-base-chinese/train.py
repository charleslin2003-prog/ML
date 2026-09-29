import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import run

if __name__ == "__main__":
    run(
        model_id="google-bert/bert-base-chinese",
        out_dir=Path(__file__).resolve().parent,
        max_length=256,
        batch_size=32,
        epochs=3,
        learning_rate=2e-5,
    )
