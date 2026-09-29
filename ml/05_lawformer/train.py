import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import run

if __name__ == "__main__":
    # Longformer 架構，attention window 512，單筆計算量遠高於 BERT，因此縮小 batch
    run(
        model_id="thunlp/Lawformer",
        out_dir=Path(__file__).resolve().parent,
        max_length=512,
        batch_size=8,
        epochs=3,
        learning_rate=2e-5,
        gradient_checkpointing=True,
    )
