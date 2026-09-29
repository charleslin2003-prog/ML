import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import run

if __name__ == "__main__":
    # ELECTRA discriminator 的 fine-tune 慣例學習率比 BERT 高一個檔次
    run(
        model_id="hfl/chinese-legal-electra-base-discriminator",
        out_dir=Path(__file__).resolve().parent,
        max_length=256,
        batch_size=32,
        epochs=3,
        learning_rate=5e-5,
    )
