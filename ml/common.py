"""五個模型資料夾共用的 fine-tune 流程，各資料夾的 train.py 只負責給參數。"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ML_ROOT = Path(__file__).resolve().parent
DATA_DIR = ML_ROOT / "data"


def load_splits(data_dir: Path = DATA_DIR):
    meta = json.loads((data_dir / "label_map.json").read_text(encoding="utf-8"))
    frames = {
        name: pd.read_csv(data_dir / f"{name}.csv", encoding="utf-8-sig")
        for name in ("train", "val", "test")
    }
    return frames, meta["labels"]


def build_metrics(labels):
    def compute(eval_pred):
        preds = eval_pred.predictions.argmax(-1)
        refs = eval_pred.label_ids
        return {
            "accuracy": (preds == refs).mean(),
            "macro_f1": f1_score(refs, preds, average="macro"),
            "weighted_f1": f1_score(refs, preds, average="weighted"),
        }

    return compute


def run(model_id: str, out_dir: Path, max_length: int = 256, batch_size: int = 32,
        epochs: int = 3, learning_rate: float = 2e-5, gradient_checkpointing: bool = False,
        data_dir: Path = DATA_DIR):
    frames, labels = load_splits(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wall_clock_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={name: i for i, name in enumerate(labels)},
    )
    num_parameters = sum(p.numel() for p in model.parameters())

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    datasets = {
        name: Dataset.from_pandas(df[["text", "label"]]).map(
            tokenize, batched=True, remove_columns=["text"]
        )
        for name, df in frames.items()
    }

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        warmup_steps=int(0.1 * epochs * len(datasets["train"]) / batch_size),
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=gradient_checkpointing,
        report_to=[],
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=build_metrics(labels),
    )
    train_output = trainer.train()

    pred = trainer.predict(datasets["test"])
    preds = pred.predictions.argmax(-1)
    refs = pred.label_ids

    wall_clock_seconds = time.time() - wall_clock_start
    if torch.cuda.is_available():
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
        peak_gpu_memory_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
    else:
        peak_gpu_memory_mb = None
        peak_gpu_memory_reserved_mb = None

    report = classification_report(
        refs, preds, target_names=labels, digits=4, zero_division=0
    )
    result = {
        "model_id": model_id,
        "max_length": max_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_parameters": int(num_parameters),
        # 整個 run() 的總耗時（含模型/tokenizer 下載、tokenize、train、test 預測），
        # 比 Trainer 內建的 train_runtime 更貼近「實際跑一次要等多久」
        "total_wall_clock_seconds": round(wall_clock_seconds, 1),
        "train_runtime_seconds": round(train_output.metrics.get("train_runtime", float("nan")), 1),
        "peak_gpu_memory_allocated_mb": round(peak_gpu_memory_mb, 1) if peak_gpu_memory_mb else None,
        "peak_gpu_memory_reserved_mb": round(peak_gpu_memory_reserved_mb, 1) if peak_gpu_memory_reserved_mb else None,
        "test_accuracy": float((preds == refs).mean()),
        "test_macro_f1": float(f1_score(refs, preds, average="macro")),
        "test_weighted_f1": float(f1_score(refs, preds, average="weighted")),
        "per_class_f1": {
            name: float(score)
            for name, score in zip(labels, f1_score(refs, preds, average=None, zero_division=0))
        },
        "confusion_matrix": confusion_matrix(refs, preds).tolist(),
    }

    (out_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "classification_report.txt").write_text(
        f"{model_id}\n\n{report}\n\nconfusion matrix (列=真實, 行=預測)\n"
        f"{np.array2string(confusion_matrix(refs, preds))}\n",
        encoding="utf-8",
    )

    print(f"\n=== {model_id} ===")
    print(report)
    print(f"macro-F1 {result['test_macro_f1']:.4f}  accuracy {result['test_accuracy']:.4f}")
    print(f"總耗時 {wall_clock_seconds/60:.1f} 分鐘  "
          f"峰值 GPU 記憶體 {result['peak_gpu_memory_allocated_mb']} MB  "
          f"參數量 {num_parameters/1e6:.0f}M")
    return result
