"""第三種實驗的訓練模組：在 common.py 的基礎上加入提升準確度的技術手段。

跟 ml/common.py 的差異（刻意獨立一份，不改動已經驗證過的 common.py）：
1. class_weight：用類別頻率倒數加權 CrossEntropyLoss，直接針對 macro-F1 最佳化
2. 可指定 epochs / lr / 模型，方便做消融實驗
3. 可另外在「清洗過的 test set」上再評估一次，報告實務部署情境的分數
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

EXP_ROOT = Path(__file__).resolve().parent


def load_splits(data_dir: Path):
    meta = json.loads((data_dir / "label_map.json").read_text(encoding="utf-8"))
    frames = {
        name: pd.read_csv(data_dir / f"{name}.csv", encoding="utf-8-sig")
        for name in ("train", "val", "test")
    }
    return frames, meta["labels"]


def build_metrics():
    def compute(eval_pred):
        preds = eval_pred.predictions.argmax(-1)
        refs = eval_pred.label_ids
        return {
            "accuracy": (preds == refs).mean(),
            "macro_f1": f1_score(refs, preds, average="macro"),
            "weighted_f1": f1_score(refs, preds, average="weighted"),
        }

    return compute


def run(run_name: str, data_dir: Path, model_id: str = "hfl/chinese-roberta-wwm-ext",
        max_length: int = 256, batch_size: int = 32, epochs: int = 3,
        learning_rate: float = 2e-5, class_weight: bool = False,
        early_stopping: bool = False, extra_test: Path | None = None):
    out_dir = EXP_ROOT / "runs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, labels = load_splits(data_dir)
    wall_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={name: i for i, name in enumerate(labels)},
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    datasets = {
        name: Dataset.from_pandas(df[["text", "label"]]).map(
            tokenize, batched=True, remove_columns=["text"])
        for name, df in frames.items()
    }

    loss_fn = None
    weights_used = None
    if class_weight:
        counts = frames["train"]["label"].value_counts().sort_index().values.astype(float)
        # 反頻率加權後正規化，讓少數類別在 loss 裡的份量跟多數類別拉平
        w = counts.sum() / (len(counts) * counts)
        weights_used = w.tolist()
        weight_tensor = torch.tensor(w, dtype=torch.float32,
                                     device="cuda" if torch.cuda.is_available() else "cpu")

        def loss_fn(outputs, labels_, num_items_in_batch=None):
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            return nn.functional.cross_entropy(
                logits.float(), labels_, weight=weight_tensor)

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
        report_to=[],
        seed=42,
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)] if early_stopping else []
    trainer = Trainer(
        model=model, args=args,
        train_dataset=datasets["train"], eval_dataset=datasets["val"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=build_metrics(),
        compute_loss_func=loss_fn,
        callbacks=callbacks,
    )
    train_out = trainer.train()

    def evaluate(ds, df_ref, tag):
        pred = trainer.predict(ds)
        preds = pred.predictions.argmax(-1)
        refs = pred.label_ids
        # 保存原始 logits，之後做集成（多模型投票）時需要
        np.save(out_dir / f"logits_{tag}.npy", pred.predictions)
        np.save(out_dir / f"refs_{tag}.npy", refs)
        rep = classification_report(refs, preds, target_names=labels, digits=4, zero_division=0)
        (out_dir / f"classification_report_{tag}.txt").write_text(
            f"{run_name} / {model_id} / {tag} ({len(df_ref)} 筆)\n\n{rep}\n", encoding="utf-8")
        return {
            "n": int(len(df_ref)),
            "accuracy": float((preds == refs).mean()),
            "macro_f1": float(f1_score(refs, preds, average="macro")),
            "weighted_f1": float(f1_score(refs, preds, average="weighted")),
            "per_class_f1": {name: float(s) for name, s in
                             zip(labels, f1_score(refs, preds, average=None, zero_division=0))},
            "confusion_matrix": confusion_matrix(refs, preds).tolist(),
        }

    result = {
        "run_name": run_name,
        "data_dir": str(data_dir.name),
        "model_id": model_id,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "class_weight": class_weight,
        "class_weights_used": weights_used,
        "early_stopping": early_stopping,
        "n_train": int(len(frames["train"])),
        "n_val": int(len(frames["val"])),
        "labels": labels,
        "total_wall_clock_seconds": round(time.time() - wall_start, 1),
        "train_runtime_seconds": round(train_out.metrics.get("train_runtime", float("nan")), 1),
        "peak_gpu_memory_allocated_mb": (
            round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
            if torch.cuda.is_available() else None),
        "main_test": evaluate(datasets["test"], frames["test"], "main"),
        "val_eval": evaluate(datasets["val"], frames["val"], "val"),
    }

    # 若另外指定一份清洗過的 test set，再評估一次（實務部署情境）
    if extra_test is not None:
        edf = pd.read_csv(extra_test / "test.csv", encoding="utf-8-sig")
        eds = Dataset.from_pandas(edf[["text", "label"]]).map(
            tokenize, batched=True, remove_columns=["text"])
        result["clean_test"] = evaluate(eds, edf, "clean")

    (out_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== {run_name} ===")
    print(f"資料: {data_dir.name}  train={result['n_train']}  模型: {model_id}")
    print(f"[主測試集 {result['main_test']['n']} 筆] macro-F1 {result['main_test']['macro_f1']:.4f}  "
          f"accuracy {result['main_test']['accuracy']:.4f}")
    if "clean_test" in result:
        print(f"[清洗後測試集 {result['clean_test']['n']} 筆] macro-F1 {result['clean_test']['macro_f1']:.4f}  "
              f"accuracy {result['clean_test']['accuracy']:.4f}")
    print(f"耗時 {result['total_wall_clock_seconds']/60:.1f} 分鐘")
    return result
