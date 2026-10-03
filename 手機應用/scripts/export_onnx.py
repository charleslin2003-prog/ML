"""把訓練好的 PyTorch 模型匯出成 ONNX 並做 INT8 動態量化，供瀏覽器端推論使用。

為什麼要量化：BERT 類模型的權重是 fp32（每個參數 4 bytes）。
動態量化把權重轉成 INT8（1 byte），體積約降為 1/4，推論也會變快，
代價是極小的準確度損失 —— 這個損失會在 verify_onnx.py 裡實測量化前後的差異。
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model"


def export(checkpoint: Path, out_name: str, max_len: int = 256):
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(checkpoint))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint)).eval()

    dummy = tok("測試句子", return_tensors="pt", padding="max_length",
                truncation=True, max_length=16)
    onnx_path = OUT / f"{out_name}.onnx"

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "token_type_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    fp32_mb = onnx_path.stat().st_size / 1024 ** 2

    # 動態量化成 INT8
    # 兩個要處理的現實問題：
    # 1. torch 匯出時把權重放在外部的 .onnx.data，但網頁端載入單一檔案比較單純，
    #    所以先用 onnx 合併回單一檔案
    # 2. onnxruntime 的量化流程會寫出 *-inferred.onnx 暫存檔，但它處理不了含中文的路徑
    #    （本專案路徑有中文），所以整個流程在純 ASCII 的暫存目錄裡做完再搬回來
    import tempfile

    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    q_path = OUT / f"{out_name}.int8.onnx"
    with tempfile.TemporaryDirectory(prefix="onnxq_") as td:
        tmp = Path(td)
        merged = tmp / "merged.onnx"
        m = onnx.load(str(onnx_path), load_external_data=True)
        onnx.save(m, str(merged), save_as_external_data=False)
        quantize_dynamic(str(merged), str(tmp / "q.onnx"), weight_type=QuantType.QUInt8)
        shutil.copy(tmp / "q.onnx", q_path)
    int8_mb = q_path.stat().st_size / 1024 ** 2

    # 詞彙表與設定一起輸出，瀏覽器端的 tokenizer 需要
    vocab_src = Path(checkpoint) / "vocab.txt"
    if not vocab_src.exists():
        tok.save_pretrained(str(OUT / "_tok_tmp"))
        vocab_src = OUT / "_tok_tmp" / "vocab.txt"
    shutil.copy(vocab_src, OUT / "vocab.txt")
    if (OUT / "_tok_tmp").exists():
        shutil.rmtree(OUT / "_tok_tmp")

    meta = {
        "model_name": out_name,
        "labels": [model.config.id2label[i] for i in range(model.config.num_labels)],
        "max_length": max_len,
        "vocab_size": tok.vocab_size,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "size_fp32_mb": round(fp32_mb, 2),
        "size_int8_mb": round(int8_mb, 2),
        "compression_ratio": round(fp32_mb / int8_mb, 2),
        "do_lower_case": getattr(tok, "do_lower_case", True),
    }
    (OUT / "model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--name", default="defamation")
    a = ap.parse_args()
    export(Path(a.checkpoint), a.name)
