"""驗證 ONNX 匯出與 INT8 量化是否損害準確度。

這一步不能省：量化是「用精度換體積」，但損失多少必須實測。
拿同一份測試集，比較三者的預測：
  PyTorch 原始模型 → ONNX fp32 → ONNX INT8
如果 INT8 的準確度掉太多，就得改用較大的模型或改量化策略。
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
TEST_CSV = ROOT.parents[0] / "ml" / "exp3_accuracy" / "data_bin_v3_source" / "test.csv"


def run_onnx(path, enc, n):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    outs = []
    for i in range(0, n, 32):
        feed = {"input_ids": enc["input_ids"][i:i+32].astype(np.int64),
                "attention_mask": enc["attention_mask"][i:i+32].astype(np.int64)}
        if "token_type_ids" in names:
            feed["token_type_ids"] = enc["token_type_ids"][i:i+32].astype(np.int64)
        outs.append(sess.run(None, feed)[0])
    return np.concatenate(outs)


def main(checkpoint):
    meta = json.loads((MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))
    df = pd.read_csv(TEST_CSV, encoding="utf-8-sig")
    tok = AutoTokenizer.from_pretrained(str(checkpoint))
    enc = tok(df["text"].astype(str).tolist(), padding="max_length", truncation=True,
              max_length=meta["max_length"], return_tensors="np")
    y = df["label"].values
    n = len(df)

    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint)).eval()
    with torch.no_grad():
        pt = []
        for i in range(0, n, 32):
            out = model(input_ids=torch.tensor(enc["input_ids"][i:i+32]),
                        attention_mask=torch.tensor(enc["attention_mask"][i:i+32]),
                        token_type_ids=torch.tensor(enc["token_type_ids"][i:i+32]))
            pt.append(out.logits.numpy())
        pt = np.concatenate(pt)

    name = meta["model_name"]
    results = {}
    for tag, logits in [
        ("PyTorch (fp32)", pt),
        ("ONNX (fp32)", run_onnx(MODEL_DIR / f"{name}.onnx", enc, n)),
        ("ONNX (INT8 量化)", run_onnx(MODEL_DIR / f"{name}.int8.onnx", enc, n)),
    ]:
        pred = logits.argmax(-1)
        results[tag] = {"accuracy": float(accuracy_score(y, pred)),
                        "macro_f1": float(f1_score(y, pred, average="macro"))}

    base = results["PyTorch (fp32)"]["accuracy"]
    print(f"{'階段':20s} {'accuracy':>10s} {'macro-F1':>10s} {'對原模型':>10s}")
    for k, v in results.items():
        print(f"{k:20s} {v['accuracy']:10.4f} {v['macro_f1']:10.4f} {v['accuracy']-base:+10.4f}")

    # 預測一致率：量化後有多少筆的預測結果改變了
    agree = (pt.argmax(-1) == run_onnx(MODEL_DIR / f"{name}.int8.onnx", enc, n).argmax(-1)).mean()
    print(f"\nINT8 與原模型的預測一致率：{agree*100:.2f}%（{n} 筆測試資料）")

    meta["verification"] = {
        "stages": results,
        "int8_vs_pytorch_agreement": float(agree),
        "n_test": int(n),
    }
    meta["accuracy"] = results["ONNX (INT8 量化)"]["accuracy"]
    meta["macro_f1"] = results["ONNX (INT8 量化)"]["macro_f1"]
    (MODEL_DIR / "model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已把驗證結果寫回 model_meta.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
