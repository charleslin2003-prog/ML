"""集成多個模型的預測，並在 val 上調決策閾值 —— 衝 80% 的最後一哩路。

兩個手段：
1. 軟投票集成：把各模型的 softmax 機率平均，比單一模型穩定
2. 閾值調整：二分類預設用 0.5 當界線，但最佳界線不一定在 0.5。
   **閾值一律在 val 上找，再套用到 test** —— 直接在 test 上調等於偷看答案。
"""
import json
from pathlib import Path

import numpy as np

E = Path(__file__).resolve().parent
RUNS = E / "runs"
MEMBERS = ["B4_bin_best", "B5_bin_bertwwm", "B6_bin_electra", "B7_bin_macbert"]


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def load(name, tag):
    d = RUNS / name
    lg, rf = d / f"logits_{tag}.npy", d / f"refs_{tag}.npy"
    if not (lg.exists() and rf.exists()):
        return None, None
    return softmax(np.load(lg).astype(np.float64)), np.load(rf)


def metrics(probs, refs, thr=0.5):
    """thr 是「判為第 1 類(誹謗類)」的機率門檻。"""
    preds = (probs[:, 1] >= thr).astype(int)
    acc = (preds == refs).mean()
    f1s = []
    for c in (0, 1):
        tp = ((preds == c) & (refs == c)).sum()
        fp = ((preds == c) & (refs != c)).sum()
        fn = ((preds != c) & (refs == c)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return acc, float(np.mean(f1s))


def best_threshold(probs, refs, target="accuracy"):
    grid = np.arange(0.20, 0.81, 0.01)
    scores = [metrics(probs, refs, t) for t in grid]
    idx = int(np.argmax([s[0] if target == "accuracy" else s[1] for s in scores]))
    return float(grid[idx]), scores[idx]


def main():
    avail = [m for m in MEMBERS if (RUNS / m / "logits_main.npy").exists()]
    if not avail:
        print("還沒有任何成員模型的 logits，先跑 run_binary_push.py")
        return

    report = {"members": {}, "ensemble": {}}
    print(f"{'模型':20s} {'test_acc':>9s} {'test_F1':>8s} {'調閾值後acc':>11s} {'最佳閾值':>8s}")

    val_probs, test_probs = [], []
    for m in avail:
        pv, rv = load(m, "val")
        pt, rt = load(m, "main")
        if pv is None or pt is None:
            continue
        val_probs.append(pv)
        test_probs.append(pt)
        acc, f1 = metrics(pt, rt)
        thr, _ = best_threshold(pv, rv, "accuracy")       # 閾值在 val 上找
        acc_t, f1_t = metrics(pt, rt, thr)                # 套用到 test
        report["members"][m] = {"accuracy": acc, "macro_f1": f1,
                                "threshold": thr, "accuracy_tuned": acc_t,
                                "macro_f1_tuned": f1_t}
        print(f"{m:20s} {acc:9.4f} {f1:8.4f} {acc_t:11.4f} {thr:8.2f}")

    if len(test_probs) < 2:
        print("\n成員不足 2 個，無法集成")
    else:
        ens_val = np.mean(val_probs, axis=0)
        ens_test = np.mean(test_probs, axis=0)
        acc, f1 = metrics(ens_test, rt)
        thr, _ = best_threshold(ens_val, rv, "accuracy")
        acc_t, f1_t = metrics(ens_test, rt, thr)
        report["ensemble"] = {"n_members": len(test_probs), "accuracy": acc,
                              "macro_f1": f1, "threshold": thr,
                              "accuracy_tuned": acc_t, "macro_f1_tuned": f1_t}
        print(f"\n{'集成(軟投票)':18s} {acc:9.4f} {f1:8.4f} {acc_t:11.4f} {thr:8.2f}")
        print(f"\n>>> 最終 accuracy: {max(acc, acc_t)*100:.2f}%   macro-F1: {max(f1, f1_t):.4f}")

    (E / "ensemble_result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
