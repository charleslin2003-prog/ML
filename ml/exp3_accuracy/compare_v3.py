"""彙總第三種實驗的所有 run，輸出 ml/exp3_accuracy/RESULTS_v3.md。"""
import json
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parent
RUNS = EXP_ROOT / "runs"
BASELINE = 0.5232   # 08_chinese-roberta-wwm-ext_merged 的 macro-F1
NOISE_FLOOR = 0.013  # 重跑同設定的正常波動（見 ml/RESULTS_rerun_check.md）

ORDER = ["E0_baseline", "E1_normalize", "E2_denoise", "E3_denoise10",
         "E4_weighted", "E5_weighted_6ep", "E6_large",
         "E7_more_len4", "E8_more_len2", "E9_more_weighted"]
DESC = {
    "E0_baseline": "未處理（驗證管線能重現基準線）",
    "E1_normalize": "空白／換行正規化",
    "E2_denoise": "正規化＋雜訊列過濾",
    "E3_denoise10": "正規化＋雜訊過濾＋最小長度10＋重新去重",
    "E4_weighted": "最佳資料＋類別加權損失",
    "E5_weighted_6ep": "最佳資料＋加權＋6 epochs＋early stopping",
    "E6_large": "最佳資料＋加權＋roberta-large (326M)",
    "E7_more_len4": "增量資料：最小長度放寬到 4 字",
    "E8_more_len2": "增量資料：最小長度放寬到 2 字",
    "E9_more_weighted": "增量資料(4字)＋類別加權",
}


def load(name):
    p = RUNS / name / "results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    rows = [(n, load(n)) for n in ORDER]
    done = [(n, r) for n, r in rows if r]
    if not done:
        print("還沒有任何結果")
        return

    L = ["# 第三種實驗：提升準確度 — 結果", "",
         f"基準線：`08_chinese-roberta-wwm-ext_merged` 的 macro-F1 = **{BASELINE:.4f}**",
         f"（3 類合併任務。重跑同設定的正常波動是 ±{NOISE_FLOOR}，"
         f"提升要超過這個幅度才算真的有效）", "",
         "## 主結果（統一在未過濾的 test set 上評分，可直接跟基準線比較）", "",
         "| 實驗 | 說明 | train 筆數 | macro-F1 | 對基準線 | accuracy | 是否顯著 |",
         "|---|---|---|---|---|---|---|"]

    for name, r in done:
        m = r["main_test"]
        delta = m["macro_f1"] - BASELINE
        sig = "✅ 有效" if delta > NOISE_FLOOR else ("❌ 變差" if delta < -NOISE_FLOOR else "— 在雜訊範圍內")
        L.append(f"| `{name}` | {DESC.get(name,'')} | {r['n_train']:,} | "
                 f"{m['macro_f1']:.4f} | {delta:+.4f} | {m['accuracy']:.4f} | {sig} |")

    L += ["", "## 實務部署情境（test set 也清洗過，只分類真的網路言論）", "",
          "> 這組**不能**跟基準線比（題目變簡單了），但能回答「上線後對真實留言有多準」。", "",
          "| 實驗 | 清洗後 test 筆數 | macro-F1 | accuracy |", "|---|---|---|---|"]
    for name, r in done:
        if "clean_test" in r:
            c = r["clean_test"]
            L.append(f"| `{name}` | {c['n']:,} | {c['macro_f1']:.4f} | {c['accuracy']:.4f} |")

    L += ["", "## 各類別 F1（主測試集）", "", "| 實驗 | 公然侮辱 | 誹謗類 | 無罪 |", "|---|---|---|---|"]
    for name, r in done:
        c = r["main_test"]["per_class_f1"]
        L.append(f"| `{name}` | {c.get('公然侮辱',0):.4f} | {c.get('誹謗類',0):.4f} | {c.get('無罪',0):.4f} |")

    L += ["", "## 成本", "", "| 實驗 | 耗時(分鐘) | 峰值GPU記憶體(GB) |", "|---|---|---|"]
    for name, r in done:
        mem = r.get("peak_gpu_memory_allocated_mb")
        L.append(f"| `{name}` | {r['total_wall_clock_seconds']/60:.1f} | "
                 f"{mem/1024:.2f} |" if mem else f"| `{name}` | {r['total_wall_clock_seconds']/60:.1f} | — |")

    missing = [n for n, r in rows if not r]
    if missing:
        L += ["", f"> 尚未完成：{', '.join(missing)}"]

    text = "\n".join(L) + "\n"
    (EXP_ROOT / "RESULTS_v3.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
