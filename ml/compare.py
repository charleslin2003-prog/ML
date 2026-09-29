"""彙總五個模型資料夾的 results.json，輸出 ml/RESULTS.md。"""
import json
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent
LABELS = json.loads((ML_ROOT / "data" / "label_map.json").read_text(encoding="utf-8"))["labels"]

GROUP = {
    "01_bert-base-chinese": "通用",
    "02_chinese-bert-wwm": "通用",
    "03_chinese-roberta-wwm-ext": "通用",
    "04_chinese-legal-electra-base": "法律領域",
    "05_lawformer": "法律領域",
}


def main() -> None:
    rows = []
    for folder in sorted(GROUP):
        path = ML_ROOT / folder / "results.json"
        if not path.exists():
            print(f"跳過 {folder}（尚未訓練）")
            continue
        rows.append((folder, json.loads(path.read_text(encoding="utf-8"))))

    if not rows:
        print("沒有任何 results.json")
        return

    rows.sort(key=lambda r: r[1]["test_macro_f1"], reverse=True)

    lines = ["# 五個模型測試集表現比較", ""]
    lines.append("以 macro-F1 排序。測試集 3,852 筆，切分時以判決為 group，train/test 無判決重疊。")
    lines.append("")
    lines.append("| 排名 | 資料夾 | 模型 | 類型 | macro-F1 | weighted-F1 | accuracy |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, (folder, r) in enumerate(rows, 1):
        lines.append(
            f"| {i} | `{folder}` | `{r['model_id']}` | {GROUP[folder]} | "
            f"{r['test_macro_f1']:.4f} | {r['test_weighted_f1']:.4f} | {r['test_accuracy']:.4f} |"
        )

    lines += ["", "## 各類別 F1", "", "| 模型 | " + " | ".join(LABELS) + " |",
              "|---" * (len(LABELS) + 1) + "|"]
    for folder, r in rows:
        cells = " | ".join(f"{r['per_class_f1'][name]:.4f}" for name in LABELS)
        lines.append(f"| `{folder}` | {cells} |")

    lines += ["", "## 運行時間與資源使用", "",
              "| 模型 | 參數量 | 總耗時 | 峰值 GPU 記憶體 | 備註 |",
              "|---|---|---|---|---|"]
    for folder, r in rows:
        params = f"{r['num_parameters']/1e6:.0f}M" if r.get("num_parameters") else "—"
        secs = r.get("total_wall_clock_seconds")
        wall = f"{secs/60:.1f} 分鐘" if secs else "—"
        mem = r.get("peak_gpu_memory_allocated_mb")
        mem_str = f"{mem/1024:.2f} GB" if mem else "回推值無此數據"
        note = "實測" if mem else "耗時為回推值，見下方說明"
        lines.append(f"| `{folder}` | {params} | {wall} | {mem_str} | {note} |")

    if any(r.get("timing_note") for _, r in rows):
        lines.append("")
        lines.append(
            "> 01~05 的耗時是從訓練佇列的 START/DONE 時間戳回推（訓練當下 `common.py` "
            "還沒加計時），GPU 記憶體無法回推；06~10（相同模型/超參數的合併標籤版）"
            "有實測數據可以參考，見 [`RESULTS_merged.md`](RESULTS_merged.md)。"
        )

    lines.append("")
    (ML_ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
