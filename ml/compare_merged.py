"""彙總 06~10（合併標籤 3 類任務）的 results.json，輸出 ml/RESULTS_merged.md，
並與對應的 5 類任務結果（01~05）並排比較。不修改既有的 compare.py / RESULTS.md。
"""
import json
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent

PAIRS = [
    ("01_bert-base-chinese", "06_bert-base-chinese_merged", "bert-base-chinese"),
    ("02_chinese-bert-wwm", "07_chinese-bert-wwm_merged", "chinese-bert-wwm"),
    ("03_chinese-roberta-wwm-ext", "08_chinese-roberta-wwm-ext_merged", "chinese-roberta-wwm-ext"),
    ("04_chinese-legal-electra-base", "09_chinese-legal-electra-base_merged", "chinese-legal-electra-base"),
    ("05_lawformer", "10_lawformer_merged", "Lawformer"),
]


def load(folder: str):
    p = ML_ROOT / folder / "results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> None:
    lines = ["# 合併標籤（3 類）vs 原始 5 類任務 比較", ""]
    lines.append(
        "把 `加重誹謗`／`散布文字誹謗`／`誹謗` 合併成單一個「誹謗類」後，"
        "測試同一組模型的 macro-F1 是否回升。詳見 [`README_合併標籤實驗.md`](README_合併標籤實驗.md)。"
    )
    lines.append("")
    lines.append("| 模型 | 5 類 macro-F1 | 3 類(合併) macro-F1 | 變化 | 5 類 acc | 3 類 acc |")
    lines.append("|---|---|---|---|---|---|")

    any_done = False
    for orig_folder, merged_folder, name in PAIRS:
        orig = load(orig_folder)
        merged = load(merged_folder)
        if orig is None and merged is None:
            continue
        any_done = True
        o_f1 = f"{orig['test_macro_f1']:.4f}" if orig else "—"
        m_f1 = f"{merged['test_macro_f1']:.4f}" if merged else "（訓練中）"
        delta = f"{merged['test_macro_f1'] - orig['test_macro_f1']:+.4f}" if orig and merged else "—"
        o_acc = f"{orig['test_accuracy']:.4f}" if orig else "—"
        m_acc = f"{merged['test_accuracy']:.4f}" if merged else "—"
        lines.append(f"| {name} | {o_f1} | {m_f1} | {delta} | {o_acc} | {m_acc} |")

    if not any_done:
        print("尚未有任何合併標籤實驗的結果，等訓練完成後再執行本腳本。")
        return

    lines.append("")
    lines.append("## 合併任務各類別 F1")
    lines.append("")
    lines.append("| 模型 | 公然侮辱 | 誹謗類 | 無罪 |")
    lines.append("|---|---|---|---|")
    for _, merged_folder, name in PAIRS:
        merged = load(merged_folder)
        if merged is None:
            continue
        c = merged["per_class_f1"]
        lines.append(f"| {name} | {c['公然侮辱']:.4f} | {c['誹謗類']:.4f} | {c['無罪']:.4f} |")

    lines += ["", "## 運行時間與資源使用（06~10，common.py 加計時後的實測數據）", "",
              "| 模型 | 參數量 | 總耗時 | Trainer train_runtime | 峰值 GPU 記憶體(已配置) | 峰值 GPU 記憶體(已保留) |",
              "|---|---|---|---|---|---|"]
    for _, merged_folder, name in PAIRS:
        merged = load(merged_folder)
        if merged is None:
            lines.append(f"| {name} | — | （訓練中） | — | — | — |")
            continue
        params = f"{merged['num_parameters']/1e6:.0f}M" if merged.get("num_parameters") else "—"
        wall = merged.get("total_wall_clock_seconds")
        wall_str = f"{wall/60:.1f} 分鐘" if wall else "—"
        runtime = merged.get("train_runtime_seconds")
        runtime_str = f"{runtime/60:.1f} 分鐘" if runtime else "—"
        alloc = merged.get("peak_gpu_memory_allocated_mb")
        reserved = merged.get("peak_gpu_memory_reserved_mb")
        alloc_str = f"{alloc/1024:.2f} GB" if alloc else "—"
        reserved_str = f"{reserved/1024:.2f} GB" if reserved else "—"
        lines.append(f"| {name} | {params} | {wall_str} | {runtime_str} | {alloc_str} | {reserved_str} |")

    lines.append("")
    lines.append(
        "> 「總耗時」含 tokenizer/模型載入、tokenize、train、test 預測的完整流程；"
        "「train_runtime」只算 Trainer 內部的訓練迴圈，兩者差距大致就是資料前處理與"
        "評估花的時間。GPU 記憶體是 fp16 + `per_device_train_batch_size` 各自不同下的峰值，"
        "Lawformer 用 512 長度、batch 8 且開了 gradient checkpointing，"
        "其餘四個用 256 長度、batch 32，數字不能只看「誰的記憶體用得少」，要連同 batch size 一起看。"
    )

    (ML_ROOT / "RESULTS_merged.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
