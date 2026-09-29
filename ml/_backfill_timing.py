"""01~05 是在 common.py 加入計時／GPU 記憶體量測之前訓練完的，results.json 裡沒有這些欄位。

這支腳本改用佇列腳本自己印出的「START/DONE + 時間戳」純文字紀錄回推耗時 ——
檔案的 mtime 在這台機器上會被背景程式（雲端同步/掃毒之類）不定期整批重寫，
不能拿來當時間依據，文字紀錄裡的時間戳才是事實依據。

跟 06~10（在 common.py 加計時之後訓練）不同，01~05 沒有 GPU 峰值記憶體數據，
留 null，並用 `timing_note` 欄位註明是從佇列紀錄回推、而非程式內實測。
"""
import datetime as dt
import json
import os
import re
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent
QUEUE_LOG = Path(
    r"C:\Users\User\AppData\Local\Temp\claude\C--Users-User-PycharmProjects-----"
    r"\4d6d4912-8425-4ad2-8a8a-7c81992e3617\tasks\bbj3ihhu5.output"
)

SEQUENCE = [
    "01_bert-base-chinese",
    "02_chinese-bert-wwm",
    "03_chinese-roberta-wwm-ext",
    "04_chinese-legal-electra-base",
    "05_lawformer",
]

NUM_PARAMETERS = {
    "01_bert-base-chinese": 102_275_330,
    "02_chinese-bert-wwm": 102_275_330,
    "03_chinese-roberta-wwm-ext": 102_275_330,
    "04_chinese-legal-electra-base": 102_270_211,
    "05_lawformer": 126_371_075,
}

RUN_DATE = dt.date(2026, 9, 29)  # 這條佇列是 2026-09-29 晚上啟動的


def parse_queue_log() -> dict[str, dt.datetime]:
    """從佇列腳本輸出解析每個資料夾的 START / DONE 時間戳，處理跨午夜的日期進位。"""
    text = QUEUE_LOG.read_text(encoding="utf-8", errors="ignore")
    events: dict[str, dt.datetime] = {}
    current_date = RUN_DATE
    last_time = None
    for m in re.finditer(
        r"##########\s+(START|DONE)\s+(\S+).*?(\d{2}:\d{2}:\d{2})\s+##########", text
    ):
        kind, folder, hms = m.group(1), m.group(2), m.group(3)
        h, mi, s = (int(x) for x in hms.split(":"))
        if last_time is not None and (h, mi, s) < last_time:
            current_date += dt.timedelta(days=1)
        last_time = (h, mi, s)
        events[f"{folder}_{kind}"] = dt.datetime.combine(
            current_date, dt.time(h, mi, s)
        )
    return events


def main() -> None:
    events = parse_queue_log()

    updated = []
    for i, folder in enumerate(SEQUENCE):
        results_path = ML_ROOT / folder / "results.json"
        if not results_path.exists():
            print(f"跳過 {folder}（尚未訓練完成）")
            continue

        if i == 0:
            checkpoints_dir = ML_ROOT / folder / "checkpoints"
            start = (
                dt.datetime.fromtimestamp(os.path.getctime(checkpoints_dir))
                if checkpoints_dir.exists()
                else None
            )
            end = events.get(f"{SEQUENCE[i + 1]}_START")
        else:
            start = events.get(f"{folder}_START")
            end = events.get(f"{folder}_DONE")

        if start is None or end is None:
            print(f"{folder}: 佇列紀錄不完整，無法回推（可能還在跑或紀錄檔缺失）")
            continue

        wall_clock_seconds = (end - start).total_seconds()

        result = json.loads(results_path.read_text(encoding="utf-8"))
        result["num_parameters"] = NUM_PARAMETERS[folder]
        result["peak_gpu_memory_allocated_mb"] = None
        result["peak_gpu_memory_reserved_mb"] = None
        result["train_runtime_seconds"] = None
        result["total_wall_clock_seconds"] = round(wall_clock_seconds, 1)
        result["timing_note"] = (
            "回推值：01~05 訓練時 common.py 尚未加入計時，"
            "改用佇列腳本自己印出的 START/DONE 時間戳回推（檔案 mtime 在這台機器上"
            "會被背景程式整批重寫，不可靠）。GPU 記憶體無法回推，"
            "06~10 用相同模型/超參數實測，可作為代表值。"
        )

        results_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        updated.append(folder)
        print(f"{folder}: {start.strftime('%H:%M:%S')} → {end.strftime('%H:%M:%S')}  "
              f"= {wall_clock_seconds/60:.1f} 分鐘")

    print(f"\n已補上 {len(updated)} 個模型的時間資訊：{updated}")


if __name__ == "__main__":
    main()
