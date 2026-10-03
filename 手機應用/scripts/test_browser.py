"""用 Playwright 模擬手機實際跑一次 PWA，驗證從載入到推論的完整流程。

Node 端驗證過的是「JS 邏輯正確」，但瀏覽器裡還有別的變數：
ONNX Runtime Web 的 WASM 後端能不能初始化、模型下載會不會逾時、
手機尺寸的介面會不會跑版 —— 這些只有真的用瀏覽器開才知道。
"""
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8000/web/"
OUT = Path(__file__).resolve().parent
CASES = ["你這個智障，廢物一個", "他把公司的錢偷偷匯到自己帳戶", "這個醫生根本沒有執照，是密醫"]


def main():
    results = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        # 模擬 iPhone 尺寸，確認手機版面正常
        ctx = browser.new_context(viewport={"width": 390, "height": 844},
                                  device_scale_factor=3, is_mobile=True,
                                  has_touch=True, locale="zh-TW")
        page = ctx.new_page()
        logs = []
        page.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"))
        page.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))

        page.goto(URL, wait_until="domcontentloaded")
        # 模型約 25MB，載入需要時間
        page.wait_for_selector("#main:not([hidden])", timeout=180_000)
        print("模型載入完成，介面已顯示")
        page.screenshot(path=str(OUT / "screenshot_ready.png"))

        for text in CASES:
            page.fill("#text", text)
            page.click("#go")
            page.wait_for_selector("#result:not([hidden])", timeout=60_000)
            page.wait_for_timeout(300)
            label = page.inner_text(".result-label")
            conf = page.inner_text(".result-conf")
            badges = page.eval_on_selector_all(".badge", "els => els.map(e => e.textContent.trim())")
            results.append({"text": text, "label": label, "conf": conf, "badges": badges})
            print(f"  「{text}」→ {label}  {conf}  {badges}")

        page.fill("#text", CASES[0])
        page.click("#go")
        page.wait_for_selector("#result:not([hidden])", timeout=60_000)
        page.screenshot(path=str(OUT / "screenshot_result.png"), full_page=True)

        errs = [l for l in logs if "error" in l.lower() or "pageerror" in l]
        print(f"\n主控台訊息 {len(logs)} 則，其中錯誤 {len(errs)} 則")
        for e in errs[:5]:
            print("  ", e[:160])
        browser.close()

    (OUT / "browser_test_result.json").write_text(
        json.dumps({"results": results, "errors": errs}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
