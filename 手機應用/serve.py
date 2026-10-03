"""本機測試伺服器。

PWA 必須透過 http(s) 提供才能運作（Service Worker 與 ES modules 在 file:// 下會被瀏覽器擋掉）。
手機測試方式：讓電腦與手機連同一個 Wi-Fi，用電腦的區網 IP 開啟網頁。

注意：Service Worker 在非 localhost 的 http 來源會被瀏覽器拒絕註冊（需要 HTTPS），
所以手機上用區網 IP 測試時，推論功能正常，但「離線快取／加到主畫面」要正式部署到
HTTPS 網域（例如 GitHub Pages）才會生效。
"""
import http.server
import socket
import socketserver
from pathlib import Path

PORT = 8000
ROOT = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def end_headers(self):
        # ONNX Runtime Web 的多執行緒 WASM 需要這兩個標頭才能啟用 SharedArrayBuffer
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        if "GET" in (args[0] if args else ""):
            super().log_message(fmt, *args)


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        print(f"電腦開啟： http://localhost:{PORT}/web/")
        print(f"手機開啟： http://{lan_ip()}:{PORT}/web/   （需與電腦同一個 Wi-Fi）")
        print("Ctrl+C 結束")
        httpd.serve_forever()
