import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.crawler_defamation import crawl_judgments

if __name__ == '__main__':
    asyncio.run(crawl_judgments(
        keyword='妨害名譽',
        max_pages=None,
        start_page=2,  # 第1頁已經抓過並寫入 CSV，從第2頁接續，避免重複
        csv_path=str(Path(__file__).resolve().parent / 'defamation_judgments.csv'),
        only_criminal=True,
        headless=True,
    ))
