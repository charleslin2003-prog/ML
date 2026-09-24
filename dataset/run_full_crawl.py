import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.crawler_defamation import crawl_exhaustive

if __name__ == '__main__':
    asyncio.run(crawl_exhaustive(
        keyword='妨害名譽',
        csv_path=str(Path(__file__).resolve().parent / 'defamation_judgments.csv'),
        only_criminal=True,
        headless=True,
    ))
