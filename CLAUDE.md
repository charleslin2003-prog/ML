# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

This project builds a machine learning training dataset for **defamation crime classification**
(妨害名譽罪名分類, e.g. 公然侮辱 vs 加重誹謗) by scraping judgments from Taiwan's judiciary
judgment query system (司法院裁判書查詢系統, judgment.judicial.gov.tw). The eventual ML task is to
predict which specific crime a given statement/post constitutes, so the dataset must be labeled at
the level of individual statements (one online post/comment = one row), not one row per judgment.

## Environment

- Python 3.14.5, managed via a `uv`-provisioned interpreter, with dependencies installed directly
  into `.venv` (no `requirements.txt`/`pyproject.toml`/lockfile present in the repo).
- Key installed packages: `playwright`, `pandas`.
- Playwright's Chromium browser binary must be installed once per machine:
  ```
  playwright install chromium
  ```

## Running the crawler

```
python dataset/crawler_defamation.py
```

- Runs headed (`headless=False`) so scraping progress is visible in a real browser window.
- Entry point calls `crawl_judgments(keyword='妨害名譽', max_pages=1)` — edit the keyword/page count
  in the `if __name__ == '__main__':` block to change scope.
- Output is written to `defamation_judgments.csv` (UTF-8 with BOM, for Excel compatibility) in the
  working directory.
- There is a 1.5s delay between detail-page fetches to avoid hammering the judiciary server —
  don't remove this when iterating.

## Architecture (`dataset/crawler_defamation.py`)

Single-file async scraper with two stages:

1. **`crawl_judgments()`** — drives Playwright against judgment.judicial.gov.tw: submits the
   keyword search, paginates the result list (`#jud tbody tr`), and opens each result's detail
   page to pull the full judgment text (`#JudFull`). Selectors are tied to the judiciary site's
   current markup and may break silently if the site is redesigned — verify with a 1-page run
   before scaling up.
2. **`extract_statements_and_crimes()`** — parses the full judgment text with regex to pull out
   `主文` (verdict) and the `事實`/`犯罪事實` (facts) section, then extracts quoted statements
   (text wrapped in `「」`/`『』`, typically followed by `等語`/`等字樣`/`等文字`) as the
   defendant's alleged statements.

**Known gap vs. the target ML dataset**: `extract_statements_and_crimes()` currently treats an
entire judgment as one record — all quoted statements found in the facts section are joined into a
single `' | '`-separated string, and the `主文` field is the raw verdict text rather than a
normalized crime label. To produce per-statement training rows, this needs to be reworked to first
segment the facts section into individual incidents (each usually carries its own date/time,
platform, account, statement, and crime), then extract per-incident fields (date, time, social
platform, platform account, statement text, fact summary, crime name, evidence/exhibit number)
from each segment individually. Because judgment wording varies a lot between cases, plan for
this to be an iterative regex effort (or regex-plus-LLM fallback for segments regex can't parse
confidently) rather than a one-shot fix.

**Category balance**: searching only the umbrella keyword `妨害名譽` will oversample the most common
crime (公然侮辱) relative to rarer ones (e.g. 加重誹謗). For a multi-class crime classifier, query
per specific crime keyword and merge/dedupe results instead of relying on one broad search.
