# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

This project builds a machine learning training dataset for **defamation crime classification**
(妨害名譽罪名分類, e.g. 公然侮辱 vs 加重誹謗) by scraping judgments from Taiwan's judiciary
judgment query system (司法院裁判書查詢系統, judgment.judicial.gov.tw). The ML task is to predict
which specific crime a given social-media statement/post constitutes, so the dataset is labeled at
the level of individual statements (one online post/comment = one row) wherever possible, not one
row per judgment.

## Environment

- Python 3.14.5, managed via a `uv`-provisioned interpreter, with dependencies installed directly
  into `.venv` (no `requirements.txt`/`pyproject.toml`/lockfile present in the repo).
- Key installed packages: `playwright`, `pandas`.
- Playwright's Chromium browser binary must be installed once per machine:
  ```
  playwright install chromium
  ```
- Git repo with an `origin` remote already configured — check `git status`/`git log` before
  assuming a clean slate.

## Running the crawler

```
python dataset/crawler_defamation.py          # runs crawl_exhaustive(keyword='妨害名譽')
python dataset/run_full_crawl.py              # same, meant for detached/background launches
```

- `crawl_exhaustive()` is the main entry point. The judiciary site caps any single query's
  paginated results at **500 hits**, even when the true match count is far larger (e.g. ~41,000
  for 案由=妨害名譽 刑事案件). To get past that cap, `crawl_exhaustive()` recursively bisects the
  裁判日期 (judgment date) range with `find_date_ranges()` until every sub-range has ≤500 hits,
  then crawls each sub-range's full result list. Progress is checkpointed to
  `<csv_path>.progress.txt` (one completed date-range per line) — re-running the same command
  skips ranges already done, so it's safe to stop/restart a multi-hour crawl.
- `crawl_judgments()` still exists for quick manual testing — it runs a single (uncapped-lookin
  g but actually 500-hit-capped) query without date partitioning. Use `max_pages` to limit scope
  when testing.
- Runs headless by default (`headless=True`) — pass `headless=False` to watch it in a real browser
  window while debugging selectors.
- Output CSV is written incrementally (one `csv_file.flush()` per judgment processed), UTF-8 with
  BOM, appending to any existing file at `csv_path` — so partial progress from an earlier run is
  never lost, but re-running from scratch on the same path will require dedup (see
  `clean_dataset.py`) since it appends rather than overwrites.
- There is a 1.5s delay between detail-page fetches and a 1s delay between listing-page turns to
  avoid hammering the judiciary server — don't remove these when iterating.
- **Scale/time expectations**: a full `crawl_exhaustive()` run over the entire judgment history is
  a multi-hour-to-multi-day background job (tens of thousands of individual page fetches at ~2s
  each). Launch it detached (e.g. `Start-Process` on Windows with `PYTHONUNBUFFERED=1` and
  redirected stdout/stderr to a log file) rather than in a foreground shell that might get killed.

## Architecture (`dataset/crawler_defamation.py`)

Two extraction paths per judgment, tried in order:

1. **`extract_incident_table()`** — some judgments (mainly social-media defamation cases) embed
   the prosecutor's indictment table verbatim in the judgment text, with columns matching almost
   exactly what this dataset wants (日期/時間/社交平台/平台帳號/言論內容/事實要旨/涉犯罪名/告證編號).
   This function detects such a table by header keywords (`日期`+`言論`+`罪名` all present), maps
   whatever exact header wording the judgment used to canonical field names via
   `INCIDENT_HEADER_KEYWORDS`, and returns one dict per table row — i.e. one dataset row per
   individual social-media post/comment, with full metadata. This is the high-quality path.
2. **`extract_statements_and_crimes()`** — fallback when no such table exists. Regex-extracts the
   主文 (verdict) and 事實/犯罪事實 (facts) sections, then pulls quoted statements (text in
   `「」`/`『』`) as the alleged defamatory statements. The raw 主文 text is then normalized to a
   clean crime label via `_extract_crime_label()` / `KNOWN_CRIME_KEYWORDS` (e.g. a verdict sentence
   like "王孋儀犯散布文字誹謗罪，處..." becomes just `散布文字誹謗`, "...無罪" becomes `無罪`) —
   without this normalization the raw sentence-length verdict text is useless as a classification
   label. This path only produces one coarse row per judgment (no date/time/platform/account), and
   many judgments still won't match any known crime keyword (kept as raw text, needs eyeballing).

Site-structure quirks worth remembering (discovered by trial — the judiciary site has no visible
"advanced search" affordance and changes markup without notice):
- Simple search (`Default.aspx`, `#txtKW`/`#btnSimpleQry`) full-text-matches the entire judgment,
  which pulls in tons of irrelevant cases that merely *mention* the keyword. Advanced search
  (`Default_AD.aspx`, `#jud_title` = 裁判案由 field) filters on the actual case-reason field and is
  far more precise — this is what the crawler uses.
- Search results render inside an `<iframe name="iframe-data">`, not the top-level page — query
  `page.frame(name='iframe-data')`, not `page` directly.
- The result table has `id="jud"` (class `jub-table`) *inside that iframe*; pagination uses the
  `#hlNext` link's `href` as a URL template (swap the `page=N` query param) rather than repeatedly
  clicking, which is both faster and resumable.
- A judgment's full text container is `div#jud` (class `int-table`) on the detail page — not
  `#JudFull`, which doesn't exist on the current site.
- 案件類別 (case category) checkboxes are `input[name="jud_sys"]`; value `M` = 刑事 (criminal).

## `dataset/clean_dataset.py`

Post-processes `defamation_judgments.csv` (never mutates it) into two derived files. Cleaning steps,
in order:
1. **Strip whitespace** on every text column (`STR_COLUMNS`).
2. **Dedup** — incident-table rows by (`原文網址`, `編號`), fallback rows by `原文網址` alone —
   since the crawler's append-only, resumable design can otherwise re-write the same judgment
   across runs (see "the crawler never stops on its own" above).
3. **Split usable vs. unresolved** — a row is usable (goes to `defamation_statements_ml_ready.csv`)
   if 言論內容 is real (not empty, not the `未明確提取` placeholder) and 案由 is non-empty.
   Everything else goes to `defamation_unresolved_judgments.csv`. This does **not** require the row
   to have come from `extract_incident_table()` — a fallback-path row with real extracted quotes
   counts as usable too, it's just coarser (whole-judgment grain, missing 日期/時間/社交平台/平台
   帳號/告證編號).
4. **`社交平台` casing** normalized to uppercase (`ig`/`IG`/`fb`/`FB` → `IG`/`FB`) — pure
   presentation normalization, not a content change.
5. **`涉犯罪名` re-normalization** — fallback-path rows carry the raw 主文 sentence (verdict text
   with defendant names/sentencing, produced *before* `_extract_crime_label()` existed in the
   crawler, or from any judgment the crawler couldn't cleanly label). `clean_dataset.py` imports and
   re-applies the crawler's own `_extract_crime_label()` to the whole `涉犯罪名` column so that
   already-scraped CSVs get the same clean labels a fresh crawl would produce, without having to
   re-crawl. Rows whose 主文 text matches no known keyword are left as raw text rather than guessed
   — see Known gaps below for what's still unresolved after this pass (mostly procedural
   dispositions like 上訴駁回/自訴不受理 that aren't crime names at all, and multi-defendant/
   multi-count judgments where only the first keyword match wins).
6. `編號`/`告證編號` cast to nullable `Int64` (only ever populated on incident-table rows).

Genuinely missing values (日期/時間/社交平台/平台帳號/告證編號 on fallback-path rows) are left as
empty — never filled or guessed, per this project's data-quality bar.

Run it after any crawl to refresh both derived files:
```
python dataset/clean_dataset.py
```

## Known gaps / next steps

- Crime-label normalization (`_extract_crime_label`) is a first-match keyword heuristic — it does
  not handle judgments where a single 主文 lists multiple different crimes for multiple
  defendants/counts (only the first matching keyword is kept).
- Category balance: even with date-range partitioning, the corpus is naturally dominated by
  公然侮辱 relative to rarer labels like 加重誹謗 — check class balance before training and
  consider querying specific crime names as 案由 directly if a class is underrepresented.
- Only a minority of judgments carry the rich `extract_incident_table()` structure; most usable
  rows currently come from the regex fallback path and lack 日期/時間/社交平台/平台帳號/告證編號.
