import asyncio
import csv
import os
import re
from datetime import date, timedelta
from urllib.parse import urljoin

from playwright.async_api import async_playwright

CSV_FIELDS = [
    '裁判字號', '裁判日期', '案由', '編號', '日期', '時間',
    '社交平台', '平台帳號', '言論內容', '事實要旨', '涉犯罪名',
    '告證編號', '原文網址',
]

ADV_SEARCH_URL = 'https://judgment.judicial.gov.tw/FJUD/Default_AD.aspx'

# 判決主文常見的罪名關鍵字，用來把整段主文原文（含被告姓名/刑度等雜訊）收斂成乾淨標籤。
# 順序很重要：較具體的名稱要排在較籠統的名稱前面（例如「加重誹謗」要先於「誹謗」比對）。
KNOWN_CRIME_KEYWORDS = [
    '加重誹謗',
    '散布文字、圖畫誹謗',
    '散布文字誹謗',
    '妨害信用',
    '恐嚇危害安全',
    '恐嚇危安',
    '公然侮辱',
    '誹謗',
]


def _find_crime_keyword(text):
  for keyword in KNOWN_CRIME_KEYWORDS:
    if keyword in text:
      return keyword
  return None


def _extract_crime_label(verdict_text, full_text=''):
  """從主文辨識出乾淨罪名標籤；辨識不出來就保留原文，供後續人工/規則再確認。

  上訴審的主文常常只寫「上訴駁回」「原判決撤銷」之類的程序性結果，不會重複寫一次罪名——
  真正的罪名藏在「理由」段落（法院說明維持或推翻原審認定時會提到）。主文本身找不到已知罪名
  關鍵字時，改去理由段落找；理由段落也找不到才維持保留原文，不用猜的填入不確定的罪名。
  """
  if not verdict_text:
    return verdict_text
  if '無罪' in verdict_text:
    return '無罪'

  match = _find_crime_keyword(verdict_text)
  if match:
    return match

  if full_text:
    reason_match = re.search(r'理\s*由\s*([\s\S]*)', full_text)
    reason_text = reason_match.group(1) if reason_match else ''
    match = _find_crime_keyword(reason_text)
    if match:
      return match

  return verdict_text


def extract_statements_and_crimes(full_text):
  """利用正則切分主文與事實，並抓出被告引號原文（找不到逐則表格時的備援方案）"""
  # 1. 抓取主文
  verdict_match = re.search(
      r'主\s*文\s*([\s\S]*?)(?:事\s*實|理\s*由|犯罪事實)', full_text
  )
  verdict = verdict_match.group(1).strip() if verdict_match else ''

  # 2. 抓取犯罪事實段落
  fact_match = re.search(
      r'(?:犯罪事實|事\s*實)\s*([\s\S]*?)(?:理\s*由|論罪科刑|應適用之法條)',
      full_text,
  )
  fact_text = fact_match.group(1).strip() if fact_match else ''

  # 3. 抽取被告原文（通常法官會用「」或『』包裹辱罵文字，後面常接「等語」或「之文字」）
  quotes = re.findall(r'[「『]([^」』]+)[」』](?:等語|等字樣|等文字)?', fact_text)
  # 過濾長度過短（例如法條名、代稱）或非言論字詞
  filtered_quotes = [q.strip() for q in quotes if len(q.strip()) >= 2]

  return {
      'verdict': _extract_crime_label(verdict, full_text),
      'fact_text': fact_text[:300] + '...' if len(fact_text) > 300 else fact_text,  # 預覽前300字
      'target_statements': (
          ' | '.join(filtered_quotes) if filtered_quotes else '未明確提取'
      ),
  }


# 部分判決書（尤其是社群平台妨害名譽案件）會把檢察官起訴書的逐筆言論附表原封不動放進全文，
# 欄位名稱可能略有出入，用關鍵字比對統一成固定欄位名稱
INCIDENT_HEADER_KEYWORDS = [
    ('告證', '告證編號'),
    ('日期', '日期'),
    ('時間', '時間'),
    ('社交平台', '社交平台'),
    ('平台帳號', '平台帳號'),
    ('帳號', '平台帳號'),
    ('言論', '言論內容'),
    ('事實', '事實要旨'),
    ('罪名', '涉犯罪名'),
    ('編號', '編號'),
]


def _map_incident_header(raw_header):
  for keyword, canonical in INCIDENT_HEADER_KEYWORDS:
    if keyword in raw_header:
      return canonical
  return raw_header


async def extract_incident_table(case_page):
  """在判決書全文裡尋找逐則言論附表（欄位含日期/言論內容/罪名），找到就回傳逐列資料"""
  tables = await case_page.query_selector_all('#jud table')
  for table in tables:
    rows = await table.query_selector_all('tr')
    if not rows:
      continue
    header_cells = await rows[0].query_selector_all('td')
    headers_raw = [(await c.inner_text()).strip() for c in header_cells]
    joined = ''.join(headers_raw)
    if '日期' not in joined or '言論' not in joined or '罪名' not in joined:
      continue

    canonical_headers = [_map_incident_header(h) for h in headers_raw]
    records = []
    for row in rows[1:]:
      cells = await row.query_selector_all('td')
      if len(cells) != len(canonical_headers):
        continue
      values = [(await c.inner_text()).strip() for c in cells]
      records.append(dict(zip(canonical_headers, values)))
    if records:
      return records
  return None


def _open_csv_writer(csv_path):
  file_exists = os.path.exists(csv_path)
  f = open(csv_path, 'a', newline='', encoding='utf-8-sig')
  writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore', restval='')
  if not file_exists:
    writer.writeheader()
    f.flush()
  return f, writer


async def _fill_advanced_search(page, keyword, only_criminal, date_range=None):
  await page.goto(ADV_SEARCH_URL)
  await page.wait_for_load_state('networkidle')
  await page.fill('#jud_title', keyword)  # 「裁判案由」欄位，比全文關鍵字搜尋精準很多
  if only_criminal:
    for cb in await page.query_selector_all('input[name="jud_sys"]'):
      if await cb.get_attribute('value') == 'M':  # M = 刑事
        await cb.check()
  if date_range:
    (sy, sm, sd), (ey, em, ed) = date_range
    await page.fill('#dy1', str(sy))
    await page.fill('#dm1', str(sm))
    await page.fill('#dd1', str(sd))
    await page.fill('#dy2', str(ey))
    await page.fill('#dm2', str(em))
    await page.fill('#dd2', str(ed))


async def _query_result_count(page, keyword, only_criminal, date_range):
  await _fill_advanced_search(page, keyword, only_criminal, date_range)
  await page.click('#btnQry')
  await page.wait_for_load_state('networkidle')
  data_frame = page.frame(name='iframe-data')
  if data_frame is None:
    return 0
  body_text = await data_frame.inner_text('body')
  m = re.search(r'共\s*([\d,]+)\s*筆', body_text)
  return int(m.group(1).replace(',', '')) if m else 0


def _roc_to_date(y, m, d):
  return date(y + 1911, m, d)


def _date_to_roc(d):
  return (d.year - 1911, d.month, d.day)


async def find_date_ranges(page, keyword, only_criminal, start_roc, end_roc, cap=500):
  """遞迴以裁判日期切分查詢區間，讓每個子區間結果都 <= cap 筆，藉此繞過網站單次查詢只顯示前500筆的限制。"""
  count = await _query_result_count(page, keyword, only_criminal, (start_roc, end_roc))
  if count == 0:
    return []
  if count <= cap:
    return [(start_roc, end_roc, count)]

  start_dt = _roc_to_date(*start_roc)
  end_dt = _roc_to_date(*end_roc)
  if start_dt >= end_dt:
    # 同一天內就超過 cap，無法再切，只能接受這天可能抓不完整（極端情況才會發生）
    return [(start_roc, end_roc, count)]

  mid_dt = start_dt + (end_dt - start_dt) // 2
  left = await find_date_ranges(page, keyword, only_criminal, start_roc, _date_to_roc(mid_dt), cap)
  next_dt = mid_dt + timedelta(days=1)
  if next_dt > end_dt:
    return left
  right = await find_date_ranges(page, keyword, only_criminal, _date_to_roc(next_dt), end_roc, cap)
  return left + right


async def _process_listing_frame(data_frame, context, writer, csv_file, counters, start_page=1):
  """逐頁讀取查詢結果清單、逐筆進入判決書全文解析並寫入 CSV。counters 是 {'judgments':n,'incidents':n} 的累加字典。"""
  next_link = await data_frame.query_selector('#hlNext')
  page_url_template = None
  if next_link is not None:
    next_href = urljoin(data_frame.url, await next_link.get_attribute('href'))
    page_url_template = re.sub(r'page=\d+', 'page={n}', next_href)

  current_page = start_page
  if current_page > 1 and page_url_template is not None:
    await data_frame.goto(page_url_template.format(n=current_page), timeout=15000)
    await data_frame.wait_for_load_state('networkidle')

  while True:
    print(f'--- 正在抓取第 {current_page} 頁清單 ---')
    rows = await data_frame.query_selector_all('#jud tbody tr')

    case_links = []
    for row in rows:
      link_elem = await row.query_selector('a#hlTitle')
      date_elem = await row.query_selector('td:nth-child(3)')
      reason_elem = await row.query_selector('td:nth-child(4)')

      if link_elem:
        title = (await link_elem.inner_text()).strip()
        href = await link_elem.get_attribute('href')
        item_date = (await date_elem.inner_text()).strip() if date_elem else ''
        reason = (await reason_elem.inner_text()).strip() if reason_elem else ''
        full_url = urljoin(data_frame.url, href)
        case_links.append(
            {'title': title, 'url': full_url, 'date': item_date, 'reason': reason}
        )

    if not case_links:
      print('本頁沒有任何案件，這個範圍抓取完畢。')
      break

    print(f'本頁共找到 {len(case_links)} 筆案件，開始讀取內文...')

    for item in case_links:
      case_page = await context.new_page()
      try:
        await case_page.goto(item['url'], timeout=15000)
        await case_page.wait_for_selector('#jud', timeout=10000)

        # 優先找判決書內附的逐則言論表格（一則言論一列）
        incident_rows = await extract_incident_table(case_page)

        if incident_rows:
          for rec in incident_rows:
            writer.writerow({
                '裁判字號': item['title'],
                '裁判日期': item['date'],
                '案由': item['reason'],
                **rec,
                '原文網址': item['url'],
            })
            counters['incidents'] += 1
          print(f"成功解析（逐則表格，共{len(incident_rows)}則）：{item['title']}")
        else:
          # 找不到現成表格，退回用正則從全文摘要（整份判決一列，粒度較粗）
          full_text = await case_page.inner_text('#jud')
          parsed = extract_statements_and_crimes(full_text)
          writer.writerow({
              '裁判字號': item['title'],
              '裁判日期': item['date'],
              '案由': item['reason'],
              '言論內容': parsed['target_statements'],
              '事實要旨': parsed['fact_text'],
              '涉犯罪名': parsed['verdict'],
              '原文網址': item['url'],
          })
          print(f"成功解析（無表格，退回摘要）：{item['title']}")

        csv_file.flush()  # 即時落盤，中斷也不會遺失已抓到的資料
        counters['judgments'] += 1
      except Exception as e:
        print(f"抓取失敗 {item['title']}: {e}")
      finally:
        await case_page.close()
        await asyncio.sleep(1.5)  # 禮貌延遲，避免對司法院伺服器造成負擔

    current_page += 1
    if page_url_template is None:
      break
    try:
      await data_frame.goto(page_url_template.format(n=current_page), timeout=15000)
      await data_frame.wait_for_load_state('networkidle')
    except Exception as e:
      print(f'跳到第 {current_page} 頁失敗：{e}，停止這個範圍的抓取。')
      break
    await asyncio.sleep(1.0)


async def crawl_judgments(
    keyword='妨害名譽',
    max_pages=None,
    start_page=1,
    csv_path='defamation_judgments.csv',
    only_criminal=True,
    headless=True,
):
  """單次查詢抓取（會受限於網站單次查詢只顯示前500筆的限制）。適合快速測試；
  要盡量抓完整個資料庫請用 crawl_exhaustive()。
  """
  csv_file, writer = _open_csv_writer(csv_path)
  counters = {'judgments': 0, 'incidents': 0}

  try:
    async with async_playwright() as p:
      browser = await p.chromium.launch(headless=headless)
      context = await browser.new_context()
      page = await context.new_page()

      print('正在前往司法院進階查詢系統...')
      await _fill_advanced_search(page, keyword, only_criminal)
      await page.click('#btnQry')
      await page.wait_for_load_state('networkidle')

      data_frame = page.frame(name='iframe-data')
      if data_frame is None:
        print('找不到查詢結果頁框（iframe-data），司法院網站結構可能又改版了。')
        return

      await _process_listing_frame(data_frame, context, writer, csv_file, counters, start_page)
      await browser.close()
  finally:
    csv_file.close()

  print(
      f"\n抓取結束。共處理 {counters['judgments']} 份判決書，"
      f"收集 {counters['incidents']} 則逐則言論資料，已存入 {csv_path}"
  )


PROGRESS_FILE_SUFFIX = '.progress.txt'


def _load_done_ranges(progress_path):
  if not os.path.exists(progress_path):
    return set()
  with open(progress_path, 'r', encoding='utf-8') as f:
    return set(line.strip() for line in f if line.strip())


def _mark_range_done(progress_path, range_key):
  with open(progress_path, 'a', encoding='utf-8') as f:
    f.write(range_key + '\n')


async def crawl_exhaustive(
    keyword='妨害名譽',
    csv_path='defamation_judgments.csv',
    only_criminal=True,
    headless=True,
    start_roc=(60, 1, 1),
    end_roc=None,
    cap=500,
):
  """用裁判日期區間遞迴切分查詢條件，逐一把每個 <=cap 筆的區間抓完，藉此突破網站單次
  查詢只顯示前500筆結果的限制，盡量把符合條件的判決書全部收錄。

  進度會記錄在 <csv_path>.progress.txt，中斷後重新執行會自動跳過已完成的區間。
  """
  if end_roc is None:
    today = date.today()
    end_roc = _date_to_roc(today)

  progress_path = csv_path + PROGRESS_FILE_SUFFIX
  done_ranges = _load_done_ranges(progress_path)

  csv_file, writer = _open_csv_writer(csv_path)
  counters = {'judgments': 0, 'incidents': 0}

  try:
    async with async_playwright() as p:
      browser = await p.chromium.launch(headless=headless)
      context = await browser.new_context()
      probe_page = await context.new_page()

      print(f'正在用裁判日期切分查詢範圍（民國{start_roc[0]}年 ~ {end_roc[0]}年），繞過500筆上限...')
      leaf_ranges = await find_date_ranges(probe_page, keyword, only_criminal, start_roc, end_roc, cap)
      print(f'切分完成，共 {len(leaf_ranges)} 個查詢區間，總筆數約 {sum(c for _, _, c in leaf_ranges)}。')

      for start_r, end_r, count in leaf_ranges:
        range_key = f'{start_r}-{end_r}'
        if range_key in done_ranges:
          print(f'區間 {range_key}（{count}筆）已抓過，略過。')
          continue

        print(f'=== 開始抓取區間 {range_key}，預期 {count} 筆 ===')
        list_page = await context.new_page()
        try:
          await _fill_advanced_search(list_page, keyword, only_criminal, (start_r, end_r))
          await list_page.click('#btnQry')
          await list_page.wait_for_load_state('networkidle')
          data_frame = list_page.frame(name='iframe-data')
          if data_frame is None:
            print(f'區間 {range_key} 找不到查詢結果頁框，略過。')
            continue
          await _process_listing_frame(data_frame, context, writer, csv_file, counters)
          _mark_range_done(progress_path, range_key)
          done_ranges.add(range_key)
        except Exception as e:
          print(f'區間 {range_key} 抓取失敗：{e}，繼續下一個區間。')
        finally:
          await list_page.close()

      await browser.close()
  finally:
    csv_file.close()

  print(
      f"\n全部區間處理完畢。共處理 {counters['judgments']} 份判決書，"
      f"收集 {counters['incidents']} 則逐則言論資料，已存入 {csv_path}"
  )


if __name__ == '__main__':
  asyncio.run(crawl_exhaustive(keyword='妨害名譽'))
