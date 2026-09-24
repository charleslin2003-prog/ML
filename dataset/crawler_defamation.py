import asyncio
import csv
import os
import re
from urllib.parse import urljoin

from playwright.async_api import async_playwright

CSV_FIELDS = [
    '裁判字號', '裁判日期', '案由', '編號', '日期', '時間',
    '社交平台', '平台帳號', '言論內容', '事實要旨', '涉犯罪名',
    '告證編號', '原文網址',
]


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
      'verdict': verdict,
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


async def crawl_judgments(
    keyword='妨害名譽',
    max_pages=None,
    start_page=1,
    csv_path='defamation_judgments.csv',
    only_criminal=True,
    headless=True,
):
  """用司法院「進階查詢」的裁判案由欄位搜尋，逐頁抓取判決書並即時寫入 CSV。

  max_pages=None 代表不設上限，抓到查無下一頁為止。
  start_page 可從指定頁數開始（例如中斷後接續抓取）。
  """
  csv_file, writer = _open_csv_writer(csv_path)
  total_judgments = 0
  total_incidents = 0

  try:
    async with async_playwright() as p:
      browser = await p.chromium.launch(headless=headless)
      context = await browser.new_context()
      page = await context.new_page()

      print('正在前往司法院進階查詢系統...')
      await page.goto('https://judgment.judicial.gov.tw/FJUD/Default_AD.aspx')
      await page.wait_for_load_state('networkidle')

      # 用「裁判案由」欄位查詢，比全文關鍵字搜尋精準很多（不會撈到只是內文提到關鍵字的無關案件）
      await page.fill('#jud_title', keyword)
      if only_criminal:
        checkboxes = await page.query_selector_all('input[name="jud_sys"]')
        for cb in checkboxes:
          if await cb.get_attribute('value') == 'M':  # M = 刑事
            await cb.check()

      await page.click('#btnQry')
      await page.wait_for_load_state('networkidle')

      # 查詢結果實際上是載入在名為 iframe-data 的子頁框內，不在主頁面上
      data_frame = page.frame(name='iframe-data')
      if data_frame is None:
        print('找不到查詢結果頁框（iframe-data），司法院網站結構可能又改版了。')
        return

      # 從「下一頁」連結取得分頁網址樣板，之後直接組網址跳頁，不用一直點擊
      next_link = await data_frame.query_selector('#hlNext')
      page_url_template = None
      if next_link is not None:
        next_href = urljoin(data_frame.url, await next_link.get_attribute('href'))
        page_url_template = re.sub(r'page=\d+', 'page={n}', next_href)

      current_page = start_page
      if current_page > 1:
        if page_url_template is None:
          print('無分頁樣板可跳頁，從第 1 頁開始。')
          current_page = 1
        else:
          await data_frame.goto(page_url_template.format(n=current_page), timeout=15000)
          await data_frame.wait_for_load_state('networkidle')

      while True:
        if max_pages is not None and current_page > start_page + max_pages - 1:
          print(f'已達本次上限 {max_pages} 頁，停止。')
          break

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
            date = (await date_elem.inner_text()).strip() if date_elem else ''
            reason = (
                (await reason_elem.inner_text()).strip() if reason_elem else ''
            )
            full_url = urljoin(data_frame.url, href)
            case_links.append(
                {'title': title, 'url': full_url, 'date': date, 'reason': reason}
            )

        if not case_links:
          print('本頁沒有任何案件，查詢結果已抓取完畢。')
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
                total_incidents += 1
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
            total_judgments += 1
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
          print(f'跳到第 {current_page} 頁失敗：{e}，停止抓取。')
          break
        await asyncio.sleep(1.0)

      await browser.close()
  finally:
    csv_file.close()

  print(
      f'\n抓取結束。共處理 {total_judgments} 份判決書，'
      f'收集 {total_incidents} 則逐則言論資料，已存入 {csv_path}'
  )


if __name__ == '__main__':
  asyncio.run(crawl_judgments(keyword='妨害名譽', max_pages=None))
