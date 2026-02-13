"""
Project: Kaohsiung E Sky Mall History Spider (V12.0 Comment Mining)
Author: Senior Data Engineer (Gemini)
Date: 2026-02-12
Key Upgrades:
    - PTT Comment Mining: Extracts all pushes/comments as individual data points.
    - Data Explosion: 1 Post -> 50+ Data Points.
    - Robust Error Handling: Fixed previous 'href' errors.
"""

import time
import random
import datetime
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Selenium Imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# --- Configuration ---
TARGET_KEYWORDS = ["義享", "義享天地", "高雄萬豪"] 
CUTOFF_DATE = datetime.datetime(2021, 3, 28) 
OUTPUT_FILE = "my_data.csv"

# Global Headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

class EskyHistorySpiderV10:
    def __init__(self):
        self.data_list = []
        self.driver = None
        self.processed_links = set()

    def _log(self, source, msg):
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [{source}] {msg}")

    def _clean_text(self, text):
        return text.strip().replace('\n', ' ').replace(',', '，')

    def _init_selenium(self):
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox") 
        chrome_options.add_argument("--disable-dev-shm-usage") 
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument(f"user-agent={HEADERS['User-Agent']}")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver

    def _parse_fuzzy_date(self, date_str):
        now = datetime.datetime.now()
        try:
            if "T" in date_str: return datetime.datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None)
            if re.match(r"\d{4}-\d{2}-\d{2}", date_str): return datetime.datetime.strptime(date_str.split(" ")[0], "%Y-%m-%d")
            if re.match(r"\d{1,2}/\d{1,2}", date_str): 
                dt = datetime.datetime.strptime(date_str, "%m/%d")
                return dt.replace(year=now.year)
        except: return None
        return None

    # ==========================
    # Module 1: PTT (Deep Mining)
    # ==========================
    def crawl_ptt(self):
        self._log("PTT", "Starting Comment Mining...")
        base_url = "https://www.ptt.cc"
        
        with requests.Session() as s:
            s.headers.update(HEADERS)
            s.cookies.update({'over18': '1'})

            for kw in TARGET_KEYWORDS:
                self._log("PTT", f"Searching: {kw}")
                # 限制抓前 3 頁，因為每頁展開後資料量會很大
                for page in range(1, 4):
                    page_url = f"{base_url}/bbs/Kaohsiung/search?page={page}&q={kw}"
                    try:
                        res = s.get(page_url, timeout=10)
                        if res.status_code != 200: break
                        
                        soup = BeautifulSoup(res.text, "html.parser")
                        r_ents = soup.select("div.r-ent")
                        
                        if not r_ents: break

                        for rent in r_ents:
                            title_tag = rent.select_one("div.title a")
                            if not title_tag: continue
                            
                            link = base_url + title_tag['href']
                            if link in self.processed_links: continue
                            
                            title = title_tag.text.strip()
                            
                            # --- 進入內頁抓推文 ---
                            try:
                                art_res = s.get(link, timeout=5)
                                art_soup = BeautifulSoup(art_res.text, "html.parser")
                                
                                # 1. 抓時間
                                metas = art_soup.select(".article-metaline .article-meta-value")
                                post_date = None
                                if metas and len(metas) >= 3:
                                    try:
                                        post_date = datetime.datetime.strptime(metas[2].text.strip(), "%a %b %d %H:%M:%S %Y")
                                    except: pass
                                
                                if post_date and post_date < CUTOFF_DATE: continue

                                date_str = post_date.strftime("%Y-%m-%d") if post_date else datetime.datetime.now().strftime("%Y-%m-%d")
                                
                                # 2. 存本文 (標題)
                                self.data_list.append({
                                    "date": date_str,
                                    "source": "PTT_Post",
                                    "content": f"[標題] {self._clean_text(title)}",
                                    "link": link
                                })
                                
                                # 3. 抓推文 (核心改動)
                                pushes = art_soup.select("div.push")
                                for p in pushes:
                                    push_tag = p.select_one("span.push-tag")
                                    push_content = p.select_one("span.push-content")
                                    
                                    if push_tag and push_content:
                                        tag = push_tag.text.strip() # 推, 噓, →
                                        content = push_content.text.strip().lstrip(': ')
                                        
                                        # 過濾掉太短的推文 (如 "推", "XD")
                                        if len(content) < 2: continue
                                        
                                        self.data_list.append({
                                            "date": date_str, # 推文時間通常沿用發文日期，或簡略處理
                                            "source": "PTT_Comment", # 標記為推文
                                            "content": f"[{tag}] {content}",
                                            "link": link # 連結共用
                                        })
                                
                                self.processed_links.add(link)
                                time.sleep(0.5) # 禮貌性延遲
                                
                            except Exception as e:
                                continue
                                
                    except Exception as e:
                        self._log("PTT", f"Page Error: {e}")
                        break

    # ==========================
    # Module 2: Mobile01 (Standard)
    # ==========================
    def crawl_mobile01(self):
        self._log("Mobile01", "Starting Selenium Crawl...")
        if not self.driver: self.driver = self._init_selenium()
        
        for kw in TARGET_KEYWORDS:
            base_search = f"https://www.mobile01.com/search.php?key={kw}&m=forum"
            # 只抓前 3 頁
            for page in range(1, 4):
                url = f"{base_search}&p={page}"
                self.driver.get(url)
                time.sleep(3)
                
                soup = BeautifulSoup(self.driver.page_source, "html.parser")
                items = soup.select(".c-searchTableList .c-listTableTr")
                
                if not items: break
                
                for item in items:
                    try:
                        t_div = item.select_one(".c-listTableTd-title a")
                        d_div = item.select_one(".o-fNotes-date")
                        if not t_div: continue
                        
                        link = "https://www.mobile01.com/" + t_div['href']
                        if link in self.processed_links: continue
                        
                        title = t_div.text.strip()
                        date_str = d_div.text.strip() if d_div else ""
                        post_date = self._parse_fuzzy_date(date_str)
                        
                        if post_date and post_date < CUTOFF_DATE: continue
                        
                        self.data_list.append({
                            "date": post_date.strftime("%Y-%m-%d") if post_date else "",
                            "source": "Mobile01",
                            "content": self._clean_text(title),
                            "link": link
                        })
                        self.processed_links.add(link)
                    except: continue

    # ==========================
    # Module 3: Dcard (Standard)
    # ==========================
    def crawl_dcard(self):
        self._log("Dcard", "Starting Selenium Crawl...")
        if not self.driver: self.driver = self._init_selenium()

        for kw in TARGET_KEYWORDS:
            url = f"https://www.dcard.tw/search/posts?query={kw}&sort=latest"
            self.driver.get(url)
            time.sleep(5)
            
            # 簡單滾動 3 次
            for _ in range(3):
                links = self.driver.find_elements(By.TAG_NAME, "a")
                for a in links:
                    try:
                        href = a.get_attribute("href")
                        if href and "/p/" in href and "/b/" not in href:
                            if href in self.processed_links: continue
                            title = a.text.strip()
                            if len(title) < 4: continue
                            
                            self.data_list.append({
                                "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                                "source": "Dcard",
                                "content": self._clean_text(title),
                                "link": href
                            })
                            self.processed_links.add(href)
                    except: continue
                
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(3)

    def close(self):
        if self.driver: self.driver.quit()

if __name__ == "__main__":
    spider = EskyHistorySpiderV10()
    try:
        spider.crawl_ptt()
        spider.crawl_mobile01()
        spider.crawl_dcard()
        
        df = pd.DataFrame(spider.data_list)
        # 不再以 Link 去重，因為同一篇文會有多個推文 (Link 相同)
        # 改以 Content 去重，避免抓到重複的推文
        if not df.empty:
            df.drop_duplicates(subset=['content'], inplace=True)
            df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
            print(f"Saved {len(df)} records (Included Comments).")
            print(df['source'].value_counts())
        else:
            print("No data found.")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        spider.close()