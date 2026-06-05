import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, urljoin
import time
import json
import os
from collections import deque
from tqdm import tqdm
from dotenv import load_dotenv
import re
import random


load_dotenv()

BASE_URL = "https://www.thss.tsinghua.edu.cn/"
DOMAIN = "thss.tsinghua.edu.cn"

OUTPUT_FILE = "app/data/raw/raw_pages.jsonl"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive"
}

CMS_SELECTORS = [
    "#vsb_content",     
    ".article-render",
    ".article",
    ".content",
    ".main",
    "#content",
    "article"
]

def extract_main_content(soup):
    # -------------------------
    # remove boilerplate globally
    # -------------------------
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    # -------------------------
    # Layer 1: CMS selectors
    # -------------------------
    for selector in CMS_SELECTORS:
        node = soup.select_one(selector)
        if node:
            text = node.get_text("\n", strip=True)
            if len(text) > 120:
                return text

    # -------------------------
    # Layer 2: heuristic scoring fallback
    # -------------------------
    candidates = soup.find_all(["div", "section", "article"])

    best = None
    best_score = 0

    for c in candidates:
        text = c.get_text(" ", strip=True)
        
        if len(text) < 120:
            continue

        links = len(c.find_all("a"))
        link_ratio = links / max(len(text), 1)

        score = len(text) * (1 - min(link_ratio, 0.5))

        if score > best_score:
            best_score = score
            best = c

    if best:
        return best.get_text("\n", strip=True)

    # -------------------------
    # Layer 3: absolute fallback
    # -------------------------
    return soup.get_text("\n", strip=True)

def is_article_page(soup, url, text):
    score = 0

    path = urlparse(url).path.lower()

    # -------------------------
    # URL signals
    # -------------------------
    if any(x in path for x in ["index", "list", "xsdt", "xydt"]):
        score -= 50

    if re.search(r"\d{4}", path):
        score += 10

    # -------------------------
    # strong article signal
    # -------------------------
    if soup.select_one("#vsb_content"):
        score += 50

    if soup.find("p") and len(text) > 500:
        score += 15

    # -------------------------
    # structure signals
    # -------------------------

    if len(text.split("。")) > 5:
        score += 10

    if len(soup.find_all("p")) > 2:
        score += 10

    # -------------------------
    # list-page signal
    # -------------------------
    links = len(soup.find_all("a"))
    link_density = links / max(len(text), 1)

    if link_density > 0.4:
        score -= 30

    if text.count("首页") > 3:
        score -= 20

    return score >= 40, score
class THSSCrawler:
    def __init__(self, max_pages=1000, delay=0.5):
        self.visited = set()
        self.queue = deque([BASE_URL])
        self.frontier_set = set([BASE_URL])
        self.max_pages = max_pages
        self.delay = delay
        self.session = requests.Session()
        self.saved_pages = 0
        self.skipped_pages = 0

        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )

        adapter = HTTPAdapter(max_retries=retry)

        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        os.makedirs("data/raw", exist_ok=True)

    # ---------------------------
    # URL validation
    # ---------------------------
    def is_valid_url(self, url):
        try:
            parsed = urlparse(url)
            return DOMAIN in parsed.netloc
        except:
            return False

    def normalize_url(self, url):
        parsed = urlparse(url)

        clean_query = urlencode(sorted(parse_qsl(parsed.query)))

        parsed = parsed._replace(fragment="")

        return urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path.rstrip("/"),
            "",
            clean_query,
            ""
        ))
    # ---------------------------
    # HTTP fetch
    # ---------------------------
    def fetch(self, url, retries=3):
        last_error = None

        for attempt in range(retries):
            try:
                resp = self.session.get(
                    url,
                    headers=HEADERS,
                    timeout=(5, 10)
                )

                # handle errors explicitly
                if resp.status_code >= 400:
                    print(f"HTTP {resp.status_code} for {url} ignoring content")
                    return None

                content_type = resp.headers.get("Content-Type", "")

                if "text/html" not in content_type.lower():
                    print(f"Skipping non-HTML content: {url} ({content_type})")
                    return None

                resp.encoding = resp.apparent_encoding or resp.encoding or "utf-8"
                return resp.text

            except Exception as e:
                last_error = str(e)
                sleep_time = (0.5 * (2 ** attempt)) + random.uniform(0, 0.5)
                print(f"Attempt {attempt + 1} failed for {url} -> {last_error}")
                time.sleep(sleep_time)

        # final failure
        print(f"[FAILED] {url} -> {last_error}")
        return None

    # ---------------------------
    # Extract links
    # ---------------------------
    def extract_links(self, html, base_url):
        soup = BeautifulSoup(html, "html.parser")
        links = []

        for a in soup.find_all("a", href=True):
            href = a.get("href")
            if not href:
                continue

            full_url = urljoin(base_url, href)
            full_url = self.normalize_url(full_url)

            if self.is_valid_url(full_url):
                links.append(full_url)

        return links

    # ---------------------------
    # Extract article content
    # ---------------------------
    def extract_page(self, html, url):
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["nav", "footer", "header", "script", "style", "noscript"]):
            tag.decompose()

        # title
        title_tag = soup.find("h1") or soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        date = None

        # 1. structured time tag
        time_tag = soup.find("time")
        if time_tag and time_tag.get("datetime"):
            date = time_tag["datetime"]

        # 2. fallback: class-based date fields
        if not date:
            date_tag = (
                soup.find(class_=re.compile("date", re.I))
                or soup.find("span", string=re.compile(r"20\d{2}"))
            )

            if date_tag:
                text = date_tag.get_text(strip=True)
                match = re.search(r"(20\d{2}[-/年]\d{1,2}[-/月]\d{1,2}日?)", text)
                if match:
                    date = match.group(1)

        # 3. last resort: full text scan
        if not date:
            text_blob = soup.get_text(" ", strip=True)
            match = re.search(r"(20\d{2}[-/年]\d{1,2}[-/月]\d{1,2}日?)", text_blob)
            if match:
                date = match.group(1)

        # main content heuristic
        content = extract_main_content(soup)

        # images
        images = []
        for img in soup.find_all("img", src=True):
            src = img["src"].lower()

            # skip obvious non-content images
            if any(x in src for x in ["logo", "icon", "avatar", "sprite", "banner"]):
                continue

            full_url = urljoin(url, img["src"])
            images.append(full_url)

        return {
            "url": url,
            "title": title,
            "date": date,
            "content": content,
            "images": images
        }

    # ---------------------------
    # Save to JSONL
    # ---------------------------
    def save(self, data):
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # ---------------------------
    # Crawl loop
    # ---------------------------
    def run(self):
        print(f"Starting crawl: {BASE_URL}")
        print(f"Max pages: {self.max_pages}")
        
        pbar = tqdm(total=self.max_pages)

        while self.queue and len(self.visited) < self.max_pages:
            url = self.normalize_url(self.queue.popleft())
            self.frontier_set.discard(url)

            if url in self.visited:
                continue

            html = self.fetch(url)
            if not html:
                continue

            self.visited.add(url)

            page_data = self.extract_page(html, url)

            soup = BeautifulSoup(html, "html.parser")

            is_article, score = is_article_page(
                soup,
                url,
                page_data["content"]
            )

            if is_article:
                self.save(page_data)
                self.saved_pages += 1
            else:
                self.skipped_pages += 1
                #print(f"SKIP non-article ({score}): {url}")

            # enqueue new links
            links = self.extract_links(html, url)
            for link in links:
                link = self.normalize_url(link)

                if link not in self.visited and link not in self.frontier_set:
                    self.queue.append(link)
                    self.frontier_set.add(link)

            pbar.update(1)
            time.sleep(random.uniform(0.3, 0.8))

        pbar.close()
        print(
            f"Crawling complete. "
            f"Fetched: {len(self.visited)}, "
            f"Saved articles: {self.saved_pages}, "
            f"Skipped: {self.skipped_pages}"
        )


if __name__ == "__main__":
    crawler = THSSCrawler(max_pages=30, delay=0.5)
    crawler.run()