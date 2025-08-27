import asyncio
import json
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
import os
import re
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

# FastAPI imports
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

# Set Gemini API Key
os.environ['GOOGLE_API_KEY'] = 'AIzaSyDypUivUcbbuJMzKtpqzGeAQII5LbUkc_k'

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Global state for scraping
SCRAPING_STATE = {
    "extracted_data": [],
    "robots_cache": {},
    "headers_list": [
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
    ],
    "current_header_index": 0
}


def get_next_headers() -> Dict[str, str]:
    """Rotate headers for each request"""
    headers = SCRAPING_STATE["headers_list"][SCRAPING_STATE["current_header_index"]].copy()
    SCRAPING_STATE["current_header_index"] = (SCRAPING_STATE["current_header_index"] + 1) % len(
        SCRAPING_STATE["headers_list"])
    headers.update({
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })
    return headers


#############################
# Scraping Helper Functions  #
#############################

# Helper: Find product links in soup
def find_product_links(soup, base_url):
    """
    Find links that look like product detail pages only.
    Returns unique absolute URLs.
    """
    keywords = ["product"]
    found = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        abs_url = urljoin(base_url, href)
        if any(kw in href.lower() for kw in keywords):
            found.add(abs_url)
    return list(found)

# Helper: Extract price from a container (with optional soup for script fallback)
def extract_price(container, soup=None):
    import re
    def clean_price(val):
        m = re.search(r'[\d]+(?:[.,]\d+)?', val)
        if m:
            price_str = m.group(0).replace(',', '.')
            if '.' not in price_str:
                if len(price_str) > 2:
                    price_str = price_str[:-2] + '.' + price_str[-2:]
            return price_str
        return None

    price = None
    price_patterns = [
        ('span', re.compile('price|amount|money|value', re.I)),
        ('div', re.compile('price|amount|money|value', re.I)),
        (None, re.compile('current-price|regular-price', re.I)),
        (None, {"itemprop": "price"}),
    ]
    for tag, pat in price_patterns:
        if isinstance(pat, dict):
            elem = container.find(attrs=pat)
        else:
            elem = container.find(tag, class_=pat) if tag else container.find(attrs={"class": pat})
        if elem:
            txt = elem.get_text(" ", strip=True)
            if txt:
                return clean_price(txt)
            for attr in ["content", "data-price", "aria-label", "value"]:
                attr_val = elem.get(attr)
                if attr_val:
                    return clean_price(attr_val.strip())
    if soup:
        for sc in soup.find_all("script", string=True):
            m = re.search(r'"price"\s*[:=]\s*"?([\d.,]+)"?', sc.string or "")
            if m:
                return clean_price(m.group(1))
    return price


# Scraping Functions
async def check_robots_txt(url: str) -> Dict[str, Any]:
    """Check robots.txt for the given URL"""
    try:
        parsed_url = urlparse(url)
        robot_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"

        if robot_url in SCRAPING_STATE["robots_cache"]:
            return SCRAPING_STATE["robots_cache"][robot_url]

        rp = RobotFileParser()
        rp.set_url(robot_url)
        rp.read()

        crawl_delay = rp.crawl_delay("*") or 1.0
        can_fetch = rp.can_fetch("*", url)

        result = {
            "url": url,
            "can_fetch": can_fetch,
            "crawl_delay": crawl_delay,
            "status": "allowed" if can_fetch else "disallowed"
        }

        SCRAPING_STATE["robots_cache"][robot_url] = result
        return result

    except Exception as e:
        return {
            "url": url,
            "status": "no_robots_txt",
            "can_fetch": True,
            "crawl_delay": 1.0,
            "error": str(e)
        }


async def analyze_website(url: str) -> Dict[str, Any]:
    """Analyze website structure"""
    analysis = {
        "url": url,
        "is_dynamic": False,
        "is_ecommerce": False,
        "detected_patterns": [],
        "recommended_method": "beautifulsoup"
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = get_next_headers()
            response = await client.get(url, headers=headers, follow_redirects=True)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Check for dynamic content
            scripts = soup.find_all('script')
            for script in scripts:
                script_text = str(script).lower()
                if any(fw in script_text for fw in ['react', 'angular', 'vue', 'next']):
                    analysis["is_dynamic"] = True
                    analysis["recommended_method"] = "selenium"
                    break

            # Check for e-commerce patterns
            ecommerce_keywords = ['price', 'product', 'cart', 'add-to-cart', 'buy', 'shop']
            for keyword in ecommerce_keywords:
                if soup.find(class_=re.compile(keyword, re.I)):
                    analysis["is_ecommerce"] = True
                    analysis["detected_patterns"].append(f"{keyword} pattern found")

            return analysis

    except Exception as e:
        analysis["error"] = str(e)
        return analysis


async def extract_with_beautifulsoup(url: str) -> Dict[str, Any]:
    """Extract data using BeautifulSoup, with product link expansion."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = get_next_headers()
            response = await client.get(url, headers=headers, follow_redirects=True)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find product/listing links before extracting
            product_links = find_product_links(soup, url)
            product_links = list(dict.fromkeys(product_links))  # unique, preserve order
            # Use product_limit from global state if set, else default to 5
            product_limit = SCRAPING_STATE.get("product_limit", 5)

            extracted_data = {
                "url": url,
                "timestamp": datetime.now().isoformat(),
                "method": "beautifulsoup",
                "data": []
            }

            # Try extracting from this page first
            container_selectors = (
                'div[class*="product"], div[class*="grid"], li[class*="product"], '
                'article[class*="product"], section[class*="product"], '
                'div[class*="collection"], div[class*="item"], div[class*="card"]'
            )
            containers = soup.select(container_selectors)
            # If multiple containers found, this is likely a listing/collection page -> skip
            if len(containers) > 1:
                containers = []

            # Extraction for each container
            for container in containers[:product_limit]:
                item_data = {}
                # --- Extract title ---
                title = None
                for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    elem = container.find(tag)
                    if elem and elem.get_text(" ", strip=True):
                        title = elem.get_text(" ", strip=True)
                        break
                title_patterns = [
                    ('a', re.compile('title|name', re.I)),
                    ('div', re.compile('title|name', re.I)),
                    ('span', re.compile('title|name', re.I)),
                    (None, re.compile('product-title|product_name|product-name', re.I)),
                    (None, {"itemprop": "name"})
                ]
                if not title:
                    for tag, pat in title_patterns:
                        if tag and isinstance(pat, re.Pattern):
                            elem = container.find(tag, class_=pat)
                        elif tag and isinstance(pat, dict):
                            elem = container.find(tag, attrs=pat)
                        elif isinstance(pat, re.Pattern):
                            elem = container.find(attrs={"class": pat})
                        elif isinstance(pat, dict):
                            elem = container.find(attrs=pat)
                        else:
                            elem = None
                        if elem and elem.get_text(" ", strip=True):
                            title = elem.get_text(" ", strip=True)
                            break
                if not title:
                    img = container.find('img')
                    if img and img.get('alt'):
                        title = img.get('alt').strip()
                if not title and soup.find('title'):
                    title = soup.find('title').get_text(" ", strip=True)
                if title:
                    item_data['title'] = title
                # --- Extract link ---
                link = None
                a_tag = container.find('a', href=True)
                if a_tag and a_tag.get('href'):
                    link = urljoin(url, a_tag['href'])
                if not link:
                    link = url
                item_data['link'] = link
                # --- Extract price ---
                price = extract_price(container, soup)
                if price:
                    item_data['price'] = price
                # Only keep title, link, price
                if item_data.get("title") or item_data.get("price"):
                    extracted_data["data"].append({
                        "title": item_data.get("title"),
                        "link": item_data.get("link"),
                        "price": item_data.get("price")
                    })

            # If no containers found, or no data, try to follow product links and extract from there
            if (not containers or not extracted_data["data"]) and product_links:
                # Limit the number of links to follow
                follow_links = product_links[:min(5, product_limit)]
                for prod_link in follow_links:
                    try:
                        headers2 = get_next_headers()
                        resp2 = await client.get(prod_link, headers=headers2, follow_redirects=True)
                        soup2 = BeautifulSoup(resp2.text, 'html.parser')
                        containers2 = soup2.select(container_selectors)
                        for container in containers2[:3]:
                            item_data = {}
                            # --- Extract title ---
                            title = None
                            for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                                elem = container.find(tag)
                                if elem and elem.get_text(" ", strip=True):
                                    title = elem.get_text(" ", strip=True)
                                    break
                            title_patterns = [
                                ('a', re.compile('title|name', re.I)),
                                ('div', re.compile('title|name', re.I)),
                                ('span', re.compile('title|name', re.I)),
                                (None, re.compile('product-title|product_name|product-name', re.I)),
                                (None, {"itemprop": "name"})
                            ]
                            if not title:
                                for tag, pat in title_patterns:
                                    if tag and isinstance(pat, re.Pattern):
                                        elem = container.find(tag, class_=pat)
                                    elif tag and isinstance(pat, dict):
                                        elem = container.find(tag, attrs=pat)
                                    elif isinstance(pat, re.Pattern):
                                        elem = container.find(attrs={"class": pat})
                                    elif isinstance(pat, dict):
                                        elem = container.find(attrs=pat)
                                    else:
                                        elem = None
                                    if elem and elem.get_text(" ", strip=True):
                                        title = elem.get_text(" ", strip=True)
                                        break
                            if not title:
                                img = container.find('img')
                                if img and img.get('alt'):
                                    title = img.get('alt').strip()
                            if not title and soup2.find('title'):
                                title = soup2.find('title').get_text(" ", strip=True)
                            if title:
                                item_data['title'] = title
                            # --- Extract link ---
                            link = prod_link
                            item_data['link'] = link
                            # --- Extract price ---
                            price = extract_price(container, soup2)
                            if price:
                                item_data['price'] = price
                            # Only keep title, link, price
                            if item_data.get("title") or item_data.get("price"):
                                extracted_data["data"].append({
                                    "title": item_data.get("title"),
                                    "link": item_data.get("link"),
                                    "price": item_data.get("price")
                                })
                    except Exception:
                        continue

            # If still no data, try LLM extraction (from containers)
            if not extracted_data["data"] and containers:
                N = min(len(containers), 3)
                html_blocks = []
                for c in containers[:N]:
                    html_blocks.append(str(c))
                html_to_analyze = "\n\n".join(html_blocks)
                prompt = (
                    "Extract product information (title, link, price) from this HTML. "
                    "Return JSON list with keys: title, link, price.\n"
                    f"HTML:\n{html_to_analyze}"
                )
                try:
                    llm_response = await llm.ainvoke(prompt)
                    import json as _json
                    resp_text = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
                    json_match = re.search(r'(\[.*\])', resp_text, re.DOTALL)
                    if json_match:
                        resp_json = json_match.group(1)
                        llm_data = _json.loads(resp_json)
                        for item in llm_data:
                            extracted_data["data"].append({
                                "title": item.get("title"),
                                "link": urljoin(url, item.get("link")) if item.get("link") else url,
                                "price": item.get("price")
                            })
                except Exception:
                    pass

            # Fallback to general page info
            if not extracted_data["data"]:
                page_data = {
                    "title": soup.find('title').get_text() if soup.find('title') else "No title",
                    "link": url,
                    "price": None
                }
                extracted_data["data"].append(page_data)

            # Only keep title, link, price in each dict
            for d in extracted_data["data"]:
                keys = list(d.keys())
                for k in keys:
                    if k not in ("title", "link", "price"):
                        del d[k]
            return extracted_data

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "data": []
        }


async def extract_with_selenium(url: str) -> Dict[str, Any]:
    """Extract data using Selenium for dynamic sites, with product link expansion."""
    driver = None
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        try:
            driver = webdriver.Chrome(options=chrome_options)
        except:
            return {
                "status": "error",
                "error": "Chrome driver not found. Please install Chrome and ChromeDriver.",
                "data": []
            }
        driver.get(url)
        time.sleep(3)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        # Find product/listing links before extracting
        product_links = find_product_links(soup, url)
        product_links = list(dict.fromkeys(product_links))
        # Use product_limit from global state if set, else default to 5
        product_limit = SCRAPING_STATE.get("product_limit", 5)
        extracted_data = {
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "method": "selenium",
            "data": []
        }
        container_selectors = (
            'div[class*="product"], div[class*="grid"], li[class*="product"], '
            'article[class*="product"], section[class*="product"], '
            'div[class*="collection"], div[class*="item"], div[class*="card"]'
        )
        containers = soup.select(container_selectors)
        # If multiple containers found, this is likely a listing/collection page -> skip
        if len(containers) > 1:
            containers = []
        for container in containers[:product_limit]:
            item_data = {}
            # --- Extract title ---
            title = None
            for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                elem = container.find(tag)
                if elem and elem.get_text(" ", strip=True):
                    title = elem.get_text(" ", strip=True)
                    break
            title_patterns = [
                ('a', re.compile('title|name', re.I)),
                ('div', re.compile('title|name', re.I)),
                ('span', re.compile('title|name', re.I)),
                (None, re.compile('product-title|product_name|product-name', re.I)),
                (None, {"itemprop": "name"})
            ]
            if not title:
                for tag, pat in title_patterns:
                    if tag and isinstance(pat, re.Pattern):
                        elem = container.find(tag, class_=pat)
                    elif tag and isinstance(pat, dict):
                        elem = container.find(tag, attrs=pat)
                    elif isinstance(pat, re.Pattern):
                        elem = container.find(attrs={"class": pat})
                    elif isinstance(pat, dict):
                        elem = container.find(attrs=pat)
                    else:
                        elem = None
                    if elem and elem.get_text(" ", strip=True):
                        title = elem.get_text(" ", strip=True)
                        break
            if not title:
                img = container.find('img')
                if img and img.get('alt'):
                    title = img.get('alt').strip()
            if not title and soup.find('title'):
                title = soup.find('title').get_text(" ", strip=True)
            if title:
                item_data['title'] = title
            # --- Extract link ---
            link = None
            a_tag = container.find('a', href=True)
            if a_tag and a_tag.get('href'):
                link = urljoin(url, a_tag['href'])
            if not link:
                link = url
            item_data['link'] = link
            # --- Extract price ---
            price = extract_price(container, soup)
            if price:
                item_data['price'] = price
            # Only keep title, link, price
            if item_data.get("title") or item_data.get("price"):
                extracted_data["data"].append({
                    "title": item_data.get("title"),
                    "link": item_data.get("link"),
                    "price": item_data.get("price")
                })
        # If no containers found, or no data, try to follow product links and extract from there
        if (not containers or not extracted_data["data"]) and product_links:
            # Limit the number of links to follow
            follow_links = product_links[:min(5, product_limit)]
            for prod_link in follow_links:
                try:
                    driver.get(prod_link)
                    time.sleep(2)
                    soup2 = BeautifulSoup(driver.page_source, 'html.parser')
                    containers2 = soup2.select(container_selectors)
                    for container in containers2[:3]:
                        item_data = {}
                        title = None
                        for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                            elem = container.find(tag)
                            if elem and elem.get_text(" ", strip=True):
                                title = elem.get_text(" ", strip=True)
                                break
                        title_patterns = [
                            ('a', re.compile('title|name', re.I)),
                            ('div', re.compile('title|name', re.I)),
                            ('span', re.compile('title|name', re.I)),
                            (None, re.compile('product-title|product_name|product-name', re.I)),
                            (None, {"itemprop": "name"})
                        ]
                        if not title:
                            for tag, pat in title_patterns:
                                if tag and isinstance(pat, re.Pattern):
                                    elem = container.find(tag, class_=pat)
                                elif tag and isinstance(pat, dict):
                                    elem = container.find(tag, attrs=pat)
                                elif isinstance(pat, re.Pattern):
                                    elem = container.find(attrs={"class": pat})
                                elif isinstance(pat, dict):
                                    elem = container.find(attrs=pat)
                                else:
                                    elem = None
                                if elem and elem.get_text(" ", strip=True):
                                    title = elem.get_text(" ", strip=True)
                                    break
                        if not title:
                            img = container.find('img')
                            if img and img.get('alt'):
                                title = img.get('alt').strip()
                        if not title and soup2.find('title'):
                            title = soup2.find('title').get_text(" ", strip=True)
                        if title:
                            item_data['title'] = title
                        link = prod_link
                        item_data['link'] = link
                        price = extract_price(container, soup2)
                        if price:
                            item_data['price'] = price
                        if item_data.get("title") or item_data.get("price"):
                            extracted_data["data"].append({
                                "title": item_data.get("title"),
                                "link": item_data.get("link"),
                                "price": item_data.get("price")
                            })
                except Exception:
                    continue
        # If still no data, try LLM extraction (from containers)
        if not extracted_data["data"] and containers:
            N = min(len(containers), 3)
            html_blocks = []
            for c in containers[:N]:
                html_blocks.append(str(c))
            html_to_analyze = "\n\n".join(html_blocks)
            prompt = (
                "Extract product information (title, link, price) from this HTML. "
                "Return JSON list with keys: title, link, price.\n"
                f"HTML:\n{html_to_analyze}"
            )
            try:
                llm_response = await llm.ainvoke(prompt)
                import json as _json
                resp_text = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
                json_match = re.search(r'(\[.*\])', resp_text, re.DOTALL)
                if json_match:
                    resp_json = json_match.group(1)
                    llm_data = _json.loads(resp_json)
                    for item in llm_data:
                        extracted_data["data"].append({
                            "title": item.get("title"),
                            "link": urljoin(url, item.get("link")) if item.get("link") else url,
                            "price": item.get("price")
                        })
            except Exception:
                pass
        if not extracted_data["data"]:
            page_data = {
                "title": soup.find('title').get_text() if soup.find('title') else "No title",
                "link": url,
                "price": None
            }
            extracted_data["data"].append(page_data)
        for d in extracted_data["data"]:
            keys = list(d.keys())
            for k in keys:
                if k not in ("title", "link", "price"):
                    del d[k]
        return extracted_data
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "data": []
        }
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass


# Define workflow state
class ScraperState(TypedDict):
    url: str
    urls: List[str]
    analysis_result: Dict
    robots_check: Dict
    extraction_method: str
    extracted_data: List[Dict]
    error_msg: str
    status: str
    final_output: Dict


# Create workflow
def create_workflow():
    workflow = StateGraph(ScraperState)

    async def analyze_node(state: ScraperState) -> ScraperState:
        url = state.get("url") or (state.get("urls", [])[0] if state.get("urls") else None)
        if not url:
            state["error_msg"] = "No URL provided"
            state["status"] = "failed"
            return state
        result = await analyze_website(url)
        state["analysis_result"] = result
        state["status"] = "analyzed"
        return state

    async def check_robots_node(state: ScraperState) -> ScraperState:
        url = state.get("url") or (state.get("urls", [])[0] if state.get("urls") else None)
        result = await check_robots_txt(url)
        state["robots_check"] = result
        return state

    async def extract_node(state: ScraperState) -> ScraperState:
        if not state.get("robots_check", {}).get("can_fetch", True):
            state["error_msg"] = "Blocked by robots.txt"
            state["status"] = "blocked"
            return state

        input_urls = state.get("urls") or [state.get("url")]
        expanded_urls = []
        product_links_seen = set()
        # For each input url, expand to product links using BeautifulSoup
        async with httpx.AsyncClient(timeout=30.0) as client:
            for url in input_urls:
                try:
                    headers = get_next_headers()
                    response = await client.get(url, headers=headers, follow_redirects=True)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    product_links = find_product_links(soup, url)
                    # Only take up to 10 unique product links per input url
                    for link in product_links:
                        if link not in product_links_seen:
                            expanded_urls.append(link)
                            product_links_seen.add(link)
                            if len(expanded_urls) >= 10:
                                break
                    # If no product links found, fall back to homepage
                    if not product_links:
                        expanded_urls.append(url)
                except Exception:
                    expanded_urls.append(url)
        # Remove duplicates, keep order
        final_urls = list(dict.fromkeys(expanded_urls))
        # Limit to 10 product links total
        final_urls = final_urls[:10]

        all_data = []
        for url in final_urls:
            if state.get("analysis_result", {}).get("is_dynamic"):
                result = await extract_with_selenium(url)
            else:
                result = await extract_with_beautifulsoup(url)
            # Only add title, link, price fields
            if result.get("data"):
                for item in result["data"]:
                    all_data.append({
                        "title": item.get("title"),
                        "link": item.get("link"),
                        "price": item.get("price")
                    })
        state["extracted_data"] = all_data
        state["status"] = "extracted" if all_data else "no_data"
        return state

    async def save_node(state: ScraperState) -> ScraperState:
        if not state.get("extracted_data"):
            state["error_msg"] = "No data to save"
            state["status"] = "failed"
            return state

        # Filter duplicates by link
        all_data = state["extracted_data"]
        seen_links = set()
        unique_data = []
        for item in all_data:
            link = item.get("link")
            if link and link not in seen_links:
                unique_data.append(item)
                seen_links.add(link)
        state["extracted_data"] = unique_data

        state["status"] = "completed"
        state["final_output"] = {
            "records": len(state["extracted_data"]),
            "timestamp": datetime.now().isoformat()
        }
        return state

    # Add nodes
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("check_robots", check_robots_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("save", save_node)

    # Set flow
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "check_robots")
    workflow.add_edge("check_robots", "extract")
    workflow.add_edge("extract", "save")
    workflow.add_edge("save", END)

    return workflow.compile()



# FastAPI app
app = FastAPI(title="Web Scraper API")

@app.get("/")
async def root():
    return {"message": "Web Scraper API is running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/analyze")
async def analyze_endpoint(url: str = Query(..., description="URL to analyze")):
    result = await analyze_website(url)
    return result

@app.post("/scrape")
async def scrape_endpoint(
    body: Dict[str, Any] = Body(...)
):
    """
    Accepts JSON body:
    {
        "urls": [ ... ],
        "product_limit": 5,
        "output_format": "json",
        "use_selenium": false
    }
    """
    urls = body.get("urls") or []
    product_limit = body.get("product_limit", 5)
    output_format = body.get("output_format", "json")
    use_selenium = body.get("use_selenium", False)
    if not urls or not isinstance(urls, list):
        raise HTTPException(status_code=400, detail="urls must be a list of URLs")
    SCRAPING_STATE["product_limit"] = product_limit
    # Compose workflow and run
    workflow = create_workflow()
    # Use the first url for analysis; override dynamic detection if use_selenium is set
    async def run_workflow():
        initial_state = {
            "urls": urls,
            "extracted_data": [],
            "status": "initialized"
        }
        result = await workflow.ainvoke(initial_state)
        # If use_selenium is forced, re-run extraction with selenium
        if use_selenium:
            all_data = []
            for url in urls:
                res = await extract_with_selenium(url)
                if res.get("data"):
                    all_data.extend(res["data"])
            result["extracted_data"] = all_data
            result["status"] = "completed"
        return result
    result = await run_workflow()
    data = result.get("extracted_data", [])
    return {"records": len(data), "data": data}

@app.post("/scrape/csv")
async def scrape_csv_endpoint(
    body: Dict[str, Any] = Body(...)
):
    """
    Accepts JSON body:
    {
        "urls": [ ... ],
        "product_limit": 5,
        "output_format": "csv",
        "use_selenium": false
    }
    Returns: CSV file response.
    """
    urls = body.get("urls") or []
    product_limit = body.get("product_limit", 5)
    use_selenium = body.get("use_selenium", False)
    if not urls or not isinstance(urls, list):
        raise HTTPException(status_code=400, detail="urls must be a list of URLs")
    SCRAPING_STATE["product_limit"] = product_limit
    workflow = create_workflow()
    async def run_workflow():
        initial_state = {
            "urls": urls,
            "extracted_data": [],
            "status": "initialized"
        }
        result = await workflow.ainvoke(initial_state)
        # If use_selenium is forced, re-run extraction with selenium
        if use_selenium:
            all_data = []
            for url in urls:
                res = await extract_with_selenium(url)
                if res.get("data"):
                    all_data.extend(res["data"])
            result["extracted_data"] = all_data
            result["status"] = "completed"
        return result
    result = await run_workflow()
    data = result.get("extracted_data", [])
    df = pd.DataFrame(data)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    headers = {
        "Content-Disposition": f'attachment; filename="scraped_{now_str}.csv"'
    }
    return StreamingResponse(iter([csv_bytes]), media_type="text/csv", headers=headers)


if __name__ == "__main__":
    uvicorn.run("webscrapper2:app", host="0.0.0.0", port=8000, reload=True)
