from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import httpx
from bs4 import BeautifulSoup
import pandas as pd
import io
import re
from urllib.parse import urlparse, urljoin
import os
import json

# FastAPI app
app = FastAPI(
    title="Web Scraper API",
    version="3.0.0",
    description="Production-ready web scraping API for Render deployment"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for scraping
SCRAPING_STATE = {
    "headers_list": [
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
        {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
    ],
    "current_header_index": 0
}

def get_next_headers() -> Dict[str, str]:
    """Rotate headers for each request"""
    headers = SCRAPING_STATE["headers_list"][SCRAPING_STATE["current_header_index"]].copy()
    SCRAPING_STATE["current_header_index"] = (SCRAPING_STATE["current_header_index"] + 1) % len(SCRAPING_STATE["headers_list"])
    headers.update({
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    })
    return headers

def find_product_links(soup, base_url):
    """Find links that look like product detail pages"""
    keywords = ["product", "item", "detail", "p/", "pd/", "dp/", "products/"]
    found = set()
    
    for a in soup.find_all("a", href=True):
        href = a["href"]
        abs_url = urljoin(base_url, href)
        
        # Check if URL contains product keywords
        if any(kw in href.lower() for kw in keywords):
            found.add(abs_url)
        
        # Check for numeric patterns (often product IDs)
        if re.search(r'/\d{3,}', href) or re.search(r'[?&]id=\d+', href):
            found.add(abs_url)
    
    return list(found)

def extract_price(container, soup=None):
    """Extract price from HTML container"""
    def clean_price(val):
        if not val:
            return None
        # Remove currency symbols and extract numbers
        val = re.sub(r'[^\d.,]', '', val)
        if not val:
            return None
        
        # Handle different decimal formats
        val = val.replace(',', '')
        try:
            price = float(val)
            if price > 0 and price < 1000000:  # Sanity check
                return f"{price:.2f}"
        except:
            pass
        return None
    
    # Look for price in various elements
    price_selectors = [
        '[class*="price"]',
        '[class*="Price"]',
        '[class*="cost"]',
        '[class*="amount"]',
        '[data-price]',
        '[itemprop="price"]',
        'span.price',
        'div.price',
        '.product-price',
        '.sale-price',
        '.regular-price'
    ]
    
    for selector in price_selectors:
        try:
            elem = container.select_one(selector)
            if elem:
                # Try text content
                text = elem.get_text(strip=True)
                price = clean_price(text)
                if price:
                    return price
                
                # Try attributes
                for attr in ["content", "data-price", "value"]:
                    val = elem.get(attr)
                    if val:
                        price = clean_price(val)
                        if price:
                            return price
        except:
            continue
    
    # Fallback: Look for price patterns in text
    text = container.get_text(" ", strip=True)
    price_patterns = [
        r'[\$₹£€]\s*[\d,]+\.?\d*',
        r'Rs\.?\s*[\d,]+\.?\d*',
        r'USD\s*[\d,]+\.?\d*',
        r'INR\s*[\d,]+\.?\d*'
    ]
    
    for pattern in price_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return clean_price(match.group())
    
    return None

def extract_title(container):
    """Extract product title from container"""
    # Try headings first
    for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
        elem = container.find(tag)
        if elem:
            text = elem.get_text(strip=True)
            if text and len(text) > 5 and len(text) < 200:
                return text
    
    # Try common title selectors
    title_selectors = [
        '[class*="title"]',
        '[class*="Title"]',
        '[class*="name"]',
        '[class*="Name"]',
        '[class*="heading"]',
        '[itemprop="name"]',
        '.product-title',
        '.product-name',
        'a[class*="link"]'
    ]
    
    for selector in title_selectors:
        try:
            elem = container.select_one(selector)
            if elem:
                text = elem.get_text(strip=True)
                if text and len(text) > 5 and len(text) < 200:
                    return text
        except:
            continue
    
    # Try image alt text
    img = container.find('img')
    if img and img.get('alt'):
        alt = img.get('alt').strip()
        if alt and len(alt) > 5 and len(alt) < 200:
            return alt
    
    # Get first meaningful text
    texts = container.get_text(separator='|').split('|')
    for text in texts:
        text = text.strip()
        if len(text) > 10 and len(text) < 200 and not re.match(r'^[\d\s\$₹£€.,]+$', text):
            return text
    
    return None

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
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, verify=False) as client:
            headers = get_next_headers()
            response = await client.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Check for e-commerce patterns
            ecommerce_keywords = ['price', 'product', 'cart', 'add-to-cart', 'buy', 'shop', 'store', 'item']
            for keyword in ecommerce_keywords:
                if soup.find(class_=re.compile(keyword, re.I)) or soup.find(id=re.compile(keyword, re.I)):
                    analysis["is_ecommerce"] = True
                    analysis["detected_patterns"].append(f"{keyword} pattern found")
            
            # Check for JavaScript frameworks
            scripts = soup.find_all('script')
            for script in scripts:
                script_text = str(script).lower()
                if any(fw in script_text for fw in ['react', 'angular', 'vue', 'next.js']):
                    analysis["is_dynamic"] = True
                    analysis["detected_patterns"].append("JavaScript framework detected")
                    break
            
            return analysis
    
    except Exception as e:
        analysis["error"] = str(e)
        return analysis

async def extract_with_beautifulsoup(url: str, product_limit: int = 10) -> Dict[str, Any]:
    """Extract data using BeautifulSoup with improved logic from webscrapper_api2.py"""
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, verify=False) as client:
            headers = get_next_headers()
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements for cleaner parsing
            for script in soup(["script", "style", "noscript"]):
                script.decompose()
            
            extracted_data = {
                "url": url,
                "timestamp": datetime.now().isoformat(),
                "method": "beautifulsoup",
                "data": []
            }
            
            # Find product links first (from your original code)
            product_links = find_product_links(soup, url)
            product_links = list(dict.fromkeys(product_links))[:product_limit * 2]  # Get extra for filtering
            
            # Strategy 1: Look for product containers on current page
            container_selectors = [
                'div[class*="product"]',
                'article[class*="product"]',
                'li[class*="product"]',
                'div[class*="item"][class*="grid"]',
                'div[class*="card"][class*="product"]',
                'div[class*="collection"]',
                'div[class*="listing"]',
                'div[data-product]',
                'article[data-product]',
                '.product-item',
                '.product-card',
                '.product-tile'
            ]
            
            containers = []
            for selector in container_selectors:
                found = soup.select(selector)
                if len(found) >= 2:  # At least 2 items to be a product listing
                    containers = found
                    break
            
            # Extract from containers on current page
            for container in containers[:product_limit]:
                item_data = {}
                
                # Extract title
                title = extract_title(container)
                if title:
                    item_data['title'] = title
                
                # Extract link
                link_elem = container.find('a', href=True)
                if link_elem:
                    item_data['link'] = urljoin(url, link_elem['href'])
                else:
                    item_data['link'] = url
                
                # Extract price
                price = extract_price(container, soup)
                if price:
                    item_data['price'] = price
                
                # Only add if we have meaningful data
                if item_data.get("title") or item_data.get("price"):
                    # Avoid duplicates
                    if not any(d.get("title") == item_data.get("title") for d in extracted_data["data"]):
                        extracted_data["data"].append(item_data)
            
            # Strategy 2: If not enough products found, follow product links
            if len(extracted_data["data"]) < product_limit and product_links:
                follow_links = product_links[:min(10, product_limit - len(extracted_data["data"]))]
                
                for prod_link in follow_links:
                    if len(extracted_data["data"]) >= product_limit:
                        break
                    
                    try:
                        resp2 = await client.get(prod_link, headers=get_next_headers())
                        soup2 = BeautifulSoup(resp2.text, 'html.parser')
                        
                        # Remove scripts from product page too
                        for script in soup2(["script", "style", "noscript"]):
                            script.decompose()
                        
                        # Look for product info on detail page
                        item_data = {}
                        
                        # Title - prioritize h1 on product pages
                        h1 = soup2.find('h1')
                        if h1:
                            item_data['title'] = h1.get_text(strip=True)[:200]
                        else:
                            # Try meta tags
                            meta_title = soup2.find('meta', {'property': 'og:title'})
                            if meta_title:
                                item_data['title'] = meta_title.get('content', '')[:200]
                            else:
                                title_elem = soup2.find('title')
                                if title_elem:
                                    item_data['title'] = title_elem.get_text(strip=True)[:200]
                        
                        item_data['link'] = prod_link
                        
                        # Price - look more broadly on product page
                        price = extract_price(soup2, soup2)
                        if price:
                            item_data['price'] = price
                        
                        if item_data.get("title") or item_data.get("price"):
                            # Avoid duplicates
                            if not any(d.get("title") == item_data.get("title") for d in extracted_data["data"]):
                                extracted_data["data"].append(item_data)
                    
                    except Exception as e:
                        print(f"Error fetching product page {prod_link}: {e}")
                        continue
            
            # Strategy 3: If still no products, try finding any div with both text and price
            if len(extracted_data["data"]) < 3:
                all_divs = soup.find_all(['div', 'article', 'section', 'li'])[:100]
                
                for div in all_divs:
                    if len(extracted_data["data"]) >= product_limit:
                        break
                    
                    text = div.get_text(strip=True)
                    if len(text) < 20 or len(text) > 500:
                        continue
                    
                    price = extract_price(div)
                    if price:
                        title = extract_title(div)
                        if title:
                            item_data = {
                                'title': title,
                                'price': price,
                                'link': url
                            }
                            
                            if not any(d.get("title") == item_data.get("title") for d in extracted_data["data"]):
                                extracted_data["data"].append(item_data)
            
            # If no products found, return page info
            if not extracted_data["data"]:
                page_title = soup.find('title')
                extracted_data["data"].append({
                    "title": page_title.get_text(strip=True) if page_title else "No products found",
                    "link": url,
                    "price": None,
                    "note": "No product data could be extracted. Try a different URL or the page may require JavaScript."
                })
            
            return extracted_data
    
    except httpx.HTTPStatusError as e:
        return {
            "status": "error",
            "error": f"HTTP {e.response.status_code}",
            "data": []
        }
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "data": []
        }

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "🕷️ Web Scraper API v3.0 is running!",
        "version": "3.0.0",
        "description": "Production-ready scraper based on webscrapper_api2.py logic",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze?url={website_url}",
            "scrape": "/scrape (POST)",
            "scrape_csv": "/scrape/csv (POST)",
            "docs": "/docs"
        },
        "test_urls": [
            "https://books.toscrape.com",
            "https://scrapeme.live/shop"
        ]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/analyze")
async def analyze_endpoint(url: str = Query(..., description="URL to analyze")):
    """Analyze website structure and patterns"""
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    result = await analyze_website(url)
    
    # Also try to extract sample products
    sample_extraction = await extract_with_beautifulsoup(url, product_limit=3)
    
    result["sample_products"] = sample_extraction.get("data", [])[:3]
    result["products_found"] = len([p for p in sample_extraction.get("data", []) if p.get("title") and "No products found" not in p.get("title", "")])
    
    return result

@app.post("/scrape")
async def scrape_endpoint(body: Dict[str, Any] = Body(...)):
    """
    Main scraping endpoint
    Accepts JSON body:
    {
        "urls": ["url1", "url2"],
        "product_limit": 10,
        "output_format": "JSON",
        "use_selenium": false
    }
    """
    urls = body.get("urls", [])
    product_limit = body.get("product_limit", 10)
    
    if not urls or not isinstance(urls, list):
        raise HTTPException(status_code=400, detail="urls must be a non-empty list")
    
    all_data = []
    errors = []
    
    for url in urls:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        print(f"🔍 Scraping: {url}")
        
        # Extract products
        result = await extract_with_beautifulsoup(url, product_limit)
        
        if result.get("data"):
            # Filter out placeholder entries
            valid_items = [
                item for item in result["data"]
                if item.get("title") and "No products found" not in item.get("title", "")
            ]
            all_data.extend(valid_items)
            print(f"✅ Extracted {len(valid_items)} products from {url}")
        
        if result.get("error"):
            errors.append(f"{url}: {result['error']}")
            print(f"❌ Error on {url}: {result['error']}")
    
    # Remove duplicates based on title
    seen_titles = set()
    unique_data = []
    for item in all_data:
        title = item.get("title", "")
        if title and title not in seen_titles:
            unique_data.append(item)
            seen_titles.add(title)
    
    return {
        "status": "success" if unique_data else "no_data",
        "records": len(unique_data),
        "data": unique_data[:product_limit * len(urls)],
        "timestamp": datetime.now().isoformat(),
        "errors": errors if errors else None
    }

@app.post("/scrape/csv")
async def scrape_csv_endpoint(body: Dict[str, Any] = Body(...)):
    """
    Export scraped data as CSV
    Accepts same JSON body as /scrape endpoint
    """
    # Use the same scraping logic
    result = await scrape_endpoint(body)
    
    if not result.get("data"):
        raise HTTPException(status_code=404, detail="No data found to export")
    
    # Convert to DataFrame
    df = pd.DataFrame(result["data"])
    
    # Create CSV in memory
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    # Return as downloadable file
    csv_bytes = csv_buffer.getvalue().encode('utf-8')
    
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        }
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting Web Scraper API v3.0 on port {port}")
    print(f"📚 Documentation available at: http://0.0.0.0:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)