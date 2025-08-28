from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import httpx
from bs4 import BeautifulSoup
from datetime import datetime
import json
import pandas as pd
import io
import re
from urllib.parse import urlparse, urljoin
import os

# Initialize FastAPI
app = FastAPI(
    title="Web Scraper API",
    version="1.0.0",
    description="Web scraping API for e-commerce products"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ScrapeRequest(BaseModel):
    urls: List[str]
    product_limit: Optional[int] = 5
    output_format: Optional[str] = "JSON"
    use_selenium: Optional[bool] = False

class ScrapeResponse(BaseModel):
    status: str
    data: List[Dict[str, Any]]
    total_items: int
    timestamp: str
    errors: Optional[List[str]] = None

# Global state
SCRAPING_STATE = {
    "headers_list": [
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
    ],
    "current_header_index": 0
}

def get_next_headers() -> Dict[str, str]:
    """Rotate headers for each request"""
    headers = SCRAPING_STATE["headers_list"][SCRAPING_STATE["current_header_index"]].copy()
    SCRAPING_STATE["current_header_index"] = (SCRAPING_STATE["current_header_index"] + 1) % len(SCRAPING_STATE["headers_list"])
    headers.update({
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })
    return headers

def find_product_links(soup, base_url):
    """Find links that look like product detail pages"""
    keywords = ["product", "item", "detail"]
    found = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        abs_url = urljoin(base_url, href)
        if any(kw in href.lower() for kw in keywords):
            found.add(abs_url)
    return list(found)

def extract_price(container, soup=None):
    """Extract price from HTML container"""
    def clean_price(val):
        if not val:
            return None
        # Extract numeric price
        m = re.search(r'[\d]+(?:[.,]\d+)?', val)
        if m:
            price_str = m.group(0).replace(',', '.')
            if '.' not in price_str and len(price_str) > 2:
                price_str = price_str[:-2] + '.' + price_str[-2:]
            return price_str
        return None
    
    price_patterns = [
        ('span', re.compile('price|amount|money|value|cost', re.I)),
        ('div', re.compile('price|amount|money|value|cost', re.I)),
        ('p', re.compile('price|amount|money|value|cost', re.I)),
        (None, {"itemprop": "price"}),
        (None, {"class": re.compile("price", re.I)}),
    ]
    
    for tag, pat in price_patterns:
        if isinstance(pat, dict):
            elem = container.find(attrs=pat)
        else:
            elem = container.find(tag, class_=pat) if tag else container.find(attrs={"class": pat})
        
        if elem:
            txt = elem.get_text(" ", strip=True)
            if txt:
                cleaned = clean_price(txt)
                if cleaned:
                    return cleaned
            
            # Check attributes
            for attr in ["content", "data-price", "aria-label", "value"]:
                attr_val = elem.get(attr)
                if attr_val:
                    cleaned = clean_price(str(attr_val).strip())
                    if cleaned:
                        return cleaned
    
    # Try to find price in script tags
    if soup:
        for sc in soup.find_all("script", string=True):
            if sc.string:
                m = re.search(r'"price"\s*[:=]\s*"?([\d.,]+)"?', sc.string)
                if m:
                    return clean_price(m.group(1))
    
    return None

def extract_product_info(container, base_url, soup=None):
    """Helper function to extract product info from a container"""
    item_data = {}
    
    # Extract title
    title = None
    
    # Try heading tags first
    for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
        elem = container.find(tag)
        if elem and elem.get_text(" ", strip=True):
            title = elem.get_text(" ", strip=True)
            break
    
    # Try common title patterns
    if not title:
        title_patterns = [
            ('a', re.compile('title|name|product-name|product-title|heading', re.I)),
            ('div', re.compile('title|name|product-name|product-title|heading', re.I)),
            ('span', re.compile('title|name|product-name|product-title|heading', re.I)),
            (None, {"itemprop": "name"}),
            (None, {"class": re.compile("product.*title|product.*name", re.I)})
        ]
        
        for tag, pat in title_patterns:
            if isinstance(pat, dict):
                elem = container.find(attrs=pat)
            elif tag:
                elem = container.find(tag, class_=pat)
            else:
                elem = container.find(attrs={"class": pat})
            
            if elem and elem.get_text(" ", strip=True):
                title = elem.get_text(" ", strip=True)[:200]  # Limit title length
                break
    
    # Try image alt text
    if not title:
        img = container.find('img')
        if img and img.get('alt'):
            title = img.get('alt').strip()[:200]
    
    if title:
        item_data['title'] = title
    
    # Extract link
    link = base_url
    a_tag = container.find('a', href=True)
    if a_tag and a_tag.get('href'):
        link = urljoin(base_url, a_tag['href'])
    item_data['link'] = link
    
    # Extract price
    price = extract_price(container, soup)
    if price:
        item_data['price'] = price
    
    return item_data

async def analyze_website(url: str) -> Dict[str, Any]:
    """Analyze website to determine if it's dynamic or e-commerce"""
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
            
            # Check for e-commerce patterns
            ecommerce_keywords = ['price', 'product', 'cart', 'add-to-cart', 'buy', 'shop', 'store', 'item']
            for keyword in ecommerce_keywords:
                if soup.find(class_=re.compile(keyword, re.I)) or soup.find(id=re.compile(keyword, re.I)):
                    analysis["is_ecommerce"] = True
                    analysis["detected_patterns"].append(f"{keyword} pattern found")
            
            return analysis
    except Exception as e:
        analysis["error"] = str(e)
        return analysis

async def extract_with_beautifulsoup(url: str, product_limit: int = 5) -> Dict[str, Any]:
    """Extract data using BeautifulSoup with improved product extraction"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = get_next_headers()
            response = await client.get(url, headers=headers, follow_redirects=True)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            extracted_data = {
                "url": url,
                "timestamp": datetime.now().isoformat(),
                "method": "beautifulsoup",
                "data": []
            }
            
            # First, try to find collection/listing pages
            listing_keywords = ["collections", "collection", "category", "shop", "store", "products", "all", "catalog"]
            listing_links = set()
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if any(kw in href.lower() for kw in listing_keywords):
                    abs_url = urljoin(url, href)
                    # Avoid external links
                    if urlparse(abs_url).netloc == urlparse(url).netloc:
                        listing_links.add(abs_url)
            
            listing_links = list(listing_links)[:3]  # Limit to first 3 collection pages
            
            # Container selectors for products - more comprehensive
            container_selectors = (
                'div[class*="product"], li[class*="product"], '
                'article[class*="product"], section[class*="product"], '
                'div[class*="item"][class*="grid"], div[class*="card"][class*="product"], '
                'div[class*="grid-item"], div[class*="collection-item"], '
                'div[class*="ProductCard"], div[class*="product-item"], '
                'div[data-product], article[data-product]'
            )
            
            # Try to extract from current page first
            containers = soup.select(container_selectors)
            
            # If this is a listing page with multiple products
            if len(containers) >= 2:  # At least 2 products to consider it a listing
                for container in containers[:product_limit]:
                    item_data = extract_product_info(container, url, soup)
                    if item_data and (item_data.get("title") or item_data.get("price")):
                        # Avoid duplicate titles
                        if not any(d.get("title") == item_data.get("title") for d in extracted_data["data"]):
                            extracted_data["data"].append(item_data)
            
            # If no data found or need more, try listing/collection pages
            if len(extracted_data["data"]) < product_limit and listing_links:
                for listing_url in listing_links:
                    if len(extracted_data["data"]) >= product_limit:
                        break
                    try:
                        resp_listing = await client.get(listing_url, headers=get_next_headers(), follow_redirects=True)
                        soup_listing = BeautifulSoup(resp_listing.text, 'html.parser')
                        containers_listing = soup_listing.select(container_selectors)
                        
                        for container in containers_listing:
                            if len(extracted_data["data"]) >= product_limit:
                                break
                            item_data = extract_product_info(container, listing_url, soup_listing)
                            if item_data and (item_data.get("title") or item_data.get("price")):
                                # Avoid duplicate titles
                                if not any(d.get("title") == item_data.get("title") for d in extracted_data["data"]):
                                    extracted_data["data"].append(item_data)
                    except Exception as e:
                        print(f"Error fetching listing page {listing_url}: {e}")
                        continue
            
            # If still no data, try to find and follow individual product links
            if len(extracted_data["data"]) < product_limit:
                product_links = find_product_links(soup, url)
                product_links = list(dict.fromkeys(product_links))[:product_limit]
                
                for prod_link in product_links:
                    if len(extracted_data["data"]) >= product_limit:
                        break
                    try:
                        resp2 = await client.get(prod_link, headers=get_next_headers(), follow_redirects=True)
                        soup2 = BeautifulSoup(resp2.text, 'html.parser')
                        
                        # Extract from product page
                        title = None
                        
                        # Try multiple ways to get title
                        h1 = soup2.find('h1')
                        if h1:
                            title = h1.get_text(" ", strip=True)
                        
                        if not title:
                            meta_title = soup2.find('meta', {'property': 'og:title'})
                            if meta_title:
                                title = meta_title.get('content', '')
                        
                        if not title:
                            page_title = soup2.find('title')
                            if page_title:
                                title = page_title.get_text(" ", strip=True)
                        
                        # Extract price
                        price = extract_price(soup2, soup2)
                        
                        if title or price:
                            item = {
                                "title": title[:200] if title else None,  # Limit title length
                                "link": prod_link,
                                "price": price
                            }
                            # Avoid duplicate titles
                            if not any(d.get("title") == item.get("title") for d in extracted_data["data"]):
                                extracted_data["data"].append(item)
                    except Exception as e:
                        print(f"Error fetching product page {prod_link}: {e}")
                        continue
            
            # Last resort - basic page info
            if not extracted_data["data"]:
                page_data = {
                    "title": soup.find('title').get_text() if soup.find('title') else "No products found",
                    "link": url,
                    "price": None
                }
                extracted_data["data"].append(page_data)
            
            return extracted_data
            
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
        "message": "🕷️ Web Scraper API is running!",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "scrape": "/scrape (POST)",
            "analyze": "/analyze (GET)",
            "docs": "/docs (Interactive API Documentation)"
        },
        "sample_request": {
            "urls": ["https://example-shop.com"],
            "product_limit": 5,
            "output_format": "JSON"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "message": "API is running smoothly!"
    }

@app.post("/scrape", response_model=ScrapeResponse)
async def scrape_website(request: ScrapeRequest):
    """
    Main scraping endpoint - extracts product information from websites
    """
    if not request.urls:
        raise HTTPException(status_code=400, detail="No URLs provided")
    
    all_data = []
    errors = []
    
    print(f"📋 Processing {len(request.urls)} URLs with limit of {request.product_limit} products each")
    
    for url in request.urls:
        try:
            # Validate URL
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            print(f"🔍 Scraping: {url}")
            
            # Check if URL is accessible
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = get_next_headers()
                test_response = await client.get(url, headers=headers, follow_redirects=True)
                if test_response.status_code >= 400:
                    errors.append(f"{url}: HTTP {test_response.status_code}")
                    print(f"❌ Failed to access {url}: HTTP {test_response.status_code}")
                    continue
            
            # Extract data
            result = await extract_with_beautifulsoup(url, request.product_limit)
            
            if result.get("data"):
                # Filter out items without meaningful data
                valid_items = []
                for item in result["data"]:
                    # Only keep items that have at least a title
                    if item.get("title") and item.get("title") != "No products found":
                        valid_items.append(item)
                
                all_data.extend(valid_items)
                print(f"✅ Extracted {len(valid_items)} items from {url}")
            elif result.get("error"):
                errors.append(f"{url}: {result.get('error')}")
                print(f"❌ Error scraping {url}: {result.get('error')}")
            else:
                print(f"⚠️ No data found for {url}")
                
        except Exception as e:
            error_msg = f"{url}: {str(e)}"
            errors.append(error_msg)
            print(f"❌ Exception for {url}: {str(e)}")
    
    # Remove duplicates based on title and link
    seen = set()
    unique_data = []
    for item in all_data:
        # Create unique key from title and link
        key = (item.get("title", ""), item.get("link", ""))
        if key not in seen and key[0]:  # Ensure title is not empty
            unique_data.append(item)
            seen.add(key)
    
    # Sort data by whether they have prices (items with prices first)
    unique_data.sort(key=lambda x: (x.get("price") is None, x.get("title", "")))
    
    print(f"\n📊 Final result: {len(unique_data)} unique items extracted")
    
    response = ScrapeResponse(
        status="success" if unique_data else "no_data",
        data=unique_data[:request.product_limit * len(request.urls)],  # Limit total results
        total_items=len(unique_data),
        timestamp=datetime.now().isoformat(),
        errors=errors if errors else None
    )
    
    return response

@app.post("/scrape/csv")
async def scrape_to_csv(request: ScrapeRequest):
    """
    Scrape and return data as CSV file
    """
    scrape_response = await scrape_website(request)
    
    if not scrape_response.data:
        raise HTTPException(status_code=404, detail="No data found to export")
    
    df = pd.DataFrame(scrape_response.data)
    
    # Create CSV in memory
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    return StreamingResponse(
        io.BytesIO(csv_buffer.getvalue().encode()),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=scraped_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        }
    )

@app.get("/analyze")
async def analyze_url(url: str):
    """
    Analyze a website to determine its characteristics
    """
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    
    # Add https if not present
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    analysis = await analyze_website(url)
    return analysis

# Main execution - for Replit
if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable (for cloud deployment)
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting Web Scraper API on port {port}")
    print(f"📚 Documentation will be available at: http://0.0.0.0:{port}/docs")
    # Run the app
    uvicorn.run(app, host="0.0.0.0", port=port)