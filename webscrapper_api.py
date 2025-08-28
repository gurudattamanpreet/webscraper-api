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
    version="2.0.0",
    description="Improved web scraping API for e-commerce products"
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
    product_limit: Optional[int] = 10
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
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    })
    return headers

def clean_price(text):
    """Extract and clean price from text"""
    if not text:
        return None
    
    # Remove common currency symbols and keep numbers
    price_text = re.sub(r'[^\d.,]', '', text)
    
    # Handle different decimal formats
    if price_text:
        # Remove thousand separators
        price_text = price_text.replace(',', '')
        try:
            return float(price_text)
        except:
            pass
    
    # Try to find price patterns in original text
    patterns = [
        r'[\$₹£€]\s*[\d,]+\.?\d*',
        r'[\d,]+\.?\d*\s*[\$₹£€]',
        r'Rs\.?\s*[\d,]+\.?\d*',
        r'USD\s*[\d,]+\.?\d*'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            price = re.sub(r'[^\d.]', '', match.group())
            try:
                return float(price)
            except:
                pass
    
    return None

def extract_product_info(container, base_url):
    """Enhanced product extraction logic"""
    product = {}
    
    # Extract title - try multiple methods
    title = None
    
    # Method 1: Look for product title classes
    title_selectors = [
        'h1', 'h2', 'h3', 'h4',
        '[class*="title"]', '[class*="name"]', '[class*="heading"]',
        '[data-testid*="title"]', '[data-testid*="name"]',
        '.product-title', '.product-name', '.item-title', '.item-name'
    ]
    
    for selector in title_selectors:
        elem = container.select_one(selector)
        if elem:
            text = elem.get_text(strip=True)
            if text and len(text) > 5:  # Avoid very short texts
                title = text[:200]  # Limit length
                break
    
    # Method 2: Look for links with text
    if not title:
        link = container.find('a')
        if link:
            text = link.get_text(strip=True)
            if text and len(text) > 5:
                title = text[:200]
    
    # Method 3: Get first substantial text
    if not title:
        texts = container.get_text(separator='|').split('|')
        for text in texts:
            text = text.strip()
            if len(text) > 10 and len(text) < 200:
                title = text
                break
    
    if title:
        product['title'] = title
    
    # Extract price - enhanced logic
    price = None
    
    # Look for price in various ways
    price_selectors = [
        '[class*="price"]', '[class*="cost"]', '[class*="amount"]',
        '[data-testid*="price"]', '[data-price]',
        '.price', '.cost', '.amount',
        'span[class*="price"]', 'div[class*="price"]'
    ]
    
    for selector in price_selectors:
        elem = container.select_one(selector)
        if elem:
            # Try text content
            text = elem.get_text(strip=True)
            price = clean_price(text)
            if price:
                break
            
            # Try attributes
            for attr in ['data-price', 'content', 'value']:
                val = elem.get(attr)
                if val:
                    price = clean_price(val)
                    if price:
                        break
    
    if price:
        product['price'] = f"${price:.2f}" if isinstance(price, (int, float)) else str(price)
    
    # Extract link
    link_elem = container.find('a', href=True)
    if link_elem:
        product['link'] = urljoin(base_url, link_elem['href'])
    else:
        product['link'] = base_url
    
    # Extract image if available
    img_elem = container.find('img')
    if img_elem:
        img_src = img_elem.get('src') or img_elem.get('data-src')
        if img_src:
            product['image'] = urljoin(base_url, img_src)
    
    return product if (product.get('title') or product.get('price')) else None

async def extract_with_beautifulsoup(url: str, product_limit: int = 10) -> Dict[str, Any]:
    """Enhanced extraction using BeautifulSoup"""
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            headers = get_next_headers()
            
            # First request to main page
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            extracted_data = {
                "url": url,
                "timestamp": datetime.now().isoformat(),
                "method": "beautifulsoup",
                "data": []
            }
            
            # Enhanced product container detection
            possible_containers = []
            
            # Strategy 1: Look for common e-commerce patterns
            selectors = [
                # Generic product selectors
                '[class*="product"]:has(a)',
                '[class*="item"]:has(a)',
                '[class*="card"]:has(a)',
                '[class*="tile"]:has(a)',
                '[class*="grid"] > div:has(a)',
                '[class*="list"] > div:has(a)',
                
                # Specific platform patterns
                'article', 'li[class*="product"]', 'div[data-product]',
                '.product-item', '.product-card', '.product-tile',
                '.grid-item', '.collection-item', '.shop-item',
                
                # Fallback to any div with both link and some text
                'div:has(a):has(span)',
                'div:has(a):has(p)',
                'section:has(a)'
            ]
            
            for selector in selectors:
                try:
                    containers = soup.select(selector)
                    if len(containers) >= 2:  # At least 2 items to be a product list
                        possible_containers = containers
                        break
                except:
                    continue
            
            # If no containers found, try finding by structure
            if not possible_containers:
                # Look for repeated structures
                all_divs = soup.find_all('div', recursive=True)
                div_classes = {}
                
                for div in all_divs:
                    classes = div.get('class', [])
                    if classes:
                        class_str = ' '.join(classes)
                        if class_str not in div_classes:
                            div_classes[class_str] = []
                        div_classes[class_str].append(div)
                
                # Find classes that appear multiple times (likely product containers)
                for class_str, divs in div_classes.items():
                    if len(divs) >= 3 and any(word in class_str.lower() for word in ['product', 'item', 'card', 'tile']):
                        possible_containers = divs[:product_limit]
                        break
            
            # Extract from containers
            products_found = []
            for container in possible_containers[:product_limit * 2]:  # Check more to filter later
                product = extract_product_info(container, url)
                if product:
                    # Avoid duplicates
                    is_duplicate = any(
                        p.get('title') == product.get('title') and 
                        p.get('price') == product.get('price') 
                        for p in products_found
                    )
                    if not is_duplicate:
                        products_found.append(product)
                
                if len(products_found) >= product_limit:
                    break
            
            # If still no products, try a different approach - look for links with prices nearby
            if len(products_found) < 3:
                all_links = soup.find_all('a', href=True)
                
                for link in all_links[:50]:  # Check first 50 links
                    # Skip navigation links
                    href = link.get('href', '').lower()
                    if any(skip in href for skip in ['#', 'javascript:', 'mailto:', 'tel:']):
                        continue
                    
                    parent = link.parent
                    if parent:
                        product = extract_product_info(parent, url)
                        if product:
                            is_duplicate = any(
                                p.get('title') == product.get('title') 
                                for p in products_found
                            )
                            if not is_duplicate:
                                products_found.append(product)
                    
                    if len(products_found) >= product_limit:
                        break
            
            extracted_data["data"] = products_found[:product_limit]
            
            # If no products found, return page info
            if not extracted_data["data"]:
                page_title = soup.find('title')
                extracted_data["data"] = [{
                    "title": page_title.get_text(strip=True) if page_title else "No products found",
                    "link": url,
                    "price": None,
                    "note": "No product data could be extracted. The page might require JavaScript or have an unusual structure."
                }]
            
            return extracted_data
            
    except httpx.HTTPStatusError as e:
        return {
            "status": "error",
            "error": f"HTTP {e.response.status_code}: {e.response.reason_phrase}",
            "data": []
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "data": []
        }

async def analyze_website(url: str) -> Dict[str, Any]:
    """Analyze website and extract sample products"""
    analysis = {
        "url": url,
        "is_dynamic": False,
        "is_ecommerce": False,
        "detected_patterns": [],
        "recommended_method": "beautifulsoup",
        "sample_products": []
    }
    
    try:
        # Get sample products
        result = await extract_with_beautifulsoup(url, product_limit=3)
        
        if result.get("data"):
            products_with_price = [p for p in result["data"] if p.get("price")]
            
            if products_with_price:
                analysis["is_ecommerce"] = True
                analysis["detected_patterns"].append(f"{len(products_with_price)} products with prices found")
                analysis["sample_products"] = result["data"][:3]
            
            if len(result["data"]) > 1:
                analysis["detected_patterns"].append(f"{len(result['data'])} product-like items found")
        
        # Check for common e-commerce indicators
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = get_next_headers()
            response = await client.get(url, headers=headers, follow_redirects=True)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            ecommerce_keywords = ['cart', 'shop', 'product', 'price', 'buy', 'sale', 'discount']
            for keyword in ecommerce_keywords:
                if soup.find(class_=re.compile(keyword, re.I)):
                    analysis["detected_patterns"].append(f"'{keyword}' pattern found")
            
            # Check for JavaScript frameworks
            if soup.find_all('script', string=re.compile('react|vue|angular', re.I)):
                analysis["is_dynamic"] = True
                analysis["detected_patterns"].append("JavaScript framework detected")
        
        return analysis
        
    except Exception as e:
        analysis["error"] = str(e)
        return analysis

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "🕷️ Web Scraper API v2.0 is running!",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "scrape": "/scrape (POST)",
            "analyze": "/analyze (GET)",
            "docs": "/docs (Interactive API Documentation)"
        },
        "improvements": "Better product extraction logic with multiple fallback strategies",
        "sample_request": {
            "urls": ["https://example-shop.com"],
            "product_limit": 10,
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
            # Validate and normalize URL
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            print(f"🔍 Scraping: {url}")
            
            # Extract data with improved logic
            result = await extract_with_beautifulsoup(url, request.product_limit)
            
            if result.get("data"):
                # Only add products with meaningful data
                valid_items = [
                    item for item in result["data"] 
                    if item.get("title") and "No products found" not in item.get("title", "")
                ]
                
                if valid_items:
                    all_data.extend(valid_items)
                    print(f"✅ Extracted {len(valid_items)} products from {url}")
                else:
                    print(f"⚠️ No valid products found on {url}")
                    errors.append(f"{url}: No products with complete data found")
                    
            elif result.get("error"):
                errors.append(f"{url}: {result.get('error')}")
                print(f"❌ Error scraping {url}: {result.get('error')}")
            else:
                print(f"⚠️ No data found for {url}")
                errors.append(f"{url}: No extractable data found")
                
        except Exception as e:
            error_msg = f"{url}: {str(e)}"
            errors.append(error_msg)
            print(f"❌ Exception for {url}: {str(e)}")
    
    # Remove duplicates
    seen = set()
    unique_data = []
    for item in all_data:
        key = (item.get("title", ""), item.get("link", ""))
        if key not in seen and key[0]:
            unique_data.append(item)
            seen.add(key)
    
    print(f"\n📊 Final result: {len(unique_data)} unique products extracted")
    
    response = ScrapeResponse(
        status="success" if unique_data else "partial" if errors else "no_data",
        data=unique_data,
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
    Analyze a website and return sample products
    """
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    
    # Add https if not present
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    analysis = await analyze_website(url)
    return analysis

# Main execution
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting Web Scraper API v2.0 on port {port}")
    print(f"📚 Documentation available at: http://0.0.0.0:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)