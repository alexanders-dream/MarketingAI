# WordPress Plugin Integration Guide - MarketingAI SaaS Platform

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Streamlit Starlette Integration](#streamlit-starlette-integration)
3. [Custom API Endpoints](#custom-api-endpoints)
4. [Business Model & Subscription Architecture](#business-model--subscription-architecture)
5. [Website Context Extraction](#website-context-extraction)
6. [Strategy & Campaign Generation](#strategy--campaign-generation)
7. [Strategy-to-Task Decomposition & Agent Assignment](#strategy-to-task-decomposition--agent-assignment)
8. [Content Scheduling & Auto-Update](#content-scheduling--auto-update)
9. [WordPress Plugin Implementation](#wordpress-plugin-implementation)
10. [Security & Authentication](#security--authentication)
11. [Deployment & Scaling](#deployment--scaling)

---

## Architecture Overview

### Recommended Architecture: Hybrid ASGI + FastAPI

The optimal architecture for this SaaS platform combines Streamlit's Starlette-based server with FastAPI for REST API endpoints:

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer / CDN                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Gateway Server                     │
│  - Authentication & API Key Validation                       │
│  - Subscription Management                                   │
│  - Rate Limiting                                             │
│  - Request Routing                                           │
└─────────────────────────────────────────────────────────────┘
              │                              │
              ▼                              ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│   Streamlit App Instance │    │   Streamlit App Instance │
│   (Marketing Agent UI)   │    │   (Marketing Agent UI)   │
│   - Strategy Generation  │    │   - Strategy Generation  │
│   - Content Creation     │    │   - Content Creation     │
│   - Market Analysis      │    │   - Market Analysis      │
└──────────────────────────┘    └──────────────────────────┘
              │                              │
              ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Shared Services Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Vector Store │  │   LLM Pool   │  │  Content Queue   │  │
│  │  (ChromaDB)  │  │  (LangChain) │  │  (Redis/Celery)  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                   WordPress Plugin Layer                     │
│  - Context Extraction                                        │
│  - Content Publishing                                        │
│  - Scheduled Updates                                         │
└─────────────────────────────────────────────────────────────┘
```

### Why This Architecture?

**Advantages of FastAPI + Streamlit Separation:**

1. **Scalability**: FastAPI handles API requests independently, allowing horizontal scaling
2. **Performance**: ASGI-based async handling for high concurrent requests
3. **Flexibility**: WordPress plugin communicates via REST API, decoupled from Streamlit UI
4. **Security**: Centralized authentication and rate limiting at the API gateway
5. **Subscription Control**: Easy to enforce API quotas and access tiers

**Alternative: Pure Streamlit with Custom Routes**

Streamlit 1.53+ supports experimental Starlette ASGI integration, allowing custom routes:

```bash
pip install streamlit[starlette]
```

```python
from streamlit.starlette import App
from starlette.routing import Route
from starlette.responses import JSONResponse

async def api_verify(request):
    return JSONResponse({"status": "ok"})

app = App(
    "market_agent.py",
    routes=[
        Route("/api/v1/auth/verify", api_verify, methods=["GET"]),
    ],
)
```

For production SaaS, FastAPI provides superior API management capabilities, but the pure Streamlit approach works for simpler deployments.

---

## Streamlit Starlette Integration

### Latest Release Features (Streamlit 1.53-1.56)

Streamlit's migration to Starlette/Uvicorn (PR #14553) enables:

1. **Custom HTTP Endpoints**: Add REST API routes alongside Streamlit UI
2. **Middleware Support**: Authentication, CORS, rate limiting
3. **Lifespan Hooks**: Startup/shutdown for resource initialization
4. **Framework Integration**: Mount Streamlit inside FastAPI or vice versa

### Implementation Example

```python
# main.py - Combined FastAPI + Streamlit Application
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from streamlit.starlette import App
from starlette.routing import Mount
import uvicorn

# Create Streamlit App first (needed for lifespan)
streamlit_app = App("market_agent.py")

# FastAPI Application (API Gateway)
# lifespan=streamlit_app.lifespan() is required for proper Streamlit runtime lifecycle
api = FastAPI(
    title="MarketingAI API",
    version="1.0.0",
    lifespan=streamlit_app.lifespan()
)

# CORS for WordPress plugin communication
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Validation Dependency
async def validate_api_key(x_api_key: str = Header(...)):
    """Validate API key against subscription database"""
    subscription = await get_subscription_by_key(x_api_key)
    if not subscription or not subscription.is_active:
        raise HTTPException(status_code=401, detail="Invalid or expired API key")
    if subscription.requests_remaining <= 0:
        raise HTTPException(status_code=429, detail="Monthly quota exceeded")
    return subscription

# API Endpoints
@api.get("/api/v1/auth/verify")
async def verify_api_key(subscription = Depends(validate_api_key)):
    return {
        "status": "active",
        "plan": subscription.plan,
        "requests_remaining": subscription.requests_remaining,
        "reset_date": subscription.reset_date.isoformat()
    }

@api.post("/api/v1/context/extract", dependencies=[Depends(validate_api_key)])
async def extract_website_context(request: ContextExtractionRequest):
    """Extract business context from WordPress website"""
    # Implementation below
    pass

@api.post("/api/v1/strategy/generate", dependencies=[Depends(validate_api_key)])
async def generate_strategy(request: StrategyRequest):
    """Generate marketing strategy"""
    pass

@api.post("/api/v1/content/generate", dependencies=[Depends(validate_api_key)])
async def generate_content(request: ContentRequest):
    """Generate marketing content"""
    pass

@api.post("/api/v1/content/schedule", dependencies=[Depends(validate_api_key)])
async def schedule_content(request: ScheduleRequest):
    """Schedule content publication"""
    pass

# Mount Streamlit App (Marketing Agent UI)
api.mount("/app", streamlit_app)

# Run with: uvicorn main:api --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=8000)
```

---

## Custom API Endpoints

### API Design

All endpoints use RESTful conventions with JSON request/response bodies.

#### Base URL
```
https://api.marketingai.com/v1
```

#### Authentication
```
X-API-Key: your_api_key_here
```

### Endpoint Reference

#### 1. Authentication & Subscription

**Verify API Key**
```http
GET /api/v1/auth/verify
Headers: X-API-Key: <api_key>
```

Response:
```json
{
  "status": "active",
  "plan": "professional",
  "requests_remaining": 450,
  "requests_limit": 1000,
  "reset_date": "2026-05-01T00:00:00Z"
}
```

#### 2. Context Extraction

**Extract Website Context**
```http
POST /api/v1/context/extract
Headers: 
  X-API-Key: <api_key>
  Content-Type: application/json
Body:
{
  "website_url": "https://clientwebsite.com",
  "wordpress_api_url": "https://clientwebsite.com/wp-json",
  "wordpress_username": "api_user",
  "wordpress_password": "app_password",
  "extract_pages": ["home", "about", "services", "contact"],
  "analyze_competitors": true
}
```

Response:
```json
{
  "context_id": "ctx_abc123",
  "company_name": "Acme Corporation",
  "industry": "Digital Marketing Services",
  "target_audience": "Small to medium businesses seeking digital transformation",
  "products_services": "SEO, Content Marketing, Social Media Management",
  "brand_description": "Full-service digital marketing agency specializing in...",
  "marketing_goals": "Increase client acquisition by 40%, expand into new markets",
  "competitive_advantages": "10+ years experience, data-driven approach, transparent reporting",
  "customer_pain_points": "Low online visibility, inconsistent lead generation",
  "keywords": ["digital marketing", "SEO services", "content strategy"],
  "competitors": ["CompetitorA", "CompetitorB", "CompetitorC"],
  "market_opportunities": ["Growing demand for video content", "Local SEO expansion"],
  "suggested_topics": [
    "How to Measure ROI from Digital Marketing",
    "5 SEO Trends Every SMB Should Know",
    "Content Marketing on a Budget"
  ]
}
```

#### 3. Strategy Generation

**Generate Marketing Strategy**
```http
POST /api/v1/strategy/generate
Headers:
  X-API-Key: <api_key>
  Content-Type: application/json
Body:
{
  "context_id": "ctx_abc123",
  "strategy_type": "comprehensive",
  "timeframe": "90_days",
  "budget_range": "5000-10000",
  "priority_channels": ["social_media", "content_marketing", "seo"]
}
```

Response:
```json
{
  "strategy_id": "strat_xyz789",
  "executive_summary": "...",
  "market_analysis": {...},
  "target_segments": [...],
  "value_proposition": "...",
  "channels_and_tactics": [...],
  "content_strategy": {
    "content_pillars": ["Education", "Thought Leadership", "Case Studies"],
    "recommended_formats": ["Blog Posts", "Infographics", "Video Tutorials"],
    "posting_frequency": {
      "linkedin": "3x per week",
      "instagram": "5x per week",
      "blog": "2x per week"
    }
  },
  "budget_allocation": {
    "content_creation": 40,
    "paid_advertising": 35,
    "tools_and_software": 15,
    "analytics": 10
  },
  "implementation_timeline": [...],
  "kpis": [...]
}
```

#### 4. Content Generation

**Generate Content**
```http
POST /api/v1/content/generate
Headers:
  X-API-Key: <api_key>
  Content-Type: application/json
Body:
{
  "context_id": "ctx_abc123",
  "content_type": "blog_post",
  "topic": "How to Measure ROI from Digital Marketing",
  "tone": "professional",
  "word_count": 1500,
  "include_keywords": ["digital marketing ROI", "marketing metrics"],
  "format": "markdown"
}
```

Response:
```json
{
  "content_id": "cnt_def456",
  "title": "How to Measure ROI from Digital Marketing: A Complete Guide",
  "content": "# How to Measure ROI from Digital Marketing...\n\n...",
  "meta_description": "Learn how to accurately measure your digital marketing ROI...",
  "featured_image_prompt": "Professional dashboard showing marketing analytics",
  "seo_score": 85,
  "keywords_used": ["digital marketing ROI", "marketing metrics", "conversion tracking"],
  "readability_score": "Grade 8",
  "estimated_reading_time": "6 minutes"
}
```

#### 5. Content Scheduling

**Schedule Content Publication**
```http
POST /api/v1/content/schedule
Headers:
  X-API-Key: <api_key>
  Content-Type: application/json
Body:
{
  "content_id": "cnt_def456",
  "wordpress_site": "https://clientwebsite.com",
  "publish_date": "2026-04-15T10:00:00Z",
  "post_status": "publish",
  "categories": ["Digital Marketing", "Analytics"],
  "tags": ["ROI", "Metrics", "Analytics"],
  "auto_update": true,
  "update_frequency": "monthly"
}
```

Response:
```json
{
  "schedule_id": "sch_ghi789",
  "status": "scheduled",
  "publish_date": "2026-04-15T10:00:00Z",
  "wordpress_post_id": 12345,
  "auto_update_enabled": true,
  "next_update_date": "2026-05-15T10:00:00Z"
}
```

#### 6. Campaign Management

**Create Campaign**
```http
POST /api/v1/campaigns/create
Headers:
  X-API-Key: <api_key>
  Content-Type: application/json
Body:
{
  "context_id": "ctx_abc123",
  "campaign_name": "Q2 Lead Generation Campaign",
  "campaign_type": "lead_generation",
  "start_date": "2026-04-01",
  "end_date": "2026-06-30",
  "budget": 15000,
  "channels": ["linkedin", "google_ads", "email"],
  "target_metrics": {
    "leads": 200,
    "conversion_rate": 3.5,
    "cost_per_lead": 75
  }
}
```

#### 7. Analytics & Reporting

**Get Usage Statistics**
```http
GET /api/v1/usage/stats
Headers: X-API-Key: <api_key>
```

Response:
```json
{
  "current_period": {
    "start": "2026-04-01T00:00:00Z",
    "end": "2026-04-30T23:59:59Z"
  },
  "requests_used": 550,
  "requests_limit": 1000,
  "content_generated": 45,
  "strategies_created": 3,
  "campaigns_active": 2,
  "scheduled_posts": 12
}
```

---

## Business Model & Subscription Architecture

### Subscription Tiers

| Tier | Price/Month | API Requests | Features |
|------|-------------|--------------|----------|
| Starter | $49 | 100 | Basic content generation, 1 website |
| Professional | $149 | 1,000 | Full strategy + content, 5 websites, auto-scheduling |
| Agency | $499 | 5,000 | White-label, unlimited websites, priority support |
| Enterprise | Custom | Unlimited | Custom integrations, dedicated instance |

### Database Schema

```sql
-- Users/Customers
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    company_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Subscriptions
CREATE TABLE subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    plan VARCHAR(50) NOT NULL, -- 'starter', 'professional', 'agency', 'enterprise'
    status VARCHAR(20) NOT NULL, -- 'active', 'cancelled', 'expired'
    stripe_customer_id VARCHAR(255),
    stripe_subscription_id VARCHAR(255),
    current_period_start TIMESTAMP,
    current_period_end TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- API Keys
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    user_id UUID REFERENCES users(id),
    subscription_id UUID REFERENCES subscriptions(id),
    name VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    last_used_at TIMESTAMP
);

-- Usage Tracking
CREATE TABLE api_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    api_key_id UUID REFERENCES api_keys(id),
    endpoint VARCHAR(255),
    request_timestamp TIMESTAMP DEFAULT NOW(),
    response_status INTEGER,
    tokens_used INTEGER,
    cost_cents INTEGER
);

-- Website Contexts
CREATE TABLE website_contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    website_url VARCHAR(500),
    context_data JSONB,
    last_updated TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Scheduled Content
CREATE TABLE scheduled_content (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    content_id VARCHAR(255),
    wordpress_site VARCHAR(500),
    wordpress_post_id INTEGER,
    scheduled_date TIMESTAMP,
    status VARCHAR(50), -- 'scheduled', 'published', 'failed'
    auto_update_enabled BOOLEAN DEFAULT false,
    update_frequency VARCHAR(50), -- 'weekly', 'monthly', 'quarterly'
    next_update_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Stripe Integration

```python
# billing.py
import stripe
from datetime import datetime, timedelta

stripe.api_key = "sk_live_..."

PLANS = {
    "starter": {
        "price_id": "price_1abc123",
        "api_requests": 100,
        "websites": 1,
        "price_cents": 4900
    },
    "professional": {
        "price_id": "price_2def456",
        "api_requests": 1000,
        "websites": 5,
        "price_cents": 14900
    },
    "agency": {
        "price_id": "price_3ghi789",
        "api_requests": 5000,
        "websites": -1,  # unlimited
        "price_cents": 49900
    }
}

async def create_subscription(user_id: str, plan: str, payment_method_id: str):
    """Create new subscription via Stripe"""
    plan_config = PLANS[plan]
    
    # Create Stripe customer if not exists
    customer = stripe.Customer.create(
        email=user.email,
        payment_method=payment_method_id,
        invoice_settings={"default_payment_method": payment_method_id}
    )
    
    # Create subscription
    subscription = stripe.Subscription.create(
        customer=customer.id,
        items=[{"price": plan_config["price_id"]}],
        payment_behavior="default_incomplete",
        expand=["latest_invoice.payment_intent"]
    )
    
    # Store in database
    await db.execute("""
        INSERT INTO subscriptions 
        (user_id, plan, status, stripe_customer_id, stripe_subscription_id, 
         current_period_start, current_period_end)
        VALUES ($1, $2, 'active', $3, $4, $5, $6)
    """, user_id, plan, customer.id, subscription.id,
        datetime.fromtimestamp(subscription.current_period_start),
        datetime.fromtimestamp(subscription.current_period_end))
    
    return subscription

async def handle_webhook_event(event: dict):
    """Handle Stripe webhook events"""
    event_type = event["type"]
    
    if event_type == "invoice.payment_succeeded":
        # Extend subscription period
        subscription_id = event["data"]["object"]["subscription"]
        await extend_subscription(subscription_id)
    
    elif event_type == "customer.subscription.deleted":
        # Cancel subscription
        subscription_id = event["data"]["object"]["id"]
        await cancel_subscription(subscription_id)
    
    elif event_type == "invoice.payment_failed":
        # Notify user of payment failure
        customer_id = event["data"]["object"]["customer"]
        await notify_payment_failure(customer_id)

async def check_api_quota(api_key: str) -> dict:
    """Check if API key has remaining quota"""
    usage = await db.fetchrow("""
        SELECT 
            s.plan,
            s.current_period_end,
            COUNT(u.id) as requests_used,
            p.api_requests as requests_limit
        FROM api_keys k
        JOIN subscriptions s ON k.subscription_id = s.id
        JOIN LATERAL (SELECT api_requests FROM plans WHERE name = s.plan) p ON true
        LEFT JOIN api_usage u ON u.api_key_id = k.id 
            AND u.request_timestamp >= s.current_period_start
        WHERE k.key_hash = $1 AND k.is_active = true
        GROUP BY s.plan, s.current_period_end, p.api_requests
    """, hash_key(api_key))
    
    if not usage:
        return {"valid": False, "reason": "Invalid API key"}
    
    remaining = usage["requests_limit"] - usage["requests_used"]
    
    return {
        "valid": True,
        "plan": usage["plan"],
        "requests_remaining": remaining,
        "requests_limit": usage["requests_limit"],
        "reset_date": usage["current_period_end"]
    }
```

### WordPress Plugin Payment Flow

```
1. User installs free WordPress plugin
2. Plugin shows "Connect to MarketingAI" button
3. User clicks -> Redirects to SaaS signup page
4. User selects plan, enters payment info (Stripe)
5. Payment successful -> API key generated
6. User returns to WordPress, enters API key
7. Plugin validates key via /api/v1/auth/verify
8. Plugin unlocks premium features
```

---

## Website Context Extraction

### Automatic Context Extraction Strategy

The WordPress plugin can automatically extract business context by:

1. **Scraping Public Pages**: Home, About, Services, Contact
2. **Analyzing WordPress Content**: Existing posts, categories, tags
3. **Extracting Metadata**: Site title, description, keywords
4. **Competitor Analysis**: Identifying competitors from industry

### WordPress Plugin Implementation

```php
// includes/class-context-extractor.php
class Marketing_AI_Context_Extractor {
    
    /**
     * Extract business context from WordPress site
     */
    public function extract_site_context() {
        $context = [
            'company_name' => $this->get_company_name(),
            'site_url' => get_site_url(),
            'site_description' => get_bloginfo('description'),
            'industry' => $this->detect_industry(),
            'existing_content' => $this->analyze_existing_content(),
            'categories' => $this->get_top_categories(),
            'tags' => $this->get_popular_tags(),
            'products_services' => $this->extract_products_services(),
            'target_audience' => $this->infer_target_audience(),
        ];
        
        return $context;
    }
    
    /**
     * Get company name from site options
     */
    private function get_company_name() {
        return get_bloginfo('name');
    }
    
    /**
     * Detect industry from content analysis
     */
    private function detect_industry() {
        $posts = get_posts([
            'numberposts' => 50,
            'post_status' => 'publish',
            'post_type' => 'post'
        ]);
        
        $content = '';
        foreach ($posts as $post) {
            $content .= ' ' . $post->post_title . ' ' . $post->post_content;
        }
        
        // Industry keyword detection
        $industries = [
            'technology' => ['software', 'tech', 'digital', 'app', 'saas'],
            'healthcare' => ['health', 'medical', 'clinic', 'wellness'],
            'finance' => ['financial', 'investment', 'insurance', 'banking'],
            'ecommerce' => ['shop', 'store', 'product', 'ecommerce'],
            'education' => ['education', 'course', 'learning', 'training'],
        ];
        
        $scores = [];
        foreach ($industries as $industry => $keywords) {
            $score = 0;
            foreach ($keywords as $keyword) {
                $score += substr_count(strtolower($content), $keyword);
            }
            $scores[$industry] = $score;
        }
        
        arsort($scores);
        return key($scores); // Return highest scoring industry
    }
    
    /**
     * Analyze existing content
     */
    private function analyze_existing_content() {
        $stats = wp_count_posts();
        $categories = get_categories(['hide_empty' => true]);
        $tags = get_tags(['hide_empty' => true]);
        
        return [
            'total_posts' => $stats->publish,
            'total_pages' => wp_count_posts('page')->publish,
            'categories_count' => count($categories),
            'tags_count' => count($tags),
            'avg_post_length' => $this->calculate_avg_post_length(),
            'posting_frequency' => $this->calculate_posting_frequency(),
        ];
    }
    
    /**
     * Extract products/services from WooCommerce or custom post types
     */
    private function extract_products_services() {
        $products = [];
        
        // Check for WooCommerce
        if (class_exists('WooCommerce')) {
            $wc_products = wc_get_products(['limit' => 20, 'status' => 'publish']);
            foreach ($wc_products as $product) {
                $products[] = [
                    'name' => $product->get_name(),
                    'description' => $product->get_description(),
                    'price' => $product->get_price(),
                ];
            }
        }
        
        // Check for services custom post type
        $services = get_posts([
            'post_type' => 'service',
            'numberposts' => 20,
            'post_status' => 'publish'
        ]);
        
        foreach ($services as $service) {
            $products[] = [
                'name' => $service->post_title,
                'description' => $service->post_content,
            ];
        }
        
        return $products;
    }
    
    /**
     * Send context to MarketingAI API
     */
    public function sync_context_to_api() {
        $context = $this->extract_site_context();
        
        $api_client = new Marketing_AI_API_Client();
        $result = $api_client->create_context($context);
        
        if (!is_wp_error($result)) {
            update_option('marketing_ai_context_id', $result['context_id']);
            return $result['context_id'];
        }
        
        return false;
    }
}
```

### Python Backend Context Processing

```python
# context_extractor.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from web_scraper import scrape_website_content
from market_analyzer import MarketAnalyzer
import asyncio

router = APIRouter()

class ContextExtractionRequest(BaseModel):
    website_url: str
    wordpress_api_url: str = None
    wordpress_username: str = None
    wordpress_password: str = None
    extract_pages: list = ["home", "about", "services", "contact"]
    analyze_competitors: bool = True

@router.post("/api/v1/context/extract")
async def extract_website_context(request: ContextExtractionRequest):
    """Extract business context from WordPress website"""
    
    try:
        # Step 1: Scrape website content
        scraped_data = await scrape_website_content(
            url=request.website_url,
            pages=request.extract_pages
        )
        
        # Step 2: If WordPress API credentials provided, get additional data
        wordpress_data = {}
        if request.wordpress_api_url and request.wordpress_username:
            wordpress_data = await fetch_wordpress_data(
                api_url=request.wordpress_api_url,
                username=request.wordpress_username,
                password=request.wordpress_password
            )
        
        # Step 3: Combine and analyze
        combined_data = {**scraped_data, **wordpress_data}
        
        # Step 4: Use LLM to extract structured context
        llm = get_llm_instance()
        analyzer = MarketAnalyzer()
        
        context = await analyzer.extract_business_context(
            llm=llm,
            website_data=combined_data
        )
        
        # Step 5: Optional competitor analysis
        if request.analyze_competitors:
            competitors = await analyzer.find_competitors(
                industry=context.get("industry"),
                location=context.get("location")
            )
            context["competitors"] = competitors
        
        # Step 6: Store context
        context_id = await store_context(context)
        
        return {
            "context_id": context_id,
            **context
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def scrape_website_content(url: str, pages: list) -> dict:
    """Scrape content from specified pages"""
    from firecrawl import FirecrawlApp
    
    app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    
    results = {}
    for page in pages:
        page_url = f"{url}/{page}" if page != "home" else url
        try:
            content = app.scrape_url(page_url, params={"formats": ["markdown"]})
            results[page] = content
        except Exception as e:
            logger.warning(f"Failed to scrape {page_url}: {e}")
    
    return results

async def fetch_wordpress_data(api_url: str, username: str, password: str) -> dict:
    """Fetch data from WordPress REST API"""
    import aiohttp
    
    auth = aiohttp.BasicAuth(username, password)
    
    async with aiohttp.ClientSession(auth=auth) as session:
        # Get recent posts
        async with session.get(f"{api_url}/wp/v2/posts?per_page=50") as resp:
            posts = await resp.json()
        
        # Get pages
        async with session.get(f"{api_url}/wp/v2/pages?per_page=20") as resp:
            pages = await resp.json()
        
        # Get categories and tags
        async with session.get(f"{api_url}/wp/v2/categories") as resp:
            categories = await resp.json()
        
        async with session.get(f"{api_url}/wp/v2/tags") as resp:
            tags = await resp.json()
        
        return {
            "posts": posts,
            "pages": pages,
            "categories": categories,
            "tags": tags
        }
```

---

## Strategy & Campaign Generation

### Strategy Generation Pipeline

```python
# strategy_generator.py
from content_generator import ContentGenerator
from market_analyzer import MarketAnalyzer
from langchain_core.prompts import ChatPromptTemplate

class StrategyGenerator:
    """Generate comprehensive marketing strategies"""
    
    def __init__(self):
        self.analyzer = MarketAnalyzer()
        self.content_gen = ContentGenerator()
    
    async def generate_strategy(self, context_id: str, params: dict) -> dict:
        """Generate complete marketing strategy"""
        
        # Get stored context
        context = await get_context(context_id)
        
        # Step 1: Market Analysis
        market_analysis = await self.analyzer.generate_market_analysis(
            llm=get_llm(),
            industry=context.get("industry"),
            company_name=context.get("company_name"),
            use_web_scraping=True
        )
        
        # Step 2: Competitive Analysis
        competitor_analysis = await self.analyzer.generate_competitor_analysis(
            llm=get_llm(),
            analysis_data=market_analysis,
            company_name=context.get("company_name"),
            industry=context.get("industry")
        )
        
        # Step 3: Generate Strategy
        strategy_prompt = ChatPromptTemplate.from_template("""
        You are a senior marketing strategist. Create a comprehensive {timeframe} 
        marketing strategy for {company_name}.
        
        Business Context:
        - Industry: {industry}
        - Target Audience: {target_audience}
        - Products/Services: {products_services}
        - Marketing Goals: {marketing_goals}
        - Budget: ${budget_range}
        
        Market Analysis:
        {market_analysis}
        
        Competitive Analysis:
        {competitor_analysis}
        
        Priority Channels: {priority_channels}
        
        Generate a detailed strategy including:
        1. Executive Summary
        2. Target Audience Segmentation
        3. Value Proposition & Positioning
        4. Channel Strategy (prioritized by ROI)
        5. Content Strategy with calendar
        6. Budget Allocation
        7. Implementation Timeline (30-60-90 day plan)
        8. KPIs & Success Metrics
        9. Risk Assessment
        """)
        
        llm = get_llm()
        chain = strategy_prompt | llm
        
        strategy = chain.invoke({
            "company_name": context.get("company_name"),
            "industry": context.get("industry"),
            "target_audience": context.get("target_audience"),
            "products_services": context.get("products_services"),
            "marketing_goals": context.get("marketing_goals"),
            "budget_range": params.get("budget_range", "5000-10000"),
            "timeframe": params.get("timeframe", "90_days"),
            "priority_channels": ", ".join(params.get("priority_channels", [])),
            "market_analysis": market_analysis,
            "competitor_analysis": competitor_analysis
        })
        
        # Step 4: Generate Content Calendar
        content_calendar = await self.generate_content_calendar(
            context=context,
            strategy=strategy,
            timeframe=params.get("timeframe", "90_days")
        )
        
        # Step 5: Store strategy
        strategy_id = await store_strategy({
            "context_id": context_id,
            "strategy": strategy,
            "content_calendar": content_calendar,
            "market_analysis": market_analysis,
            "competitor_analysis": competitor_analysis
        })
        
        return {
            "strategy_id": strategy_id,
            "strategy": strategy,
            "content_calendar": content_calendar
        }
    
    async def generate_content_calendar(self, context: dict, strategy: str, 
                                       timeframe: str) -> list:
        """Generate detailed content calendar"""
        
        calendar_prompt = ChatPromptTemplate.from_template("""
        Based on this marketing strategy, create a detailed content calendar.
        
        Strategy: {strategy}
        
        Timeframe: {timeframe}
        
        Generate a calendar with:
        - Specific post dates
        - Content topics and titles
        - Platform assignments (LinkedIn, Instagram, Blog, etc.)
        - Content types (educational, promotional, thought leadership)
        - Calls-to-action
        
        Format as a list of content items with all details.
        """)
        
        llm = get_llm()
        chain = calendar_prompt | llm
        
        calendar = chain.invoke({
            "strategy": strategy,
            "timeframe": timeframe
        })
        
        return self._parse_calendar(calendar)
```

---

## Strategy-to-Task Decomposition & Agent Assignment

When a user selects a marketing strategy or campaign for implementation, the system automatically decomposes the strategy into actionable tasks, creates a calendar-based to-do list in WordPress, and assigns each task to specialized AI agents for execution.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Selects Strategy                     │
│              (WordPress Plugin UI / Streamlit)               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Strategy Decomposition Engine                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  1. Parse Strategy Structure                          │   │
│  │  2. Identify Task Categories & Dependencies          │   │
│  │  3. Generate Dynamic Agent Prompts                   │   │
│  │  4. Create Task Objects with Metadata                │   │
│  │  5. Build Execution Timeline                         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  Content Agent   │ │   SEO Agent      │ │  Social Agent    │
│  - Blog Posts    │ │  - Keyword Opt   │ │  - LinkedIn      │
│  - Social Copy   │ │  - Meta Tags     │ │  - Instagram     │
│  - Email Copy    │ │  - Schema Markup │ │  - Twitter/X     │
└──────────────────┘ └──────────────────┘ └──────────────────┘
              │               │               │
              ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│              WordPress Calendar & Task Manager               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  Task Queue     │  │  Calendar View  │  │  Progress    │ │
│  │  (Pending)      │  │  (Scheduled)    │  │  Tracking    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Task Decomposition Engine

The core engine that converts a strategy into executable tasks:

```python
# strategy_decomposer.py
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json

class AgentType(str, Enum):
    CONTENT_WRITER = "content_writer"
    SEO_SPECIALIST = "seo_specialist"
    SOCIAL_MEDIA_MANAGER = "social_media_manager"
    EMAIL_MARKETER = "email_marketer"
    GRAPHIC_DESIGNER = "graphic_designer"
    ANALYTICS_TRACKER = "analytics_tracker"
    CAMPAIGN_MANAGER = "campaign_manager"

class TaskPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class TaskDependency(BaseModel):
    task_id: str
    dependency_type: str  # "blocks", "informs", "optional"

class TaskDefinition(BaseModel):
    task_id: str
    title: str
    description: str
    agent_type: AgentType
    priority: TaskPriority
    estimated_duration_hours: float
    scheduled_date: datetime
    deadline: datetime
    dependencies: List[TaskDependency] = []
    dynamic_prompt_template: str
    input_context: Dict[str, Any] = {}
    expected_output_format: str
    wordpress_calendar_entry: bool = True
    auto_execute: bool = False
    requires_approval: bool = True

class StrategyDecomposer:
    """Decomposes marketing strategies into executable tasks with dynamic prompts"""
    
    def __init__(self):
        self.agent_prompt_templates = self._load_agent_templates()
        self.task_patterns = self._load_task_patterns()
    
    def _load_agent_templates(self) -> Dict[str, str]:
        """Load dynamic prompt templates for each agent type"""
        return {
            AgentType.CONTENT_WRITER: """
You are a professional content writer executing a specific task within a marketing campaign.

## Campaign Context
- Company: {company_name}
- Industry: {industry}
- Target Audience: {target_audience}
- Campaign: {campaign_name}
- Campaign Goal: {campaign_goal}

## Your Task
{task_description}

## Content Requirements
- Tone: {tone}
- Word Count: {word_count}
- Keywords to Include: {keywords}
- Brand Voice Guidelines: {brand_voice}

## Strategic Context
This task is part of a larger {campaign_type} campaign running from {start_date} to {end_date}.
Your output should align with the overall campaign messaging: {campaign_messaging}

## Output Format
{expected_output_format}

## Constraints
- Do not mention competitors by name unless specified
- Maintain consistent brand voice throughout
- Include the specified call-to-action
- Optimize for the target platform
""",
            
            AgentType.SEO_SPECIALIST: """
You are an SEO specialist optimizing content for search engine visibility.

## Campaign Context
- Company: {company_name}
- Industry: {industry}
- Target Keywords: {target_keywords}

## Your Task
{task_description}

## Content to Optimize
{content_to_optimize}

## SEO Requirements
- Primary Keyword: {primary_keyword}
- Secondary Keywords: {secondary_keywords}
- Target Search Intent: {search_intent}
- Competitor Analysis: {competitor_keywords}

## Deliverables
{expected_output_format}

## Technical SEO Checklist
- [ ] Title tag optimized (50-60 chars, primary keyword)
- [ ] Meta description compelling (150-160 chars)
- [ ] H1 contains primary keyword
- [ ] H2/H3 structure logical with secondary keywords
- [ ] Internal linking opportunities identified
- [ ] Image alt text suggestions provided
- [ ] Schema markup recommendations included
""",
            
            AgentType.SOCIAL_MEDIA_MANAGER: """
You are a social media manager creating platform-specific content.

## Campaign Context
- Company: {company_name}
- Campaign: {campaign_name}
- Campaign Hashtag: {campaign_hashtag}

## Your Task
{task_description}

## Platform Specifications
- Platform: {platform}
- Optimal Post Length: {post_length}
- Best Posting Time: {best_posting_time}
- Hashtag Strategy: {hashtag_strategy}

## Content Requirements
- Visual Description: {visual_requirements}
- Caption/Copy: {copy_requirements}
- Call-to-Action: {cta}
- Engagement Prompt: {engagement_prompt}

## Output Format
{expected_output_format}
""",
            
            AgentType.EMAIL_MARKETER: """
You are an email marketing specialist creating campaign emails.

## Campaign Context
- Company: {company_name}
- Campaign: {campaign_name}
- Email Sequence Position: {sequence_position}

## Your Task
{task_description}

## Audience Segment
- Segment: {audience_segment}
- Pain Points: {pain_points}
- Previous Interactions: {previous_interactions}

## Email Requirements
- Subject Line Options (provide 3): {subject_requirements}
- Preheader Text: {preheader_requirements}
- Body Copy: {body_requirements}
- CTA Button Text: {cta_requirements}

## Output Format
{expected_output_format}
""",
            
            AgentType.GRAPHIC_DESIGNER: """
You are a creative designer generating visual concepts for a marketing campaign.

## Campaign Context
- Company: {company_name}
- Brand Colors: {brand_colors}
- Brand Style: {brand_style}
- Campaign: {campaign_name}

## Your Task
{task_description}

## Design Requirements
- Format: {format}
- Dimensions: {dimensions}
- Style Direction: {style_direction}
- Key Message: {key_message}

## Visual Elements
- Primary Visual: {primary_visual}
- Supporting Elements: {supporting_elements}
- Text Overlay: {text_overlay}
- Logo Placement: {logo_placement}

## Output Format
Provide detailed design brief and image generation prompt:
{expected_output_format}
""",
            
            AgentType.ANALYTICS_TRACKER: """
You are an analytics specialist setting up tracking and reporting for a campaign.

## Campaign Context
- Company: {company_name}
- Campaign: {campaign_name}
- Campaign Goals: {campaign_goals}

## Your Task
{task_description}

## KPIs to Track
- Primary KPIs: {primary_kpis}
- Secondary KPIs: {secondary_kpis}
- Conversion Events: {conversion_events}

## Deliverables
{expected_output_format}
""",
            
            AgentType.CAMPAIGN_MANAGER: """
You are a campaign coordinator ensuring all tasks align with the overall strategy.

## Campaign Context
- Company: {company_name}
- Campaign: {campaign_name}
- Budget: {budget}
- Timeline: {timeline}

## Your Task
{task_description}

## Coordination Requirements
- Tasks to Coordinate: {coordination_tasks}
- Dependencies: {dependencies}
- Approval Workflow: {approval_workflow}

## Deliverables
{expected_output_format}
"""
        }
    
    def _load_task_patterns(self) -> Dict[str, List[Dict]]:
        """Load task patterns for different strategy types"""
        return {
            "content_marketing": [
                {
                    "pattern": "content strategy",
                    "tasks": [
                        {"agent": AgentType.CONTENT_WRITER, "title": "Create blog post: {topic}", "duration": 4},
                        {"agent": AgentType.SEO_SPECIALIST, "title": "Optimize blog post for SEO", "duration": 1},
                        {"agent": AgentType.SOCIAL_MEDIA_MANAGER, "title": "Create social promotion posts", "duration": 2},
                        {"agent": AgentType.GRAPHIC_DESIGNER, "title": "Design featured image for blog post", "duration": 1.5},
                    ]
                },
                {
                    "pattern": "email campaign",
                    "tasks": [
                        {"agent": AgentType.EMAIL_MARKETER, "title": "Write email sequence: {sequence_name}", "duration": 3},
                        {"agent": AgentType.CONTENT_WRITER, "title": "Create landing page copy", "duration": 3},
                        {"agent": AgentType.ANALYTICS_TRACKER, "title": "Set up email tracking", "duration": 1},
                    ]
                }
            ],
            "social_media_campaign": [
                {
                    "pattern": "social media strategy",
                    "tasks": [
                        {"agent": AgentType.SOCIAL_MEDIA_MANAGER, "title": "Create {platform} content calendar", "duration": 4},
                        {"agent": AgentType.GRAPHIC_DESIGNER, "title": "Design social media graphics batch", "duration": 3},
                        {"agent": AgentType.CONTENT_WRITER, "title": "Write social media copy batch", "duration": 3},
                        {"agent": AgentType.ANALYTICS_TRACKER, "title": "Set up social media tracking", "duration": 1},
                    ]
                }
            ],
            "seo_campaign": [
                {
                    "pattern": "seo strategy",
                    "tasks": [
                        {"agent": AgentType.SEO_SPECIALIST, "title": "Conduct keyword research for {topic_cluster}", "duration": 4},
                        {"agent": AgentType.CONTENT_WRITER, "title": "Write optimized content for {keyword}", "duration": 4},
                        {"agent": AgentType.SEO_SPECIALIST, "title": "Technical SEO audit and fixes", "duration": 6},
                        {"agent": AgentType.CONTENT_WRITER, "title": "Create internal linking structure", "duration": 2},
                    ]
                }
            ],
            "comprehensive_campaign": [
                {
                    "pattern": "comprehensive strategy",
                    "tasks": [
                        {"agent": AgentType.CAMPAIGN_MANAGER, "title": "Create campaign coordination plan", "duration": 2},
                        {"agent": AgentType.CONTENT_WRITER, "title": "Develop core messaging framework", "duration": 3},
                        {"agent": AgentType.SEO_SPECIALIST, "title": "SEO keyword strategy and optimization plan", "duration": 3},
                        {"agent": AgentType.SOCIAL_MEDIA_MANAGER, "title": "Social media content calendar", "duration": 4},
                        {"agent": AgentType.EMAIL_MARKETER, "title": "Email nurture sequence", "duration": 3},
                        {"agent": AgentType.GRAPHIC_DESIGNER, "title": "Campaign visual identity package", "duration": 4},
                        {"agent": AgentType.ANALYTICS_TRACKER, "title": "Campaign tracking dashboard setup", "duration": 2},
                    ]
                }
            ]
        }
    
    async def decompose_strategy(self, strategy_id: str, strategy_data: Dict[str, Any]) -> List[TaskDefinition]:
        """Main entry point: decompose a strategy into executable tasks"""
        
        # Step 1: Use LLM to identify strategy type and extract task structure
        strategy_type, task_structure = await self._analyze_strategy_structure(strategy_data)
        
        # Step 2: Generate tasks based on patterns and LLM analysis
        tasks = await self._generate_tasks(strategy_data, task_structure)
        
        # Step 3: Build dependency graph
        tasks = self._build_dependencies(tasks, strategy_data)
        
        # Step 4: Schedule tasks on calendar
        tasks = self._schedule_tasks(tasks, strategy_data)
        
        # Step 5: Generate dynamic prompts for each task
        tasks = self._generate_dynamic_prompts(tasks, strategy_data)
        
        # Step 6: Store tasks and return
        stored_tasks = await self._store_tasks(strategy_id, tasks)
        
        return stored_tasks
    
    async def _analyze_strategy_structure(self, strategy_data: Dict[str, Any]) -> tuple:
        """Use LLM to analyze strategy and identify task categories"""
        
        analysis_prompt = ChatPromptTemplate.from_template("""
You are a marketing operations expert analyzing a marketing strategy to identify all executable tasks.

## Strategy Document
{strategy_text}

## Business Context
- Company: {company_name}
- Industry: {industry}
- Target Audience: {target_audience}
- Campaign Duration: {duration}

## Task: Analyze this strategy and return a structured breakdown:

Return ONLY valid JSON with this exact structure:
{{
    "strategy_type": "content_marketing|social_media_campaign|seo_campaign|comprehensive_campaign|custom",
    "phases": [
        {{
            "phase_name": "Phase 1: Awareness",
            "phase_order": 1,
            "start_week": 1,
            "end_week": 4,
            "objectives": ["objective1", "objective2"],
            "channels": ["blog", "social", "email"],
            "content_pieces": [
                {{
                    "type": "blog_post",
                    "topic": "Topic here",
                    "platform": "website",
                    "estimated_effort_hours": 4
                }}
            ]
        }}
    ],
    "total_estimated_tasks": 25,
    "required_agents": ["content_writer", "seo_specialist", "social_media_manager"]
}}
""")
        
        llm = get_llm()
        parser = JsonOutputParser()
        chain = analysis_prompt | llm | parser
        
        result = chain.invoke({
            "strategy_text": strategy_data.get("strategy", ""),
            "company_name": strategy_data.get("context", {}).get("company_name", ""),
            "industry": strategy_data.get("context", {}).get("industry", ""),
            "target_audience": strategy_data.get("context", {}).get("target_audience", ""),
            "duration": strategy_data.get("timeframe", "90_days")
        })
        
        return result.get("strategy_type", "comprehensive_campaign"), result
    
    async def _generate_tasks(self, strategy_data: Dict, structure: Dict) -> List[TaskDefinition]:
        """Generate individual task definitions from strategy structure"""
        tasks = []
        task_counter = 0
        
        strategy_type = structure.get("strategy_type", "comprehensive_campaign")
        phases = structure.get("phases", [])
        
        for phase in phases:
            phase_name = phase.get("phase_name", "")
            phase_order = phase.get("phase_order", 1)
            content_pieces = phase.get("content_pieces", [])
            channels = phase.get("channels", [])
            
            for piece in content_pieces:
                task_counter += 1
                task_id = f"task_{task_counter:04d}"
                
                # Determine agent type based on content type and channel
                agent_type = self._determine_agent_type(piece.get("type", ""), channels)
                
                # Create task definition
                task = TaskDefinition(
                    task_id=task_id,
                    title=self._generate_task_title(piece, phase_name),
                    description=self._generate_task_description(piece, phase_name, strategy_data),
                    agent_type=agent_type,
                    priority=self._determine_priority(piece, phase_order),
                    estimated_duration_hours=piece.get("estimated_effort_hours", 2),
                    scheduled_date=datetime.now(),  # Will be updated in scheduling
                    deadline=datetime.now(),
                    expected_output_format=self._get_output_format(piece.get("type", "")),
                    input_context={
                        "strategy_data": strategy_data,
                        "phase": phase,
                        "content_piece": piece
                    }
                )
                
                tasks.append(task)
        
        # Add coordination and setup tasks
        tasks.extend(self._generate_coordination_tasks(structure, strategy_data))
        
        return tasks
    
    def _determine_agent_type(self, content_type: str, channels: List[str]) -> AgentType:
        """Determine which agent should handle a task"""
        type_mapping = {
            "blog_post": AgentType.CONTENT_WRITER,
            "article": AgentType.CONTENT_WRITER,
            "social_post": AgentType.SOCIAL_MEDIA_MANAGER,
            "email": AgentType.EMAIL_MARKETER,
            "landing_page": AgentType.CONTENT_WRITER,
            "graphic": AgentType.GRAPHIC_DESIGNER,
            "infographic": AgentType.GRAPHIC_DESIGNER,
            "video_script": AgentType.CONTENT_WRITER,
            "seo_optimization": AgentType.SEO_SPECIALIST,
            "analytics_setup": AgentType.ANALYTICS_TRACKER,
        }
        
        return type_mapping.get(content_type, AgentType.CONTENT_WRITER)
    
    def _generate_task_title(self, piece: Dict, phase_name: str) -> str:
        """Generate a descriptive task title"""
        content_type = piece.get("type", "content")
        topic = piece.get("topic", "general")
        
        titles = {
            "blog_post": f"Write blog post: {topic}",
            "social_post": f"Create social content: {topic}",
            "email": f"Write email: {topic}",
            "graphic": f"Design graphic: {topic}",
            "seo_optimization": f"SEO optimize: {topic}",
        }
        
        return titles.get(content_type, f"Create {content_type}: {topic}")
    
    def _generate_task_description(self, piece: Dict, phase_name: str, strategy_data: Dict) -> str:
        """Generate detailed task description"""
        return (
            f"Execute the following task as part of '{phase_name}':\n\n"
            f"Content Type: {piece.get('type', 'content')}\n"
            f"Topic: {piece.get('topic', 'TBD')}\n"
            f"Platform: {piece.get('platform', 'website')}\n\n"
            f"Align with the overall campaign strategy for {strategy_data.get('context', {}).get('company_name', 'the client')}."
        )
    
    def _determine_priority(self, piece: Dict, phase_order: int) -> TaskPriority:
        """Determine task priority based on phase and content type"""
        if phase_order == 1:
            return TaskPriority.HIGH
        elif piece.get("type") in ["blog_post", "landing_page"]:
            return TaskPriority.HIGH
        elif phase_order <= 2:
            return TaskPriority.MEDIUM
        return TaskPriority.LOW
    
    def _get_output_format(self, content_type: str) -> str:
        """Define expected output format for each content type"""
        formats = {
            "blog_post": "Markdown formatted blog post with H1, H2, H3 structure, meta description, and suggested featured image description",
            "social_post": "Platform-specific post copy with hashtags, visual description, and optimal posting time",
            "email": "Complete email with subject line options, preheader, body copy, and CTA button text",
            "graphic": "Detailed design brief with dimensions, color palette, layout description, and image generation prompt",
            "seo_optimization": "SEO audit report with specific recommendations for title tags, meta descriptions, headings, internal links, and schema markup",
        }
        return formats.get(content_type, "Structured content appropriate for the specified task")
    
    def _build_dependencies(self, tasks: List[TaskDefinition], strategy_data: Dict) -> List[TaskDefinition]:
        """Build task dependency graph"""
        # SEO tasks depend on content being written first
        # Social promotion depends on content being ready
        # Graphics depend on content brief
        
        content_tasks = [t for t in tasks if t.agent_type == AgentType.CONTENT_WRITER]
        seo_tasks = [t for t in tasks if t.agent_type == AgentType.SEO_SPECIALIST]
        social_tasks = [t for t in tasks if t.agent_type == AgentType.SOCIAL_MEDIA_MANAGER]
        design_tasks = [t for t in tasks if t.agent_type == AgentType.GRAPHIC_DESIGNER]
        
        # SEO depends on content
        for seo_task in seo_tasks:
            matching_content = next((ct for ct in content_tasks if self._tasks_related(ct, seo_task)), None)
            if matching_content:
                seo_task.dependencies.append(TaskDependency(
                    task_id=matching_content.task_id,
                    dependency_type="blocks"
                ))
        
        # Social promotion depends on content
        for social_task in social_tasks:
            matching_content = next((ct for ct in content_tasks if self._tasks_related(ct, social_task)), None)
            if matching_content:
                social_task.dependencies.append(TaskDependency(
                    task_id=matching_content.task_id,
                    dependency_type="blocks"
                ))
        
        # Design tasks depend on content brief
        for design_task in design_tasks:
            matching_content = next((ct for ct in content_tasks if self._tasks_related(ct, design_task)), None)
            if matching_content:
                design_task.dependencies.append(TaskDependency(
                    task_id=matching_content.task_id,
                    dependency_type="informs"
                ))
        
        return tasks
    
    def _tasks_related(self, content_task: TaskDefinition, other_task: TaskDefinition) -> bool:
        """Check if two tasks are related by topic"""
        content_topic = content_task.input_context.get("content_piece", {}).get("topic", "").lower()
        other_topic = other_task.input_context.get("content_piece", {}).get("topic", "").lower()
        return content_topic and other_topic and (content_topic in other_topic or other_topic in content_topic)
    
    def _schedule_tasks(self, tasks: List[TaskDefinition], strategy_data: Dict) -> List[TaskDefinition]:
        """Schedule tasks on calendar based on dependencies and priorities"""
        
        # Parse strategy timeframe
        timeframe = strategy_data.get("timeframe", "90_days")
        days_map = {"30_days": 30, "60_days": 60, "90_days": 90, "180_days": 180}
        total_days = days_map.get(timeframe, 90)
        
        start_date = datetime.now() + timedelta(days=1)
        
        # Sort tasks by priority and dependencies
        priority_order = {TaskPriority.HIGH: 0, TaskPriority.MEDIUM: 1, TaskPriority.LOW: 2}
        tasks.sort(key=lambda t: (priority_order.get(t.priority, 2), t.estimated_duration_hours))
        
        # Schedule respecting dependencies
        scheduled_dates = {}
        
        for task in tasks:
            # Calculate earliest start based on dependencies
            earliest_start = start_date
            for dep in task.dependencies:
                if dep.task_id in scheduled_dates:
                    if dep.dependency_type == "blocks":
                        earliest_start = max(earliest_start, scheduled_dates[dep.task_id] + timedelta(hours=4))
            
            task.scheduled_date = earliest_start
            task.deadline = earliest_start + timedelta(hours=task.estimated_duration_hours)
            scheduled_dates[task.task_id] = task.deadline
        
        return tasks
    
    def _generate_dynamic_prompts(self, tasks: List[TaskDefinition], strategy_data: Dict) -> List[TaskDefinition]:
        """Generate dynamic prompts for each task based on strategy context"""
        
        context = strategy_data.get("context", {})
        strategy = strategy_data.get("strategy", {})
        
        for task in tasks:
            template = self.agent_prompt_templates.get(task.agent_type, "")
            
            # Fill in template with context
            prompt = template.format(
                company_name=context.get("company_name", "the company"),
                industry=context.get("industry", "the industry"),
                target_audience=context.get("target_audience", "the target audience"),
                campaign_name=strategy_data.get("campaign_name", "Marketing Campaign"),
                campaign_goal=context.get("marketing_goals", "increase brand awareness"),
                task_description=task.description,
                tone="professional",
                word_count=str(task.input_context.get("content_piece", {}).get("word_count", "1000-1500")),
                keywords=", ".join(context.get("keywords", [])[:5]),
                brand_voice=context.get("brand_description", ""),
                campaign_type=strategy.get("type", "comprehensive"),
                campaign_messaging=strategy.get("value_proposition", ""),
                start_date=strategy_data.get("start_date", ""),
                end_date=strategy_data.get("end_date", ""),
                expected_output_format=task.expected_output_format,
                # SEO specific
                target_keywords=", ".join(context.get("keywords", [])[:10]),
                content_to_optimize="[Content will be provided from content writer task]",
                primary_keyword=context.get("keywords", [""])[0] if context.get("keywords") else "",
                secondary_keywords=", ".join(context.get("keywords", [])[1:6]),
                search_intent="informational",
                competitor_keywords=", ".join(context.get("competitors", [])),
                # Social specific
                platform=task.input_context.get("content_piece", {}).get("platform", "LinkedIn"),
                post_length="150-300 words",
                best_posting_time="10:00 AM Tuesday-Thursday",
                hashtag_strategy="Mix of branded, industry, and trending hashtags",
                visual_requirements="Professional, on-brand imagery",
                copy_requirements="Engaging, value-driven copy with clear CTA",
                cta="Learn more at our website",
                engagement_prompt="End with a thought-provoking question",
                campaign_hashtag=f"#{context.get('company_name', 'Brand')}{strategy_data.get('campaign_name', 'Campaign')}".replace(" ", ""),
                # Email specific
                sequence_position="1 of 5",
                audience_segment=context.get("target_audience", ""),
                pain_points=context.get("customer_pain_points", ""),
                previous_interactions="None - new subscriber",
                subject_requirements="Compelling, under 50 characters, personalized",
                preheader_requirements="Supporting text, under 100 characters",
                body_requirements="Value-focused, scannable, mobile-optimized",
                cta_requirements="Action-oriented, under 4 words",
                # Design specific
                brand_colors="#1f77b4, #ff7f0e, #2ca02c",
                brand_style="Modern, professional, clean",
                format="1200x628px social media graphic",
                dimensions="1200x628",
                style_direction="Clean, modern, professional",
                key_message=task.input_context.get("content_piece", {}).get("topic", ""),
                primary_visual="Professional imagery related to topic",
                supporting_elements="Brand logo, subtle background pattern",
                text_overlay="Headline text, minimal",
                logo_placement="Bottom right corner",
                # Analytics specific
                campaign_goals=context.get("marketing_goals", ""),
                primary_kpis="Traffic, engagement, conversions",
                secondary_kpis="Time on page, bounce rate, social shares",
                conversion_events="Form submissions, downloads, purchases",
                # Campaign manager specific
                budget=strategy_data.get("budget", "TBD"),
                timeline=strategy_data.get("timeframe", "90 days"),
                coordination_tasks="All tasks in current phase",
                dependencies="See task dependency graph",
                approval_workflow="Content -> SEO -> Campaign Manager -> Publish"
            )
            
            task.dynamic_prompt_template = prompt
        
        return tasks
    
    def _generate_coordination_tasks(self, structure: Dict, strategy_data: Dict) -> List[TaskDefinition]:
        """Generate meta-tasks for coordination and setup"""
        tasks = []
        task_counter = 9000  # High number to avoid collision
        
        # Initial setup task
        task_counter += 1
        tasks.append(TaskDefinition(
            task_id=f"task_{task_counter:04d}",
            title="Campaign Setup & Brief Review",
            description="Review the complete strategy document, ensure all team members have access to brand guidelines, and confirm campaign timeline.",
            agent_type=AgentType.CAMPAIGN_MANAGER,
            priority=TaskPriority.HIGH,
            estimated_duration_hours=2,
            scheduled_date=datetime.now() + timedelta(hours=1),
            deadline=datetime.now() + timedelta(hours=3),
            expected_output_format="Campaign kickoff checklist with confirmation of all prerequisites",
            auto_execute=False,
            requires_approval=False
        ))
        
        # Weekly review task
        task_counter += 1
        tasks.append(TaskDefinition(
            task_id=f"task_{task_counter:04d}",
            title="Weekly Campaign Performance Review",
            description="Review all campaign metrics, identify underperforming areas, and recommend adjustments for the following week.",
            agent_type=AgentType.ANALYTICS_TRACKER,
            priority=TaskPriority.MEDIUM,
            estimated_duration_hours=2,
            scheduled_date=datetime.now() + timedelta(weeks=1),
            deadline=datetime.now() + timedelta(weeks=1, hours=2),
            expected_output_format="Weekly performance report with insights and recommendations",
            auto_execute=True,
            requires_approval=False
        ))
        
        return tasks
    
    async def _store_tasks(self, strategy_id: str, tasks: List[TaskDefinition]) -> List[Dict]:
        """Store tasks in database and prepare WordPress calendar entries"""
        stored_tasks = []
        
        for task in tasks:
            task_data = {
                "task_id": task.task_id,
                "strategy_id": strategy_id,
                "title": task.title,
                "description": task.description,
                "agent_type": task.agent_type.value,
                "priority": task.priority.value,
                "estimated_duration_hours": task.estimated_duration_hours,
                "scheduled_date": task.scheduled_date.isoformat(),
                "deadline": task.deadline.isoformat(),
                "dependencies": [d.dict() for d in task.dependencies],
                "dynamic_prompt": task.dynamic_prompt_template,
                "expected_output_format": task.expected_output_format,
                "wordpress_calendar_entry": task.wordpress_calendar_entry,
                "auto_execute": task.auto_execute,
                "requires_approval": task.requires_approval,
                "status": "pending",
                "created_at": datetime.now().isoformat()
            }
            
            # Store in database
            await db.execute("""
                INSERT INTO campaign_tasks 
                (task_id, strategy_id, title, description, agent_type, priority,
                 estimated_duration_hours, scheduled_date, deadline, dependencies,
                 dynamic_prompt, expected_output_format, status, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """, task_data["task_id"], task_data["strategy_id"], task_data["title"],
                task_data["description"], task_data["agent_type"], task_data["priority"],
                task_data["estimated_duration_hours"], task_data["scheduled_date"],
                task_data["deadline"], json.dumps(task_data["dependencies"]),
                task_data["dynamic_prompt"], task_data["expected_output_format"],
                task_data["status"], task_data["created_at"])
            
            stored_tasks.append(task_data)
        
        return stored_tasks


class TaskExecutor:
    """Executes tasks by running dynamic prompts through appropriate agents"""
    
    def __init__(self):
        self.decomposer = StrategyDecomposer()
    
    async def execute_task(self, task_data: Dict) -> Dict:
        """Execute a single task using its dynamic prompt"""
        
        task_id = task_data["task_id"]
        agent_type = task_data["agent_type"]
        dynamic_prompt = task_data["dynamic_prompt"]
        
        # Update status to in_progress
        await self._update_task_status(task_id, "in_progress")
        
        try:
            # Select appropriate LLM based on task complexity
            llm = self._select_llm_for_task(task_data)
            
            # Create prompt and execute
            prompt = ChatPromptTemplate.from_template(dynamic_prompt)
            chain = prompt | llm
            
            result = chain.invoke({})
            
            # Parse and store result
            task_result = {
                "task_id": task_id,
                "output": result,
                "completed_at": datetime.now().isoformat(),
                "status": "completed"
            }
            
            await self._store_task_result(task_result)
            
            # If WordPress calendar entry, update it
            if task_data.get("wordpress_calendar_entry"):
                await self._update_wordpress_calendar(task_id, "completed")
            
            # Check if any dependent tasks can now start
            await self._trigger_dependent_tasks(task_id)
            
            return task_result
            
        except Exception as e:
            await self._update_task_status(task_id, "failed", str(e))
            raise
    
    async def execute_task_batch(self, strategy_id: str, agent_type: str = None) -> List[Dict]:
        """Execute all pending tasks, optionally filtered by agent type"""
        
        query = """
            SELECT * FROM campaign_tasks 
            WHERE strategy_id = $1 AND status = 'pending'
        """
        params = [strategy_id]
        
        if agent_type:
            query += " AND agent_type = $2"
            params.append(agent_type)
        
        query += " ORDER BY scheduled_date ASC"
        
        tasks = await db.fetch(query, *params)
        
        results = []
        for task in tasks:
            # Check dependencies
            if await self._dependencies_met(task["task_id"]):
                result = await self.execute_task(dict(task))
                results.append(result)
        
        return results
    
    def _select_llm_for_task(self, task_data: Dict):
        """Select appropriate LLM based on task type and complexity"""
        agent_type = task_data.get("agent_type", "")
        duration = task_data.get("estimated_duration_hours", 2)
        
        # Complex/long tasks get more capable models
        if duration > 4 or agent_type in ["campaign_manager", "analytics_tracker"]:
            return get_llm(provider="openai", model="gpt-4")
        
        # Standard tasks
        return get_llm()
    
    async def _dependencies_met(self, task_id: str) -> bool:
        """Check if all blocking dependencies are completed"""
        deps = await db.fetch("""
            SELECT dep->>'task_id' as dep_id, dep->>'dependency_type' as dep_type
            FROM campaign_tasks, jsonb_array_elements(dependencies) as dep
            WHERE task_id = $1
        """, task_id)
        
        for dep in deps:
            if dep["dep_type"] == "blocks":
                status = await db.fetchval(
                    "SELECT status FROM campaign_tasks WHERE task_id = $1",
                    dep["dep_id"]
                )
                if status != "completed":
                    return False
        
        return True
    
    async def _update_task_status(self, task_id: str, status: str, error: str = None):
        """Update task status in database"""
        await db.execute("""
            UPDATE campaign_tasks SET status = $1, error = $2, updated_at = NOW()
            WHERE task_id = $3
        """, status, error, task_id)
    
    async def _store_task_result(self, result: Dict):
        """Store task execution result"""
        await db.execute("""
            INSERT INTO task_results (task_id, output, completed_at, status)
            VALUES ($1, $2, $3, $4)
        """, result["task_id"], result["output"], result["completed_at"], result["status"])
    
    async def _update_wordpress_calendar(self, task_id: str, status: str):
        """Update WordPress calendar entry for task"""
        # This would call the WordPress plugin API
        pass
    
    async def _trigger_dependent_tasks(self, completed_task_id: str):
        """Check and trigger tasks that were waiting for this task"""
        dependent_tasks = await db.fetch("""
            SELECT ct.* FROM campaign_tasks ct
            WHERE ct.dependencies @> $1::jsonb AND ct.status = 'pending'
        """, json.dumps([{"task_id": completed_task_id}]))
        
        for task in dependent_tasks:
            if await self._dependencies_met(task["task_id"]):
                # Auto-execute if configured
                if task.get("auto_execute"):
                    await self.execute_task(dict(task))
```

### WordPress Plugin: Calendar & Task Manager

```php
// includes/class-task-calendar.php
class Marketing_AI_Task_Calendar {
    
    /**
     * Create calendar entries for all tasks in a strategy
     */
    public function create_strategy_calendar($strategy_id, $tasks) {
        $calendar_entries = [];
        
        foreach ($tasks as $task) {
            $entry = $this->create_calendar_entry($task);
            $calendar_entries[] = $entry;
        }
        
        // Store in WordPress as custom post type
        $this->save_calendar_entries($strategy_id, $calendar_entries);
        
        return $calendar_entries;
    }
    
    /**
     * Create a calendar entry from task data
     */
    private function create_calendar_entry($task) {
        return [
            'post_type' => 'marketing_task',
            'post_title' => $task['title'],
            'post_content' => $task['description'],
            'post_status' => $task['status'] === 'completed' ? 'publish' : 'draft',
            'post_date' => $task['scheduled_date'],
            'meta_input' => [
                '_task_id' => $task['task_id'],
                '_strategy_id' => $task['strategy_id'],
                '_agent_type' => $task['agent_type'],
                '_priority' => $task['priority'],
                '_estimated_hours' => $task['estimated_duration_hours'],
                '_deadline' => $task['deadline'],
                '_dependencies' => wp_json_encode($task['dependencies']),
                '_dynamic_prompt' => $task['dynamic_prompt'],
                '_expected_output' => $task['expected_output_format'],
                '_auto_execute' => $task['auto_execute'] ? 'yes' : 'no',
                '_requires_approval' => $task['requires_approval'] ? 'yes' : 'no',
                '_task_status' => $task['status'],
            ]
        ];
    }
    
    /**
     * Save calendar entries as custom post type
     */
    private function save_calendar_entries($strategy_id, $entries) {
        foreach ($entries as $entry) {
            $post_id = wp_insert_post($entry);
            
            if ($post_id && !is_wp_error($post_id)) {
                // Link to strategy
                update_post_meta($post_id, '_strategy_id', $strategy_id);
                
                // If task is scheduled, create WordPress cron event
                if ($entry['meta_input']['_auto_execute'] === 'yes') {
                    $scheduled_time = strtotime($entry['post_date']);
                    wp_schedule_single_event(
                        $scheduled_time,
                        'marketing_ai_execute_task',
                        [$post_id, $entry['meta_input']['_task_id']]
                    );
                }
            }
        }
    }
    
    /**
     * Display task calendar in admin
     */
    public function render_calendar_view() {
        $strategy_id = isset($_GET['strategy_id']) ? intval($_GET['strategy_id']) : 0;
        
        if (!$strategy_id) {
            echo '<p>Select a strategy to view its task calendar.</p>';
            return;
        }
        
        $tasks = $this->get_strategy_tasks($strategy_id);
        
        ?>
        <div class="wrap marketing-ai-task-calendar">
            <h1>Task Calendar - Strategy #<?php echo $strategy_id; ?></h1>
            
            <!-- Calendar Filter -->
            <div class="calendar-filters">
                <form method="get">
                    <input type="hidden" name="page" value="marketing-ai-tasks">
                    <select name="agent_filter">
                        <option value="">All Agents</option>
                        <option value="content_writer">Content Writer</option>
                        <option value="seo_specialist">SEO Specialist</option>
                        <option value="social_media_manager">Social Media Manager</option>
                        <option value="email_marketer">Email Marketer</option>
                        <option value="graphic_designer">Graphic Designer</option>
                    </select>
                    <select name="status_filter">
                        <option value="">All Statuses</option>
                        <option value="pending">Pending</option>
                        <option value="in_progress">In Progress</option>
                        <option value="completed">Completed</option>
                        <option value="failed">Failed</option>
                    </select>
                    <input type="submit" class="button" value="Filter">
                </form>
            </div>
            
            <!-- Task Timeline -->
            <div class="task-timeline">
                <?php $this->render_task_timeline($tasks); ?>
            </div>
            
            <!-- Task Board (Kanban) -->
            <div class="task-board">
                <?php $this->render_task_board($tasks); ?>
            </div>
        </div>
        <?php
    }
    
    /**
     * Render task timeline view
     */
    private function render_task_timeline($tasks) {
        // Group tasks by week
        $weeks = [];
        foreach ($tasks as $task) {
            $week = date('Y-W', strtotime($task['scheduled_date']));
            if (!isset($weeks[$week])) {
                $weeks[$week] = [];
            }
            $weeks[$week][] = $task;
        }
        
        foreach ($weeks as $week => $week_tasks) {
            echo '<div class="timeline-week">';
            echo '<h3>Week ' . $week . '</h3>';
            
            foreach ($week_tasks as $task) {
                $this->render_task_card($task);
            }
            
            echo '</div>';
        }
    }
    
    /**
     * Render individual task card
     */
    private function render_task_card($task) {
        $status_class = 'status-' . $task['_task_status'];
        $priority_class = 'priority-' . $task['_priority'];
        $agent_icon = $this->get_agent_icon($task['_agent_type']);
        
        ?>
        <div class="task-card <?php echo $status_class . ' ' . $priority_class; ?>" 
             data-task-id="<?php echo esc_attr($task['_task_id']); ?>">
            <div class="task-header">
                <span class="agent-icon"><?php echo $agent_icon; ?></span>
                <span class="task-title"><?php echo esc_html($task['post_title']); ?></span>
                <span class="task-priority"><?php echo esc_html($task['_priority']); ?></span>
            </div>
            <div class="task-meta">
                <span class="scheduled-date">
                    📅 <?php echo date('M d, Y H:i', strtotime($task['post_date'])); ?>
                </span>
                <span class="estimated-hours">
                    ⏱️ <?php echo esc_html($task['_estimated_hours']); ?>h
                </span>
            </div>
            <div class="task-actions">
                <?php if ($task['_task_status'] === 'pending'): ?>
                    <button class="button button-primary execute-task" 
                            data-task-id="<?php echo esc_attr($task['_task_id']); ?>">
                        ▶ Execute
                    </button>
                <?php elseif ($task['_task_status'] === 'completed'): ?>
                    <span class="dashicons dashicons-yes-alt"></span> Completed
                <?php endif; ?>
                
                <a href="<?php echo admin_url('post.php?post=' . $task['ID'] . '&action=edit'); ?>" 
                   class="button">View Details</a>
            </div>
        </div>
        <?php
    }
    
    /**
     * Register custom post type for tasks
     */
    public function register_task_post_type() {
        register_post_type('marketing_task', [
            'labels' => [
                'name' => 'Marketing Tasks',
                'singular_name' => 'Marketing Task',
                'add_new' => 'Add New Task',
                'add_new_item' => 'Add New Task',
                'edit_item' => 'Edit Task',
                'view_item' => 'View Task',
            ],
            'public' => false,
            'show_ui' => true,
            'show_in_menu' => 'marketing-ai',
            'supports' => ['title', 'editor', 'custom-fields'],
            'menu_icon' => 'dashicons-calendar-alt',
        ]);
    }
}

// Register cron action for auto-executing tasks
add_action('marketing_ai_execute_task', function($post_id, $task_id) {
    $api_client = new Marketing_AI_API_Client();
    $result = $api_client->execute_task($task_id);
    
    if ($result && !is_wp_error($result)) {
        update_post_meta($post_id, '_task_status', 'completed');
        update_post_meta($post_id, '_completed_at', current_time('mysql'));
        update_post_meta($post_id, '_task_output', $result['output']);
    } else {
        update_post_meta($post_id, '_task_status', 'failed');
        update_post_meta($post_id, '_task_error', $result->get_error_message());
    }
}, 10, 2);
```

### API Endpoints for Task Management

```python
# Add to FastAPI router

@router.post("/api/v1/strategies/{strategy_id}/decompose")
async def decompose_strategy_for_execution(
    strategy_id: str,
    subscription = Depends(validate_api_key)
):
    """Decompose a strategy into executable tasks"""
    
    strategy_data = await get_strategy(strategy_id)
    if not strategy_data:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    decomposer = StrategyDecomposer()
    tasks = await decomposer.decompose_strategy(strategy_id, strategy_data)
    
    # Create WordPress calendar entries
    wordpress_calendar = await create_wordpress_calendar(strategy_id, tasks)
    
    return {
        "strategy_id": strategy_id,
        "total_tasks": len(tasks),
        "tasks": tasks,
        "calendar_entries": wordpress_calendar
    }

@router.post("/api/v1/tasks/{task_id}/execute")
async def execute_task(
    task_id: str,
    subscription = Depends(validate_api_key)
):
    """Execute a specific task"""
    
    task_data = await get_task(task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")
    
    executor = TaskExecutor()
    result = await executor.execute_task(task_data)
    
    return result

@router.post("/api/v1/strategies/{strategy_id}/execute-batch")
async def execute_task_batch(
    strategy_id: str,
    agent_type: str = None,
    subscription = Depends(validate_api_key)
):
    """Execute all pending tasks for a strategy, optionally filtered by agent"""
    
    executor = TaskExecutor()
    results = await executor.execute_task_batch(strategy_id, agent_type)
    
    return {
        "strategy_id": strategy_id,
        "executed_tasks": len(results),
        "results": results
    }

@router.get("/api/v1/strategies/{strategy_id}/tasks")
async def get_strategy_tasks(
    strategy_id: str,
    status: str = None,
    agent_type: str = None,
    subscription = Depends(validate_api_key)
):
    """Get all tasks for a strategy with optional filters"""
    
    query = "SELECT * FROM campaign_tasks WHERE strategy_id = $1"
    params = [strategy_id]
    
    if status:
        query += f" AND status = ${len(params)+1}"
        params.append(status)
    
    if agent_type:
        query += f" AND agent_type = ${len(params)+1}"
        params.append(agent_type)
    
    query += " ORDER BY scheduled_date ASC"
    
    tasks = await db.fetch(query, *params)
    
    return {
        "strategy_id": strategy_id,
        "total_tasks": len(tasks),
        "tasks_by_status": {
            "pending": len([t for t in tasks if t["status"] == "pending"]),
            "in_progress": len([t for t in tasks if t["status"] == "in_progress"]),
            "completed": len([t for t in tasks if t["status"] == "completed"]),
            "failed": len([t for t in tasks if t["status"] == "failed"]),
        },
        "tasks": tasks
    }

@router.get("/api/v1/tasks/{task_id}/output")
async def get_task_output(
    task_id: str,
    subscription = Depends(validate_api_key)
):
    """Get the output of a completed task"""
    
    result = await db.fetchrow(
        "SELECT * FROM task_results WHERE task_id = $1", task_id
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Task output not found")
    
    return result
```

### Database Schema Additions

```sql
-- Campaign Tasks Table
CREATE TABLE campaign_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id VARCHAR(50) UNIQUE NOT NULL,
    strategy_id UUID REFERENCES strategies(id),
    title VARCHAR(500) NOT NULL,
    description TEXT,
    agent_type VARCHAR(50) NOT NULL,
    priority VARCHAR(20) DEFAULT 'medium',
    estimated_duration_hours FLOAT,
    scheduled_date TIMESTAMP,
    deadline TIMESTAMP,
    dependencies JSONB DEFAULT '[]',
    dynamic_prompt TEXT,
    expected_output_format TEXT,
    status VARCHAR(20) DEFAULT 'pending',
    error TEXT,
    wordpress_post_id INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Task Results Table
CREATE TABLE task_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id VARCHAR(50) REFERENCES campaign_tasks(task_id),
    output TEXT,
    completed_at TIMESTAMP DEFAULT NOW(),
    status VARCHAR(20),
    execution_time_seconds FLOAT
);

-- Index for performance
CREATE INDEX idx_campaign_tasks_strategy ON campaign_tasks(strategy_id);
CREATE INDEX idx_campaign_tasks_status ON campaign_tasks(status);
CREATE INDEX idx_campaign_tasks_scheduled ON campaign_tasks(scheduled_date);
CREATE INDEX idx_task_results_task ON task_results(task_id);
```

### Dynamic Prompt Generation Workflow

```
1. User selects strategy → API receives strategy_id
2. StrategyDecomposer analyzes strategy with LLM
   ├── Identifies strategy type (content, SEO, social, comprehensive)
   ├── Extracts phases, content pieces, channels
   └── Determines required agents
3. For each content piece/task:
   ├── Determine agent type (content_writer, seo_specialist, etc.)
   ├── Select prompt template for agent type
   ├── Fill template with:
   │   ├── Business context (company, industry, audience)
   │   ├── Campaign context (name, goals, timeline)
   │   ├── Task-specific details (topic, platform, format)
   │   └── Output format requirements
   └── Store dynamic prompt with task
4. Build dependency graph:
   ├── Content → SEO optimization (blocks)
   ├── Content → Social promotion (blocks)
   └── Content brief → Design (informs)
5. Schedule tasks on calendar:
   ├── Respect dependencies
   ├── Prioritize by phase and importance
   └── Set deadlines based on effort estimates
6. Create WordPress calendar entries:
   ├── Custom post type: marketing_task
   ├── Meta fields for all task data
   └── Cron events for auto-execution
7. Return task list to user for approval
```

### Example: Decomposed Strategy Output

```json
{
  "strategy_id": "strat_abc123",
  "total_tasks": 18,
  "tasks": [
    {
      "task_id": "task_0001",
      "title": "Campaign Setup & Brief Review",
      "description": "Review the complete strategy document...",
      "agent_type": "campaign_manager",
      "priority": "high",
      "estimated_duration_hours": 2,
      "scheduled_date": "2026-04-08T10:00:00",
      "deadline": "2026-04-08T12:00:00",
      "dependencies": [],
      "dynamic_prompt": "You are a campaign coordinator...\n\n## Campaign Context\n- Company: Acme Corp\n...",
      "expected_output_format": "Campaign kickoff checklist...",
      "status": "pending",
      "auto_execute": false,
      "requires_approval": false
    },
    {
      "task_id": "task_0002",
      "title": "Write blog post: How to Measure ROI from Digital Marketing",
      "description": "Execute the following task as part of 'Phase 1: Awareness'...",
      "agent_type": "content_writer",
      "priority": "high",
      "estimated_duration_hours": 4,
      "scheduled_date": "2026-04-08T12:00:00",
      "deadline": "2026-04-08T16:00:00",
      "dependencies": [],
      "dynamic_prompt": "You are a professional content writer...\n\n## Campaign Context\n- Company: Acme Corp\n- Industry: Digital Marketing Services\n...",
      "expected_output_format": "Markdown formatted blog post...",
      "status": "pending",
      "auto_execute": false,
      "requires_approval": true
    },
    {
      "task_id": "task_0003",
      "title": "SEO optimize: How to Measure ROI from Digital Marketing",
      "agent_type": "seo_specialist",
      "priority": "medium",
      "dependencies": [
        {"task_id": "task_0002", "dependency_type": "blocks"}
      ],
      "status": "pending",
      "auto_execute": true,
      "requires_approval": false
    },
    {
      "task_id": "task_0004",
      "title": "Create social content: How to Measure ROI from Digital Marketing",
      "agent_type": "social_media_manager",
      "priority": "medium",
      "dependencies": [
        {"task_id": "task_0002", "dependency_type": "blocks"}
      ],
      "status": "pending",
      "auto_execute": true,
      "requires_approval": false
    }
  ],
  "tasks_by_status": {
    "pending": 18,
    "in_progress": 0,
    "completed": 0,
    "failed": 0
  }
}
```

---

## Content Scheduling & Auto-Update

### WordPress Content Publishing

```php
// includes/class-content-publisher.php
class Marketing_AI_Content_Publisher {
    
    private $api_client;
    
    public function __construct() {
        $this->api_client = new Marketing_AI_API_Client();
    }
    
    /**
     * Schedule content for publication
     */
    public function schedule_content($content_data, $schedule_date) {
        $post_data = [
            'post_title' => $content_data['title'],
            'post_content' => $content_data['content'],
            'post_status' => 'future',
            'post_date' => date('Y-m-d H:i:s', strtotime($schedule_date)),
            'post_type' => 'post',
            'post_author' => get_current_user_id(),
            'meta_input' => [
                '_marketing_ai_content_id' => $content_data['content_id'],
                '_marketing_ai_auto_update' => true,
                '_marketing_ai_update_frequency' => 'monthly'
            ]
        ];
        
        $post_id = wp_insert_post($post_data);
        
        if ($post_id && !is_wp_error($post_id)) {
            // Set categories and tags
            if (!empty($content_data['categories'])) {
                wp_set_post_categories($post_id, $content_data['categories']);
            }
            
            if (!empty($content_data['tags'])) {
                wp_set_post_tags($post_id, $content_data['tags']);
            }
            
            // Set featured image if provided
            if (!empty($content_data['featured_image_url'])) {
                $this->set_featured_image($post_id, $content_data['featured_image_url']);
            }
            
            // Schedule auto-update cron
            $this->schedule_auto_update($post_id, $content_data['content_id']);
            
            return $post_id;
        }
        
        return false;
    }
    
    /**
     * Schedule automatic content update
     */
    public function schedule_auto_update($post_id, $content_id) {
        $next_update = strtotime('+1 month');
        
        wp_schedule_single_event(
            $next_update,
            'marketing_ai_auto_update_content',
            [$post_id, $content_id]
        );
    }
    
    /**
     * Auto-update content handler
     */
    public function auto_update_content($post_id, $content_id) {
        $post = get_post($post_id);
        if (!$post) return;
        
        // Get current content from API
        $api_client = new Marketing_AI_API_Client();
        $updated_content = $api_client->get_content($content_id);
        
        if (!$updated_content || is_wp_error($updated_content)) {
            return;
        }
        
        // Create revision before updating
        wp_save_post_revision($post_id);
        
        // Update post content
        wp_update_post([
            'ID' => $post_id,
            'post_content' => $updated_content['content'],
            'post_title' => $updated_content['title']
        ]);
        
        // Update meta
        update_post_meta($post_id, '_marketing_ai_last_updated', current_time('mysql'));
        update_post_meta($post_id, '_marketing_ai_update_count', 
            (get_post_meta($post_id, '_marketing_ai_update_count', true) ?: 0) + 1);
        
        // Schedule next update
        $this->schedule_auto_update($post_id, $content_id);
        
        // Notify admin
        $this->send_update_notification($post_id, $updated_content);
    }
    
    /**
     * Send update notification email
     */
    private function send_update_notification($post_id, $updated_content) {
        $admin_email = get_option('admin_email');
        $post_url = get_permalink($post_id);
        
        wp_mail(
            $admin_email,
            'MarketingAI: Content Auto-Updated',
            sprintf(
                "Your post has been automatically updated with fresh content.\n\nPost: %s\nURL: %s\n\nChanges:\n%s",
                get_the_title($post_id),
                $post_url,
                $updated_content['update_summary'] ?? 'Content refreshed with latest information'
            )
        );
    }
    
    /**
     * Set featured image from URL
     */
    private function set_featured_image($post_id, $image_url) {
        require_once(ABSPATH . 'wp-admin/includes/image.php');
        require_once(ABSPATH . 'wp-admin/includes/file.php');
        require_once(ABSPATH . 'wp-admin/includes/media.php');
        
        $attachment_id = media_sideload_image($image_url, $post_id, null, 'id');
        
        if (!is_wp_error($attachment_id)) {
            set_post_thumbnail($post_id, $attachment_id);
        }
    }
}

// Register cron action
add_action('marketing_ai_auto_update_content', function($post_id, $content_id) {
    $publisher = new Marketing_AI_Content_Publisher();
    $publisher->auto_update_content($post_id, $content_id);
}, 10, 2);
```

### Scheduled Content Queue (Backend)

```python
# content_scheduler.py
from fastapi import APIRouter
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import aiohttp

router = APIRouter()
scheduler = AsyncIOScheduler()

class ContentScheduler:
    """Manage scheduled content publication"""
    
    def __init__(self):
        self.scheduler = scheduler
        self.scheduler.start()
    
    async def schedule_wordpress_post(self, schedule_data: dict):
        """Schedule content publication to WordPress"""
        
        post_date = schedule_data["publish_date"]
        content_id = schedule_data["content_id"]
        
        # Schedule the job
        self.scheduler.add_job(
            func=self.publish_to_wordpress,
            trigger="date",
            run_date=post_date,
            kwargs={
                "content_id": content_id,
                "wordpress_site": schedule_data["wordpress_site"],
                "wordpress_credentials": schedule_data["wordpress_credentials"],
                "post_data": schedule_data["post_data"]
            },
            id=f"post_{content_id}"
        )
        
        return {"status": "scheduled", "publish_date": post_date}
    
    async def publish_to_wordpress(self, content_id: str, wordpress_site: str,
                                   wordpress_credentials: dict, post_data: dict):
        """Publish content to WordPress via REST API"""
        
        # Get content
        content = await get_content(content_id)
        
        # WordPress REST API endpoint
        wp_api_url = f"{wordpress_site}/wp-json/wp/v2/posts"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                wp_api_url,
                auth=aiohttp.BasicAuth(
                    wordpress_credentials["username"],
                    wordpress_credentials["application_password"]
                ),
                json={
                    "title": content["title"],
                    "content": content["content"],
                    "status": post_data.get("status", "publish"),
                    "categories": post_data.get("categories", []),
                    "tags": post_data.get("tags", []),
                    "meta": {
                        "marketing_ai_content_id": content_id
                    }
                }
            ) as response:
                result = await response.json()
                
                if response.status == 201:
                    # Store WordPress post ID for future updates
                    await update_schedule_status(
                        content_id=content_id,
                        wordpress_post_id=result["id"],
                        status="published"
                    )
                    
                    # Schedule auto-update if enabled
                    if post_data.get("auto_update"):
                        await self.schedule_auto_update(
                            content_id=content_id,
                            wordpress_post_id=result["id"],
                            frequency=post_data.get("update_frequency", "monthly")
                        )
                else:
                    logger.error(f"WordPress publish failed: {result}")

# Initialize scheduler
content_scheduler = ContentScheduler()
```

---

## WordPress Plugin Implementation

### Plugin File Structure

```
wordpress-plugin/
├── marketing-ai.php              # Main plugin file
├── readme.txt                    # WordPress.org readme
├── includes/
│   ├── class-marketing-ai.php    # Core plugin class
│   ├── class-api-client.php      # API communication
│   ├── class-context-extractor.php # Context extraction
│   ├── class-content-publisher.php # Content publishing
│   ├── class-subscription-manager.php # Subscription handling
│   └── class-admin-ui.php        # Admin interface
├── admin/
│   ├── css/
│   │   └── admin.css
│   ├── js/
│   │   └── admin.js
│   └── views/
│       ├── dashboard.php
│       ├── settings.php
│       ├── content-generator.php
│       └── analytics.php
├── blocks/
│   └── content-generator/
│       ├── block.json
│       ├── index.js
│       └── editor.css
└── assets/
    └── icon.svg
```

### Main Plugin File

```php
<?php
/**
 * Plugin Name:       Marketing AI
 * Plugin URI:        https://marketingai.com/wordpress-plugin
 * Description:       AI-powered marketing content generation with strategy, scheduling, and auto-updates.
 * Version:           2.0.0
 * Requires at least: 5.8
 * Requires PHP:      7.4
 * Author:            MarketingAI
 * License:           GPL v2 or later
 */

if (!defined('WPINC')) {
    die;
}

define('MARKETING_AI_VERSION', '2.0.0');
define('MARKETING_AI_PLUGIN_DIR', plugin_dir_path(__FILE__));
define('MARKETING_AI_PLUGIN_URL', plugin_dir_url(__FILE__));
define('MARKETING_AI_API_BASE_URL', 'https://api.marketingai.com/v1');

// Core classes
require_once MARKETING_AI_PLUGIN_DIR . 'includes/class-marketing-ai.php';
require_once MARKETING_AI_PLUGIN_DIR . 'includes/class-api-client.php';
require_once MARKETING_AI_PLUGIN_DIR . 'includes/class-context-extractor.php';
require_once MARKETING_AI_PLUGIN_DIR . 'includes/class-content-publisher.php';
require_once MARKETING_AI_PLUGIN_DIR . 'includes/class-subscription-manager.php';
require_once MARKETING_AI_PLUGIN_DIR . 'includes/class-admin-ui.php';

function run_marketing_ai() {
    $plugin = new Marketing_AI();
    $plugin->run();
}
run_marketing_ai();
```

### Admin Dashboard

```php
// admin/views/dashboard.php
?>
<div class="wrap marketing-ai-dashboard">
    <h1>🎯 Marketing AI Dashboard</h1>
    
    <?php if (!$api_key_valid): ?>
    <div class="notice notice-warning">
        <p>Please enter your API key to activate Marketing AI.</p>
        <a href="<?php echo admin_url('admin.php?page=marketing-ai-settings'); ?>" 
           class="button button-primary">
            Configure API Key
        </a>
    </div>
    <?php else: ?>
    
    <!-- Subscription Status -->
    <div class="card subscription-status">
        <h2>Subscription Status</h2>
        <div class="status-grid">
            <div class="status-item">
                <span class="label">Plan:</span>
                <span class="value"><?php echo esc_html($subscription['plan']); ?></span>
            </div>
            <div class="status-item">
                <span class="label">API Requests:</span>
                <span class="value">
                    <?php echo esc_html($subscription['requests_used']); ?> / 
                    <?php echo esc_html($subscription['requests_limit']); ?>
                </span>
            </div>
            <div class="status-item">
                <span class="label">Reset Date:</span>
                <span class="value">
                    <?php echo date('M d, Y', strtotime($subscription['reset_date'])); ?>
                </span>
            </div>
        </div>
    </div>
    
    <!-- Quick Actions -->
    <div class="card quick-actions">
        <h2>Quick Actions</h2>
        <div class="action-buttons">
            <a href="<?php echo admin_url('admin.php?page=marketing-ai-context'); ?>" 
               class="button button-large">
                📊 Extract Website Context
            </a>
            <a href="<?php echo admin_url('admin.php?page=marketing-ai-strategy'); ?>" 
               class="button button-large">
                📈 Generate Strategy
            </a>
            <a href="<?php echo admin_url('admin.php?page=marketing-ai-content'); ?>" 
               class="button button-large">
                ✍️ Generate Content
            </a>
            <a href="<?php echo admin_url('admin.php?page=marketing-ai-schedule'); ?>" 
               class="button button-large">
                📅 View Scheduled Posts
            </a>
        </div>
    </div>
    
    <!-- Recent Content -->
    <div class="card recent-content">
        <h2>Recently Generated Content</h2>
        <table class="wp-list-table widefat fixed striped">
            <thead>
                <tr>
                    <th>Title</th>
                    <th>Type</th>
                    <th>Generated</th>
                    <th>Status</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                <?php foreach ($recent_content as $content): ?>
                <tr>
                    <td><?php echo esc_html($content['title']); ?></td>
                    <td><?php echo esc_html($content['type']); ?></td>
                    <td><?php echo human_time_diff(strtotime($content['created_at'])); ?> ago</td>
                    <td>
                        <span class="status-badge status-<?php echo esc_attr($content['status']); ?>">
                            <?php echo esc_html($content['status']); ?>
                        </span>
                    </td>
                    <td>
                        <a href="<?php echo admin_url('post.php?post=' . $content['post_id'] . '&action=edit'); ?>">
                            Edit
                        </a>
                    </td>
                </tr>
                <?php endforeach; ?>
            </tbody>
        </table>
    </div>
    
    <?php endif; ?>
</div>
```

---

## Security & Authentication

### API Key Security

```python
# security.py
import hashlib
import secrets
from fastapi import Depends, HTTPException, Header

def hash_api_key(api_key: str) -> str:
    """Hash API key for storage"""
    return hashlib.sha256(api_key.encode()).hexdigest()

def generate_api_key() -> tuple[str, str]:
    """Generate new API key (returns raw key and hash)"""
    raw_key = f"mkai_{secrets.token_urlsafe(32)}"
    key_hash = hash_api_key(raw_key)
    return raw_key, key_hash

async def authenticate_request(x_api_key: str = Header(...)):
    """Authenticate API request"""
    key_hash = hash_api_key(x_api_key)
    
    subscription = await db.fetchrow("""
        SELECT s.*, u.email, u.company_name
        FROM api_keys k
        JOIN subscriptions s ON k.subscription_id = s.id
        JOIN users u ON s.user_id = u.id
        WHERE k.key_hash = $1 
          AND k.is_active = true
          AND s.status = 'active'
          AND s.current_period_end > NOW()
    """, key_hash)
    
    if not subscription:
        raise HTTPException(status_code=401, detail="Invalid or expired API key")
    
    # Check rate limit
    rate_limit_exceeded = await check_rate_limit(subscription["id"])
    if rate_limit_exceeded:
        raise HTTPException(status_code=429, detail="Monthly quota exceeded")
    
    # Record usage
    await record_api_usage(subscription["id"], x_api_key)
    
    return subscription

async def check_rate_limit(subscription_id: str) -> bool:
    """Check if subscription has exceeded rate limit"""
    usage = await db.fetchrow("""
        SELECT COUNT(*) as count
        FROM api_usage
        WHERE subscription_id = $1
          AND request_timestamp >= DATE_TRUNC('month', NOW())
    """, subscription_id)
    
    plan = await db.fetchrow("SELECT api_requests FROM plans WHERE id = (SELECT plan_id FROM subscriptions WHERE id = $1)", subscription_id)
    
    return usage["count"] >= plan["api_requests"]
```

### WordPress Application Passwords

```php
// Use WordPress Application Passwords for API authentication
class Marketing_AI_WordPress_Auth {
    
    /**
     * Create application password for API access
     */
    public function create_api_password($user_id, $name = 'MarketingAI API') {
        if (!function_exists('wp_create_application_password')) {
            return new WP_Error('no_app_passwords', 'Application passwords not available');
        }
        
        $result = wp_create_application_password($user_id, $name);
        
        if (is_wp_error($result)) {
            return $result;
        }
        
        list($password, $item) = $result;
        
        return [
            'password' => $password, // Show only once!
            'item' => $item
        ];
    }
    
    /**
     * Verify API credentials
     */
    public function verify_credentials($username, $application_password) {
        $user = get_user_by('login', $username);
        
        if (!$user) {
            return new WP_Error('invalid_user', 'User not found');
        }
        
        $valid = wp_check_application_passwords($user->ID, null, $application_password);
        
        if (is_wp_error($valid)) {
            return $valid;
        }
        
        return $user;
    }
}
```

---

## Deployment & Scaling

### Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    command: uvicorn main:api --host 0.0.0.0 --port 8000 --workers 4
    volumes:
      - .:/app
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/marketingai
      - REDIS_URL=redis://redis:6379
      - GROQ_API_KEY=${GROQ_API_KEY}
      - STRIPE_SECRET_KEY=${STRIPE_SECRET_KEY}
    depends_on:
      - db
      - redis
    ports:
      - "8000:8000"

  streamlit:
    build: .
    command: streamlit run market_agent.py --server.port=8501 --server.address=0.0.0.0
    environment:
      - API_URL=http://api:8000
    ports:
      - "8501:8501"

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=marketingai
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  worker:
    build: .
    command: celery -A tasks worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/marketingai
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

volumes:
  postgres_data:
```

### Nginx Configuration

```nginx
# nginx.conf
upstream api_backend {
    server api:8000;
}

upstream streamlit_backend {
    server streamlit:8501;
}

server {
    listen 80;
    server_name api.marketingai.com;
    
    location / {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

server {
    listen 80;
    server_name app.marketingai.com;
    
    location / {
        proxy_pass http://streamlit_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## Summary & Recommendations

### Is This Architecture Optimal?

**Yes, with the following rationale:**

1. **FastAPI + Streamlit Separation**: Provides clean API/ UI separation, enabling WordPress plugin to use REST API while users interact with Streamlit UI
2. **Starlette Integration**: Streamlit's native Starlette support (v1.53+) enables custom routes if needed
3. **Subscription via Stripe**: Industry-standard payment processing with webhook-based subscription management
4. **Context Extraction**: Automated website scraping + WordPress API integration minimizes manual setup
5. **Auto-Update via Cron**: WordPress cron + backend scheduler ensures content stays fresh

### Key Implementation Steps

1. **Phase 1**: Set up FastAPI gateway with authentication
2. **Phase 2**: Implement Stripe subscription management
3. **Phase 3**: Build context extraction pipeline
4. **Phase 4**: Develop WordPress plugin with API integration
5. **Phase 5**: Add content scheduling and auto-update features
6. **Phase 6**: Deploy with Docker, set up monitoring

### Monetization Strategy

- **Freemium**: Free WordPress plugin with limited features
- **Premium Tiers**: $49-$499/month based on usage
- **White-Label**: Agency plan for resellers
- **Enterprise**: Custom deployments for large clients

This architecture provides scalability, flexibility, and a clear path to monetization while leveraging Streamlit's latest Starlette integration capabilities.
