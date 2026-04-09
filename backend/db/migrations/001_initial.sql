-- 001_initial.sql

-- Enable pgcrypto for gen_random_uuid() if not on PG13+ (though mostly standard now)
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Core Tables
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    api_key_hash VARCHAR(255) UNIQUE NOT NULL,
    plan VARCHAR(50) DEFAULT 'free',
    region VARCHAR(10) DEFAULT 'global',
    preferred_currency VARCHAR(5) DEFAULT 'USD',
    phone VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Subscriptions 
CREATE TABLE IF NOT EXISTS subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    payment_provider VARCHAR(20) DEFAULT 'stripe',
    provider_subscription_id VARCHAR(255),
    pesapal_order_tracking_id VARCHAR(255),
    pesapal_merchant_reference VARCHAR(255),
    status VARCHAR(50) NOT NULL,
    current_period_start TIMESTAMP NOT NULL,
    current_period_end TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Usage tracking for rate limiting/quotas
CREATE TABLE IF NOT EXISTS usage_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL, -- 'api_call', 'social_post', 'content_generation'
    event_count INTEGER DEFAULT 1,
    period_month VARCHAR(7) NOT NULL, -- YYYY-MM
    created_at TIMESTAMP DEFAULT NOW()
);

-- Context
CREATE TABLE IF NOT EXISTS website_contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    website_url VARCHAR(255) NOT NULL,
    company_name VARCHAR(255),
    industry VARCHAR(100),
    target_audience TEXT,
    brand_voice VARCHAR(100) DEFAULT 'professional',
    keywords JSONB DEFAULT '[]',
    competitors JSONB DEFAULT '[]',
    context_data JSONB DEFAULT '{}', -- full scraped data, product images, etc.
    webhook_url VARCHAR(255),
    webhook_secret VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Strategies
CREATE TABLE IF NOT EXISTS strategies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    context_id UUID REFERENCES website_contexts(id) ON DELETE SET NULL,
    title VARCHAR(255) NOT NULL,
    goal TEXT NOT NULL,
    kpis JSONB DEFAULT '[]',
    timeframe VARCHAR(50) DEFAULT '90_days',
    channels JSONB DEFAULT '[]',
    strategy_data JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'draft', -- draft, executing, completed
    created_at TIMESTAMP DEFAULT NOW()
);

-- Generated Content
CREATE TABLE IF NOT EXISTS generated_content (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255),
    content TEXT NOT NULL,
    content_type VARCHAR(50) NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Social accounts and publishes structure (simplified for storing Upload-Post mappings)
CREATE TABLE IF NOT EXISTS social_publishes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    content_id UUID REFERENCES generated_content(id) ON DELETE SET NULL,
    platforms JSONB NOT NULL, -- ['instagram', 'linkedin']
    upload_post_id VARCHAR(255), -- ID returned by Upload-Post API
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Strategy Tasks
CREATE TABLE IF NOT EXISTS strategy_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id UUID REFERENCES strategies(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    title VARCHAR(500) NOT NULL,
    description TEXT,
    task_type VARCHAR(50) NOT NULL,
    assigned_agent VARCHAR(50) NOT NULL,
    
    scheduled_date DATE NOT NULL,
    scheduled_time TIME,
    deadline DATE,
    priority VARCHAR(20) DEFAULT 'medium',
    
    status VARCHAR(30) DEFAULT 'pending',
    celery_task_id VARCHAR(255),
    
    generated_prompt TEXT,
    prompt_context JSONB DEFAULT '{}',
    
    output_content_id UUID REFERENCES generated_content(id) ON DELETE SET NULL,
    output_social_publish_id UUID REFERENCES social_publishes(id) ON DELETE SET NULL,
    execution_log TEXT,
    error_message TEXT,
    
    depends_on UUID[] DEFAULT '{}',
    
    target_platform VARCHAR(50),
    target_content_type VARCHAR(50),
    
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- Indexes
CREATE INDEX idx_tasks_user_date ON strategy_tasks(user_id, scheduled_date);
CREATE INDEX idx_tasks_status ON strategy_tasks(status, scheduled_date);
CREATE INDEX idx_tasks_strategy ON strategy_tasks(strategy_id);
CREATE INDEX idx_usage_user_period ON usage_events(user_id, period_month);

-- ROW LEVEL SECURITY (RLS)

-- Application role used by FastAPI (NOT a superuser). 
-- In a real setup, this would be the DB user FastAPI connects as.
-- For local dev with postgres:postgres, RLS still applies if we SET role or enforce it.
-- We must explicitly ENABLE ROW LEVEL SECURITY on tenant tables.
ALTER TABLE subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE usage_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE website_contexts ENABLE ROW LEVEL SECURITY;
ALTER TABLE strategies ENABLE ROW LEVEL SECURITY;
ALTER TABLE generated_content ENABLE ROW LEVEL SECURITY;
ALTER TABLE social_publishes ENABLE ROW LEVEL SECURITY;
ALTER TABLE strategy_tasks ENABLE ROW LEVEL SECURITY;

-- Note: users table might not need RLS if queries are carefully written 
-- or we can enable it there as well. We'll leave it off for `users` for login/registration ease 
-- since we look up by email/api key.

-- Create Policies
-- Note: 'app.current_user_id' is set via `SET LOCAL app.current_user_id = 'xxxx'` in the session

CREATE POLICY tenant_isolation_subscriptions ON subscriptions
    FOR ALL
    USING (user_id = current_setting('app.current_user_id', true)::uuid);

CREATE POLICY tenant_isolation_usage ON usage_events
    FOR ALL
    USING (user_id = current_setting('app.current_user_id', true)::uuid);

CREATE POLICY tenant_isolation_contexts ON website_contexts
    FOR ALL
    USING (user_id = current_setting('app.current_user_id', true)::uuid);

CREATE POLICY tenant_isolation_strategies ON strategies
    FOR ALL
    USING (user_id = current_setting('app.current_user_id', true)::uuid);

CREATE POLICY tenant_isolation_content ON generated_content
    FOR ALL
    USING (user_id = current_setting('app.current_user_id', true)::uuid);

CREATE POLICY tenant_isolation_social ON social_publishes
    FOR ALL
    USING (user_id = current_setting('app.current_user_id', true)::uuid);

CREATE POLICY tenant_isolation_tasks ON strategy_tasks
    FOR ALL
    USING (user_id = current_setting('app.current_user_id', true)::uuid);

-- If the connection is a superuser (e.g. postgres user in dev), RLS is bypassed. 
-- To test RLS with postgres superuser, we need to enforce it by either using a separate role 
-- or testing carefully. The code will set `app.current_user_id`, which is a good practice.
