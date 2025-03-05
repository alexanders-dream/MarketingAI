# marketing_functions.py


def generate_strategy(llm, brand_description, target_audience):
    """Generate a marketing strategy."""
    prompt = f"""
    You are a seasoned marketing strategist with extensive experience in boosting brand visibility and customer engagement. Based on the details provided below, develop a comprehensive, multi-channel marketing strategy that includes:
    - Clear objectives and measurable KPIs
    - Tactical recommendations across digital, social, and content marketing channels
    - Messaging guidelines tailored to the target audience
    - Consideration of current market trends and competitive analysis
    
    Brand Description: {brand_description}
    Target Audience: {target_audience}
    
    Please present the strategy in a structured, step-by-step format.
    """
    return llm.invoke([{"role": "user", "content": prompt}]).content

def generate_campaign(llm, product_service, goals):
    """Generate digital marketing campaign ideas."""
    prompt = f"""
    As an experienced digital marketing specialist, design innovative campaign ideas for the following product/service. Your response should include:
    - A list of creative campaign concepts with brief descriptions
    - Recommended digital channels (e.g., social media, email, PPC, SEO)
    - Messaging strategies that align with the stated goals
    - Key performance indicators to measure campaign success
    - Any emerging trends or unique tactics that could be leveraged
    
    Product/Service: {product_service}
    Goals: {goals}
    
    Provide your ideas in a clear and organized format.
    """
    return llm.invoke([{"role": "user", "content": prompt}]).content

def generate_content(llm, platform, topic, tone, target_audience):
    """Generate social media content."""
    prompt = f"""
    You are a creative content strategist specialized in {platform}. Develop an engaging and original post centered on the topic "{topic}" using a {tone} tone. Ensure that your post:
    - Captures attention quickly and holds reader interest
    - Is concise and tailored to the {target_audience} audience
    - Includes relevant hashtags, a call-to-action, or other engagement elements if applicable
    - Reflects the brand’s voice and aligns with current content trends
    
    Please produce the content in a format that suits {platform}.
    Post:
    """
    return llm.invoke([{"role": "user", "content": prompt}]).content

def optimize_seo(llm, content, keywords):
    """Provide SEO optimization suggestions."""
    prompt = f"""
    You are an expert SEO consultant. Analyze the content provided below and propose actionable recommendations to enhance its search engine performance. Focus on:
    - Integrating and positioning the following keywords effectively: {keywords}
    - Improving on-page elements such as title tags, meta descriptions, and header tags
    - Optimizing keyword density and readability while preserving natural language
    - Suggestions for internal linking, backlink opportunities, and additional off-page SEO tactics
    
    Content: {content}
    
    Please list your recommendations in a clear, step-by-step manner.
    """
    return llm.invoke([{"role": "user", "content": prompt}]).content