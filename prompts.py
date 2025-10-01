from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class MarketingImagePromptGenerator:
    """
    Generates detailed and platform-optimized image prompts for marketing content.
    """
    business_name: str
    business_description: str
    target_audience: str
    branding_guidelines: str
    content_purpose: str
    content_format: str
    key_message: str
    platform: str
    image_style: str
    image_mood: str
    image_composition: str
    image_color_palette: str
    image_negative_space: bool
    image_text_overlay: str
    image_call_to_action: str
    image_aspect_ratio: str

    PLATFORM_OPTIMIZATIONS: Dict[str, str] = field(default_factory=lambda: {
        "LinkedIn": "Professional, clean, and informative. Use high-quality stock photos or custom graphics. Avoid overly casual or cluttered images.",
        "Twitter": "Eye-catching and concise. Use bold colors and clear imagery. Memes and GIFs can be effective if relevant to the brand.",
        "Facebook": "Engaging and shareable. Use images that evoke emotion or tell a story. User-generated content can be very effective.",
        "Instagram": "Visually stunning and high-resolution. Use a consistent aesthetic and color palette. Lifestyle and behind-the-scenes shots work well.",
        "Pinterest": "Inspirational and aspirational. Use vertical images with text overlays. Infographics and tutorials are popular.",
        "Blog": "Relevant and high-quality. Use images that break up text and illustrate concepts. Custom graphics and charts are valuable.",
    })

    def generate_prompt(self) -> str:
        """
        Generates a comprehensive image prompt based on the provided marketing context.
        """
        platform_optimization = self.PLATFORM_OPTIMIZATIONS.get(self.platform, "General best practices for web images.")

        prompt = f"""
        **Image Prompt for {self.business_name}**

        **1. Core Concept:**
        - **Business:** {self.business_name} ({self.business_description})
        - **Target Audience:** {self.target_audience}
        - **Content Purpose:** {self.content_purpose}
        - **Key Message:** {self.key_message}
        - **Platform:** {self.platform} ({platform_optimization})

        **2. Image Style and Mood:**
        - **Overall Style:** {self.image_style}
        - **Mood and Tone:** {self.image_mood}
        - **Color Palette:** {self.image_color_palette}

        **3. Composition and Framing:**
        - **Composition:** {self.image_composition}
        - **Aspect Ratio:** {self.image_aspect_ratio}
        - **Negative Space:** {'Yes, include ample negative space for text or graphics.' if self.image_negative_space else 'No, a full-frame image is preferred.'}

        **4. Subject and Elements:**
        - **Primary Subject:** [Describe the main subject of the image, e.g., a person, a product, an abstract concept]
        - **Secondary Elements:** [Describe any supporting elements, e.g., background, props, text]
        - **Lighting:** [Describe the desired lighting, e.g., bright and airy, dramatic and moody, natural light]

        **5. Text and Branding:**
        - **Text Overlay:** {self.image_text_overlay}
        - **Call to Action (CTA):** {self.image_call_to_action}
        - **Branding Guidelines:** {self.branding_guidelines}

        **6. Negative Keywords (to avoid):**
        - [List any elements, styles, or concepts to exclude from the image]

        **Final Prompt (for image generation model):**
        A visually compelling {self.image_style} image for {self.business_name} targeting {self.target_audience}.
        The image should convey a feeling of {self.image_mood} and adhere to a {self.image_color_palette} color palette.
        It's for {self.platform}, so it needs to be {platform_optimization.lower()}.
        The composition is {self.image_composition} with a {self.image_aspect_ratio} aspect ratio.
        The main subject is [Detailed description of the primary subject].
        [Include details about secondary elements, lighting, and any text or CTA].
        Ensure the image follows these branding guidelines: {self.branding_guidelines}.
        Avoid [List of negative keywords].
        """
        return prompt.strip()

class Prompts:
    """Generates prompts for various marketing tasks."""

    def get_document_extraction_prompt(self, text: str) -> str:
        """Prompt to extract business context from a document."""
        return f"""
        Analyze the following document and extract the key business context information.
        Format the output as a JSON object with EXACTLY these keys (use empty string "" if information is not found):
        - "company_name": The name of the company or organization
        - "industry": The industry or sector the company operates in
        - "target_audience": Description of the target customers or audience
        - "products_services": Description of products and/or services offered
        - "brand_description": Description of the brand, mission, values, and unique selling points
        - "marketing_goals": The marketing objectives and goals
        - "existing_content": Any existing marketing content, campaigns, or channels mentioned
        - "keywords": Relevant keywords for marketing purposes
        - "market_opportunities": Potential market opportunities or growth areas
        - "competitive_advantages": Competitive advantages and differentiators
        - "customer_pain_points": Customer problems or pain points that the business solves
        - "suggested_topics": 5-7 suggested marketing topics based on the business context, products, and target audience

        IMPORTANT: Return ONLY valid JSON with these exact field names. Do not include any explanatory text outside the JSON.

        Document content:
        ---
        {text}
        ---
        """

    def get_website_extraction_prompt(self, content: str, platform: str) -> str:
        """Prompt to extract business context from a website."""
        return f"""
        Analyze the following website content from {platform} and extract the key business context information.
        Format the output as a JSON object with the following keys:
        - "company_name"
        - "industry"
        - "target_audience"
        - "products_services"
        - "brand_description"
        - "suggested_topics": 5-7 suggested marketing topics based on the business context, products, and target audience

        Website content:
        ---
        {content}
        ---
        """

    def get_business_context_suggestion_prompt(self, current_context: Dict[str, Any]) -> str:
        """Prompt to generate AI suggestions for business context."""
        return f"""
        Given the following business context, analyze it and provide suggestions for improvement.
        Format the output as a JSON object with the same keys as the input, but with improved values.
        Include a "reasoning" key to explain your suggestions.

        Current context:
        ---
        {json.dumps(current_context, indent=2)}
        ---
        """
