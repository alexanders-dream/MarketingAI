"""
Image Generation using Google Gemini for Marketing AI v3
"""
import logging
import requests
import base64
from typing import Optional, Dict, Any, List
from io import BytesIO
from PIL import Image
import google.generativeai as genai

from config import AppConfig
from prompts import MarketingImagePromptGenerator

logger = logging.getLogger(__name__)


class GeminiImageGenerator:
    """Generates images using Google Gemini Vision API"""

    def __init__(self, app_config: AppConfig = None):
        self.app_config = app_config or AppConfig()
        self.api_key = self.app_config.get_api_key("GEMINI")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        else:
            self.model = None

    def _configure_api(self):
        if self.api_key:
            genai.configure(api_key=self.api_key)

    def get_image_settings(self):
        return {
            "aspect_ratios": ["1.91:1", "1:1", "4:5", "9:16"],
            "styles": ["Photorealistic", "Illustration", "3D Render", "Abstract"]
        }
    def generate_image(self, prompt: str, aspect_ratio: str = "1:1",
                      style: str = "natural") -> Optional[bytes]:
        """
        Generate an image using Gemini

        Args:
            prompt: Text description of the image to generate
            aspect_ratio: Image aspect ratio (1:1, 16:9, 4:3, etc.)
            style: Image style (natural, vivid, etc.)

        Returns:
            Image bytes or None if generation failed
        """
        if not self.model:
            logger.error("Gemini model not initialized - check API key")
            return None

        try:
            # Enhanced prompt for better marketing images
            enhanced_prompt = self._enhance_prompt(prompt, style)

            # Generate image
            response = self.model.generate_content(
                f"Generate an image: {enhanced_prompt}",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                )
            )

            # Extract image from response
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            # Decode base64 image data
                            image_data = base64.b64decode(part.inline_data.data)
                            return self._process_image(image_data, aspect_ratio)

            logger.warning("No image data found in Gemini response")
            return None

        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            return None

    def _enhance_prompt(self, base_prompt: str, style: str) -> str:
        """
        Enhance the prompt for better image generation

        Args:
            base_prompt: Original user prompt
            style: Desired style

        Returns:
            Enhanced prompt
        """
        style_modifiers = {
            "natural": "photorealistic, natural lighting, professional photography",
            "vivid": "bright colors, vibrant, eye-catching, dynamic composition",
            "minimal": "clean, minimalistic, simple, elegant, white background",
            "corporate": "professional, corporate, business-appropriate, clean design",
            "creative": "artistic, creative, unique, visually striking",
            "social": "optimized for social media, engaging, shareable, trendy"
        }

        style_modifier = style_modifiers.get(style.lower(), style_modifiers["natural"])

        enhanced = f"""
        Create a high-quality marketing image with the following specifications:
        - Style: {style_modifier}
        - Subject: {base_prompt}
        - Quality: Professional, high-resolution, suitable for marketing materials
        - Composition: Well-balanced, visually appealing, clear focal point
        - Colors: Brand-appropriate, harmonious color scheme
        - Details: Sharp, clear, professional finish
        """

        return enhanced.strip()

    def _process_image(self, image_data: bytes, aspect_ratio: str) -> bytes:
        """
        Process and resize image to desired aspect ratio

        Args:
            image_data: Raw image bytes
            aspect_ratio: Desired aspect ratio

        Returns:
            Processed image bytes
        """
        try:
            # Open image with PIL
            image = Image.open(BytesIO(image_data))

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Parse aspect ratio
            if ':' in aspect_ratio:
                width_ratio, height_ratio = map(int, aspect_ratio.split(':'))
            else:
                # Default to square
                width_ratio, height_ratio = 1, 1

            # Calculate new dimensions maintaining aspect ratio
            current_width, current_height = image.size
            current_ratio = current_width / current_height
            target_ratio = width_ratio / height_ratio

            if current_ratio > target_ratio:
                # Image is too wide, crop width
                new_width = int(current_height * target_ratio)
                offset = (current_width - new_width) // 2
                image = image.crop((offset, 0, offset + new_width, current_height))
            elif current_ratio < target_ratio:
                # Image is too tall, crop height
                new_height = int(current_width / target_ratio)
                offset = (current_height - new_height) // 2
                image = image.crop((0, offset, current_width, offset + new_height))

            # Resize to standard marketing sizes
            if aspect_ratio == "1:1":
                image = image.resize((1080, 1080), Image.Resampling.LANCZOS)
            elif aspect_ratio == "16:9":
                image = image.resize((1920, 1080), Image.Resampling.LANCZOS)
            elif aspect_ratio == "4:3":
                image = image.resize((1600, 1200), Image.Resampling.LANCZOS)
            elif aspect_ratio == "9:16":
                image = image.resize((1080, 1920), Image.Resampling.LANCZOS)

            # Save to bytes
            output_buffer = BytesIO()
            image.save(output_buffer, format='JPEG', quality=95, optimize=True)
            return output_buffer.getvalue()

        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            return image_data  # Return original if processing fails

    def generate_image_variations(self, base_prompt: str, count: int = 3,
                                aspect_ratio: str = "1:1") -> List[bytes]:
        """
        Generate multiple variations of an image

        Args:
            base_prompt: Base image description
            count: Number of variations to generate
            aspect_ratio: Image aspect ratio

        Returns:
            List of image bytes
        """
        variations = []
        styles = ["natural", "vivid", "corporate", "creative"]

        for i in range(min(count, len(styles))):
            style = styles[i % len(styles)]
            variation_prompt = f"{base_prompt} (style variation {i+1}: {style})"

            image_bytes = self.generate_image(variation_prompt, aspect_ratio, style)
            if image_bytes:
                variations.append(image_bytes)

        return variations

    def generate_marketing_bundle(self, content_description: str,
                                content_type: str = "social") -> Dict[str, bytes]:
        """
        Generate a complete marketing image bundle

        Args:
            content_description: Description of the content/marketing piece
            content_type: Type of content (social, blog, email, etc.)

        Returns:
            Dictionary of image types and their bytes
        """
        bundle = {}

        # Define image types based on content type
        image_specs = {
            "social": [
                ("main_image", "1:1", "vivid"),
                ("story_image", "9:16", "natural"),
                ("carousel_1", "1:1", "corporate"),
                ("carousel_2", "1:1", "creative")
            ],
            "blog": [
                ("featured_image", "16:9", "natural"),
                ("thumbnail", "1:1", "vivid"),
                ("social_share", "1:1", "corporate")
            ],
            "email": [
                ("header_image", "16:9", "corporate"),
                ("product_image", "1:1", "natural")
            ]
        }

        specs = image_specs.get(content_type, image_specs["social"])

        for image_name, aspect_ratio, style in specs:
            prompt = f"Marketing image for: {content_description}"
            image_bytes = self.generate_image(prompt, aspect_ratio, style)
            if image_bytes:
                bundle[image_name] = image_bytes

        return bundle

    def enhance_prompt_with_brand(self, base_prompt: str, brand_description: str) -> str:
        """
        Enhance image prompt with brand context

        Args:
            base_prompt: Original image prompt
            brand_description: Brand description for context

        Returns:
            Enhanced prompt incorporating brand elements
        """
        enhancement = f"""
        Create an image that incorporates these brand elements: {brand_description}

        Original concept: {base_prompt}

        Ensure the image reflects the brand's personality, values, and visual identity.
        """

        return enhancement.strip()
