"""
Parsers and text processing utilities for Marketing AI v3
"""
import json
import re
from typing import Dict, Any, Union

def extract_json_from_text(text: Any) -> Dict[str, Any]:
    """
    Extract JSON object from LLM response text using robust parsing with fallbacks.
    Handles cases where JSON is embedded in other text or has formatting issues.
    Also maps field names to ensure consistency.
    """
    # Handle various input types gracefully
    if not text:
        return {}
    
    # Convert to string if it's not already
    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return {}
    
    # Clean the text first - remove markdown code blocks and extra whitespace
    cleaned_text = text.strip()
    
    # Remove markdown code blocks if present
    if cleaned_text.startswith('```'):
        lines = cleaned_text.split('\n')
        if len(lines) > 2 and lines[0].startswith('```') and lines[-1].startswith('```'):
            cleaned_text = '\n'.join(lines[1:-1]).strip()
    
    # Try direct parsing first with comprehensive error handling
    try:
        parsed_data = json.loads(cleaned_text)
        return map_field_names(parsed_data)
    except json.JSONDecodeError:
        # Log the error but continue with fallback methods
        pass
    
    # Enhanced JSON extraction with multiple fallback strategies
    extraction_attempts = [
        _extract_json_using_brackets,
        _extract_json_using_regex,
        _extract_json_using_llm_fallback
    ]
    
    for extraction_method in extraction_attempts:
        try:
            result = extraction_method(cleaned_text)
            if result:
                return map_field_names(result)
        except (json.JSONDecodeError, ValueError):
            continue
    
    # Final fallback: try to extract any JSON-like structure
    try:
        # Look for the first { and last } in the text with better error handling
        start_idx = cleaned_text.find('{')
        end_idx = cleaned_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = cleaned_text[start_idx:end_idx+1]
            # Try to fix common JSON issues
            json_str = _fix_common_json_issues(json_str)
            parsed_data = json.loads(json_str)
            return map_field_names(parsed_data)
    except (json.JSONDecodeError, ValueError):
        pass
    
    return {}


def _extract_json_using_brackets(text: str) -> Dict[str, Any]:
    """Extract JSON using bracket matching with validation"""
    stack = []
    start_idx = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start_idx = i
            stack.append(char)
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start_idx != -1:
                    # Found complete JSON object
                    json_str = text[start_idx:i+1]
                    return json.loads(json_str)
    
    raise ValueError("No complete JSON object found")


def _extract_json_using_regex(text: str) -> Dict[str, Any]:
    """Extract JSON using regex patterns with validation"""
    # More robust regex pattern for JSON objects
    json_pattern = r'\{[\s\S]*?\}(?=\s*\{|$)'  # Match complete JSON objects
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            # Clean and validate the match
            match = match.strip()
            if match.startswith('{') and match.endswith('}'):
                # Try to fix common issues before parsing
                match = _fix_common_json_issues(match)
                return json.loads(match)
        except (json.JSONDecodeError, ValueError):
            continue
    
    raise ValueError("No valid JSON found via regex")


def _extract_json_using_llm_fallback(text: str) -> Dict[str, Any]:
    """
    Fallback method that uses simple string processing when JSON parsing fails.
    This is a last resort for badly formatted but still parseable content.
    """
    # Look for key-value patterns in the text
    lines = text.split('\n')
    result = {}
    
    for line in lines:
        line = line.strip()
        if ':' in line and not line.startswith('#'):
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip().strip('"\'')
                value = parts[1].strip().strip('"\'')
                if key and value:
                    result[key] = value
    
    if result:
        return result
    else:
        raise ValueError("No parseable key-value pairs found in text")


def _fix_common_json_issues(json_str: str) -> str:
    """Fix common JSON formatting issues"""
    if not json_str:
        return json_str
    
    # Fix missing quotes around keys
    json_str = re.sub(r'(\w+)(\s*:\s*)', r'"\1"\2', json_str)
    
    # Fix single quotes to double quotes
    json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
    
    # Fix trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Remove comments (JSON doesn't support comments)
    json_str = re.sub(r'//.*', '', json_str)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    
    return json_str


def map_field_names(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map various field name variations to the expected field names.
    """
    if not isinstance(data, dict):
        return {}

    # Define field name mappings (common variations to expected names)
    field_mappings = {
        # Company name variations
        "company": "company_name",
        "company_name": "company_name",
        "organization": "company_name",
        "business_name": "company_name",

        # Industry variations
        "industry": "industry",
        "sector": "industry",
        "field": "industry",

        # Target audience variations
        "target_audience": "target_audience",
        "audience": "target_audience",
        "customers": "target_audience",
        "target_market": "target_audience",

        # Products/Services variations
        "products_services": "products_services",
        "products": "products_services",
        "services": "products_services",
        "offerings": "products_services",

        # Brand description variations
        "brand_description": "brand_description",
        "brand": "brand_description",
        "description": "brand_description",
        "about": "brand_description",
        "mission": "brand_description",

        # Marketing goals variations
        "marketing_goals": "marketing_goals",
        "goals": "marketing_goals",
        "objectives": "marketing_goals",
        "marketing_objectives": "marketing_goals",

        # Existing content variations
        "existing_content": "existing_content",
        "content": "existing_content",
        "current_content": "existing_content",
        "marketing_content": "existing_content",

        # Keywords variations
        "keywords": "keywords",
        "keyword": "keywords",
        "seo_keywords": "keywords",

        # Market opportunities variations
        "market_opportunities": "market_opportunities",
        "opportunities": "market_opportunities",
        "growth_areas": "market_opportunities",

        # Competitive advantages variations
        "competitive_advantages": "competitive_advantages",
        "advantages": "competitive_advantages",
        "competitors": "competitive_advantages",
        "differentiators": "competitive_advantages",

        # Customer pain points variations
        "customer_pain_points": "customer_pain_points",
        "pain_points": "customer_pain_points",
        "problems": "customer_pain_points",
        "challenges": "customer_pain_points",
    }

    # Create mapped result
    mapped_result = {}

    for key, value in data.items():
        # Map the key if it exists in mappings, otherwise keep original
        mapped_key = field_mappings.get(key.lower(), key)
        mapped_result[mapped_key] = value

    return mapped_result
