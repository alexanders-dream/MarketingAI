import json
import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def parse_json_output(output: str) -> Dict[str, Any]:
    """
    Safely extract and parse JSON from an LLM output string.
    Handles markdown code blocks and malformed strings.
    """
    try:
        # Direct parsing if it's already clean JSON
        return json.loads(output)
    except json.JSONDecodeError:
        pass

    try:
        # Extract from code block
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", output, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except Exception as e:
        logger.warning(f"Failed to parse JSON from markdown code block: {str(e)}")

    try:
        # Attempt to find JSON-like object bounds
        start_idx = output.find('{')
        end_idx = output.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return json.loads(output[start_idx : end_idx + 1])
    except Exception as e:
        logger.error(f"Failed to extract JSON object: {str(e)}")

    # Fallback if nothing else works
    return {"raw_output": output, "error": "Failed to parse json"}
