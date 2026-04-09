def _sanitize_for_template(value) -> str:
    """Escape curly braces in user-supplied content before str.format() interpolation.

    Without this guard, a WordPress site whose `brand_voice` field contains a string
    like '{strategy_goal}' would either raise KeyError or allow adversarial content
    to override prompt variables (prompt injection via crafted site content).
    """
    if not isinstance(value, str):
        value = str(value)
    return value.replace("{", "{{").replace("}", "}}")
