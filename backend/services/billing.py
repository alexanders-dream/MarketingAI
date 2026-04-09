import httpx
from core.config import get_settings

_FALLBACK_KES_PER_USD = 128.0

async def usd_to_kes(usd_amount: float) -> float:
    """Fetch the current USD to KES rate and return the converted amount."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                "https://api.exchangerate.host/convert",
                params={"from": "USD", "to": "KES", "amount": usd_amount}
            )
            data = resp.json()
            return round(data["result"], 2)
    except Exception:
        return round(usd_amount * _FALLBACK_KES_PER_USD, 2)
