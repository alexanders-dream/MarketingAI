import hashlib
import hmac
import secrets
import string
from core.config import get_settings

def generate_api_key(prefix: str = "mkai_") -> str:
    """Generate a secure random API key."""
    alphabet = string.ascii_letters + string.digits
    secure_string = ''.join(secrets.choice(alphabet) for _ in range(32))
    return f"{prefix}{secure_string}"

def hash_api_key(raw_key: str) -> str:
    """Hash the raw API key for storage."""
    return hashlib.sha256(raw_key.encode()).hexdigest()

def verify_webhook_signature(payload: bytes, secret: str, received_sig: str) -> bool:
    """Verify HMAC signature for incoming webhooks (e.g., Stripe, Pesapal)."""
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, received_sig)

def sign_webhook_payload(payload: bytes, secret: str) -> str:
    """Sign payload for outgoing webhooks (e.g., calling WP Plugin)."""
    return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

