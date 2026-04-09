import httpx
from datetime import datetime, timedelta

class PesapalClient:
    """Pesapal API 3.0 client for African payments"""
    
    SANDBOX_URL = "https://cybqa.pesapal.com/pesapalv3"
    LIVE_URL = "https://pay.pesapal.com/v3"
    
    def __init__(self, consumer_key: str, consumer_secret: str, sandbox=False):
        self.base_url = self.SANDBOX_URL if sandbox else self.LIVE_URL
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self._token = None
        self._token_expiry = None
    
    async def _get_token(self) -> str:
        """Get or refresh OAuth bearer token (5-minute TTL)"""
        if self._token and self._token_expiry > datetime.utcnow():
            return self._token
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/api/Auth/RequestToken",
                json={
                    "consumer_key": self.consumer_key,
                    "consumer_secret": self.consumer_secret
                }
            )
            data = resp.json()
            self._token = data["token"]
            self._token_expiry = datetime.utcnow() + timedelta(minutes=4)
            return self._token
    
    async def register_ipn(self, callback_url: str) -> str:
        """Register IPN endpoint - returns ipn_id for future transactions"""
        token = await self._get_token()
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/api/URLSetup/RegisterIPN",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "url": callback_url,
                    "ipn_notification_type": "POST"
                }
            )
            return resp.json()["ipn_id"]
    
    async def submit_order(
        self, order_id: str, amount: float, currency: str,
        description: str, customer_email: str, customer_phone: str,
        ipn_id: str, callback_url: str,
        account_number: str = None  # For recurring payments
    ) -> dict:
        """Submit payment order - returns redirect URL for payment page"""
        token = await self._get_token()
        
        payload = {
            "id": order_id,
            "currency": currency,  # KES, UGX, TZS, USD
            "amount": amount,
            "description": description,
            "callback_url": callback_url,
            "notification_id": ipn_id,
            "billing_address": {
                "email_address": customer_email,
                "phone_number": customer_phone,
            }
        }
        
        if account_number:
            payload["account_number"] = account_number
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/api/Transactions/SubmitOrderRequest",
                headers={"Authorization": f"Bearer {token}"},
                json=payload
            )
            return resp.json()
    
    async def get_transaction_status(self, order_tracking_id: str) -> dict:
        """Check payment status: COMPLETED, FAILED, PENDING"""
        token = await self._get_token()
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/api/Transactions/GetTransactionStatus",
                headers={"Authorization": f"Bearer {token}"},
                params={"orderTrackingId": order_tracking_id}
            )
            return resp.json()
