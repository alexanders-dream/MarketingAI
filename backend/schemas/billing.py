from pydantic import BaseModel
from typing import Optional, Any, Dict

class SubscribeRequest(BaseModel):
    payment_provider: str = 'stripe' # 'stripe' or 'pesapal'
    plan_id: str
    email: str
    phone: Optional[str] = None

class SubscribeResponse(BaseModel):
    payment_provider: str
    status: str
    redirect_url: Optional[str] = None
    client_secret: Optional[str] = None
    order_tracking_id: Optional[str] = None
