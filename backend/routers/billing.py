from fastapi import APIRouter, Depends, HTTPException, Request, Header
from backend.db.session import get_db
from backend.routers.auth import get_current_user
from backend.schemas.billing import SubscribeRequest, SubscribeResponse
from backend.services.billing import usd_to_kes
from backend.services.pesapal_client import PesapalClient
from core.config import get_settings
import uuid

router = APIRouter()

def get_pesapal_client():
    settings = get_settings()
    return PesapalClient(settings.pesapal_consumer_key, settings.pesapal_consumer_secret, sandbox=settings.pesapal_sandbox)

@router.post("/subscribe", response_model=SubscribeResponse)
async def subscribe(req: SubscribeRequest, user = Depends(get_current_user), conn = Depends(get_db)):
    if req.payment_provider == "pesapal":
        client = get_pesapal_client()
        # Mock USD amount based on plan
        amount_usd = 29.99 if req.plan_id == "pro" else 99.99
        amount_kes = await usd_to_kes(amount_usd)
        
        callback_url = "https://marketingai.com/api/v1/billing/webhook/pesapal"
        try:
            ipn_id = await client.register_ipn(callback_url)
            order_id = str(uuid.uuid4())
            result = await client.submit_order(
                order_id=order_id,
                amount=amount_kes,
                currency="KES",
                description=f"MarketingAI {req.plan_id.capitalize()} Plan",
                customer_email=req.email,
                customer_phone=req.phone or "0000000000",
                ipn_id=ipn_id,
                callback_url="https://marketingai.com/dashboard?payment=success"
            )
            return SubscribeResponse(
                payment_provider="pesapal",
                status="redirect",
                redirect_url=result.get("redirect_url"),
                order_tracking_id=result.get("order_tracking_id")
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
            
    elif req.payment_provider == "stripe":
        # Stripe integration mock
        return SubscribeResponse(
            payment_provider="stripe",
            status="requires_payment_method",
            client_secret="pi_mock_secret_key"
        )
    
    raise HTTPException(status_code=400, detail="Invalid payment provider")

@router.get("/webhook/pesapal")
async def pesapal_webhook(OrderTrackingId: str, OrderNotificationType: str, OrderMerchantReference: str, conn = Depends(get_db)):
    client = get_pesapal_client()
    try:
        status = await client.get_transaction_status(OrderTrackingId)
        # Update Subscription Repository based on status
        return {"status": "ok", "payment_status": status.get("payment_status_description")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
