from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class TransactionInput(BaseModel):
    transaction_id: str = Field(..., example="TXN_100001")
    user_id: str = Field(..., example="user_1")
    timestamp: datetime = Field(default_factory=datetime.now)
    amount: float = Field(..., gt=0)
    device_type: str = Field(..., example="mobile_app")
    location: str = Field(..., example="VN")
    payment_method: str = Field(..., example="credit_card")
    is_guest_checkout: int = Field(..., ge=0, le=1)
    time_since_last_login_hours: float = Field(..., ge=0)

class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    status: str = "success"
    model_version: str = "1.0.0"

    model_config = {'protected_namespaces': ()}
