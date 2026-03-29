from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.app.service import predict_churn

router = APIRouter()


class CustomerRequest(BaseModel):
    SeniorCitizen: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    FamilyStatus: str
    Churn: Optional[str] = None


@router.post("/predict")
def predict(request: CustomerRequest):
    result = predict_churn(request.model_dump())
    if result.get("error"):
        raise HTTPException(status_code=422, detail=result["error"])
    return result