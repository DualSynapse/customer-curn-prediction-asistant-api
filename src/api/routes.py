from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.app.service import predict_churn

router = APIRouter()


class CustomerRequest(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float


@router.post("/predict")
def predict(request: CustomerRequest):
    result = predict_churn(request.model_dump())
    if result.get("error"):
        raise HTTPException(status_code=422, detail=result["error"])
    return result