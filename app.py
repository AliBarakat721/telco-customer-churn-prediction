import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# 1. تحديد المسار الأساسي للمشروع
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. تحميل جميع الملفات المدربة (النموذج، الترميزات، المقياس، وأسماء الأعمدة)
loaded_model = joblib.load(os.path.join(BASE_DIR, "models", "best_model.pkl"))
encoders = joblib.load(os.path.join(BASE_DIR, "models", "encoder.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))

# 3. تهيئة تطبيق FastAPI وإضافة CORS
app = FastAPI(title="Telco Customer Churn Prediction")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # في الإنتاج حدد النطاقات المسموحة بدل *
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


# 4. صفحة الواجهة الرئيسية (UI) - تم تحديثها لتناسب Starlette/FastAPI الحديثة
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


# 5. دالة التنبؤ (مع معالجة الأخطاء وترتيب الأعمدة)
def make_prediction(input_data: dict):
    # تحويل المدخلات إلى DataFrame
    input_df = pd.DataFrame([input_data])

    # إعادة ترتيب الأعمدة بنفس ترتيب التدريب
    input_df = input_df[feature_columns]

    # تطبيق الترميز (Label Encoding) مع التقاط الأخطاء
    try:
        for column, encoder in encoders.items():
            if column in input_df.columns:
                input_df[column] = encoder.transform(input_df[column])
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"قيمة غير صالحة في العمود '{column}': {e}. تأكد من إدخال قيمة موجودة في التدريب.",
        )

    # تطبيق التطبيع (Standardization) على الأعمدة العددية
    numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

    # التنبؤ
    prediction = loaded_model.predict(input_df)[0]
    probability = float(loaded_model.predict_proba(input_df)[0, 1])

    result = "Churn" if prediction == 1 else "No Churn"
    return result, probability


# 6. هيكل البيانات القادمة من المستخدم (Pydantic Model)
class PredictionRequest(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
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
    TotalCharges: float


# 7. نقطة نهاية التنبؤ (API Endpoint)
@app.post("/predict")
async def predict(data: PredictionRequest):
    # التعامل مع إصدارات Pydantic المختلفة
    input_data = data.model_dump() if hasattr(data, "model_dump") else data.dict()

    prediction, probability = make_prediction(input_data)

    return {
        "prediction": prediction,
        "probability": probability,
    }
