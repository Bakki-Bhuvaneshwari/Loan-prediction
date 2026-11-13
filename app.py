
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import joblib

# Project path
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "classification_model.pkl"

# Load model
if not MODEL_PATH.exists():
    raise FileNotFoundError("classification_model.pkl not found!")

model = joblib.load(MODEL_PATH)

# FastAPI application
app = FastAPI(
    title="Loan Prediction",
    description="Loan Approval Prediction using Machine Learning",
    version="1.0"
)

# HTML templates
templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)

# Input features
class LoanInput(BaseModel):
    age: int
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: int
    loan_amount: int
    loan_term: int
    cibil_score: int
    residential_assets_value: int
    commercial_assets_value: int
    luxury_assets_value: int
    bank_asset_value: int
    employment_type: str
    years_employed: int
    loan_purpose: str
    property_area: str
    credit_history_years: int
    existing_loans: int
    previous_defaults: int
    monthly_expenses: int
    savings_balance: int
    monthly_emi: int
    debt_to_income_ratio: float
    interest_rate: float


# Home page
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="classification.html",
        context={}
    )


# Prediction API
@app.post("/predict")
def predict(data: LoanInput):

    try:
        # Convert input into DataFrame
        input_df = pd.DataFrame([data.model_dump()])

        # Match the model's training column order
        if hasattr(model, "feature_names_in_"):
            expected_columns = list(model.feature_names_in_)

            missing_columns = [
                col for col in expected_columns
                if col not in input_df.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"Missing model features: {missing_columns}"
                )

            input_df = input_df[expected_columns]

        print("Input DataFrame:")
        print(input_df)

        # Prediction
        prediction = model.predict(input_df)[0]

        # Convert prediction to readable status
        if prediction == 1:
            loan_status = "Approved"
        elif prediction == 0:
            loan_status = "Rejected"
        else:
            loan_status = str(prediction)

        print("Prediction:", prediction)
        print("Loan Status:", loan_status)

        return {
            "prediction": str(prediction),
            "loan_status": loan_status,
            "message": "Prediction completed successfully"
        }

    except Exception as e:
        print("Prediction Error:", str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# Health check
@app.get("/health")
def health():
    return {
        "status": "Application is running",
        "model_loaded": model is not None
    }
